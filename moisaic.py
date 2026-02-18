import math
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# HEIC/HEIF support
from pillow_heif import register_heif_opener
register_heif_opener()

PHI = (1 + 5 ** 0.5) / 2  # golden ratio ~1.618

# 1/4 US Letter at 300 DPI (half width & half height of letter @300dpi)
LETTER_QUARTER_PX = (1275, 1650)  # (W, H)
DEFAULT_HUE_START_DEG = 30.0


def _rgb_to_hv_keys(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized RGB -> Hue (0..1), Value (0..1)."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 1e-10

    idx = nz & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = nz & (cmax == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = nz & (cmax == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0

    h = (h / 6.0) % 1.0
    v = cmax
    return h.astype(np.float32), v.astype(np.float32)


def _count_pixels_from_bytes(file_bytes_list: List[bytes], skip_alpha: bool) -> int:
    total = 0
    for b in file_bytes_list:
        im = Image.open(BytesIO(b)).convert("RGBA")
        arr = np.asarray(im)
        if skip_alpha:
            total += int(np.count_nonzero(arr[..., 3] > 0))
        else:
            total += arr.shape[0] * arr.shape[1]
    return total


def _choose_golden_dims_minpad(n: int) -> Tuple[int, int, int, float]:
    """
    Choose W,H with:
      - W*H >= n
      - aspect close to phi
      - padding minimized (to avoid big white bands)
    """
    h0 = int(round(math.sqrt(n / PHI)))
    best = None

    # Search window: enough to find near-optimal padding/aspect without huge compute
    for h in range(max(1, h0 - 2000), h0 + 2000):
        w = (n + h - 1) // h  # ceil(n/h) => minimal padding for this h
        area = w * h
        pad = area - n
        aspect = w / h
        cost = pad + int(1_000_000 * abs(aspect - PHI))
        if best is None or cost < best[0]:
            best = (cost, w, h, pad, aspect)

    _, w, h, pad, aspect = best
    return w, h, pad, aspect


def _load_pixels_memmap(file_bytes_list: List[bytes], total_pixels: int, skip_alpha: bool, mm_path: Path):
    pixels = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(total_pixels, 3))
    pos = 0
    for b in file_bytes_list:
        im = Image.open(BytesIO(b)).convert("RGBA")
        arr = np.asarray(im)
        if skip_alpha:
            rgb = arr[..., :3][arr[..., 3] > 0]
        else:
            rgb = arr[..., :3].reshape(-1, 3)

        n = rgb.shape[0]
        pixels[pos:pos+n] = rgb
        pos += n

    pixels.flush()
    if pos != total_pixels:
        # Rare guard; truncate view
        return pixels[:pos]
    return pixels


def _build_scores_memmap(pixels_mm: np.ndarray, hue_start_degrees: float, scores_path: Path, chunk: int = 8_000_000):
    n = pixels_mm.shape[0]
    scores = np.memmap(scores_path, dtype=np.float32, mode="w+", shape=(n,))
    h0 = (hue_start_degrees % 360.0) / 360.0

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        rgb = np.asarray(pixels_mm[start:end])
        h, v = _rgb_to_hv_keys(rgb)
        h_rot = (h - h0) % 1.0
        # “top-left -> bottom-right” progression:
        # dark first, then hue
        scores[start:end] = (1.0 - v) + h_rot

    scores.flush()
    return scores


def generate_sorted_mosaic(
    file_bytes_list: List[bytes],
    skip_alpha: bool = True,
    hue_start_degrees: float = DEFAULT_HUE_START_DEG,
) -> Image.Image:
    """
    Returns the FULL golden-ish mosaic image (can be huge),
    filled with all pixels, with minimal padding (padding is white tail).
    """
    total = _count_pixels_from_bytes(file_bytes_list, skip_alpha=skip_alpha)
    if total <= 0:
        raise ValueError("No pixels found in uploads.")

    w, h, pad, aspect = _choose_golden_dims_minpad(total)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pixels_path = td / "pixels.u8.memmap"
        scores_path = td / "scores.f32"

        pixels_mm = _load_pixels_memmap(file_bytes_list, total, skip_alpha, pixels_path)
        scores_mm = _build_scores_memmap(pixels_mm, hue_start_degrees, scores_path)

        order = np.argsort(np.asarray(scores_mm), kind="stable")

        total_cells = w * h
        out = np.empty((total_cells, 3), dtype=np.uint8)
        out[:] = 255  # white padding
        out[:pixels_mm.shape[0]] = np.asarray(pixels_mm)[order]

        img_arr = out.reshape(h, w, 3)
        return Image.fromarray(img_arr, mode="RGB")


def compose_quarter_letter_with_qr(
    mosaic_img: Image.Image,
    qr_img: Image.Image,
    target_px: Tuple[int, int] = LETTER_QUARTER_PX,
) -> Image.Image:
    """
    Create a single image sized to 1/4 Letter @300DPI (1275x1650):
      - Mosaic on the left (fit)
      - QR code on the right, with whitespace margin
    """
    W, H = target_px
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    # Layout:
    # left area: ~78% width for mosaic
    # right area: ~22% width for QR + margin
    left_w = int(W * 0.78)
    right_w = W - left_w

    # Fit mosaic into left area, preserving aspect
    mosaic_fit = mosaic_img.copy()
    mosaic_fit.thumbnail((left_w, H), Image.Resampling.LANCZOS)

    # Center mosaic vertically within left area
    mx = (left_w - mosaic_fit.size[0]) // 2
    my = (H - mosaic_fit.size[1]) // 2
    canvas.paste(mosaic_fit, (mx, my))

    # QR: make it square and comfortably sized
    # Keep margin around it
    margin = int(min(right_w, H) * 0.12)
    qr_box = min(right_w - 2 * margin, H - 2 * margin)
    qr_fit = qr_img.copy()
    qr_fit = qr_fit.resize((qr_box, qr_box), Image.Resampling.NEAREST)

    qx = left_w + (right_w - qr_box) // 2
    qy = (H - qr_box) // 2
    canvas.paste(qr_fit, (qx, qy))

    return canvas
