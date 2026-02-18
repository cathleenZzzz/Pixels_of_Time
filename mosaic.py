import math
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()

PHI = (1 + 5 ** 0.5) / 2  # ~1.618

# 1/4 US Letter @300DPI
LETTER_QUARTER_PX = (1275, 1650)  # (W,H)
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
            total += int(arr.shape[0] * arr.shape[1])
    return total


def _choose_golden_dims_minpad(n: int) -> Tuple[int, int, int, float]:
    """
    Choose W,H:
      - W*H >= n
      - aspect close to phi
      - padding minimized (reduces visible white bands)
    """
    h0 = int(round(math.sqrt(n / PHI)))
    best = None

    # Search around ideal height
    for h in range(max(1, h0 - 2000), h0 + 2000):
        w = (n + h - 1) // h  # ceil(n/h)
        area = w * h
        pad = area - n
        aspect = w / h
        cost = pad + int(1_000_000 * abs(aspect - PHI))
        if best is None or cost < best[0]:
            best = (cost, w, h, pad, aspect)

    _, w, h, pad, aspect = best
    return w, h, pad, aspect


def _sample_pixels_fast_mode(
    file_bytes_list: List[bytes],
    total_pixels: int,
    max_pixels: int,
    skip_alpha: bool,
    seed: int = 0,
) -> np.ndarray:
    """
    Fast-mode sampling without saving uploads:
      - compute sampling probability p = max_pixels / total_pixels
      - for each image, draw k ~ Binomial(n_i, p)
      - sample k pixels from that image
    """
    if total_pixels <= max_pixels:
        # no need to sample
        all_pixels = []
        for b in file_bytes_list:
            im = Image.open(BytesIO(b)).convert("RGBA")
            arr = np.asarray(im)
            rgb = arr[..., :3][arr[..., 3] > 0] if skip_alpha else arr[..., :3].reshape(-1, 3)
            if rgb.size:
                all_pixels.append(rgb)
        return np.vstack(all_pixels) if all_pixels else np.zeros((0, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    p = max_pixels / float(total_pixels)

    chunks = []
    used = 0

    for b in file_bytes_list:
        im = Image.open(BytesIO(b)).convert("RGBA")
        arr = np.asarray(im)
        rgb = arr[..., :3][arr[..., 3] > 0] if skip_alpha else arr[..., :3].reshape(-1, 3)
        n = rgb.shape[0]
        if n == 0:
            continue

        # how many from this image?
        k = int(rng.binomial(n, p))
        if k <= 0:
            continue

        # avoid going over hard cap by too much
        remaining = max_pixels - used
        if remaining <= 0:
            break
        if k > remaining:
            k = remaining

        idx = rng.choice(n, size=k, replace=False)
        chunks.append(rgb[idx])
        used += k

        if used >= max_pixels:
            break

    if not chunks:
        return np.zeros((0, 3), dtype=np.uint8)

    pixels = np.vstack(chunks)

    # If we're slightly under the cap due to binomial variance, that's fine.
    return pixels


def _load_pixels_memmap(file_bytes_list: List[bytes], total_pixels: int, skip_alpha: bool, mm_path: Path):
    pixels = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(total_pixels, 3))
    pos = 0

    for b in file_bytes_list:
        im = Image.open(BytesIO(b)).convert("RGBA")
        arr = np.asarray(im)
        rgb = arr[..., :3][arr[..., 3] > 0] if skip_alpha else arr[..., :3].reshape(-1, 3)

        n = rgb.shape[0]
        pixels[pos:pos+n] = rgb
        pos += n

    pixels.flush()
    if pos != total_pixels:
        return pixels[:pos]
    return pixels


def _build_scores(pixels: np.ndarray, hue_start_degrees: float) -> np.ndarray:
    h0 = (hue_start_degrees % 360.0) / 360.0
    h, v = _rgb_to_hv_keys(pixels)
    h_rot = (h - h0) % 1.0
    # “top-left -> bottom-right” progression:
    # dark first, then hue
    return (1.0 - v) + h_rot


def generate_sorted_mosaic(
    file_bytes_list: List[bytes],
    skip_alpha: bool = True,
    hue_start_degrees: float = DEFAULT_HUE_START_DEG,
    full_mode_cap_pixels: int = 120_000_000,
    fast_mode: bool = False,
    fast_mode_sample_pixels: int = 15_000_000,
) -> Tuple[Image.Image, Dict]:
    """
    Returns: (mosaic_img, meta)
    - Full mode uses ALL pixels (disk-backed temp memmaps) BUT enforces a safety cap.
    - Fast mode samples up to fast_mode_sample_pixels and sorts those.

    Padding cells (if any) are white at the end.
    """
    input_pixels = _count_pixels_from_bytes(file_bytes_list, skip_alpha=skip_alpha)
    if input_pixels <= 0:
        raise ValueError("No pixels found in uploads.")

    # Safety: force fast mode if too large, or tell user to enable it
    if (not fast_mode) and input_pixels > full_mode_cap_pixels:
        raise ValueError(
            f"Too many pixels for Full Mode ({input_pixels:,} pixels). "
            f"Please enable Fast Mode (samples up to {fast_mode_sample_pixels:,} pixels), "
            f"or upload fewer/smaller photos."
        )

    if fast_mode:
        pixels = _sample_pixels_fast_mode(
            file_bytes_list=file_bytes_list,
            total_pixels=input_pixels,
            max_pixels=fast_mode_sample_pixels,
            skip_alpha=skip_alpha,
            seed=0,
        )
        used_pixels = int(pixels.shape[0])
        if used_pixels <= 0:
            raise ValueError("Fast Mode sampling produced no pixels. Try different photos or disable skip-alpha.")
        w, h, pad, aspect = _choose_golden_dims_minpad(used_pixels)

        scores = _build_scores(pixels, hue_start_degrees)
        order = np.argsort(scores, kind="stable")

        total_cells = w * h
        out = np.empty((total_cells, 3), dtype=np.uint8)
        out[:] = 255
        out[:used_pixels] = pixels[order]
        img_arr = out.reshape(h, w, 3)
        meta = {"mode": "fast", "input_pixels": int(input_pixels), "used_pixels": used_pixels}
        return Image.fromarray(img_arr, mode="RGB"), meta

    # Full mode: all pixels via temp memmap; deleted automatically after request
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pixels_path = td / "pixels.u8.memmap"

        pixels_mm = _load_pixels_memmap(file_bytes_list, input_pixels, skip_alpha, pixels_path)
        used_pixels = int(pixels_mm.shape[0])

        w, h, pad, aspect = _choose_golden_dims_minpad(used_pixels)

        # Scores computed in RAM in chunks would be nicer; for simplicity we build once.
        # Since full mode is capped, this is acceptable.
        scores = _build_scores(np.asarray(pixels_mm), hue_start_degrees)
        order = np.argsort(scores, kind="stable")

        total_cells = w * h
        out = np.empty((total_cells, 3), dtype=np.uint8)
        out[:] = 255
        out[:used_pixels] = np.asarray(pixels_mm)[order]
        img_arr = out.reshape(h, w, 3)

        meta = {"mode": "full", "input_pixels": int(input_pixels), "used_pixels": used_pixels}
        return Image.fromarray(img_arr, mode="RGB"), meta


def compose_quarter_letter_with_qr(
    mosaic_img: Image.Image,
    qr_img: Image.Image,
    target_px: Tuple[int, int] = LETTER_QUARTER_PX,
) -> Image.Image:
    """
    Output is EXACTLY 1/4 Letter @300DPI: 1275×1650 px.
    Layout:
      - left ~78%: mosaic (fit)
      - right ~22%: QR (fit) centered
    """
    W, H = target_px
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    left_w = int(W * 0.78)
    right_w = W - left_w

    mosaic_fit = mosaic_img.copy()
    mosaic_fit.thumbnail((left_w, H), Image.Resampling.LANCZOS)
    mx = (left_w - mosaic_fit.size[0]) // 2
    my = (H - mosaic_fit.size[1]) // 2
    canvas.paste(mosaic_fit, (mx, my))

    margin = int(min(right_w, H) * 0.12)
    qr_box = min(right_w - 2 * margin, H - 2 * margin)
    qr_fit = qr_img.copy().resize((qr_box, qr_box), Image.Resampling.NEAREST)
    qx = left_w + (right_w - qr_box) // 2
    qy = (H - qr_box) // 2
    canvas.paste(qr_fit, (qx, qy))

    return canvas
