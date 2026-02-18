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
LETTER_QUARTER_PX = (1500, 1159)  # (W,H)
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
    max_pixels: int,
    skip_alpha: bool,
    hue_bins: int = 36,
    val_bins: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """
    Representative sampling:
      - Spatially uniform sampling (grid stride) per image
      - Then stratify by HSV (hue x value) to preserve color/brightness distribution
    """
    rng = np.random.default_rng(seed)
    n_files = len(file_bytes_list)
    if n_files == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    # Budget per image (roughly even)
    per_img_budget = max(10_000, max_pixels // n_files)

    all_chunks = []
    used = 0

    for b in file_bytes_list:
        if used >= max_pixels:
            break

        im = Image.open(BytesIO(b)).convert("RGBA")
        arr = np.asarray(im)  # HxWx4 uint8

        # Optional alpha filter
        if skip_alpha:
            mask = arr[..., 3] > 0
            rgb_full = arr[..., :3][mask]
            if rgb_full.size == 0:
                continue
            # Spatial information is lost after masking; for photos this is usually fine.
            rgb = rgb_full
        else:
            # Keep spatial structure for stride sampling
            rgb = arr[..., :3].reshape(-1, 3)

        n = rgb.shape[0]
        if n == 0:
            continue

        # Step 1: spatial-ish uniform sampling via stride
        # Choose a stride so we get a manageable candidate pool.
        # Candidate pool capped around ~4x per-image budget.
        target_candidates = min(n, per_img_budget * 4)
        stride = max(1, n // target_candidates)
        candidates = rgb[::stride]
        if candidates.shape[0] > target_candidates:
            candidates = candidates[:target_candidates]

        # Step 2: stratify by HSV bins to preserve distribution
        h, v = _rgb_to_hv_keys(candidates)
        hbin = np.clip((h * (hue_bins - 1)).astype(np.int32), 0, hue_bins - 1)
        vbin = np.clip((v * (val_bins - 1)).astype(np.int32), 0, val_bins - 1)
        bin_id = vbin * hue_bins + hbin
        nbins = hue_bins * val_bins

        # How many to keep from each bin
        img_budget = min(per_img_budget, max_pixels - used)
        per_bin = max(1, img_budget // nbins)

        chosen_idx = []
        for bid in range(nbins):
            idxs = np.where(bin_id == bid)[0]
            if idxs.size == 0:
                continue
            k = min(per_bin, idxs.size)
            pick = rng.choice(idxs, size=k, replace=False)
            chosen_idx.append(pick)

        if chosen_idx:
            chosen_idx = np.concatenate(chosen_idx)
        else:
            # fallback: random sample
            k = min(img_budget, candidates.shape[0])
            chosen_idx = rng.choice(candidates.shape[0], size=k, replace=False)

        # If we’re under budget (bins were sparse), top up randomly from candidates
        if chosen_idx.size < img_budget and candidates.shape[0] > chosen_idx.size:
            remaining = img_budget - chosen_idx.size
            pool = np.setdiff1d(np.arange(candidates.shape[0]), chosen_idx, assume_unique=False)
            if pool.size > 0:
                extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
                chosen_idx = np.concatenate([chosen_idx, extra])

        chunk = candidates[chosen_idx]
        all_chunks.append(chunk)
        used += chunk.shape[0]

        # free big arrays ASAP
        del arr, rgb, candidates

    if not all_chunks:
        return np.zeros((0, 3), dtype=np.uint8)

    pixels = np.vstack(all_chunks)

    # Hard cap, just in case
    if pixels.shape[0] > max_pixels:
        idx = rng.choice(pixels.shape[0], size=max_pixels, replace=False)
        pixels = pixels[idx]

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
