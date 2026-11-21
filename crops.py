"""
One crop per segmented nucleus, plus its DAB / haematoxylin optical density.

Called at the start of main.py; the result is cached in
outputs/<dataset>/crops/<mode>/ (crops/*.png and cells.csv).

Crop modes
----------
All fixed-size modes use the same square box for every cell in the dataset:
S = max(bbox side over the whole dataset) + 2 * margin, rounded up to even
(64 x 64 px on MIST). Each cell keeps its true size inside that box.

mode            | box placement                    | pixels that are not the target cell
--------------- | -------------------------------- | ---------------------------------------
isolated_median | centred on the cell              | global median background colour 
isolated_white  | centred on the cell              | white
real            | centred, clamped inside the tile | kept as photographed
centered        | centred, tile mirror-padded      | kept as photographed, mirrored at edges

isolated_median is the default (crops.mode in the config). 

The global median background is the median over every background pixel (mask == 0) of
every tile, so the fill carries no information about which tile a cell came from.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from natsort import natsorted
from skimage.color import rgb2hed
from skimage.io import imread, imsave
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm


MODES = ("isolated_median", "isolated_white", "real", "centered")
WHITE = 255


def read_rgb(path):
    """RGB uint8 (H, W, 3), whatever the file is stored as (greyscale, RGBA, ...)."""
    a = imread(path)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.shape[2] > 3:
        a = a[:, :, :3]
    if a.shape[2] != 3:
        raise ValueError(f"{path}: unexpected shape {a.shape}")
    return a


def canonical_key(path):
    """
    .../100M2004069_15_28_r1_c0_cell_0007.png -> 100M2004069_15_28_r1_c0_cell_7

    Every nucleus is identified by this key, so cached feature matrices are realigned
    to the cell table by key and never by row position.
    """
    s = str(path).replace("\\", "/").split("/")[-1].lower()
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    marker = "_cell_"
    i = s.rfind(marker)
    if i != -1 and s[i + len(marker):].isdigit():
        return f"{s[:i]}{marker}{int(s[i + len(marker):])}"
    return s


def load_cells(csv_path):
    """Read a cells.csv and add the crop_key column."""
    cells = pd.read_csv(csv_path)
    cells["image_name"] = cells["image_name"].astype(str)
    cells["crop_key"] = cells["crop_path"].map(canonical_key)
    if not cells["crop_key"].is_unique:
        raise ValueError("crop_key collision - two cells share a filename")
    return cells


def list_masks(masks_dir):
    files = natsorted(Path(masks_dir).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no .tif masks in {masks_dir} - run step 1 first")
    return files


def read_tile(tiles_dir, stem, ext=".png"):
    return read_rgb(Path(tiles_dir) / f"{stem}{ext}")


def crop_size(mask_files, margin=1, fixed=None):
    """Side of the square box: the largest nucleus in the dataset plus margin per side."""
    if fixed is not None:
        return int(fixed)
    largest = 0
    for f in mask_files:
        for p in regionprops(tifffile.imread(f)):
            y0, x0, y1, x1 = p.bbox
            largest = max(largest, y1 - y0, x1 - x0)
    s = int(largest) + 2 * margin
    return s + s % 2


def global_background(mask_files, tiles_dir, ext=".png", subsample=17):
    """Median RGB over the background pixels (mask == 0) of every tile."""
    px = []
    for f in mask_files:
        tile = read_tile(tiles_dir, f.stem, ext)
        bg = tile[tifffile.imread(f) == 0]
        if len(bg):
            px.append(bg[::subsample])
    return np.median(np.concatenate(px), axis=0).astype(np.uint8)


def stain_scores(rgb_pixels):
    """Mean haematoxylin and DAB optical density of a set of RGB uint8 pixels (N, 3)."""
    if len(rgb_pixels) == 0:
        return np.nan, np.nan
    hed = rgb2hed(np.clip(rgb_pixels.reshape(-1, 1, 3) / 255.0, 1e-6, 1.0)).reshape(-1, 3)
    return float(hed[:, 0].mean()), float(hed[:, 2].mean())


def _fill_colour(mode, global_bg):
    if mode == "isolated_median":
        return global_bg.astype(np.uint8)
    if mode == "isolated_white":
        return np.full(3, WHITE, dtype=np.uint8)
    return None   # 'real' / 'centered': surroundings kept


def _pad(tile, mask, pad, bg):
    """
    Mirror-pad when the surroundings are kept, constant-pad when the cell is isolated.

    Constant padding matters for isolated crops: mirroring would copy the cell's own label
    across the tile edge and paste a reflection of it back into the crop.
    """
    if bg is None:
        return (np.pad(tile, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"),
                np.pad(mask, ((pad, pad), (pad, pad)), mode="reflect"))
    tile_p = np.empty((tile.shape[0] + 2 * pad, tile.shape[1] + 2 * pad, 3), tile.dtype)
    tile_p[:] = bg
    tile_p[pad:pad + tile.shape[0], pad:pad + tile.shape[1]] = tile
    mask_p = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), mask.dtype)
    mask_p[pad:pad + mask.shape[0], pad:pad + mask.shape[1]] = mask
    return tile_p, mask_p


def extract_crops(mask_files, tiles_dir, out_dir, mode="isolated_median", S=None,
                  global_bg=None, ext=".png", mask_dilate=0):
    """
    Write one PNG per nucleus to out_dir/crops and return the cell table.

    The table (also saved as out_dir/cells.csv) has one row per nucleus with its tile,
    instance label, crop geometry, morphology, and the mean DAB / haematoxylin optical
    density measured inside its own (undilated) mask on the original tile pixels.
    """
    if mode not in MODES:
        raise ValueError(f"unknown crop mode {mode!r}; choose one of {MODES}")
    if S is None:
        raise ValueError("S (crop size) is required")
    if global_bg is None and mode == "isolated_median":
        raise ValueError("isolated_median needs the global background colour")

    out_dir = Path(out_dir)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    bg = _fill_colour(mode, global_bg)
    isolated = bg is not None
    pad = S
    records = []

    for f in tqdm(mask_files, desc=f"crops [{mode}]"):
        stem = f.stem
        tile = read_tile(tiles_dir, stem, ext)
        mask = tifffile.imread(f)
        H, W = mask.shape
        if tile.shape[:2] != (H, W):
            raise ValueError(f"{stem}: tile {tile.shape[:2]} != mask {(H, W)}")
        if S > H or S > W:
            raise ValueError(f"{stem}: crop size {S} larger than tile {(H, W)}")
        tile_p, mask_p = _pad(tile, mask, pad, bg)

        for p in regionprops(mask):
            y0, x0, y1, x1 = p.bbox
            h, w = y1 - y0, x1 - x0
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0

            iy0, ix0 = int(round(cy - S / 2.0)), int(round(cx - S / 2.0))
            if mode == "real":                       # keep the box inside the tile
                ny0, nx0 = min(max(iy0, 0), H - S), min(max(ix0, 0), W - S)
            else:
                ny0, nx0 = iy0, ix0
            raw = tile_p[ny0 + pad:ny0 + pad + S, nx0 + pad:nx0 + pad + S]
            win = mask_p[ny0 + pad:ny0 + pad + S, nx0 + pad:nx0 + pad + S]

            sel = win == p.label
            if mask_dilate:
                sel = binary_dilation(sel, disk(mask_dilate))
            crop = raw if bg is None else np.where(sel[:, :, None], raw, bg).astype(raw.dtype)

            name = f"{stem}_cell_{p.label:04d}.png"
            imsave(crop_dir / name, crop, check_contrast=False)

            hema, dab = stain_scores(tile[mask == p.label])
            records.append(dict(
                image_name=stem,
                cell_label=int(p.label),
                crop_path=str(crop_dir / name),
                crop_mode=mode,
                crop_y0=int(ny0), crop_x0=int(nx0),
                crop_size=int(S),
                bbox_center_y=int(round(cy)), bbox_center_x=int(round(cx)),
                centroid_y=float(p.centroid[0]), centroid_x=float(p.centroid[1]),
                bbox_h=int(h), bbox_w=int(w),
                cell_size=int(p.area),
                equiv_diameter=float(getattr(p, "equivalent_diameter_area", None) or p.equivalent_diameter),
                eccentricity=float(p.eccentricity),
                solidity=float(p.solidity),
                is_border=bool(y0 == 0 or x0 == 0 or y1 == H or x1 == W),
                was_clamped=bool((ny0, nx0) != (iy0, ix0)),
                target_frac=float(sel.mean()),
                n_neighbours=0 if isolated else int((np.unique(win) > 0).sum() - 1),
                dab_score=dab,
                hema_score=hema,
            ))

    cells = pd.DataFrame.from_records(records)
    cells.to_csv(out_dir / "cells.csv", index=False)
    return cells


def run_crops(tiles_dir, masks_dir, out_dir, mode="isolated_median", margin=1, size=None,
              ext=".png", mask_dilate=0):
    """Whole step: crop size, global background, crops. -> cell table (with crop_key)."""
    mask_files = list_masks(masks_dir)
    S = crop_size(mask_files, margin=margin, fixed=size)
    bg = global_background(mask_files, tiles_dir, ext)
    print(f"{len(mask_files)} masks | crop size {S} x {S} | "
          f"global background {tuple(int(v) for v in bg)}")
    extract_crops(mask_files, tiles_dir, out_dir, mode=mode, S=S, global_bg=bg, ext=ext,
                  mask_dilate=mask_dilate)
    return load_cells(Path(out_dir) / "cells.csv")
