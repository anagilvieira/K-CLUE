"""
Features option 2: handcrafted colour descriptors.

Every statistic is computed over the pixels inside one nucleus' own instance mask, read
directly from the original tile, so neighbouring nuclei and background never contribute.

Each nucleus is described in three colour spaces (skimage conversions):

hed - haematoxylin, eosin, DAB optical density (Ruifrok & Johnston deconvolution)
hsv - hue (treated as a circular quantity), saturation, value
lab - CIE L*a*b*

Each linear channel is summarised by mean, sd, p10, p50 and p90. Hue gets a circular
mean, circular sd, and the mean cos / sin of the angle (k-means cannot use a wrapped
coordinate). The plain arithmetic mean of hue is also measured (hsv_hue_arithmean) but
excluded from the *_clean sets.

Feature sets
------------
hed, hsv, lab, hed_hsv_lab     - one colour space, or all three concatenated
hsv_clean, hed_hsv_lab_clean   - the same without hsv_hue_arithmean
dab_otsu                       - mean DAB only, split by a global Otsu threshold
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.color import rgb2hed, rgb2hsv, rgb2lab
from tqdm import tqdm

from crops import read_rgb

LIN_STATS = ("mean", "sd", "p10", "p50", "p90")
HUE_STATS = ("circmean", "circsd", "cos", "sin", "arithmean")

CHANNELS = {
    "hed": [("hema", False), ("eosin", False), ("dab", False)],
    "hsv": [("hue", True), ("sat", False), ("val", False)],
    "lab": [("L", False), ("a", False), ("b", False)],
}
GROUPS = list(CHANNELS)
GROUP_COLUMNS = {g: [f"{g}_{c}_{s}" for c, circ in ch for s in (HUE_STATS if circ else LIN_STATS)]
                 for g, ch in CHANNELS.items()}
ALL_COLUMNS = [c for g in GROUPS for c in GROUP_COLUMNS[g]]
DROP_COLUMNS = {"hsv_hue_arithmean"}

FEATURE_SETS = {
    "hed": ["hed"],
    "hsv": ["hsv"],
    "lab": ["lab"],
    "hed_hsv_lab": ["hed", "hsv", "lab"],
    "hsv_clean": ["hsv"],
    "hed_hsv_lab_clean": ["hed", "hsv", "lab"],
}
OTSU_SET = "dab_otsu"


def _lin_stats(v):
    if v.size == 0:
        return [0.0] * len(LIN_STATS)
    return [float(v.mean()), float(v.std()), *(float(x) for x in np.percentile(v, (10, 50, 90)))]


def _hue_stats(h):
    """Directional statistics for hue in [0, 1], where 0 and 1 are the same colour."""
    if h.size == 0:
        return [0.0] * len(HUE_STATS)
    a = h * 2.0 * np.pi
    cos_m, sin_m = float(np.cos(a).mean()), float(np.sin(a).mean())
    R = np.sqrt(cos_m ** 2 + sin_m ** 2)
    circmean = float(np.arctan2(sin_m, cos_m) / (2.0 * np.pi) % 1.0)
    circsd = float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-8, 1.0))) / (2.0 * np.pi))
    return [circmean, circsd, cos_m, sin_m, float(h.mean())]


def _colour_spaces(rgb_pixels_u8):
    """(N, 3) uint8 -> {group: (N, 3) float}."""
    rgb = np.clip(rgb_pixels_u8.reshape(-1, 1, 3) / 255.0, 0.0, 1.0)
    return {"hed": rgb2hed(np.clip(rgb, 1e-6, 1.0)).reshape(-1, 3),
            "hsv": rgb2hsv(rgb).reshape(-1, 3),
            "lab": rgb2lab(rgb).reshape(-1, 3)}


def cell_features(rgb_pixels_u8):
    """All ALL_COLUMNS statistics for the pixels of one nucleus."""
    spaces = _colour_spaces(rgb_pixels_u8)
    row = []
    for g, chans in CHANNELS.items():
        for k, (_, circ) in enumerate(chans):
            v = spaces[g][:, k]
            row += _hue_stats(v) if circ else _lin_stats(v)
    return row


def extract_handcrafted(cells, tiles_dir, masks_dir, ext=".png"):
    """(N, len(ALL_COLUMNS)) feature matrix, one row per row of cells."""
    by_tile = defaultdict(list)
    for i, (name, label) in enumerate(zip(cells["image_name"], cells["cell_label"])):
        by_tile[str(name)].append((i, int(label)))

    X = np.zeros((len(cells), len(ALL_COLUMNS)), dtype=np.float64)
    for name, rows in tqdm(by_tile.items(), desc="handcrafted"):
        tile = read_rgb(Path(tiles_dir) / f"{name}{ext}")
        mask = tifffile.imread(Path(masks_dir) / f"{name}.tif")
        for i, label in rows:
            X[i] = cell_features(tile[mask == label])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feature_set(X, name):
    """Columns of the full handcrafted matrix that make up one feature set."""
    groups = FEATURE_SETS[name]
    cols = [c for g in groups for c in GROUP_COLUMNS[g]]
    if name.endswith("_clean"):
        cols = [c for c in cols if c not in DROP_COLUMNS]
    idx = [ALL_COLUMNS.index(c) for c in cols]
    return X[:, idx], cols
