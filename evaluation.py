"""
Evaluation against point annotations (the only place annotations are used).

Matching:
Predicted nuclei (centroids of the instance masks) are matched to ground-truth points by
one-to-one Hungarian assignment. A pair is accepted when the centroids are at most
tol_um (3 um) apart. Distances are measured in microns so the same rule applies to
datasets scanned at different resolutions.

Two evaluations share this matching:
- end-to-end (main.py) - per class, a matched nucleus with the right class is a TP; 
with the wrong class it is a FP for the predicted class and a FN for the true class; 
an unmatched prediction is a FP and an unmatched GT point is a FN.

- matched nuclei (analysis/evaluate_encoders.py) - only nuclei matched to a GT point, 
so segmentation errors do not count; plus MCC.

Both report the per-tile counting MAE, which does not depend on the matching.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment


# ground truth --------------------------------------------------------------------------- 
def load_gt_slidescore(path):
    """MIST: SlideScore export (tab-separated, one row per image and question)."""
    pts, seen = defaultdict(list), defaultdict(set)
    ann = pd.read_csv(path, sep="\t")
    for _, row in ann.iterrows():
        name = Path(str(row["Image Name"])).stem
        q = str(row["Question"]).lower()
        cls = 1 if "positive" in q else 0 if "negative" in q else None
        if cls is None:
            raise ValueError(f"unrecognised Question: {row['Question']!r}")
        seen[name].add(cls)
        for p in json.loads(row["Answer"]):
            pts[name].append((p["x"], p["y"], cls))
    partial = sorted(n for n, s in seen.items() if len(s) < 2)
    if partial:
        print(f"WARNING: {len(partial)} image(s) annotated for one class only: {partial}")
    return {n: (np.array(v, float)[:, :2], np.array(v, float)[:, 2].astype(int))
            for n, v in pts.items()}


def load_gt_centroids(path):
    """BCData (or any dataset): CSV with columns image, x, y, label (positive / negative)."""
    df = pd.read_csv(path)
    missing = {"image", "x", "y", "label"} - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    unknown = set(df["label"].unique()) - {"positive", "negative"}
    if unknown:
        raise ValueError(f"unrecognised label(s): {sorted(unknown)}")
    df = df.assign(_stem=df["image"].map(lambda v: Path(str(v)).stem),
                   _cls=(df["label"] == "positive").astype(int))
    return {stem: (g[["x", "y"]].values.astype(float), g["_cls"].values.astype(int))
            for stem, g in df.groupby("_stem")}


def load_gt(path, fmt):
    if fmt == "slidescore":
        return load_gt_slidescore(path)
    if fmt == "centroids":
        return load_gt_centroids(path)
    raise ValueError(f"unknown gt_format: {fmt!r}")


# matching ------------------------------------------------------------------------------ 
def gt_to_pixels(gt_xy, mask_shape, gt_canvas):
    """GT points from the annotation frame to mask pixels, as (row, col)."""
    scale = gt_canvas / mask_shape[0]                       # square tiles
    return np.stack([gt_xy[:, 1] / scale, gt_xy[:, 0] / scale], axis=1)


def match_tile(gt_xy, mask_shape, pred_rc, gt_canvas, um_per_px, tol_um):
    """
    Hungarian matching for one tile. -> list of (pred index, gt index) pairs.

    pred_rc are the (row, col) centroids of the predicted nuclei. Pairs further apart
    than tol_um are discarded after solving.
    """
    if len(gt_xy) == 0 or len(pred_rc) == 0:
        return []
    rc = gt_to_pixels(gt_xy, mask_shape, gt_canvas)
    d_um = np.linalg.norm(rc[:, None, :] - pred_rc[None, :, :], axis=2) * um_per_px
    big = float(d_um.max() + tol_um) * 1e3
    gi, pj = linear_sum_assignment(np.where(d_um <= tol_um, d_um, big))
    return [(int(j), int(i)) for i, j in zip(gi, pj) if d_um[i, j] <= tol_um]


def match_dataset(cells, gt, masks_dir, gt_canvas, um_per_px, tol_um=3.0):
    """
    Match every tile once; the result is reused to score any number of predictions.

    Returns gt_class (per cell: class of its matched GT point, or -1 if unmatched),
    the classes of the GT points left unmatched, and the GT counts per tile.
    """
    cells = cells.reset_index(drop=True)
    gt_class = np.full(len(cells), -1, dtype=int)
    unmatched_gt = []
    gt_counts = {}

    for name in sorted(set(gt) | set(cells["image_name"])):
        gt_xy, gt_cls = gt.get(name, (np.empty((0, 2)), np.empty(0, int)))
        gt_counts[name] = (int((gt_cls == 1).sum()), int((gt_cls == 0).sum()))
        grp = cells.index[cells["image_name"] == name].to_numpy()
        mpath = Path(masks_dir) / f"{name}.tif"
        if not mpath.exists():                 # no segmentation: every GT point is missed
            unmatched_gt += list(gt_cls)
            continue

        shape = tifffile.imread(mpath).shape
        rc = cells.loc[grp, ["centroid_y", "centroid_x"]].to_numpy(float)
        pairs = match_tile(gt_xy, shape, rc, gt_canvas, um_per_px, tol_um)
        for j, i in pairs:
            gt_class[grp[j]] = int(gt_cls[i])
        matched_gt = {i for _, i in pairs}
        unmatched_gt += [int(gt_cls[i]) for i in range(len(gt_xy)) if i not in matched_gt]

    counts = pd.DataFrame.from_dict(gt_counts, orient="index", columns=["gt_pos", "gt_neg"])
    return dict(gt_class=gt_class, unmatched_gt=np.array(unmatched_gt, dtype=int),
                gt_counts=counts, image_name=cells["image_name"].to_numpy())


# metrics ------------------------------------------------------------------------------- 
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def mcc(tp, tn, fp, fn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def counting_mae(pred, match):
    """Per-tile counting MAE for (Ki67+, Ki67-), from all predicted nuclei (no matching)."""
    counts = match["gt_counts"]
    s = pd.Series(pred, index=match["image_name"]).groupby(level=0)
    q_pos = s.sum().reindex(counts.index).fillna(0).to_numpy(float)
    q_neg = (s.size() - s.sum()).reindex(counts.index).fillna(0).to_numpy(float)
    return (float(np.mean(np.abs(counts["gt_pos"].to_numpy(float) - q_pos))),
            float(np.mean(np.abs(counts["gt_neg"].to_numpy(float) - q_neg))))


def _confusion(pred, match):
    y, m = match["gt_class"], match["gt_class"] >= 0
    yt, yp = y[m], pred[m]
    return (int(((yt == 1) & (yp == 1)).sum()), int(((yt == 0) & (yp == 0)).sum()),
            int(((yt == 0) & (yp == 1)).sum()), int(((yt == 1) & (yp == 0)).sum()), m)


def score_end_to_end(pred, match):
    """End-to-end Precision / Recall / F1 and MAE per class for one 0/1 prediction vector."""
    pred = np.asarray(pred, dtype=int)
    tp, tn, fp, fn, m = _confusion(pred, match)
    un_pred, un_gt = pred[~m], match["unmatched_gt"]
    mae_pos, mae_neg = counting_mae(pred, match)
    rows = []
    for cls, name, (a, b, c), mae in [(1, "Ki67+", (tp, fp, fn), mae_pos),
                                      (0, "Ki67-", (tn, fn, fp), mae_neg)]:
        p, r, f = prf(a, b + int((un_pred == cls).sum()), c + int((un_gt == cls).sum()))
        rows.append(dict(Class=name, Precision=p, Recall=r, F1=f, MAE=mae))
    return rows


def score_matched(pred, match):
    """Precision / Recall / F1 per class and MCC over matched nuclei only, plus MAE."""
    pred = np.asarray(pred, dtype=int)
    tp, tn, fp, fn, _ = _confusion(pred, match)
    m_cc = mcc(tp, tn, fp, fn)
    mae_pos, mae_neg = counting_mae(pred, match)
    rows = []
    for name, (a, b, c), mae in [("Ki67+", (tp, fp, fn), mae_pos),
                                 ("Ki67-", (tn, fn, fp), mae_neg)]:
        p, r, f = prf(a, b, c)
        rows.append(dict(Class=name, Precision=p, Recall=r, F1=f, MAE=mae, MCC=m_cc))
    return rows


def evaluate(labels, match, scorer):
    """
    Mean over seeds of scorer for every feature. labels: {name: (n_seeds, N)}.

    Precision / Recall / F1 in percent.
    """
    rows = [dict(Model=name, **r)
            for name, P in labels.items() for pred in np.atleast_2d(P) for r in scorer(pred, match)]
    df = pd.DataFrame(rows).groupby(["Model", "Class"], sort=False).mean()
    for col in ("Precision", "Recall", "F1"):
        df[col] = (100 * df[col]).round(2)
    df["MAE"] = df["MAE"].round(2)
    if "MCC" in df:
        df["MCC"] = df["MCC"].round(4)
    return df
