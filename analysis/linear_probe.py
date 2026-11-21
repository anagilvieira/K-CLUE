"""Supervised linear probe on the frozen embeddings.

Shows that Ki67 status is linearly decodable from the raw (no PCA, no whitening)
embeddings of every encoder, including those whose k-means clustering fails. Labels are
used for this analysis only; the probe is not part of K-CLUE.

The probe uses the nuclei matched to a ground-truth point (the same set the clustering is
scored on), nested cross-validation grouped by tile, and class-balanced logistic regression.

    python analysis/linear_probe.py --config configs/mist.yaml
    python analysis/linear_probe.py --config configs/mist.yaml --pc-sweep
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import DEFAULT_ENCODERS
from evaluation import load_gt, match_dataset
from main import get_cells, load_config, load_matrix, paths

C_GRID = np.logspace(-6, 3, 10)


def make_clf(C, standardize):
    lr = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",  # ~1:2.5 imbalance in MIST
    )
    steps = [("scaler", StandardScaler())] if standardize else []
    return Pipeline(steps + [("lr", lr)])


def nested_probe(X, y, groups, standardize=False, n_outer=5, n_inner=3):
    """Nested grouped CV. Returns out-of-fold decision scores and chosen Cs."""
    n_outer = min(n_outer, len(np.unique(groups)))
    outer = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan, dtype=float)
    chosen = []

    for tr, te in outer.split(X, y, groups):
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        n_in = min(n_inner, len(np.unique(gtr)))
        inner = GroupKFold(n_splits=n_in)
        inner_splits = list(inner.split(Xtr, ytr, gtr))

        best_C, best_auc = C_GRID[0], -np.inf
        for C in C_GRID:
            aucs = []
            for itr, ite in inner_splits:
                if len(np.unique(ytr[itr])) < 2 or len(np.unique(ytr[ite])) < 2:
                    continue
                clf = make_clf(C, standardize).fit(Xtr[itr], ytr[itr])
                aucs.append(roc_auc_score(ytr[ite], clf.decision_function(Xtr[ite])))
            if aucs and np.mean(aucs) > best_auc:
                best_auc, best_C = np.mean(aucs), C

        clf = make_clf(best_C, standardize).fit(Xtr, ytr)
        oof[te] = clf.decision_function(X[te])
        chosen.append(float(best_C))

    return oof, chosen


def per_image_ci(y, pred, image_id, n_boot=2000, seed=0):
    """95% CI for F1-positive, bootstrapped over IMAGES (not over seeds)."""
    images = np.unique(image_id)
    per_img = {}
    for im in images:
        m = image_id == im
        if y[m].sum() == 0:            # no positives -> F1+ undefined
            continue
        per_img[im] = f1_score(y[m], pred[m], zero_division=0)
    vals = np.array(list(per_img.values()))
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return vals.mean(), *np.percentile(boots, [2.5, 97.5])


def pc_subspace_sweep(X, y, groups, ks, standardize=False):
    """Probe restricted to the top-k principal components.

    PCA is refit inside each training fold to avoid leakage. This curve says how
    much Ki67 information lives in the leading variance directions -- i.e. the
    part of the space k-means can actually see. It is the quantitative bridge to
    Experiment 2.
    """
    rows = []
    for k in ks:
        if k > min(X.shape):
            continue
        n_outer = min(5, len(np.unique(groups)))
        outer = GroupKFold(n_splits=n_outer)
        oof = np.full(len(y), np.nan)
        for tr, te in outer.split(X, y, groups):
            pca = PCA(n_components=k, svd_solver="randomized",
                      random_state=0).fit(X[tr])
            Ztr, Zte = pca.transform(X[tr]), pca.transform(X[te])
            clf = make_clf(1.0, standardize).fit(Ztr, y[tr])
            oof[te] = clf.decision_function(Zte)
        rows.append({"k": k, "auc": roc_auc_score(y, oof),
                     "f1_pos": f1_score(y, (oof > 0).astype(int), zero_division=0)})
    return rows


def run_encoder(name, X, y, groups, standardize, do_sweep):
    oof, chosen = nested_probe(X, y, groups, standardize=standardize)
    pred = (oof > 0).astype(int)
    f1_mean, lo, hi = per_image_ci(y, pred, groups)
    row = {
        "encoder": name,
        "n_nuclei": int(len(y)),
        "dim": int(X.shape[1]),
        "pos_rate": float(y.mean()),
        "probe_auc": float(roc_auc_score(y, oof)),
        "probe_ap": float(average_precision_score(y, oof)),
        "probe_f1_pos_pooled": float(f1_score(y, pred, zero_division=0)),
        "probe_f1_pos_per_image": float(f1_mean),
        "probe_f1_ci_lo": float(lo),
        "probe_f1_ci_hi": float(hi),
        "probe_bal_acc": float(balanced_accuracy_score(y, pred)),
        "chosen_C": json.dumps(chosen),
        "standardized": standardize,
    }
    sweep = None
    if do_sweep:
        ks = [1, 2, 3, 5, 10, 20, 50, 100]
        sweep = pd.DataFrame(pc_subspace_sweep(X, y, groups, ks, standardize))
        sweep.insert(0, "encoder", name)
    return row, sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    ap.add_argument("--mode", help="crop mode (default: crops.mode in the config)")
    ap.add_argument("--standardize", action="store_true",
                    help="per-dimension z-scoring inside the CV (secondary row)")
    ap.add_argument("--pc-sweep", action="store_true",
                    help="also compute the top-k PC subspace probe curve")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    mode = args.mode or cfg["crops"]["mode"]
    cells = get_cells(cfg, mode)
    p = paths(cfg, mode)
    gt = load_gt(ev["gt_path"], ev["gt_format"])
    match = match_dataset(cells, gt, p["masks"], ev["gt_canvas"], ev["um_per_px"], ev["tol_um"])
    keep = match["gt_class"] >= 0
    y = match["gt_class"][keep]
    groups = cells["image_name"].to_numpy(str)[keep]          # grouped by tile
    print(f"{int(keep.sum())} matched nuclei on {len(np.unique(groups))} tiles, {y.mean():.1%} Ki67+")

    rows, sweeps = [], []
    for name in args.encoders:
        X = load_matrix(p["features"] / f"{name}.npy", cells)[keep].astype(np.float64)
        print(f"[probe] {name} ...", flush=True)
        row, sweep = run_encoder(name, X, y, groups, args.standardize, args.pc_sweep)
        rows.append(row)
        if sweep is not None:
            sweeps.append(sweep)
        print(f"    AUC={row['probe_auc']:.4f}  F1+={row['probe_f1_pos_pooled']:.4f}", flush=True)

    out = Path(cfg["output_dir"]) / "analysis" / mode
    out.mkdir(parents=True, exist_ok=True)
    tag = "_std" if args.standardize else ""
    df = pd.DataFrame(rows)
    df.to_csv(out / f"linear_probe{tag}.csv", index=False)
    print(df[["encoder", "dim", "probe_auc", "probe_ap", "probe_f1_pos_pooled",
              "probe_bal_acc"]].to_string(index=False))
    if sweeps:
        pd.concat(sweeps).to_csv(out / f"linear_probe{tag}_pc_sweep.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
