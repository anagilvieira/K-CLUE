"""
K-CLUE: crops -> features -> PCA + whitening -> k-means -> DAB assignment -> end-to-end evaluation.

Run segment.py first. Crops and features are computed on the first run and cached in
outputs/<dataset>/, so later runs (e.g. --no-whiten) only redo the clustering.

    # K-CLUE with the seven default encoders (whitened)
    python main.py --config configs/mist.yaml

    # the same without whitening
    python main.py --config configs/mist.yaml --no-whiten

    # handcrafted colour features and the DAB + Otsu baseline
    python main.py --config configs/mist.yaml --features hed hsv_clean lab hed_hsv_lab_clean dab_otsu

Clustering
----------
1. PCA, keeping the leading components that reach var_cut (90%) of the variance.
2. Whitening (default): each component is rescaled to unit variance. 
    --no-whiten leaves the PCA components unchanged. 
    --alpha divides each component by std ** alpha instead (partial whitening).
3. k-means (k = 2) on the pooled nuclei of all tiles: one global decision boundary.
   Run once per seed, n_init restarts each.
4. The cluster with the higher mean DAB optical density becomes Ki67-positive.
No annotation is used before the evaluation.

Results go to outputs/<dataset>/results/<mode>/pca90_whitened/ (or pca90_unwhitened, pca90_alpha0.5, ...):
  <feature>.npy     Ki67 labels (1 = positive), one row per seed, aligned with cells.csv
  predictions.csv   centroid and label of every nucleus (first seed)
  end_to_end.csv    end-to-end Precision / Recall / F1 (%) and counting MAE per class, mean over seeds
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu

from crops import MODES, load_cells, read_rgb, run_crops
from embeddings import DEFAULT_ENCODERS, extract_embeddings
from evaluation import evaluate, load_gt, match_dataset, score_end_to_end
from handcrafted import FEATURE_SETS, OTSU_SET, extract_handcrafted, feature_set

MAX_COMPONENTS = 500


# config and caching ---------------------------------------------------------------------- 
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def paths(cfg, mode):
    root = Path(cfg["output_dir"])
    return dict(masks=root / "masks", crops=root / "crops" / mode,
                features=root / "features" / mode, results=root / "results" / mode)


def save_matrix(path, matrix, keys):
    """Save an (N, D) matrix next to the cell keys it was built in."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))
    np.save(path.with_name(path.stem + "_keys.npy"), np.asarray(keys, dtype=object))


def load_matrix(path, cells):
    """Load a matrix saved with save_matrix, rows reordered to match cells."""
    path = Path(path)
    emb = np.load(path)
    keys = [str(k) for k in np.load(path.with_name(path.stem + "_keys.npy"), allow_pickle=True)]
    pos = {k: i for i, k in enumerate(keys)}
    missing = [k for k in cells["crop_key"] if k not in pos]
    if len(keys) != len(emb) or missing:
        raise ValueError(f"{path.name} does not match cells.csv - delete it and re-run")
    return emb[[pos[k] for k in cells["crop_key"]]]


def get_cells(cfg, mode):
    """The cell table; runs the crop step if it has not been run for this mode."""
    p = paths(cfg, mode)
    if (p["crops"] / "cells.csv").exists():
        return load_cells(p["crops"] / "cells.csv")
    if not any(p["masks"].glob("*.tif")):
        raise FileNotFoundError(f"no masks in {p['masks']} - run segment.py first")
    c = cfg["crops"]
    return run_crops(cfg["tiles_dir"], p["masks"], p["crops"], mode=mode, margin=c["margin"],
                     size=c["size"], ext=cfg["image_ext"], mask_dilate=c["mask_dilate"])


def get_embedding(cfg, mode, cells, key, device=None):
    """Embeddings of one encoder; extracted on first use."""
    npy = paths(cfg, mode)["features"] / f"{key}.npy"
    if not npy.exists():
        print(f"[{key}] extracting embeddings ...")
        crops = np.stack([read_rgb(p) for p in cells["crop_path"]])
        enc = cfg["encoders"]
        emb = extract_embeddings(crops, key, device=device, batch_size=enc["batch_size"],
                                 hovernext_ckpt=enc.get("hovernext_ckpt"))
        save_matrix(npy, emb, cells["crop_key"])
    return load_matrix(npy, cells)


def get_handcrafted(cfg, mode, cells):
    """All handcrafted colour statistics; measured on first use."""
    npy = paths(cfg, mode)["features"] / "handcrafted.npy"
    if not npy.exists():
        X = extract_handcrafted(cells, cfg["tiles_dir"], paths(cfg, mode)["masks"],
                                ext=cfg["image_ext"])
        save_matrix(npy, X, cells["crop_key"])
    return load_matrix(npy, cells)


# clustering ------------------------------------------------------------------------------ 
def clustering_space(X, var_cut=0.90, alpha=1.0, standardize=False,
                     max_components=MAX_COMPONENTS):
    """PCA + whitening. standardize drops constant columns and z-scores (handcrafted)."""
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        X = X[:, X.std(axis=0) > 1e-12]
        X = StandardScaler().fit_transform(X)

    n_max = min(max_components, *X.shape)
    pca = PCA(n_components=n_max, random_state=0).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, var_cut) + 1)
    if n > n_max:
        raise RuntimeError(f"{var_cut:.2f} of the variance needs more than {n_max} components")

    Z = pca.transform(X)[:, :n]
    if alpha > 0:
        Z = Z / (np.maximum(pca.explained_variance_[:n], 1e-12) ** (alpha / 2.0))
    return Z


def assign_by_dab(labels, dab):
    """Cluster ids are arbitrary: the cluster with the higher mean DAB becomes Ki67+ (1)."""
    means = {c: float(dab[labels == c].mean()) for c in np.unique(labels)}
    pos = max(means, key=means.get)
    return (labels == pos).astype(int)


def cluster_seeds(Z, dab, seeds=(0, 1, 2, 3, 4), k=2, n_init=20):
    """k-means once per seed. -> Ki67 labels, shape (n_seeds, N)."""
    return np.vstack([
        assign_by_dab(KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(Z).labels_, dab)
        for s in seeds])


# main ------------------------------------------------------------------------------------ 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", nargs="+", default=DEFAULT_ENCODERS,
                    help=f"encoders {DEFAULT_ENCODERS} and/or handcrafted sets "
                         f"{list(FEATURE_SETS) + [OTSU_SET]}")
    ap.add_argument("--mode", choices=MODES, help="crop mode (default: crops.mode in the config)")
    ap.add_argument("--no-whiten", action="store_true", help="cluster the PCA components without whitening")
    ap.add_argument("--alpha", type=float,
                    help="partial whitening exponent in [0, 1]; overrides --no-whiten")
    ap.add_argument("--var-cut", type=float, help="PCA variance kept (default: config)")
    ap.add_argument("--device", help="cuda / cpu for the encoders (default: cuda if available)")
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cc = cfg["clustering"]
    mode = args.mode or cfg["crops"]["mode"]
    alpha = args.alpha if args.alpha is not None else (0.0 if args.no_whiten else 1.0)
    var_cut = cc["var_cut"] if args.var_cut is None else args.var_cut

    cells = get_cells(cfg, mode)
    dab = cells["dab_score"].to_numpy(float)
    space = {1.0: "whitened", 0.0: "unwhitened"}.get(alpha, f"alpha{alpha:g}")
    out = paths(cfg, mode)["results"] / f"pca{int(round(var_cut * 100))}_{space}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} nuclei | mode={mode} | PCA {var_cut:.0%} | {space}")

    labels = {}
    for name in args.features:
        if name == OTSU_SET:
            P = np.tile((dab > threshold_otsu(dab)).astype(int), (len(cc["seeds"]), 1))
        else:
            if name in FEATURE_SETS:
                X, _ = feature_set(get_handcrafted(cfg, mode, cells), name)
            else:
                X = get_embedding(cfg, mode, cells, name, device=args.device)
            Z = clustering_space(X, var_cut=var_cut, alpha=alpha, standardize=name in FEATURE_SETS)
            P = cluster_seeds(Z, dab, seeds=cc["seeds"], k=cc["k"], n_init=cc["n_init"])
            print(f"[{name}] clustered on {Z.shape[1]} principal components")
        labels[name] = P
        np.save(out / f"{name}.npy", P.astype(np.int8))

    preds = cells[["image_name", "cell_label", "centroid_x", "centroid_y"]].rename(
        columns={"centroid_x": "x", "centroid_y": "y"})
    for name, P in labels.items():
        preds[name] = np.where(P[0] == 1, "positive", "negative")
    preds.to_csv(out / "predictions.csv", index=False, float_format="%.2f")

    if args.no_eval:
        print(f"-> {out}")
        return

    ev = cfg["evaluation"]
    gt = load_gt(ev["gt_path"], ev["gt_format"])
    match = match_dataset(cells, gt, paths(cfg, mode)["masks"], ev["gt_canvas"],
                          ev["um_per_px"], ev["tol_um"])
    table = evaluate(labels, match, score_end_to_end)
    table.to_csv(out / "end_to_end.csv")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print("\nEnd-to-end evaluation (mean over seeds)")
        print(table)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
