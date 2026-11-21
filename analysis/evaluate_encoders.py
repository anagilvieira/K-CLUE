"""Encoder comparison over matched nuclei.

Only nuclei matched to a ground-truth point are scored, so segmentation errors do not
count and the comparison isolates the representation. Reports Precision / Recall / F1 per
class, MCC, and the per-tile counting MAE, as the mean over the k-means seeds.

Reads the labels saved by main.py (nothing is re-run):

    python main.py --config configs/mist.yaml --no-whiten             # before whitening
    python analysis/evaluate_encoders.py --config configs/mist.yaml --run pca90_unwhitened

    python main.py --config configs/mist.yaml                         # after whitening
    python analysis/evaluate_encoders.py --config configs/mist.yaml --run pca90_whitened
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate, load_gt, match_dataset, score_matched
from main import get_cells, load_config, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default="pca90_whitened", help="results folder written by main.py")
    ap.add_argument("--mode", help="crop mode (default: crops.mode in the config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    mode = args.mode or cfg["crops"]["mode"]
    run_dir = paths(cfg, mode)["results"] / args.run
    files = sorted(run_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no labels in {run_dir} - run main.py first")

    cells = get_cells(cfg, mode)
    gt = load_gt(ev["gt_path"], ev["gt_format"])
    match = match_dataset(cells, gt, paths(cfg, mode)["masks"], ev["gt_canvas"],
                          ev["um_per_px"], ev["tol_um"])
    print(f"{int((match['gt_class'] >= 0).sum())} nuclei matched to a GT point")

    table = evaluate({f.stem: np.load(f) for f in files}, match, score_matched)
    table.to_csv(run_dir / "matched_nuclei.csv")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(table)
    print(f"-> {run_dir / 'matched_nuclei.csv'}")


if __name__ == "__main__":
    main()
