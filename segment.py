"""
Nuclei instance segmentation with Cellpose (Cellpose-DINOv3 ViT-B by default).

Saves one <tile>.tif instance mask per tile in outputs/<dataset>/masks/

    python segment.py --config configs/mist.yaml [--overlays]
"""

import argparse
from pathlib import Path

import yaml
from natsort import natsorted
from tqdm import tqdm


def save_overlay(img, masks, path):
    """Tile with the predicted nuclear outlines drawn in red (visual QC only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cellpose import utils

    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    for o in utils.outlines_list(masks):
        ax.plot(o[:, 0], o[:, 1], color="red", linewidth=1.5)
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def segment_tiles(tiles_dir, masks_dir, model="cpdino-vitb", diameter=None,
                  flow_threshold=0.4, cellprob_threshold=0.0, tile_norm_blocksize=0,
                  ext=".png", overlays_dir=None):
    from cellpose import core, io, models

    files = natsorted(f for f in Path(tiles_dir).glob(f"*{ext}")
                      if "_masks" not in f.name and "_flows" not in f.name)
    if not files:
        raise FileNotFoundError(f"no '*{ext}' tiles in {tiles_dir}")
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if overlays_dir is not None:
        Path(overlays_dir).mkdir(parents=True, exist_ok=True)

    print(f"Cellpose model: {model} | {len(files)} tiles")
    net = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=model)
    normalize = {"tile_norm_blocksize": tile_norm_blocksize} if tile_norm_blocksize > 0 else True

    for f in tqdm(files, desc="Segmenting"):
        img = io.imread(str(f))
        masks, _, _ = net.eval(img, diameter=diameter, flow_threshold=flow_threshold,
                               cellprob_threshold=cellprob_threshold, normalize=normalize)
        io.imsave(str(masks_dir / f"{f.stem}.tif"), masks)
        if overlays_dir is not None:
            save_overlay(img, masks, Path(overlays_dir) / f"{f.stem}.png")
    print(f"masks -> {masks_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlays", action="store_true", help="also save outline overlays for QC")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    s = cfg["segmentation"]
    out = Path(cfg["output_dir"])
    segment_tiles(cfg["tiles_dir"], out / "masks", model=s["model"], diameter=s["diameter"],
                  flow_threshold=s["flow_threshold"], cellprob_threshold=s["cellprob_threshold"],
                  tile_norm_blocksize=s["tile_norm_blocksize"], ext=cfg["image_ext"],
                  overlays_dir=out / "overlays" if args.overlays else None)


if __name__ == "__main__":
    main()
