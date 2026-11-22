# K-CLUE: A Segmentation-to-Clustering Framework for Ki67 Cell Typing
 
Official code for the BD-MedVis workshop paper at **BMVC 2026**:
 
> **K-CLUE: A Segmentation-to-Clustering Framework for Ki67 Cell Typing**
> Ana Beatriz Vieira, Sara P. Oliveira, Diana Montezuma, João Fraga, Jaime S. Cardoso, Arlindo L. Oliveira
 
K-CLUE is an **annotation-free** pipeline that separates Ki67-positive from Ki67-negative nuclei in IHC (H-DAB) images. It uses no task-specific training and no manual annotations. Annotations are used only for evaluation.
 
## Method
 
1. **Segmentation.** [Cellpose](https://github.com/mouseland/cellpose) with a DINOv3 ViT-B backbone (Cellpose-DINOv3) produces nuclear instance masks.
2. **Crops.** Each nucleus is cropped in a fixed 64×64 box centred on it. Pixels outside its mask are replaced with the dataset's global median background colour. The crop is then upsampled to 224×224.
3. **Representation.** A frozen pretrained encoder embeds each crop.
4. **PCA + whitening.** The components that explain 90% of the variance are kept, and each one is rescaled to unit variance.
5. **Clustering.** k-means (k = 2) runs on the pooled nuclei of all tiles. The cluster with the higher mean DAB optical density is labelled Ki67-positive.


## Installation
 
```bash
git clone https://github.com/anagilvieira/K-CLUE.git
cd K-CLUE
conda create -n kclue python=3.10 -y
conda activate kclue
pip install -r requirements.txt
```
 
**Encoder weights**
- ResNet, EfficientNet, ConvNeXt-V2 and DINOv3 download automatically through `timm`.
- UNI2-h and Virchow2 are gated. Request access on their Hugging Face pages, then run `huggingface-cli login`.
- HoVer-NeXt uses the encoder of the Lizard ConvNeXt-V2 Base checkpoint, available in the HoVer-NeXt_Inference repository. Download it and set its path in `encoders.hovernext_ckpt` in the config.
## Data
 
The tiles and point annotations used in the paper will be made available soon. Place them as follows (or edit the paths in `configs/*.yaml`):
 
```
data/
├── MIST/
│   ├── tiles/               # 56 RGB tiles (.png), 256 x 256
│   └── 56_ANN_MIST.txt      # point annotations (SlideScore export)
└── BCData/
    ├── images/test/         # 640 x 640 .png images
    └── centroids_640.csv    # point annotations (image, x, y, label)
```
 
## Usage
 
Run every command from the repository root. Paths and parameters are in `configs/<dataset>.yaml`, and every output goes to `outputs/<dataset>/`.
 
```bash
# 1. nuclei segmentation (once per dataset)
python segment.py --config configs/mist.yaml
 
# 2. K-CLUE: crops -> embeddings -> PCA + whitening -> k-means -> evaluation
python main.py --config configs/mist.yaml
```
 
The first run of `main.py` creates the crops and the features. They are cached, so later runs only redo the clustering and the evaluation. Results are saved to `outputs/<dataset>/results/<crop mode>/pca90_whitened/` (or `pca90_unwhitened/`). The main file is `end_to_end.csv`, with the end-to-end Precision, Recall and F1 (%) and the per-tile counting MAE for each class, averaged over the 5 k-means seeds.
 
To compare encoders as in the paper's ablations, run `analysis/evaluate_encoders.py` on a run folder. It reports Precision, Recall, F1 and MCC over the nuclei matched to a ground-truth point, plus the MAE:
 
```bash
python analysis/evaluate_encoders.py --config configs/mist.yaml --run pca90_whitened
```
 
Useful options:
 
```bash
python main.py --config configs/mist.yaml --no-whiten                   # no whitening
python main.py --config configs/mist.yaml --features uni2 virchow2      # chosen encoders
python main.py --config configs/mist.yaml --mode isolated_white         # other crop fill
python main.py --config configs/mist.yaml \
    --features hed hsv_clean lab hed_hsv_lab_clean dab_otsu             # handcrafted colour
```

## Repository structure
 
```
K-CLUE/
├── segment.py          # nuclei segmentation with Cellpose (run once)
├── main.py             # K-CLUE: crops -> features -> PCA, whitening, k-means -> evaluation
├── crops.py            # per-nucleus crops + mean DAB / haematoxylin (called by main.py)
├── embeddings.py       # features 1: frozen pretrained encoders
├── handcrafted.py      # features 2: HED / HSV / Lab colour descriptors
├── evaluation.py       # GT loading, Hungarian matching, metrics
├── configs/            # one YAML per dataset
└── analysis/           # encoder evaluation, linear probe
```

## License
 
The code is released under the MIT License (see [LICENSE](LICENSE)). Pretrained models and datasets are subject to their own licenses.
