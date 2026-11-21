"""
Features option 1: frozen pretrained encoders, one feature vector per nuclear crop.

Every network runs in inference mode with its classification head removed. Crops are
upsampled (bicubic) to 224 x 224 and normalised with each model's own mean / std.

Encoders:
Generic: EfficientNet-B2, ResNet-101, ConvNeXt-V2 Tiny, DINOv3 ViT-S/16, 
Pathology: HoVer-NeXt, UNI2-h, Virchow2
"""

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ENCODERS = ["efficientnet_b2", "resnet101", "convnextv2_tiny", "dinov3_small_cls",
                  "hovernext", "uni2", "virchow2"]

INPUT_SIZE = 224
INTERPOLATION = "bicubic"

_DINOV3_NAMES = {"small": "vit_small_patch16_dinov3.lvd1689m",
                 "base": "vit_base_patch16_dinov3.lvd1689m",
                 "large": "vit_large_patch16_dinov3.lvd1689m"}


def encoder_spec(key, hovernext_ckpt=None):
    """Recipe for one encoder key (nothing is downloaded here)."""
    if re.fullmatch(r"resnet(34|50|101)", key):
        return dict(kind="timm", name=key)
    if re.fullmatch(r"efficientnet_b[0-3]", key):
        return dict(kind="timm", name=key)
    m = re.fullmatch(r"convnextv2_(tiny|base|large)", key)
    if m:
        return dict(kind="timm", name=f"{key}.fcmae_ft_in1k")
    m = re.fullmatch(r"dinov3_(small|base|large)_(cls|avg|cls_avg)", key)
    if m:
        return dict(kind="dinov3", name=_DINOV3_NAMES[m.group(1)], pool=m.group(2))
    if key == "hovernext":
        if hovernext_ckpt is None or not Path(hovernext_ckpt).exists():
            raise FileNotFoundError(
                f"hovernext needs the HoVer-NeXt Lizard checkpoint (encoders.hovernext_ckpt "
                f"in the config); got {hovernext_ckpt!r}")
        # HoVer-NeXt's own code feeds images scaled to [0, 1] at 256 px
        return dict(kind="hovernext", name="convnextv2_base", ckpt=str(hovernext_ckpt),
                    size=256, norm="unit")
    if key == "uni2":
        return dict(kind="uni2", name="hf-hub:MahmoodLab/UNI2-h")
    if key == "virchow2":
        return dict(kind="virchow2", name="hf-hub:paige-ai/Virchow2")
    raise ValueError(f"unknown encoder {key!r}")


class Virchow2Features(torch.nn.Module):
    """
    Virchow2's documented readout: CLS token concatenated with the mean patch token.

    Tokens 1-4 are register tokens and are skipped, hence out[:, 5:].
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone.forward_features(x)
        return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)


class DinoV3Features(torch.nn.Module):
    """DINOv3 readout: CLS token, mean patch token, or both concatenated (registers skipped)."""

    def __init__(self, backbone, pool):
        super().__init__()
        self.backbone, self.pool = backbone, pool

    def forward(self, x):
        out = self.backbone.forward_features(x)
        cls, patch = out[:, 0], out[:, self.backbone.num_prefix_tokens:].mean(1)
        if self.pool == "cls":
            return cls
        if self.pool == "avg":
            return patch
        return torch.cat([cls, patch], dim=-1)


def build_encoder(spec, device):
    """-> (module in eval mode on device, preprocessing cfg)."""
    import timm

    kind = spec["kind"]
    if kind == "timm":
        model = timm.create_model(spec["name"], pretrained=True, num_classes=0)
        pcfg = model.pretrained_cfg

    elif kind == "hovernext":
        # The checkpoint holds the whole segmentation network; only the encoder is kept.
        # Its keys are in timm's features_only form (stem_0, stages_0, ...) and are renamed
        # to the dotted form so they load into a num_classes=0 ConvNeXt-V2 Base, whose
        # parameter-free head.norm (identity at init) is applied after global pooling.
        state = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)["model_state_dict"]
        prefix = "encoder.model."
        enc = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not enc:
            raise KeyError(f"no tensors under {prefix!r} in {spec['ckpt']}")
        enc = {re.sub(r"^(stem|stages)_(\d+)\.", r"\1.\2.", k): v for k, v in enc.items()}
        model = timm.create_model(spec["name"], pretrained=False, num_classes=0)
        missing, unexpected = model.load_state_dict(enc, strict=False)
        assert not unexpected, f"unexpected tensors: {sorted(unexpected)[:5]}"
        assert set(missing) <= {"head.norm.weight", "head.norm.bias"}, sorted(missing)[:5]
        pcfg = model.pretrained_cfg

    elif kind == "uni2":
        model = timm.create_model(
            spec["name"], pretrained=True,
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, dynamic_img_size=True)
        pcfg = model.pretrained_cfg

    elif kind == "virchow2":
        backbone = timm.create_model(spec["name"], pretrained=True, num_classes=0,
                                     mlp_layer=timm.layers.SwiGLUPacked,
                                     act_layer=torch.nn.SiLU)
        pcfg = backbone.pretrained_cfg
        model = Virchow2Features(backbone)

    elif kind == "dinov3":
        # pretrained_cfg says 256, but every encoder sees the same 224 px input here
        backbone = timm.create_model(spec["name"], pretrained=True, num_classes=0)
        pcfg = backbone.pretrained_cfg
        model = DinoV3Features(backbone, spec["pool"])

    else:
        raise ValueError(kind)

    if spec.get("norm") == "unit":
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        mean, std = pcfg["mean"], pcfg["std"]
    cfg = dict(size=spec.get("size", INPUT_SIZE),
               interp=spec.get("interp", INTERPOLATION),
               mean=torch.tensor(mean, device=device).view(1, 3, 1, 1),
               std=torch.tensor(std, device=device).view(1, 3, 1, 1))
    return model.eval().to(device), cfg


def preprocess(batch_uint8, cfg, device):
    """(B, H, W, 3) uint8 -> normalised float tensor, resized to the encoder's input size."""
    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(cfg["size"], cfg["size"]),
                      mode=cfg["interp"], align_corners=False).clamp_(0, 1)
    return (x - cfg["mean"]) / cfg["std"]


@torch.no_grad()
def embed(crops, model, cfg, device, batch_size=256):
    """(N, H, W, 3) uint8 crops -> (N, D) float32 embeddings."""
    out = []
    for i in range(0, len(crops), batch_size):
        out.append(model(preprocess(crops[i:i + batch_size], cfg, device)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def extract_embeddings(crops, key, device=None, batch_size=256, hovernext_ckpt=None):
    """(N, H, W, 3) uint8 crops -> (N, D) embeddings with the encoder key."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_encoder(encoder_spec(key, hovernext_ckpt), device)
    emb = embed(crops, model, cfg, device, batch_size=batch_size)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return emb
