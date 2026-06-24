#!/usr/bin/env python3
# Copyright (c) 2026 Ant Group and the KiD-Seg authors.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file or
# https://creativecommons.org/licenses/by-nc/4.0/ for details.
# For research and non-commercial use only; not for clinical use.
"""
Standalone test_rfnet_liver.py – RFNet for multi-modal liver segmentation with missing modalities.

Evaluates the best model on the TEST split (same split as train_rfnet_liver.py) and reports:
  • Per-combination Dice and HD95: mean ± std (median)
  • Average Dice and HD95 across all 8 TEST_COMBINATIONS

Generates a comprehensive visualization figure per patient:
  Row 1 : 8 MRI modalities (middle axial slice)
  Row 2 : 8 TEST_COMBINATION overlay maps (pred=green, GT=red, overlap=yellow) on T2WI
  Row 3-L: t-SNE – Class Separability under Severe Missingness (G1-only)
  Row 3-R: t-SNE – Feature Invariance to Input Permutations across combinations

Usage:
  python test_rfnet_liver.py --datapath /path/to/preprocess_nii_256x32_1 \\
                              --checkpoint ./output_rfnet_liver/model_best.pth \\
                              --savepath ./test_output \\
                              --resize_x 256 --resize_y 256 --resize_z 32

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — RFNet testing
══════════════════════════════════════════════════════════════════════════════

### test

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=0
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

nohup python -u test_rfnet_liver.py --datapath ./data/preprocess_nii_256x32 --checkpoint ./output_rfnet_liver/model_best.pth --savepath ./output_rfnet_liver/test_output > <PROJECT_ROOT>/logs/test_rfnet_liver.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_rfnet_liver.log
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import logging
import random
from pathlib import Path


import warnings
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.GradScaler",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast",
    category=FutureWarning
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Blowfish.*")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from torch.amp import autocast as _autocast
    def make_autocast(enabled):
        return _autocast('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    def make_autocast(enabled):
        return _autocast(enabled=enabled)

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")

from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import wilcoxon
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & GROUP DEFINITIONS  (identical to train_rfnet_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════
MODALITY_NAMES = ["T2WI", "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase"]
MODALITY_SUFFIXES = MODALITY_NAMES[:]
NUM_MODALITIES = len(MODALITY_NAMES)  # 8

GROUP_INDICES = {
    "G1": [0],            # T2WI – always present
    "G2": [1, 2, 3, 4],  # contrast phases
    "G3": [5],            # DWI
    "G4": [6, 7],         # InPhase / OutPhase
}

GROUP_LABELS = {
    0: "G1", 1: "G2", 2: "G2", 3: "G2", 4: "G2",
    5: "G3", 6: "G4", 7: "G4",
}

# All 8 valid test combinations (G1 always on; G2/G3/G4 on/off)
TEST_COMBINATIONS = [
    ("G1",              (False, False, False)),
    ("G1+G2",           (True,  False, False)),
    ("G1+G3",           (False, True,  False)),
    ("G1+G4",           (False, False, True )),
    ("G1+G2+G3",        (True,  True,  False)),
    ("G1+G2+G4",        (True,  False, True )),
    ("G1+G3+G4",        (False, True,  True )),
    ("G1+G2+G3+G4",     (True,  True,  True )),
]


def groups_to_mask(g2: bool, g3: bool, g4: bool):
    """Return boolean list of length NUM_MODALITIES."""
    m = [False] * NUM_MODALITIES
    for i in GROUP_INDICES["G1"]:
        m[i] = True
    if g2:
        for i in GROUP_INDICES["G2"]:
            m[i] = True
    if g3:
        for i in GROUP_INDICES["G3"]:
            m[i] = True
    if g4:
        for i in GROUP_INDICES["G4"]:
            m[i] = True
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA  (identical to train_rfnet_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════
def discover_patients(root: str):
    root = Path(root)
    patients = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        img_dir, lbl_dir = d / "images", d / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        ok = all(
            (img_dir / f"{d.name}_{s}.nii.gz").exists() and
            (lbl_dir / f"{d.name}_{s}.nii.gz").exists()
            for s in MODALITY_SUFFIXES
        )
        if ok:
            patients.append(d.name)
    patients.sort()
    return patients


def split_patients(patients, train_ratio=0.7, val_ratio=0.2):
    n = len(patients)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n, 0, 0
    return patients[:n_train], patients[n_train:n_train + n_val], patients[n_train + n_val:]


def load_nifti(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def resize_volume(vol, target_shape, order=1):
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return vol
    return scipy_zoom(vol, factors, order=order)


def normalize_volume(vol):
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-8:
        return vol
    return (vol - vmin) / (vmax - vmin)


def preprocess_patient(root, name, shape):
    pdir = Path(root) / name
    imgs = []
    for suf in MODALITY_SUFFIXES:
        v = load_nifti(pdir / "images" / f"{name}_{suf}.nii.gz")
        v = normalize_volume(resize_volume(v, shape, order=1))
        imgs.append(v)
    images = np.stack(imgs, 0).astype(np.float32)
    lbl = load_nifti(pdir / "labels" / f"{name}_T2WI.nii.gz")
    lbl = (resize_volume(lbl, shape, order=0) > 0.5).astype(np.int64)
    return images, lbl


class LiverTestDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, label, name = self.data[idx]
        return (torch.from_numpy(images.copy()),
                torch.from_numpy(label.copy().astype(np.uint8)),
                name)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL LAYERS  (RFNet – identical to train_rfnet_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════
basic_dims = 16


def normalization(planes, norm="in"):
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    elif norm == "gn":
        return nn.GroupNorm(4, planes)
    elif norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"Unsupported norm: {norm}")


class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1,
                 pad_type="reflect", norm="in", act_type="lrelu", relufactor=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride, padding,
                              padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        self.activation = (nn.ReLU(inplace=True) if act_type == "relu"
                           else nn.LeakyReLU(relufactor, inplace=True))

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


# ── PRM generators ───────────────────────────────────────────────────────────

class prm_generator_laststage(nn.Module):
    """Bottleneck PRM: uses all stacked modality features at the deepest scale."""
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0),
            general_conv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1),
            general_conv3d(in_channel // 4, in_channel, k_size=1, padding=0))
        self.prm_layer = nn.Sequential(
            general_conv3d(in_channel, 16, k_size=1, padding=0),
            nn.Conv3d(16, num_cls, 1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x, mask):
        """x: (B, K, C, H, W, Z),  mask: (B, K) bool."""
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return self.prm_layer(self.embedding_layer(y.view(B, -1, H, W, Z)))


class prm_generator(nn.Module):
    """Mid-scale PRM: conditioned on the current decoder feature + encoder feature."""
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0),
            general_conv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1),
            general_conv3d(in_channel // 4, in_channel, k_size=1, padding=0))
        self.prm_layer = nn.Sequential(
            general_conv3d(in_channel * 2, 16, k_size=1, padding=0),
            nn.Conv3d(16, num_cls, 1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        """x1: decoder feat (B,C,H,W,Z), x2: encoder feats (B,K,C,H,W,Z), mask: (B,K) bool."""
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        return self.prm_layer(torch.cat((x1, self.embedding_layer(y.view(B, -1, H, W, Z))), 1))


# ── Modal / Region fusion ────────────────────────────────────────────────────

class modal_fusion(nn.Module):
    """Attention-weighted modal fusion for a single tumour/tissue region."""
    def __init__(self, in_channel=64, num_modal=8):
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 128, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, num_modal, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        """
        x   : (B, K, C, H, W, Z) – region-weighted modality features
        prm : (B, 1, H, W, Z)    – region probability for this class
        """
        B, K, C, H, W, Z = x.size()
        prm_avg = torch.mean(prm, dim=(2, 3, 4), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg.unsqueeze(-1)
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg.view(B, 1, 1, 1, 1)), dim=1)
        weight = self.weight_layer(feat_avg).view(B, K, 1, 1, 1, 1)
        weight = self.sigmoid(weight)
        return torch.sum(x * weight, dim=1)


class region_fusion(nn.Module):
    """Fuse per-region features into a single feature map via 1×1→3×3→1×1 convs."""
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel, k_size=1, padding=0),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0))

    def forward(self, region_feats):
        return self.fusion_layer(torch.cat(region_feats, dim=1))


class region_aware_modal_fusion(nn.Module):
    """Memory-efficient region-aware modal fusion (RFM).

    Iterates over classes to avoid materialising the (B, K, cls, C, H, W, Z)
    tensor of the original RFNet implementation.
    """
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.num_cls = num_cls
        self.num_modal = num_modal
        self.modal_fusions = nn.ModuleList(
            [modal_fusion(in_channel, num_modal) for _ in range(num_cls)])
        self.region_fuse = region_fusion(in_channel, num_cls)
        self.short_cut = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel, k_size=1, padding=0),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0))

    def forward(self, x, prm, mask):
        """
        x    : (B, K, C, H, W, Z)       stacked modality features
        prm  : (B, num_cls, H, W, Z)    region probability maps
        mask : (B, K) bool              modality-presence mask
        """
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        region_feats = []
        for c in range(self.num_cls):
            prm_c = prm[:, c:c+1, :, :, :]
            region_modal = y * prm_c.unsqueeze(2)
            region_feats.append(self.modal_fusions[c](region_modal, prm_c))
        return torch.cat((self.region_fuse(region_feats),
                          self.short_cut(y.view(B, -1, H, W, Z))), dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL – Encoder / Decoder / RFNet_Liver
# ═══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """4-scale residual UNet encoder (identical to original RFNet)."""
    def __init__(self):
        super().__init__()
        self.e1_c1 = general_conv3d(1, basic_dims, pad_type="reflect")
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type="reflect")
        self.e2_c1 = general_conv3d(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e3_c1 = general_conv3d(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e4_c1 = general_conv3d(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")

    def forward(self, x):
        x1 = self.e1_c1(x);  x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1); x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x3 = self.e3_c1(x2); x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3); x4 = x4 + self.e4_c3(self.e4_c2(x4))
        return x1, x2, x3, x4


class Decoder_sep(nn.Module):
    """Shared decoder used for per-modality segmentation regularisation loss."""
    def __init__(self, num_cls=2):
        super().__init__()
        c = basic_dims
        self.d3     = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1  = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_c2  = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_out = general_conv3d(c * 4, c * 4, k_size=1, padding=0, pad_type="reflect")
        self.d2     = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1  = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_c2  = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_out = general_conv3d(c * 2, c * 2, k_size=1, padding=0, pad_type="reflect")
        self.d1     = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1  = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_c2  = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_out = general_conv3d(c, c, k_size=1, padding=0, pad_type="reflect")
        self.seg    = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        d = self.d3_c1(self.d3(x4))
        d = self.d3_out(self.d3_c2(torch.cat((d, x3), 1)))
        d = self.d2_c1(self.d2(d))
        d = self.d2_out(self.d2_c2(torch.cat((d, x2), 1)))
        d = self.d1_c1(self.d1(d))
        d = self.d1_out(self.d1_c2(torch.cat((d, x1), 1)))
        return self.softmax(self.seg(d))


class Decoder_fuse(nn.Module):
    """
    Pure RFNet fusion decoder.

    At each of the 4 decoder scales:
      1. Generate a PRM (region probability map) from the multi-modal features.
      2. Apply region-aware modal fusion (RFM) guided by the detached PRM.
      3. Upsample and concatenate with the previous scale.

    Returns
    -------
    pred      : (B, num_cls, H, W, Z)  softmax segmentation
    prm_preds : 4-tuple of PRM predictions, all upsampled to H×W×Z
    fuse_x4   : (B, C, H/8, W/8, D/8) bottleneck RFM feature (for t-SNE invariance)
    """
    def __init__(self, num_cls=2, num_modal=8):
        super().__init__()
        c = basic_dims
        # decoder convolution blocks
        self.d3_c1  = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_c2  = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_out = general_conv3d(c * 4, c * 4, k_size=1, padding=0, pad_type="reflect")
        self.d2_c1  = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_c2  = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_out = general_conv3d(c * 2, c * 2, k_size=1, padding=0, pad_type="reflect")
        self.d1_c1  = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_c2  = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_out = general_conv3d(c, c, k_size=1, padding=0, pad_type="reflect")
        self.seg     = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        # upsamplers
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        # region-aware modal fusion at each scale
        self.RFM4 = region_aware_modal_fusion(c * 8, num_cls, num_modal)
        self.RFM3 = region_aware_modal_fusion(c * 4, num_cls, num_modal)
        self.RFM2 = region_aware_modal_fusion(c * 2, num_cls, num_modal)
        self.RFM1 = region_aware_modal_fusion(c * 1, num_cls, num_modal)
        # PRM generators at each scale
        self.prm4 = prm_generator_laststage(c * 8, num_cls, num_modal)
        self.prm3 = prm_generator(c * 4, num_cls, num_modal)
        self.prm2 = prm_generator(c * 2, num_cls, num_modal)
        self.prm1 = prm_generator(c * 1, num_cls, num_modal)

    def forward(self, x1, x2, x3, x4, mask):
        """
        x1-x4 : (B, K, C, H, W, Z)   multi-modal encoder features at 4 scales
        mask   : (B, K) bool           modality-presence mask

        Returns: pred, prm_preds, fuse_x4
        """
        # ── Scale 4 (bottleneck) ──────────────────────────────────────
        p4 = self.prm4(x4, mask)
        d4 = self.RFM4(x4, p4.detach(), mask)
        fuse_x4 = d4                          # save for t-SNE invariance
        d4 = self.d3_c1(self.up2(d4))

        # ── Scale 3 ───────────────────────────────────────────────────
        p3 = self.prm3(d4, x3, mask)
        d3 = self.RFM3(x3, p3.detach(), mask)
        d3 = self.d3_out(self.d3_c2(torch.cat((d3, d4), 1)))
        d3 = self.d2_c1(self.up2(d3))

        # ── Scale 2 ───────────────────────────────────────────────────
        p2 = self.prm2(d3, x2, mask)
        d2 = self.RFM2(x2, p2.detach(), mask)
        d2 = self.d2_out(self.d2_c2(torch.cat((d2, d3), 1)))
        d2 = self.d1_c1(self.up2(d2))

        # ── Scale 1 ───────────────────────────────────────────────────
        p1 = self.prm1(d2, x1, mask)
        d1 = self.RFM1(x1, p1.detach(), mask)
        d1 = self.d1_out(self.d1_c2(torch.cat((d1, d2), 1)))

        pred      = self.softmax(self.seg(d1))
        prm_preds = (p1, self.up2(p2), self.up4(p3), self.up8(p4))
        return pred, prm_preds, fuse_x4


class RFNet_Liver(nn.Module):
    """
    RFNet generalised to num_modal modalities for liver segmentation.

    Inference mode (is_training=False):
        Returns fuse_pred only.
    Training mode (is_training=True):
        Returns (fuse_pred, sep_preds, prm_preds)
    """
    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=False):
        super().__init__()
        self.num_modal = num_modal
        self.num_cls = num_cls
        self.use_ckpt = use_checkpoint

        self.encoders     = nn.ModuleList([Encoder() for _ in range(num_modal)])
        self.decoder_fuse = Decoder_fuse(num_cls, num_modal)
        self.decoder_sep  = Decoder_sep(num_cls)

        self.is_training = False
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def _encode_one(self, encoder, x_m):
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def forward(self, x, mask):
        feats = [self._encode_one(self.encoders[m], x[:, m:m+1])
                 for m in range(self.num_modal)]

        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        fuse_pred, prm_preds, _ = self.decoder_fuse(x1, x2, x3, x4, mask)

        if not self.is_training:
            return fuse_pred

        sep_preds = [self.decoder_sep(*feats[m]) for m in range(self.num_modal)]
        return fuse_pred, sep_preds, prm_preds

    def forward_with_features(self, x, mask):
        """Forward pass returning prediction + multi-scale features for t-SNE.

        Returns:
            fuse_pred : (B, num_cls, H, W, D)
            fuse_x4   : (B, C, H/8, W/8, D/8)  bottleneck RFM feature (invariance t-SNE)
            x1        : (B, K, C1, H, W, D)     full-res encoder features (class sep t-SNE)
        """
        feats = [self._encode_one(self.encoders[m], x[:, m:m+1])
                 for m in range(self.num_modal)]

        x1 = torch.stack([f[0] for f in feats], 1)  # (B, K, C, H, W, D) full-res
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        fuse_pred, _, fuse_x4 = self.decoder_fuse(x1, x2, x3, x4, mask)
        return fuse_pred, fuse_x4, x1


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS – Dice & Hausdorff Distance 95
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Dice coefficient for binary masks."""
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    intersection = np.sum(p & g)
    return float((2.0 * intersection + eps) / (p.sum() + g.sum() + eps))


def compute_hd95(pred: np.ndarray, gt: np.ndarray, voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    """
    95th-percentile Hausdorff Distance between binary masks.
    Returns np.inf if either mask is empty.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return np.inf

    # Surface voxels = boundary
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_b & ~binary_erosion(pred_b, structure=struct)
    gt_surface = gt_b & ~binary_erosion(gt_b, structure=struct)

    # Fallback: if erosion removes everything, use the mask itself
    if pred_surface.sum() == 0:
        pred_surface = pred_b
    if gt_surface.sum() == 0:
        gt_surface = gt_b

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_b, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt_b, sampling=voxel_spacing)

    # Distances from surfaces
    d_gt2pred = dt_pred[gt_surface]
    d_pred2gt = dt_gt[pred_surface]

    all_dist = np.concatenate([d_gt2pred, d_pred2gt])
    return float(np.percentile(all_dist, 95))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TEST ALL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_all_combinations(model, test_loader, args, save_dir):
    """
    Returns dicts for Dice and HD95, each keyed by combination name + "Average".
    Each entry: {"mean": float, "std": float, "median": float, "scores": list}
    """
    model.eval()
    model.is_training = False

    dice_results = {}
    hd95_results = {}
    all_dice, all_hd95 = [], []

    for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
        mask_list = groups_to_mask(g2, g3, g4)
        present = [MODALITY_NAMES[i] for i, v in enumerate(mask_list) if v]
        logging.info(f"Testing {combo_name}: {present}")

        dice_scores, hd95_scores = [], []
        for x, y_int, _ in test_loader:
            x, y_int = x.cuda(), y_int.cuda()
            mask = torch.tensor([mask_list] * x.size(0), dtype=torch.bool, device=x.device)
            with make_autocast(args.amp):
                pred = model(x, mask)
            pl = pred.argmax(1).cpu().numpy()
            gt = y_int.cpu().numpy()

            for b in range(x.size(0)):
                p_bin = (pl[b] == 1).astype(np.uint8)
                g_bin = (gt[b] == 1).astype(np.uint8)
                dice_scores.append(compute_dice(p_bin, g_bin))
                hd95_scores.append(compute_hd95(p_bin, g_bin))

        d_arr = np.array(dice_scores)
        h_arr = np.array(hd95_scores)
        # Replace inf with nan for stats (but keep raw)
        h_finite = np.where(np.isinf(h_arr), np.nan, h_arr)

        dice_results[combo_name] = {
            "mean": d_arr.mean(), "std": d_arr.std(), "median": np.median(d_arr),
            "scores": d_arr.tolist()
        }
        hd95_results[combo_name] = {
            "mean": float(np.nanmean(h_finite)), "std": float(np.nanstd(h_finite)),
            "median": float(np.nanmedian(h_finite)), "scores": h_arr.tolist()
        }
        all_dice.append(d_arr)
        all_hd95.append(h_finite)

        logging.info(f"  {combo_name} Dice : {d_arr.mean():.3f} ± {d_arr.std():.3f}"
                     f" ({np.median(d_arr):.3f})")
        logging.info(f"  {combo_name} HD95 : {np.nanmean(h_finite):.3f} ± {np.nanstd(h_finite):.3f}"
                     f" ({np.nanmedian(h_finite):.3f})")

    # Average across all combinations
    flat_d = np.concatenate(all_dice)
    flat_h = np.concatenate(all_hd95)
    dice_results["Average"] = {
        "mean": flat_d.mean(), "std": flat_d.std(), "median": float(np.median(flat_d))
    }
    hd95_results["Average"] = {
        "mean": float(np.nanmean(flat_h)), "std": float(np.nanstd(flat_h)),
        "median": float(np.nanmedian(flat_h))
    }

    return dice_results, hd95_results


def print_results(dice_results, hd95_results):
    """Print results to console (plain mean +/- std (median))."""
    print("\n" + "=" * 90)
    print(f"{'Combination':25s} | {'Dice':45s} | {'HD95':45s}")
    print("=" * 90)
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str = f"{d['mean']:.3f} ± {d['std']:.3f} ({d['median']:.3f})"
        h_str = f"{h['mean']:.3f} ± {h['std']:.3f} ({h['median']:.3f})"
        print(f"  {name:23s} | {d_str:43s} | {h_str:43s}")
    print("-" * 90)
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str = f"{d['mean']:.3f} ± {d['std']:.3f} ({d['median']:.3f})"
    h_str = f"{h['mean']:.3f} ± {h['std']:.3f} ({h['median']:.3f})"
    print(f"  {'Average':23s} | {d_str:43s} | {h_str:43s}")
    print("=" * 90)


def save_results_json(dice_results, hd95_results, save_dir):
    """Save results to JSON (without raw scores)."""
    out = {}
    for name in list(dice_results.keys()):
        out[name] = {
            "dice": {k: float(v) for k, v in dice_results[name].items() if k != "scores"},
            "hd95": {k: float(v) for k, v in hd95_results[name].items() if k != "scores"},
        }
    path = os.path.join(save_dir, "test_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logging.info(f"Results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. STATISTICAL SIGNIFICANCE — two-sided Wilcoxon signed-rank test (p < 0.05)
# ═══════════════════════════════════════════════════════════════════════════════

def paired_wilcoxon(scores_a, scores_b):
    """Two-sided Wilcoxon signed-rank test on paired per-subject scores.

    Pairs holding a non-finite value (e.g. HD95 = inf when a mask is empty) are
    dropped before testing; no subject is excluded based on its performance.
    Returns (statistic, p_value, n_pairs). When the test is undefined (no usable
    pairs, or all paired differences are zero) the p-value defaults to 1.0.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size == 0 or np.allclose(a, b):
        return float("nan"), 1.0, int(a.size)
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return float("nan"), 1.0, int(a.size)
    return float(stat), float(p), int(a.size)


def _fmt_pvalue(p):
    """Format a p-value with an explicit verdict (no dagger / star markers)."""
    if p != p:  # NaN → undefined
        return "n/a"
    return f"{p:.4g} ({'significant' if p < 0.05 else 'n.s.'})"


def save_per_subject_scores(dice_results, hd95_results, save_dir):
    """Persist per-subject Dice/HD95 scores so another method can be compared
    against this one via --compare_json (paired Wilcoxon signed-rank test)."""
    out = {"dice": {}, "hd95": {}}
    for combo_name, _ in TEST_COMBINATIONS:
        out["dice"][combo_name] = dice_results.get(combo_name, {}).get("scores", [])
        out["hd95"][combo_name] = hd95_results.get(combo_name, {}).get("scores", [])
    path = os.path.join(save_dir, "per_subject_scores.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logging.info(f"Per-subject scores saved to {path}")


def statistical_significance_test(dice_results, hd95_results, args,
                                  ref_combo="G1+G2+G3+G4"):
    """Two-sided Wilcoxon signed-rank tests (p < 0.05), reported as p-values.

    (A) Within-model: each modality combination vs. the full-modality reference,
        paired per patient (always runs).
    (B) Cross-method: when --compare_json is given, compare this model against a
        reference method's per-subject scores, per combination and overall — this
        reproduces the manuscript's method-vs-method significance analysis.

    Significance is reported as numeric p-values with an explicit verdict; no
    dagger or star markers are printed.
    """
    def _scores(results, combo):
        return results.get(combo, {}).get("scores", [])

    print("\n" + "=" * 88)
    print("STATISTICAL SIGNIFICANCE - two-sided Wilcoxon signed-rank test (p < 0.05)")
    print("=" * 88)

    # ── (A) Within-model: each combination vs. the full-modality reference ──
    print(f"(A) Each combination vs. full-modality '{ref_combo}' (paired per patient):")
    print(f"  {'Combination':16s} | {'DSC p-value':26s} | {'HD95 p-value':26s} | n")
    print("-" * 88)
    ref_d, ref_h = _scores(dice_results, ref_combo), _scores(hd95_results, ref_combo)
    for combo_name, _ in TEST_COMBINATIONS:
        if combo_name == ref_combo:
            continue
        _, pd_, nd = paired_wilcoxon(_scores(dice_results, combo_name), ref_d)
        _, ph_, _ = paired_wilcoxon(_scores(hd95_results, combo_name), ref_h)
        print(f"  {combo_name:16s} | {_fmt_pvalue(pd_):26s} | {_fmt_pvalue(ph_):26s} | {nd}")

    # ── (B) Cross-method comparison (optional, --compare_json) ──
    compare_path = getattr(args, "compare_json", None)
    if not compare_path:
        print("\n(B) Cross-method test skipped - pass --compare_json "
              "<other_method>/per_subject_scores.json to enable.")
        return
    if not os.path.isfile(compare_path):
        print(f"\n(B) Cross-method test skipped - file not found: {compare_path}")
        return

    with open(compare_path) as f:
        ref = json.load(f)
    ref_dice, ref_hd95 = ref.get("dice", {}), ref.get("hd95", {})
    print(f"\n(B) This model vs. reference '{compare_path}' (paired per patient):")
    print(f"  {'Combination':16s} | {'DSC p-value':26s} | {'HD95 p-value':26s} | n")
    print("-" * 88)
    pool_a_d, pool_a_h, pool_b_d, pool_b_h = [], [], [], []
    for combo_name, _ in TEST_COMBINATIONS:
        a_d, a_h = _scores(dice_results, combo_name), _scores(hd95_results, combo_name)
        b_d, b_h = ref_dice.get(combo_name, []), ref_hd95.get(combo_name, [])
        _, pd_, nd = paired_wilcoxon(a_d, b_d)
        _, ph_, _ = paired_wilcoxon(a_h, b_h)
        print(f"  {combo_name:16s} | {_fmt_pvalue(pd_):26s} | {_fmt_pvalue(ph_):26s} | {nd}")
        pool_a_d += list(a_d); pool_b_d += list(b_d)
        pool_a_h += list(a_h); pool_b_h += list(b_h)
    _, pd_o, nd_o = paired_wilcoxon(pool_a_d, pool_b_d)
    _, ph_o, _ = paired_wilcoxon(pool_a_h, pool_b_h)
    print("-" * 88)
    print(f"  {'Overall':16s} | {_fmt_pvalue(pd_o):26s} | {_fmt_pvalue(ph_o):26s} | {nd_o}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE EXTRACTION FOR t-SNE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features_for_tsne(model, test_loader, args):
    """
    Extract t-SNE features PER PATIENT.

    Returns:
      class_features_by_patient: dict[name] -> list of (feature_vec, label:0/1)
          Features are voxel-level encoder features under severe missingness (G1-only).

      invariance_features_by_patient: dict[name] -> list of (feature_vec, combo_name)
          Features are tumor voxel encoder features sampled across all TEST_COMBINATIONS.
    """
    model.eval()
    model.is_training = False

    class_features_by_patient = {}
    invariance_features_by_patient = {}

    # -----------------------------
    # (A) Class Separability (G1 only)
    # -----------------------------
    g1_mask_list = groups_to_mask(False, False, False)

    for x, y_int, names in test_loader:
        x, y_int = x.cuda(), y_int.cuda()
        if isinstance(names, str):
            names = [names]
        mask = torch.tensor([g1_mask_list] * x.size(0), dtype=torch.bool, device=x.device)

        with make_autocast(args.amp):
            _, _, x1_full = model.forward_with_features(x, mask)

        B = x.size(0)
        for b in range(B):
            name = names[b]
            class_features_by_patient.setdefault(name, [])

            gt = y_int[b].cpu().numpy()  # (H, W, D)
            mask_b = mask[b]             # (K,)
            x1_b = x1_full[b]            # (K, C, H, W, D)
            x1_avail = x1_b[mask_b]      # (K_avail, C, H, W, D)
            feat_full = x1_avail.mean(dim=0).cpu().numpy()  # (C, H, W, D)

            tumor_coords = np.argwhere(gt > 0)
            bg_coords = np.argwhere(gt == 0)
            n_sample = min(500, len(tumor_coords), len(bg_coords))
            if n_sample < 2:
                continue

            if len(tumor_coords) > n_sample:
                idx = np.random.choice(len(tumor_coords), n_sample, replace=False)
                tumor_coords = tumor_coords[idx]
            if len(bg_coords) > n_sample:
                idx = np.random.choice(len(bg_coords), n_sample, replace=False)
                bg_coords = bg_coords[idx]

            for tc in tumor_coords:
                class_features_by_patient[name].append(
                    (feat_full[:, tc[0], tc[1], tc[2]], 1)
                )
            for bc in bg_coords:
                class_features_by_patient[name].append(
                    (feat_full[:, bc[0], bc[1], bc[2]], 0)
                )

    # -----------------------------
    # (B) Invariance to Input Permutations (per patient)
    # -----------------------------
    N_PATCHES = 50  # tumor voxels per (patient, combination)

    for x, y_int, names in test_loader:
        x, y_int = x.cuda(), y_int.cuda()
        if isinstance(names, str):
            names = [names]
        B = x.size(0)

        for b in range(B):
            name = names[b]
            invariance_features_by_patient.setdefault(name, [])

            gt = y_int[b].cpu().numpy()
            tumor_coords = np.argwhere(gt > 0)
            if len(tumor_coords) < 2:
                continue

            xb = x[b:b+1]
            for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
                mask_list = groups_to_mask(g2, g3, g4)
                mask_t = torch.tensor([mask_list], dtype=torch.bool, device=x.device)

                with make_autocast(args.amp):
                    _, _, x1_full = model.forward_with_features(xb, mask_t)

                mask_b = mask_t[0]
                x1_avail = x1_full[0][mask_b]  # (K_avail, C, H, W, D)
                feat_full = x1_avail.mean(dim=0).cpu().numpy()  # (C, H, W, D)

                n_samp = min(N_PATCHES, len(tumor_coords))
                idx = np.random.choice(len(tumor_coords), n_samp, replace=False)
                for i in idx:
                    tc = tumor_coords[i]
                    invariance_features_by_patient[name].append(
                        (feat_full[:, tc[0], tc[1], tc[2]], combo_name)
                    )

    return class_features_by_patient, invariance_features_by_patient


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  COMPREHENSIVE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_slice(s):
    """Normalize a 2D slice to [0, 1]."""
    s = s.astype(np.float32)
    s -= s.min()
    mx = s.max()
    if mx > 0:
        s /= mx
    return s


def _orient_slice(s):
    """Rotate right 90° then flip horizontal (standard radiological orientation)."""
    return np.fliplr(np.rot90(s, k=-1))


def _overlay_pred_gt(base_rgb, pred_slice, gt_slice, alpha=0.55):
    """
    Overlay prediction and GT on RGB base image.
    Green = GT only, Red = Pred only, Yellow = Overlap.
    """
    ov = base_rgb.copy()
    pred_b = pred_slice.astype(bool)
    gt_b = gt_slice.astype(bool)
    gt_only = gt_b & ~pred_b
    pred_only = pred_b & ~gt_b
    both = gt_b & pred_b

    # GT only → red
    ov[gt_only, 0] = ov[gt_only, 0] * (1 - alpha) + alpha
    ov[gt_only, 1] *= (1 - alpha)
    ov[gt_only, 2] *= (1 - alpha)
    # Pred only → green
    ov[pred_only, 0] *= (1 - alpha)
    ov[pred_only, 1] = ov[pred_only, 1] * (1 - alpha) + alpha
    ov[pred_only, 2] *= (1 - alpha)
    # Overlap → yellow
    ov[both, 0] = ov[both, 0] * (1 - alpha) + alpha
    ov[both, 1] = ov[both, 1] * (1 - alpha) + alpha
    ov[both, 2] *= (1 - alpha)

    return np.clip(ov, 0, 1)


def _fit_tsne_2d(X: np.ndarray, random_state: int = 42):
    """
    sklearn-version-compatible t-SNE wrapper:
    - handles `max_iter` (newer sklearn) vs `n_iter` (older sklearn)
    - falls back from learning_rate='auto' if unsupported
    """
    if len(X) < 4:
        return None
    perp = min(30, max(2, len(X) // 4))
    base_kwargs = dict(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
    )

    for iter_key in ("max_iter", "n_iter"):
        for lr_val in ("auto", 200.0):
            kwargs = dict(base_kwargs)
            kwargs[iter_key] = 1000
            kwargs["learning_rate"] = lr_val
            try:
                tsne = TSNE(**kwargs)
                return tsne.fit_transform(X)
            except TypeError:
                continue
            except Exception:
                continue

    try:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
        return tsne.fit_transform(X)
    except Exception:
        return None


@torch.no_grad()
def generate_comprehensive_visualization(model, test_loader, args, save_dir,
                                         class_features_by_patient,
                                         invariance_features_by_patient):
    """
    Generate ONE comprehensive figure PER TEST PATIENT:
      Row 1 (8 cols): All 8 modality images (middle axial slice)
      Row 2 (8 cols): All 8 TEST_COMBINATION overlays on T2WI
      Row 3 Left:     Per-patient t-SNE – Class Separability (G1-only)
      Row 3 Right:    Per-patient t-SNE – Feature Invariance to Input Permutations

    Returns:
      list of saved figure paths.
    """
    model.eval()
    model.is_training = False
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    combo_names_ordered = [c[0] for c in TEST_COMBINATIONS]
    saved_paths = []

    for x, y_int, names in test_loader:
        if isinstance(names, str):
            names = [names]

        for b in range(x.size(0)):
            x_pt = x[b]
            gt_pt = y_int[b].numpy()
            name = names[b]

            H, W, D = gt_pt.shape
            z_mid = D // 2

            modality_slices = []
            for m in range(NUM_MODALITIES):
                sl = _orient_slice(_norm_slice(x_pt[m, :, :, z_mid].numpy()))
                modality_slices.append(sl)

            t2_base = modality_slices[0]
            t2_rgb = np.stack([t2_base] * 3, axis=-1)

            combo_overlays = []
            xb = x[b:b+1].cuda()
            for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
                mask_list = groups_to_mask(g2, g3, g4)
                mask = torch.tensor([mask_list], dtype=torch.bool, device=xb.device)
                with make_autocast(args.amp):
                    pred = model(xb, mask)

                pred_slice = _orient_slice(
                    (pred.argmax(1)[0].cpu().numpy()[:, :, z_mid] > 0).astype(np.uint8)
                )
                gt_slice = _orient_slice((gt_pt[:, :, z_mid] > 0).astype(np.uint8))
                ov = _overlay_pred_gt(t2_rgb.copy(), pred_slice, gt_slice)
                combo_overlays.append((combo_name, ov))

            class_features = class_features_by_patient.get(name, [])
            invariance_features = invariance_features_by_patient.get(name, [])

            cf_emb, cf_y = None, None
            if len(class_features) >= 4:
                cf_X = np.array([f[0] for f in class_features])
                cf_y = np.array([f[1] for f in class_features])
                if len(cf_X) > 4000:
                    idx = np.random.choice(len(cf_X), 4000, replace=False)
                    cf_X, cf_y = cf_X[idx], cf_y[idx]
                cf_emb = _fit_tsne_2d(cf_X, random_state=42)

            inv_emb, inv_y = None, None
            if len(invariance_features) >= 4:
                inv_X = np.array([f[0] for f in invariance_features])
                inv_labels = [f[1] for f in invariance_features]
                inv_y = np.array([combo_names_ordered.index(l) for l in inv_labels])
                if len(inv_X) > 4000:
                    idx = np.random.choice(len(inv_X), 4000, replace=False)
                    inv_X, inv_y = inv_X[idx], inv_y[idx]
                inv_emb = _fit_tsne_2d(inv_X, random_state=42)

            fig = plt.figure(figsize=(28, 14), dpi=150)
            gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.35, wspace=0.15,
                                   height_ratios=[1, 1, 1.3])

            group_colors = {"G1": "#2196F3", "G2": "#FF9800", "G3": "#4CAF50", "G4": "#9C27B0"}
            for i in range(8):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(modality_slices[i], cmap="gray", interpolation="nearest")
                grp = GROUP_LABELS[i]
                ax.set_title(MODALITY_NAMES[i], fontsize=9, fontweight="bold",
                             color=group_colors[grp])
                ax.axis("off")

            for i, (cname, ov) in enumerate(combo_overlays):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(ov, interpolation="nearest")
                ax.set_title(cname, fontsize=8, fontweight="bold")
                ax.axis("off")

            ax_tsne1 = fig.add_subplot(gs[2, :4])
            if cf_emb is not None and cf_y is not None:
                bg_mask = cf_y == 0
                tm_mask = cf_y == 1
                ax_tsne1.scatter(cf_emb[bg_mask, 0], cf_emb[bg_mask, 1],
                                 c='#3498db', alpha=0.5, s=12, label='Background', edgecolors='none')
                ax_tsne1.scatter(cf_emb[tm_mask, 0], cf_emb[tm_mask, 1],
                                 c='#e74c3c', alpha=0.5, s=12, label='Tumor', edgecolors='none')
                ax_tsne1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                                borderaxespad=0, framealpha=0.8)
            else:
                ax_tsne1.text(0.5, 0.5, "Insufficient samples for per-patient t-SNE",
                              ha='center', va='center', fontsize=10, transform=ax_tsne1.transAxes)
            ax_tsne1.set_title("Per-Patient t-SNE: Class Separability (G1 only)",
                               fontsize=11, fontweight="bold")
            ax_tsne1.set_xlabel("t-SNE 1", fontsize=9)
            ax_tsne1.set_ylabel("t-SNE 2", fontsize=9)
            ax_tsne1.tick_params(labelsize=7)
            ax_tsne1.set_box_aspect(1)
            ax_tsne1.grid(True, alpha=0.2)

            ax_tsne2 = fig.add_subplot(gs[2, 4:])

            def _mathcal_combo(name_):
                parts = name_.split("+")
                tex_parts = [r"$\mathcal{G}_" + p[1:] + "$" for p in parts]
                return "+".join(tex_parts)

            if inv_emb is not None and inv_y is not None:
                cmap = plt.get_cmap("tab10", len(combo_names_ordered))
                for ci, cname in enumerate(combo_names_ordered):
                    sel = inv_y == ci
                    if sel.sum() > 0:
                        ax_tsne2.scatter(inv_emb[sel, 0], inv_emb[sel, 1],
                                         c=[cmap(ci)], alpha=0.7, s=30,
                                         label=_mathcal_combo(cname),
                                         edgecolors='k', linewidths=0.3)
                ax_tsne2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                                borderaxespad=0, framealpha=0.8, ncol=1)
            else:
                ax_tsne2.text(0.5, 0.5, "Insufficient samples for per-patient t-SNE",
                              ha='center', va='center', fontsize=10, transform=ax_tsne2.transAxes)
            ax_tsne2.set_title("Per-Patient t-SNE: Feature Invariance to Input Permutations\n"
                               "(Tumor features across modality availability)",
                               fontsize=11, fontweight="bold")
            ax_tsne2.set_xlabel("t-SNE 1", fontsize=9)
            ax_tsne2.set_ylabel("t-SNE 2", fontsize=9)
            ax_tsne2.tick_params(labelsize=7)
            ax_tsne2.set_box_aspect(1)
            ax_tsne2.grid(True, alpha=0.2)

            fig.suptitle(f"RFNet Test Visualization — Patient: {name} — Axial Slice z={z_mid}",
                         fontsize=14, fontweight="bold", y=0.98)

            out_path = os.path.join(vis_dir, f"{name}_comprehensive_z{z_mid}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved_paths.append(out_path)
            logging.info(f"Comprehensive visualization saved to {out_path}")

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="RFNet Liver – Standalone Test Script")
    p.add_argument("--datapath", default='./data/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--checkpoint", default="./output_rfnet_liver/model_best.pth",
                   help="Path to best model checkpoint")
    p.add_argument("--savepath", default="./output_rfnet_liver/test_output")

    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)

    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--num_modal", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    p.add_argument("--compare_json", default=None,
                   help="Path to another method's per_subject_scores.json. When given, "
                        "a paired two-sided Wilcoxon signed-rank test (p<0.05) is run "
                        "against it, per combination and overall.")
    return p.parse_args()


def setup_logging(path):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        filename=os.path.join(path, "test.log"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("").addHandler(ch)


def main():
    args = parse_args()
    setup_logging(args.savepath)
    logging.info(f"Test Args: {args}")

    # Reproducibility (must match training seed for identical split)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # ── Discover & split patients (identical logic to train_rfnet_liver.py) ──
    patients = discover_patients(args.datapath)
    logging.info(f"Found {len(patients)} patients: {patients}")
    if len(patients) < 3:
        logging.warning("Very few patients – using all for test")
        test_p = patients
    else:
        _, _, test_p = split_patients(patients)
    logging.info(f"Test patients ({len(test_p)}): {test_p}")

    # ── Preprocess test set ──
    shape = (args.resize_x, args.resize_y, args.resize_z)
    logging.info(f"Preprocessing to {shape} ...")
    test_data = []
    for n in test_p:
        logging.info(f"  Loading {n} ...")
        imgs, lbl = preprocess_patient(args.datapath, n, shape)
        test_data.append((imgs, lbl, n))
    logging.info(f"Loaded {len(test_data)} test patients into RAM.")

    test_set = LiverTestDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers,
                             pin_memory=True)

    # ── Load model ──
    model = RFNet_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=False  # no gradient checkpointing needed at test time
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"RFNet_Liver model: {nparams:.2f}M params")

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    # Handle DataParallel state_dict
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    logging.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ck.get('epoch', '?')})")

    # ══════════════════════════════════════════════════════════════════════
    # A.  TEST ALL COMBINATIONS → Dice & HD95
    # ══════════════════════════════════════════════════════════════════════
    logging.info("=" * 70)
    logging.info("EVALUATING ALL TEST COMBINATIONS ...")
    logging.info("=" * 70)
    dice_results, hd95_results = test_all_combinations(model, test_loader, args,
                                                        args.savepath)
    print_results(dice_results, hd95_results)
    save_results_json(dice_results, hd95_results, args.savepath)
    save_per_subject_scores(dice_results, hd95_results, args.savepath)
    statistical_significance_test(dice_results, hd95_results, args)

    # ══════════════════════════════════════════════════════════════════════
    # B.  EXTRACT FEATURES FOR t-SNE
    # ══════════════════════════════════════════════════════════════════════
    logging.info("Extracting features for t-SNE ...")
    class_features_by_patient, invariance_features_by_patient = extract_features_for_tsne(
        model, test_loader, args)
    total_class = sum(len(v) for v in class_features_by_patient.values())
    total_inv = sum(len(v) for v in invariance_features_by_patient.values())
    logging.info(f"  Class separability samples (total): {total_class}")
    logging.info(f"  Invariance samples (total): {total_inv}")
    for pn in sorted(class_features_by_patient.keys()):
        logging.info(f"    {pn}: class={len(class_features_by_patient[pn])}, inv={len(invariance_features_by_patient.get(pn, []))}")

    # ══════════════════════════════════════════════════════════════════════
    # C.  COMPREHENSIVE VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════
    logging.info("Generating comprehensive visualization ...")
    fig_paths = generate_comprehensive_visualization(
        model, test_loader, args, args.savepath,
        class_features_by_patient, invariance_features_by_patient)
    logging.info(f"Done. Generated {len(fig_paths)} figures in {os.path.join(args.savepath, 'visualizations')}")


if __name__ == "__main__":
    main()
