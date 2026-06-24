#!/usr/bin/env python3
# Copyright (c) 2026 Ant Group and the KiD-Seg authors.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file or
# https://creativecommons.org/licenses/by-nc/4.0/ for details.
# For research and non-commercial use only; not for clinical use.
"""
Standalone test.py – DC-Seg + A²CL + DKD + HAC for multi-modal liver segmentation
with missing modalities.

Evaluates the best model on the TEST split (same split as training) and reports:
  • Per-combination Dice and HD95: mean ± std (median)
  • Average Dice and HD95 across all 8 TEST_COMBINATIONS

Generates a comprehensive visualization figure per patient:
  Row 1 : 8 MRI modalities (middle axial slice)
  Row 2 : 8 TEST_COMBINATION overlay maps (pred=green, GT=red, overlap=yellow) on T2WI
  Row 3-L: t-SNE – Class Separability under Severe Missingness (G1-only)
  Row 3-R: t-SNE – Feature Invariance to Input Permutations across combinations

Model: DC_Seg_Liver with
  - HierarchicalGroupFusion (HAC) gated residuals at every decoder scale
  - Kinetic Disentanglement (KiD): DifferenceEncoder + KineticAttentionPool
  - Decoder_fuse with KiD injection and HGF gates

Usage:
  python test_kidseg.py --datapath /path/to/preprocess_nii_256x32

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — Ablation variants, one script, --variant controlled
--variant MUST match the variant used to train the checkpoint. When --checkpoint
and --savepath are omitted they auto-resolve to the matching training folder
(./output_dcseg_liver_a2cl_dkd_hac_<variant>/...), so no manual paths are needed.
══════════════════════════════════════════════════════════════════════════════

### test

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=0
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

# Row 1 — Baseline (DC-Seg)
nohup python -u test_kidseg.py --variant baseline --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/test_baseline.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_baseline.log

# Row 2 — + AGCL
nohup python -u test_kidseg.py --variant agcl --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/test_agcl.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_agcl.log

# Row 3 — + AGCL + DKD
nohup python -u test_kidseg.py --variant agcl_dkd --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/test_agcl_dkd.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_agcl_dkd.log

# Row 4 — + AGCL + DKD + HGF  (Full KiD-Seg)
nohup python -u test_kidseg.py --variant full --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/test_full.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_full.log

kill <PID>   # stop a background run (PID printed by nohup, or: ps aux | grep test_kidseg)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import json
import logging
import random
from pathlib import Path

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

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & GROUP DEFINITIONS
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
# 2.  DATA
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
# 3.  MODEL LAYERS  (DC-Seg + A²CL + KiD + HAC)
# ═══════════════════════════════════════════════════════════════════════════════
basic_dims = 16

# ── KiD constants ────────────────────────────────────────────────────────────
G2_BASELINE_IDX  = 1           # C-pre  (within the 8-modality channel dim)
G2_PHASE_INDICES = [2, 3, 4]   # C+A, C+V, C+Delay
NUM_KINETIC_PHASES = len(G2_PHASE_INDICES)
KID_CH = basic_dims * 2        # Output channels of DifferenceEncoder


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


class BasicConv(nn.Module):
    def __init__(self, inp, out, ks, stride=1, padding=0, relu=True, norm=True, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(inp, out, ks, stride, padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Adaptive_instance_norm(nn.Module):
    def forward(self, content, gamma, beta, epsilon=1e-5):
        c_mean = torch.mean(content, [2, 3, 4], keepdim=True)
        c_std = torch.std(content, [2, 3, 4], keepdim=True)
        return gamma * ((content - c_mean) / (c_std + epsilon)) + beta


class Adaptive_resblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = BasicConv(in_planes, out_planes, 3, 1, 1, relu=False, norm=False)
        self.i_norm1 = Adaptive_instance_norm()
        self.conv2 = BasicConv(in_planes, out_planes, 3, 1, 1, relu=False, norm=False)
        self.i_norm2 = Adaptive_instance_norm()

    def forward(self, x_init, mu, sigma):
        x = F.relu(self.i_norm1(self.conv1(x_init), sigma, mu), inplace=True)
        x = self.i_norm2(self.conv2(x), sigma, mu)
        return x + x_init


# ── PRM generators ───────────────────────────────────────────────────────────

class prm_generator_laststage(nn.Module):
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
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return self.prm_layer(self.embedding_layer(y.view(B, -1, H, W, Z)))


class prm_generator(nn.Module):
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
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        return self.prm_layer(torch.cat((x1, self.embedding_layer(y.view(B, -1, H, W, Z))), 1))


# ── Modal / Region fusion — MEMORY-EFFICIENT (no 7-D tensor) ─────────────────

class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modal=8):
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 128, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, num_modal, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        B, K, C, H, W, Z = x.size()
        prm_avg = torch.mean(prm, dim=(2, 3, 4), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg.unsqueeze(-1)
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg.view(B, 1, 1, 1, 1)), dim=1)
        weight = self.weight_layer(feat_avg).view(B, K, 1, 1, 1, 1)
        weight = self.sigmoid(weight)
        return torch.sum(x * weight, dim=1)


class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel, k_size=1, padding=0),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0))

    def forward(self, region_feats):
        return self.fusion_layer(torch.cat(region_feats, dim=1))


class region_aware_modal_fusion_gen(nn.Module):
    """Memory-efficient region-aware modal fusion (no 7-D tensor)."""
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
# 3b. HIERARCHICAL ATOMIC-GROUP FUSION – HAC
# ═══════════════════════════════════════════════════════════════════════════════

class IntraGroupAttentionPool(nn.Module):
    """Lightweight attention-gated pooling Φ_g."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_ch, max(in_ch // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_ch // 4, 4), 1, 1, bias=False),
        )

    def forward(self, feats: list):
        if len(feats) == 1:
            return feats[0]
        stacked = torch.stack(feats, dim=1)
        B, G, C, H, W, Z = stacked.shape
        scores = self.attn(
            stacked.reshape(B * G, C, H, W, Z)
        ).reshape(B, G, 1, H, W, Z)
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)


class InterGroupCrossAttention(nn.Module):
    """Masked cross-attention Ψ (Eq. intergroup).

    Implements Ψ(F_{T2WI}, H_2, H_3, H_4; r) as multi-head cross-attention
    where every spatial voxel independently attends across G=4 group tokens.
    """
    NUM_GROUPS = 4  # G1(anchor) + G2 + G3 + G4

    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        assert in_ch % num_heads == 0, \
            f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.d_k = in_ch // num_heads
        self.scale = self.d_k ** -0.5

        self.proj_q = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.proj_k = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.proj_v = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)

        self.proj_out = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(in_ch),
        )

    def forward(self, f_anchor, h_groups, r):
        """
        f_anchor : (B, C, H, W, Z) – T2WI anchor (always present)
        h_groups : list of 3 tensors [H_2, H_3, H_4], each (B, C, H, W, Z)
        r        : (B, 4) bool – per-sample group-presence flags [G1, G2, G3, G4]
        Returns  : (B, C, H, W, Z)
        """
        B, C, H, W, Z = f_anchor.shape
        G  = self.NUM_GROUPS
        nh = self.num_heads
        dk = self.d_k
        N  = H * W * Z

        groups = torch.stack([f_anchor] + h_groups, dim=1)  # (B, G, C, H, W, Z)

        q = self.proj_q(f_anchor).view(B, nh, dk, N)

        groups_flat = groups.view(B * G, C, H, W, Z)
        k = self.proj_k(groups_flat).view(B, G, nh, dk, N)
        v = self.proj_v(groups_flat).view(B, G, nh, dk, N)

        scores = (q.unsqueeze(1) * k).sum(dim=3) * self.scale  # (B, G, nh, N)

        attn_mask = r.view(B, G, 1, 1)
        scores = scores.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=1).nan_to_num(0.0)  # (B, G, nh, N)

        out = (attn_weights.unsqueeze(3) * v).sum(dim=1)  # (B, nh, dk, N)

        out = out.reshape(B, C, H, W, Z)
        return f_anchor + self.proj_out(out)


class HierarchicalGroupFusion(nn.Module):
    """Full hierarchical fusion for one encoder scale level.

    1. Intra-group pooling: G2's 4 contrast phases → H₂, G4's 2 phases → H₄,
       G3 (DWI, single modality) → H₃ = identity.
    2. Inter-group masked cross-attention: F̃ = Ψ(F_{T2WI}, H₂, H₃, H₄; r)
    """

    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        self.intra_g2 = IntraGroupAttentionPool(in_ch)
        self.intra_g4 = IntraGroupAttentionPool(in_ch)
        self.inter = InterGroupCrossAttention(in_ch, num_heads=num_heads)

    def forward(self, x_stacked, mask):
        """
        x_stacked : (B, 8, C, H, W, Z) – per-modality encoder features
        mask      : (B, 8) bool         – per-modality presence
        Returns   : (B, C, H, W, Z)
        """
        f_t2wi = x_stacked[:, 0]

        g2_feats = [x_stacked[:, i] for i in GROUP_INDICES["G2"]]
        g2_present = mask[:, GROUP_INDICES["G2"][0]]
        h2 = self.intra_g2(g2_feats)
        h2 = h2 * g2_present.float().view(-1, 1, 1, 1, 1)

        h3 = x_stacked[:, GROUP_INDICES["G3"][0]]
        g3_present = mask[:, GROUP_INDICES["G3"][0]]
        h3 = h3 * g3_present.float().view(-1, 1, 1, 1, 1)

        g4_feats = [x_stacked[:, i] for i in GROUP_INDICES["G4"]]
        g4_present = mask[:, GROUP_INDICES["G4"][0]]
        h4 = self.intra_g4(g4_feats)
        h4 = h4 * g4_present.float().view(-1, 1, 1, 1, 1)

        r = torch.stack([
            mask[:, 0],
            g2_present, g3_present, g4_present,
        ], dim=1)  # (B, 4)

        return self.inter(f_t2wi, [h2, h3, h4], r)


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. KINETIC DISENTANGLEMENT (KiD)
# ═══════════════════════════════════════════════════════════════════════════════

class DifferenceEncoder(nn.Module):
    """Lightweight 3D encoder E_Δ for single-channel difference maps.
    3 down-sampling stages → 1/8 spatial resolution (matching bottleneck)."""

    def __init__(self, out_ch: int = KID_CH):
        super().__init__()
        c = basic_dims
        self.net = nn.Sequential(
            general_conv3d(1, c, pad_type="reflect"),
            general_conv3d(c, c * 2, stride=2, pad_type="reflect"),
            general_conv3d(c * 2, c * 2, pad_type="reflect"),
            general_conv3d(c * 2, out_ch, stride=2, pad_type="reflect"),
            general_conv3d(out_ch, out_ch, pad_type="reflect"),
            general_conv3d(out_ch, out_ch, stride=2, pad_type="reflect"),
        )

    def forward(self, x):
        """x: (B, 1, H, W, Z) → (B, out_ch, H/8, W/8, Z/8)."""
        return self.net(x)


class KineticAttentionPool(nn.Module):
    """Attention pooling across kinetic phases → z_k per voxel."""

    def __init__(self, in_ch: int = KID_CH):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch // 4, 1, 1, bias=False),
        )

    def forward(self, phase_feats: list):
        """phase_feats: list of Q tensors (B, C, H, W, Z) → (B, C, H, W, Z)."""
        stacked = torch.stack(phase_feats, dim=1)      # (B, Q, C, H, W, Z)
        B, Q, C, H, W, Z = stacked.shape
        flat = stacked.reshape(B * Q, C, H, W, Z)
        scores = self.attn_fc(flat).reshape(B, Q, 1, H, W, Z)
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL – Encoder / Decoder / DC-Seg
# ═══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
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
    """Shared decoder for per-modality regularisation."""
    def __init__(self, num_cls=2):
        super().__init__()
        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")
        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")
        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")
        self.seg = nn.Conv3d(basic_dims, num_cls, 1, bias=True)
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
    """Fusion decoder with region-aware modal fusion + hierarchical group fusion.

    HGF is added at each scale as a gated residual. Gates are
    initialised near-zero so the decoder starts identical to baseline.
    Optionally injects kinetic features z_k (KiD) at the bottleneck.
    """
    def __init__(self, num_cls=2, num_modal=8, kid_ch=0, use_hgf=True):
        super().__init__()
        c = basic_dims
        self.use_hgf = use_hgf
        self.d3_c1 = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_c2 = general_conv3d(c * 8, c * 4, pad_type="reflect")
        self.d3_out = general_conv3d(c * 4, c * 4, k_size=1, padding=0, pad_type="reflect")
        self.d2_c1 = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_c2 = general_conv3d(c * 4, c * 2, pad_type="reflect")
        self.d2_out = general_conv3d(c * 2, c * 2, k_size=1, padding=0, pad_type="reflect")
        self.d1_c1 = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_c2 = general_conv3d(c * 2, c, pad_type="reflect")
        self.d1_out = general_conv3d(c, c, k_size=1, padding=0, pad_type="reflect")
        self.seg = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.RFM4 = region_aware_modal_fusion_gen(c * 8, num_cls, num_modal)
        self.RFM3 = region_aware_modal_fusion_gen(c * 4, num_cls, num_modal)
        self.RFM2 = region_aware_modal_fusion_gen(c * 2, num_cls, num_modal)
        self.RFM1 = region_aware_modal_fusion_gen(c * 1, num_cls, num_modal)
        self.prm4 = prm_generator_laststage(c * 8, num_cls, num_modal)
        self.prm3 = prm_generator(c * 4, num_cls, num_modal)
        self.prm2 = prm_generator(c * 2, num_cls, num_modal)
        self.prm1 = prm_generator(c * 1, num_cls, num_modal)

        # ── Hierarchical Group Fusion at each scale ──
        # Built only when HGF is enabled (ablation control), so the baseline
        # variant has an identical parameter count to DC-Seg.
        if self.use_hgf:
            self.hgf4 = HierarchicalGroupFusion(c * 8)
            self.hgf3 = HierarchicalGroupFusion(c * 4)
            self.hgf2 = HierarchicalGroupFusion(c * 2)
            self.hgf1 = HierarchicalGroupFusion(c * 1)
            # Gates initialised near-zero: sigmoid(-3) ≈ 0.047
            self.hgf_gate4 = nn.Parameter(torch.tensor(-3.0))
            self.hgf_gate3 = nn.Parameter(torch.tensor(-3.0))
            self.hgf_gate2 = nn.Parameter(torch.tensor(-3.0))
            self.hgf_gate1 = nn.Parameter(torch.tensor(-3.0))

        # ── KiD gated injection (optional) ──
        self.has_kid = kid_ch > 0
        if self.has_kid:
            self.kid_proj = nn.Sequential(
                general_conv3d(kid_ch, c * 4, k_size=1, padding=0, pad_type="reflect"),
                general_conv3d(c * 4, c * 8, k_size=1, padding=0, pad_type="reflect"),
            )
            self.kid_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x1, x2, x3, x4, mask, z_k=None, g2_present=None,
                return_logits=False):
        p4 = self.prm4(x4, mask)
        d4 = self.RFM4(x4, p4.detach(), mask)

        # HGF at scale 4
        if self.use_hgf:
            hgf4_out = self.hgf4(x4, mask)
            d4 = d4 + torch.sigmoid(self.hgf_gate4) * hgf4_out

        # Inject kinetic features
        if self.has_kid and z_k is not None:
            gate = torch.sigmoid(self.kid_gate_logit)
            z_k_proj = self.kid_proj(z_k)
            if g2_present is not None and not g2_present.all():
                z_k_proj = z_k_proj * g2_present.float().view(-1, 1, 1, 1, 1)
            d4 = d4 + gate * z_k_proj

        fuse_x4 = d4
        d4 = self.d3_c1(self.up2(d4))

        p3 = self.prm3(d4, x3, mask)
        d3 = self.RFM3(x3, p3.detach(), mask)
        if self.use_hgf:
            d3 = d3 + torch.sigmoid(self.hgf_gate3) * self.hgf3(x3, mask)
        d3 = self.d3_out(self.d3_c2(torch.cat((d3, d4), 1)))

        d3 = self.d2_c1(self.up2(d3))
        p2 = self.prm2(d3, x2, mask)
        d2 = self.RFM2(x2, p2.detach(), mask)
        if self.use_hgf:
            d2 = d2 + torch.sigmoid(self.hgf_gate2) * self.hgf2(x2, mask)
        d2 = self.d2_out(self.d2_c2(torch.cat((d2, d3), 1)))

        d2 = self.d1_c1(self.up2(d2))
        p1 = self.prm1(d2, x1, mask)
        d1 = self.RFM1(x1, p1.detach(), mask)
        if self.use_hgf:
            d1 = d1 + torch.sigmoid(self.hgf_gate1) * self.hgf1(x1, mask)
        d1 = self.d1_out(self.d1_c2(torch.cat((d1, d2), 1)))

        logits = self.seg(d1)
        pred = self.softmax(logits)
        prm_preds = (p1, self.up2(p2), self.up4(p3), self.up8(p4))

        if return_logits:
            return pred, prm_preds, fuse_x4, logits
        return pred, prm_preds, fuse_x4


class Style_encoder(nn.Module):
    def __init__(self, in_channels=1, n=32):
        super().__init__()
        self.encoder = nn.Sequential(
            BasicConv(in_channels, n, 7, 1, 3, relu=True, norm=False),
            BasicConv(n, n * 2, 4, 2, 1, relu=True, norm=False),
            BasicConv(n * 2, n * 4, 4, 2, 1, relu=True, norm=False),
            BasicConv(n * 4, n * 4, 4, 2, 1, relu=True, norm=False),
            BasicConv(n * 4, n * 4, 4, 2, 1, relu=True, norm=False))
        self.final = BasicConv(n * 4, n * 4, 1, 2, 0, relu=False, norm=False)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, [2, 3, 4], keepdim=True)
        return self.final(x)


class MLP(nn.Module):
    def __init__(self, in_ch=128, mlp_ch=128):
        super().__init__()
        self.ch = mlp_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, mlp_ch), nn.ReLU(True),
            nn.Linear(mlp_ch, mlp_ch), nn.ReLU(True))
        self.l_mu = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)

    def forward(self, s):
        x = self.net(s.view(s.size(0), -1))
        return (self.l_mu(x).view(-1, self.ch, 1, 1, 1),
                self.l_sigma(x).view(-1, self.ch, 1, 1, 1))


class Image_decoder(nn.Module):
    def __init__(self, in_style=128, in_content=128, mlp_ch=128):
        super().__init__()
        ch = mlp_ch
        self.mlp = MLP(in_style, mlp_ch)
        self.res_blocks = nn.ModuleList([Adaptive_resblock(in_content, ch) for _ in range(4)])
        dec, c = [], ch
        for _ in range(3):
            dec.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear"),
                BasicConv(c, c // 2, 5, 1, 2, relu=False, norm=False)))
            c //= 2
        self.dec_blocks = nn.ModuleList(dec)
        self.final = BasicConv(c, 1, 7, 1, 3, relu=False, norm=False)

    def forward(self, style, content):
        mu, sigma = self.mlp(style)
        x = content
        for rb in self.res_blocks:
            x = rb(x, mu, sigma)
        for db in self.dec_blocks:
            x = db(x)
            x = F.layer_norm(x, x.shape[1:])
            x = F.relu(x, inplace=True)
        return self.final(x), mu, sigma


class DC_Seg_Liver(nn.Module):
    """DC-Seg-Liver with A²CL + KiD + HAC.

    At test time (is_training=False), forward() returns only fuse_pred.
    forward_with_features() returns pred + bottleneck features for t-SNE.
    """
    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=False,
                 use_agcl=True, use_dkd=True, use_hgf=True):
        super().__init__()
        self.num_modal = num_modal
        self.num_cls = num_cls
        self.use_ckpt = use_checkpoint

        # ── Ablation switches – must match training configuration ──
        self.use_agcl = use_agcl
        self.use_dkd = use_dkd
        self.use_hgf = use_hgf

        self.encoders = nn.ModuleList([Encoder() for _ in range(num_modal)])
        self.style_encoders = nn.ModuleList([Style_encoder() for _ in range(num_modal)])
        self.img_decoders = nn.ModuleList([Image_decoder() for _ in range(num_modal)])
        kid_ch = KID_CH if use_dkd else 0
        self.decoder_fuse = Decoder_fuse(num_cls, num_modal, kid_ch=kid_ch,
                                         use_hgf=use_hgf)
        self.decoder_sep = Decoder_sep(num_cls)

        # ── Kinetic Disentanglement (KiD) ──
        if self.use_dkd:
            self.diff_encoder = DifferenceEncoder(out_ch=KID_CH)
            self.kinetic_pool = KineticAttentionPool(in_ch=KID_CH)

        self.is_training = False
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def _encode_one(self, encoder, x_m):
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def _compute_z_k(self, x, mask):
        """Compute kinetic embedding z_k from G2 difference maps.

        Returns (z_k, g2_present) where z_k is None if G2 is fully absent.
        When DKD is disabled (ablation), returns (None, None).
        """
        if not self.use_dkd:
            return None, None

        g2_present = mask[:, G2_BASELINE_IDX]
        for pidx in G2_PHASE_INDICES:
            g2_present = g2_present & mask[:, pidx]

        if not g2_present.any():
            return None, g2_present

        baseline = x[:, G2_BASELINE_IDX: G2_BASELINE_IDX + 1]
        phase_feats = []
        for pidx in G2_PHASE_INDICES:
            diff_map = x[:, pidx: pidx + 1] - baseline
            if not g2_present.all():
                diff_map = diff_map * g2_present.float().view(-1, 1, 1, 1, 1)
            phase_feats.append(self.diff_encoder(diff_map))
        z_k = self.kinetic_pool(phase_feats)
        return z_k, g2_present

    def forward(self, x, mask):
        """
        x:    (B, M, H, W, D)
        mask: (B, M) bool
        At test time returns: fuse_pred (B, num_cls, H, W, D)
        """
        z_k, g2_present = self._compute_z_k(x, mask)

        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))

        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        fuse_pred, prm_preds, fuse_x4 = self.decoder_fuse(
            x1, x2, x3, x4, mask, z_k=z_k, g2_present=g2_present)

        if not self.is_training:
            return fuse_pred

        # ── per-modality seg (regulariser) ──
        sep_preds = []
        for m in range(self.num_modal):
            sep_preds.append(self.decoder_sep(*feats[m]))

        # ── style + reconstruction ──
        recon_list, mu_list, sigma_list, styles_raw = [], [], [], []
        for m in range(self.num_modal):
            st = self.style_encoders[m](x[:, m:m+1])
            styles_raw.append(st)
            rec, mu, sig = self.img_decoders[m](st, fuse_x4)
            recon_list.append(rec)
            mu_list.append(mu)
            sigma_list.append(sig)

        recon_out = torch.cat(recon_list, 1)
        contents = x4
        styles = torch.stack(styles_raw, 1).squeeze(-1).squeeze(-1).squeeze(-1)

        return (fuse_pred, sep_preds, prm_preds, recon_out,
                mu_list, sigma_list, contents, styles,
                z_k, g2_present)

    def forward_with_features(self, x, mask):
        """Forward pass that returns prediction + multi-scale features for t-SNE.

        Returns:
            fuse_pred:  (B, num_cls, H, W, D)
            fuse_x4:    (B, C, H/8, W/8, D/8) bottleneck (for GAP-pooled invariance)
            x1_full:    (B, K, C1, H, W, D)    full-res encoder features (for voxel class sep.)
        """
        z_k, g2_present = self._compute_z_k(x, mask)

        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))

        x1 = torch.stack([f[0] for f in feats], 1)  # (B, K, C, H, W, D) full-res
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        fuse_pred, prm_preds, fuse_x4 = self.decoder_fuse(
            x1, x2, x3, x4, mask, z_k=z_k, g2_present=g2_present)
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

    Each entry contains:
      • Stats over ALL patients: mean, std, median, scores
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
        h_finite = np.where(np.isinf(h_arr), np.nan, h_arr)

        dice_results[combo_name] = {
            "mean": float(d_arr.mean()), "std": float(d_arr.std()),
            "median": float(np.median(d_arr)), "scores": d_arr.tolist(),
        }
        hd95_results[combo_name] = {
            "mean": float(np.nanmean(h_finite)), "std": float(np.nanstd(h_finite)),
            "median": float(np.nanmedian(h_finite)), "scores": h_arr.tolist(),
        }
        all_dice.append(d_arr)
        all_hd95.append(h_finite)

        logging.info(f"  {combo_name} Dice : "
                     f"{d_arr.mean():.3f} +/- {d_arr.std():.3f}"
                     f"  ({np.median(d_arr):.3f})")
        logging.info(f"  {combo_name} HD95 : "
                     f"{np.nanmean(h_finite):.3f} +/- {np.nanstd(h_finite):.3f}"
                     f"  ({np.nanmedian(h_finite):.3f})")

    flat_d = np.concatenate(all_dice)
    flat_h = np.concatenate(all_hd95)

    dice_results["Average"] = {
        "mean": float(flat_d.mean()), "std": float(flat_d.std()),
        "median": float(np.median(flat_d)),
    }
    hd95_results["Average"] = {
        "mean": float(np.nanmean(flat_h)), "std": float(np.nanstd(flat_h)),
        "median": float(np.nanmedian(flat_h)),
    }

    return dice_results, hd95_results


def print_results(dice_results, hd95_results):
    """Print results to terminal."""
    def _fmt(d, h):
        dm, ds, dmd = d["mean"], d["std"], d["median"]
        hm, hs, hmd = h["mean"], h["std"], h["median"]
        d_str = f"{dm:.3f} +/- {ds:.3f}  ({dmd:.3f})"
        h_str = f"{hm:.3f} +/- {hs:.3f}  ({hmd:.3f})"
        return d_str, h_str

    print("\n" + "=" * 90)
    print(f"{'Combination':25s} | {'Dice':35s} | {'HD95':35s}")
    print("=" * 90)
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str, h_str = _fmt(d, h)
        print(f"  {name:23s} | {d_str:33s} | {h_str:33s}")
    print("-" * 90)
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str, h_str = _fmt(d, h)
    print(f"  {'Average':23s} | {d_str:33s} | {h_str:33s}")
    print("=" * 90)
def save_results_json(dice_results, hd95_results, save_dir):
    """Save results to JSON."""
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

    # GT only → green
    ov[gt_only, 0] *= (1 - alpha)
    ov[gt_only, 1] = ov[gt_only, 1] * (1 - alpha) + alpha
    ov[gt_only, 2] *= (1 - alpha)
    # Pred only → red
    ov[pred_only, 0] = ov[pred_only, 0] * (1 - alpha) + alpha
    ov[pred_only, 1] *= (1 - alpha)
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
                ax.set_title(f"{MODALITY_NAMES[i]}\n({grp})", fontsize=9, fontweight="bold",
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

            fig.suptitle(
                f"DC-Seg (A²CL+KiD+HAC) Test Visualization — Patient: {name} — Axial Slice z={z_mid}\n"
                f"Overlays: Green = GT only,  Red = Pred only,  Yellow = Overlap",
                fontsize=14, fontweight="bold", y=0.99)

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
    p = argparse.ArgumentParser(
        description="DC-Seg (A²CL+KiD+HAC) Liver – Standalone Test Script")
    p.add_argument("--datapath", default="./data/preprocess_nii_256x32")
    p.add_argument("--checkpoint", default=None,
                   help="Path to best model checkpoint. If omitted, defaults to "
                        "./output_dcseg_liver_a2cl_dkd_hac_<variant>/model_best.pth")
    p.add_argument("--savepath", default=None,
                   help="Output dir. If omitted, defaults to "
                        "./output_dcseg_liver_a2cl_dkd_hac_<variant>/test_output")

    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)

    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--num_modal", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    # ── Ablation control – MUST match the trained checkpoint ──────
    #   baseline → DC-Seg | agcl → +AGCL | agcl_dkd → +AGCL+DKD |
    #   full     → +AGCL+DKD+HGF (Full KiD-Seg, default)
    p.add_argument("--variant", choices=["baseline", "agcl", "agcl_dkd", "full"],
                   default="full",
                   help="Ablation preset. Must match the checkpoint's "
                        "training variant. Overridable by --use_agcl/--use_dkd/--use_hgf.")
    p.add_argument("--use_agcl", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) AGCL (else follow --variant)")
    p.add_argument("--use_dkd", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) DKD/KiD (else follow --variant)")
    p.add_argument("--use_hgf", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) HGF (else follow --variant)")

    p.add_argument("--compare_json", default=None,
                   help="Path to another method's per_subject_scores.json. When given, "
                        "a paired two-sided Wilcoxon signed-rank test (p<0.05) is run "
                        "against it, per combination and overall.")

    return resolve_ablation(p.parse_args())


def resolve_ablation(args):
    """Resolve --variant + per-component overrides into boolean ablation flags.

    Mirrors the training-side resolver so the test model is rebuilt with exactly
    the same architecture as the loaded checkpoint. Mutates and returns ``args``.
    """
    presets = {
        "baseline": (False, False, False),  # DC-Seg
        "agcl":     (True,  False, False),  # + AGCL
        "agcl_dkd": (True,  True,  False),  # + AGCL + DKD
        "full":     (True,  True,  True),   # + AGCL + DKD + HGF (Full KiD-Seg)
    }
    a, d, h = presets[args.variant]
    if args.use_agcl is not None:
        a = bool(args.use_agcl)
    if args.use_dkd is not None:
        d = bool(args.use_dkd)
    if args.use_hgf is not None:
        h = bool(args.use_hgf)
    args.use_agcl, args.use_dkd, args.use_hgf = a, d, h

    # Mirror the training-side variant-suffixed defaults so the test script
    # finds the matching checkpoint/output without manual path juggling
    # (explicit --checkpoint / --savepath are respected as-is).
    base = f"./output_dcseg_liver_a2cl_dkd_hac_{args.variant}"
    if args.checkpoint is None:
        args.checkpoint = f"{base}/model_best.pth"
    if args.savepath is None:
        args.savepath = f"{base}/test_output"
    return args


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

    # ── Discover & split patients (identical logic to training) ──
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

    # ── Load model (DC-Seg + A²CL + KiD + HAC) ──
    logging.info(f"Ablation variant='{args.variant}'  "
                 f"AGCL={args.use_agcl}  DKD={args.use_dkd}  HGF={args.use_hgf}")
    model = DC_Seg_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=False,  # no gradient checkpointing at test time
        use_agcl=args.use_agcl,
        use_dkd=args.use_dkd,
        use_hgf=args.use_hgf,
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model: {nparams:.2f}M params")

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    # Handle DataParallel state_dict
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    logging.info(f"Loaded checkpoint: {args.checkpoint} (epoch {ck.get('epoch', '?')})")

    # ══════════════════════════════════════════════════════════════════════
    # A.  TEST ALL COMBINATIONS → Dice & HD95
    # ══════════════════════════════════════════════════════════════════════
    logging.info("=" * 70)
    logging.info("EVALUATING ALL TEST COMBINATIONS ...")

    logging.info("=" * 70)
    dice_results, hd95_results = test_all_combinations(
        model, test_loader, args, args.savepath)
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
