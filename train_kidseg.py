#!/usr/bin/env python3
"""
Standalone train.py – DC-Seg for multi-modal liver segmentation with missing modalities.

Memory-efficient version:
  - AMP (mixed precision fp16)
  - Gradient checkpointing on encoders
  - No 7-D tensor in fusion (sequential region processing)
  - Sequential modality encoding/decoding
  - No DataParallel overhead

Adapted from:
  DC-Seg: Disentangled Contrastive Learning for Brain Tumor Segmentation
  with Missing Modalities

Modality groups:
  G1 = [T2WI]                          (always present)
  G2 = [C-pre, C+A, C+V, C+Delay]
  G3 = [DWI]
  G4 = [InPhase, OutPhase]

Usage:
  python train_kidseg.py --datapath /path/to/preprocess_nii_256x32

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — Ablation variants, one script, --variant controlled
Each variant trains to its own folder: ./output_dcseg_liver_a2cl_dkd_hac_<variant>
══════════════════════════════════════════════════════════════════════════════

### train

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=1
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

# Row 1 — Baseline (DC-Seg): no AGCL, no DKD, no HGF
nohup python -u train_kidseg.py --variant baseline --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/train_baseline.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_baseline.log

# Row 2 — + AGCL
nohup python -u train_kidseg.py --variant agcl --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/train_agcl.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_agcl.log

# Row 3 — + AGCL + DKD
nohup python -u train_kidseg.py --variant agcl_dkd --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/train_agcl_dkd.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_agcl_dkd.log

# Row 4 — + AGCL + DKD + HGF  (Full KiD-Seg)
nohup python -u train_kidseg.py --variant full --datapath <PROJECT_ROOT>/preprocess_nii_256x32 > <PROJECT_ROOT>/logs/train_full.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_full.log

kill <PID>   # stop a background run (PID printed by nohup, or: ps aux | grep train_kidseg)

# Per-component override (takes precedence over --variant), e.g. full minus HGF:
#   nohup python -u train_kidseg.py --variant full --use_hgf 0 ... &
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import math
import os
import time
import logging
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast('cuda', enabled=enabled)
    def make_scaler(enabled): return _GradScaler('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast(enabled=enabled)
    def make_scaler(enabled): return _GradScaler(enabled=enabled)

# custom_fwd shim: prefer the modern torch.amp API (device_type='cuda'),
# fall back to the deprecated torch.cuda.amp API on older PyTorch.
try:
    from torch.amp import custom_fwd as _custom_fwd
    def amp_custom_fwd(cast_inputs=torch.float32):
        return _custom_fwd(device_type="cuda", cast_inputs=cast_inputs)
except (ImportError, TypeError):
    from torch.cuda.amp import custom_fwd as _custom_fwd
    def amp_custom_fwd(cast_inputs=torch.float32):
        return _custom_fwd(cast_inputs=cast_inputs)
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")
from scipy.ndimage import zoom as scipy_zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def random_group_mask():
    """Random mask: G1 always on, G2/G3/G4 each 50 % on."""
    return groups_to_mask(
        random.random() < 0.5,
        random.random() < 0.5,
        random.random() < 0.5,
    )


# ── Structured Group Dropout ──────────────────────────────────────────
VALID_GROUP_COMBOS = [
    (False, False, False),  # G1 only
    (True,  False, False),  # G1+G2
    (False, True,  False),  # G1+G3
    (False, False, True ),  # G1+G4
    (True,  True,  False),  # G1+G2+G3
    (True,  False, True ),  # G1+G2+G4
    (False, True,  True ),  # G1+G3+G4
    (True,  True,  True ),  # G1+G2+G3+G4  (full)
]


def structured_group_dropout(p_full: float = 0.3):
    """Sample a valid group combination with structured probabilities."""
    if random.random() < p_full:
        return groups_to_mask(True, True, True)
    combo = random.choice(VALID_GROUP_COMBOS[:-1])
    return groups_to_mask(*combo)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA – discovery, split, preprocess, dataset
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


def split_patients(patients, train_ratio=0.8, val_ratio=0.1):
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
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    m, s = vol[mask].mean(), vol[mask].std() + 1e-8
    return (vol - m) / s


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


class LiverDataset(Dataset):
    def __init__(self, data_list, num_cls=2, is_train=True, crop_size=None,
                 p_full=0.3):
        self.data = data_list
        self.num_cls = num_cls
        self.is_train = is_train
        self.crop_size = crop_size
        self.p_full = p_full

    def __len__(self):
        return len(self.data)

    def _random_crop(self, x, y):
        if self.crop_size is None:
            return x, y
        _, X, Y, Z = x.shape
        cx, cy, cz = [min(c, s) for c, s in zip(self.crop_size, (X, Y, Z))]
        sx = random.randint(0, max(0, X - cx))
        sy = random.randint(0, max(0, Y - cy))
        sz = random.randint(0, max(0, Z - cz))
        return x[:, sx:sx+cx, sy:sy+cy, sz:sz+cz], y[sx:sx+cx, sy:sy+cy, sz:sz+cz]

    def _augment(self, x, y):
        for ax in [1, 2, 3]:
            if random.random() < 0.5:
                x = np.flip(x, axis=ax)
                y = np.flip(y, axis=ax - 1)
        for c in range(x.shape[0]):
            x[c] = x[c] * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def __getitem__(self, idx):
        images, label, name = self.data[idx]
        x, y = images.copy(), label.copy()
        x, y = self._random_crop(x, y)
        if self.is_train:
            x, y = self._augment(x, y)
        H, W, Z = y.shape
        yo = np.eye(self.num_cls, dtype=np.float32)[y.ravel()].reshape(H, W, Z, self.num_cls)
        yo = yo.transpose(3, 0, 1, 2)
        mask = np.array(structured_group_dropout(p_full=self.p_full) if self.is_train
                        else [True] * NUM_MODALITIES, dtype=bool)
        mask_max = np.array([True] * NUM_MODALITIES, dtype=bool)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(yo)),
                torch.from_numpy(mask),
                torch.from_numpy(mask_max), name)


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
# 3.  MODEL LAYERS
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


# ── Modal / Region fusion — MEMORY-EFFICIENT (no 7-D tensor) ────────────────

class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modal=8):
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 128, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, num_modal, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        """
        x:   (B, K, C, H, W, Z)  – region-weighted modality features
        prm: (B, 1, H, W, Z)     – region probability for this class
        """
        B, K, C, H, W, Z = x.size()
        prm_avg = torch.mean(prm, dim=(2, 3, 4), keepdim=False) + 1e-7  # (B, 1)
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg.unsqueeze(-1)  # (B, K, C)
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg.view(B, 1, 1, 1, 1)), dim=1)
        weight = self.weight_layer(feat_avg).view(B, K, 1, 1, 1, 1)
        weight = self.sigmoid(weight)
        return torch.sum(x * weight, dim=1)  # (B, C, H, W, Z)


class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel, k_size=1, padding=0),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0))

    def forward(self, region_feats):
        """region_feats: list of num_cls tensors, each (B, C, H, W, Z)."""
        return self.fusion_layer(torch.cat(region_feats, dim=1))


class region_aware_modal_fusion_gen(nn.Module):
    """Memory-efficient region-aware modal fusion.

    KEY FIX: Instead of building the massive 7-D tensor
        (B, num_modal, num_cls, C, H, W, Z)
    we iterate over regions (num_cls=2), and for each region multiply
    the (B, K, C, H, W, Z) features by the (B, 1, H, W, Z) region prob
    using broadcasting — never expanding to 7-D.
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
        x:    (B, K, C, H, W, Z)     stacked modality features
        prm:  (B, num_cls, H, W, Z)  region probability maps
        mask: (B, K)                  boolean modality mask
        """
        B, K, C, H, W, Z = x.size()

        # zero-out missing modalities
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        # per-region fusion — iterate over classes (typically 2), NOT K*cls
        region_feats = []
        for c in range(self.num_cls):
            prm_c = prm[:, c:c+1, :, :, :]                     # (B, 1, H, W, Z)
            # broadcast: (B, K, C, H, W, Z) * (B, 1, 1, H, W, Z) → (B, K, C, H, W, Z)
            region_modal = y * prm_c.unsqueeze(2)                # no new big dim!
            region_feats.append(self.modal_fusions[c](region_modal, prm_c))

        return torch.cat((self.region_fuse(region_feats),
                          self.short_cut(y.view(B, -1, H, W, Z))), dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. HIERARCHICAL ATOMIC-GROUP FUSION
# ═══════════════════════════════════════════════════════════════════════════════

class IntraGroupAttentionPool(nn.Module):
    """Lightweight attention-gated pooling Φ_g (Eq. intragroup)."""

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
    r"""Masked cross-attention :math:`\Psi` (Eq.\ intergroup).

    Implements :math:`\Psi(F_{\texttt{T2WI}}, H_2, H_3, H_4;\, \mathbf{r})`
    as multi-head cross-attention where every spatial voxel independently
    attends across ``G = 4`` group tokens at the *same* position.

    Architecture
    ────────────
    The anchor feature queries all four group tokens (including itself) to
    dynamically select *which* groups are informative at each spatial
    location — far more expressive than static gated summation.

    .. math::

        Q = W_q \cdot f_{\text{anchor}}            \quad (B, n_h, d_k, N) \\
        K = W_k \cdot [f_{\text{anchor}}; H_2; H_3; H_4]  \quad (B, G, n_h, d_k, N) \\
        V = W_v \cdot [f_{\text{anchor}}; H_2; H_3; H_4]  \quad (B, G, n_h, d_k, N)

    Per-voxel scores:

    .. math::

        s_g = \frac{Q \cdot K_g}{\sqrt{d_k}}, \quad
        s_g = -\infty \text{if} r_g = 0, \quad
        \alpha = \text{softmax}(s, \dim=G)

    Output with residual:

    .. math::

        \widetilde{F} = f_{\text{anchor}} + W_o \!\left(
            \sum_{g} \alpha_g \cdot V_g \right)

    The attention matrix is ``(1 × G)`` per voxel per head — negligible
    memory overhead regardless of volume size (G = 4 is tiny).

    Parameters
    ----------
    in_ch : int
        Channel dimension C (must be divisible by ``num_heads``).
    num_heads : int
        Number of attention heads (default 4).
    """

    NUM_GROUPS = 4  # G1(anchor) + G2 + G3 + G4

    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        assert in_ch % num_heads == 0, \
            f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.d_k = in_ch // num_heads
        self.scale = self.d_k ** -0.5

        # ── Linear projections via 1×1×1 Conv3d (spatially local) ──────
        # Q from anchor only; K, V from all group tokens
        self.proj_q = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.proj_k = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.proj_v = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)

        # ── Output projection + normalisation ──────────────────────────
        self.proj_out = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(in_ch),
        )

    # ────────────────────────────────────────────────────────────────────

    def forward(self, f_anchor, h_groups, r):
        """
        Parameters
        ----------
        f_anchor : (B, C, H, W, Z)
            T2WI anchor encoder feature (always present).
        h_groups : list of 3 tensors ``[H_2, H_3, H_4]``
            Each ``(B, C, H, W, Z)``.  Already zero-masked by the caller
            when the corresponding group is absent.
        r : (B, 4) bool
            Per-sample group-presence flags ``[G1, G2, G3, G4]``.

        Returns
        -------
        (B, C, H, W, Z) – fused feature with residual connection to anchor.
        """
        B, C, H, W, Z = f_anchor.shape
        G  = self.NUM_GROUPS       # 4
        nh = self.num_heads
        dk = self.d_k
        N  = H * W * Z             # spatial sequence length

        # ── Stack all group tokens: [anchor, H_2, H_3, H_4] ──────────
        # (B, G, C, H, W, Z)
        groups = torch.stack([f_anchor] + h_groups, dim=1)

        # ── Q from anchor  →  (B, nh, dk, N) ─────────────────────────
        q = self.proj_q(f_anchor)                         # (B, C, H, W, Z)
        q = q.view(B, nh, dk, N)                          # flatten spatial

        # ── K, V from all G groups  →  (B, G, nh, dk, N) ─────────────
        groups_flat = groups.view(B * G, C, H, W, Z)      # batch-project
        k = self.proj_k(groups_flat).view(B, G, nh, dk, N)
        v = self.proj_v(groups_flat).view(B, G, nh, dk, N)

        # ── Per-voxel attention scores ────────────────────────────────
        #   q:   (B, 1, nh, dk, N)   broadcast over G
        #   k:   (B, G, nh, dk, N)
        #   →  element-wise multiply, sum over dk  →  (B, G, nh, N)
        scores = (q.unsqueeze(1) * k).sum(dim=3) * self.scale

        # ── Mask absent groups  →  -inf before softmax ────────────────
        # r: (B, G) bool  →  (B, G, 1, 1) for broadcast over (nh, N)
        attn_mask = r.view(B, G, 1, 1)
        scores = scores.masked_fill(~attn_mask, float('-inf'))

        # ── Softmax over the group dimension ──────────────────────────
        attn_weights = F.softmax(scores, dim=1)            # (B, G, nh, N)
        # Guard: if all groups masked at some position (shouldn't happen
        # since G1 is always present), nan_to_num prevents NaN propagation.
        attn_weights = attn_weights.nan_to_num(0.0)

        # ── Weighted sum of values ────────────────────────────────────
        #   attn:  (B, G, nh, 1, N)   broadcast over dk
        #   v:     (B, G, nh, dk, N)
        #   → sum over G  →  (B, nh, dk, N)
        out = (attn_weights.unsqueeze(3) * v).sum(dim=1)

        # ── Reshape back to spatial + output projection ───────────────
        out = out.reshape(B, C, H, W, Z)
        return f_anchor + self.proj_out(out)


class HierarchicalGroupFusion(nn.Module):
    """Full hierarchical fusion for one encoder scale level.

    1. **Intra-group pooling**: G2's 4 contrast phases → H₂, G4's 2 phases → H₄,
       G3 (DWI single modality) → H₃ = identity.
    2. **Inter-group masked cross-attention**: F̃ = Ψ(F_{T2WI}, H₂, H₃, H₄; r)
    """

    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        self.intra_g2 = IntraGroupAttentionPool(in_ch)
        self.intra_g4 = IntraGroupAttentionPool(in_ch)
        # G3 (DWI) is a single modality → H₃ = F_DWI (no pooling needed)
        self.inter = InterGroupCrossAttention(in_ch, num_heads=num_heads)

    def forward(self, x_stacked, mask):
        """
        x_stacked : (B, 8, C, H, W, Z) – per-modality encoder features.
        mask      : (B, 8) bool         – per-modality presence.
        Returns   : (B, C, H, W, Z)     – hierarchically fused feature.
        """
        # ── Anchor (always present) ──
        f_t2wi = x_stacked[:, 0]

        # ── Intra-group: G2 (contrast phases) ──
        g2_feats = [x_stacked[:, i] for i in GROUP_INDICES["G2"]]
        g2_present = mask[:, GROUP_INDICES["G2"][0]]
        h2 = self.intra_g2(g2_feats)
        h2 = h2 * g2_present.float().view(-1, 1, 1, 1, 1)

        # ── Intra-group: G3 (DWI, single modality) ──
        h3 = x_stacked[:, GROUP_INDICES["G3"][0]]
        g3_present = mask[:, GROUP_INDICES["G3"][0]]
        h3 = h3 * g3_present.float().view(-1, 1, 1, 1, 1)

        # ── Intra-group: G4 (InPhase / OutPhase) ──
        g4_feats = [x_stacked[:, i] for i in GROUP_INDICES["G4"]]
        g4_present = mask[:, GROUP_INDICES["G4"][0]]
        h4 = self.intra_g4(g4_feats)
        h4 = h4 * g4_present.float().view(-1, 1, 1, 1, 1)

        # ── Group-presence vector r ──
        r = torch.stack([
            mask[:, 0],                           # G1 (always True)
            g2_present, g3_present, g4_present,
        ], dim=1)                                  # (B, 4)

        # ── Inter-group masked cross-attention ──
        return self.inter(f_t2wi, [h2, h3, h4], r)


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

    HGF is added at each scale as a gated residual.  Gates are
    initialised near-zero so the decoder starts identical to baseline.
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


# ═══════════════════════════════════════════════════════════════════════════════
#  DC_Seg_Liver – main model (memory-efficient)
# ═══════════════════════════════════════════════════════════════════════════════

class DC_Seg_Liver(nn.Module):
    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=True,
                 use_agcl=True, use_dkd=True, use_hgf=True):
        super().__init__()
        self.num_modal = num_modal
        self.num_cls = num_cls
        self.use_ckpt = use_checkpoint

        # ── Ablation switches ──
        #   use_agcl : Asymmetric Anchor-Guided Contrastive Learning (loss-only)
        #   use_dkd  : Differential Kinetic Disentanglement (DifferenceEncoder +
        #              kinetic pooling + gated injection + KCL)
        #   use_hgf  : Hierarchical Atomic-Group Fusion (gated residuals)
        self.use_agcl = use_agcl
        self.use_dkd = use_dkd
        self.use_hgf = use_hgf

        self.encoders = nn.ModuleList([Encoder() for _ in range(num_modal)])
        self.style_encoders = nn.ModuleList([Style_encoder() for _ in range(num_modal)])
        self.img_decoders = nn.ModuleList([Image_decoder() for _ in range(num_modal)])
        # KiD modules / injection are only constructed when DKD is enabled so the
        # ablated variants have the correct (smaller) parameter count.
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
        """Encode single modality with optional gradient checkpointing."""
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def _compute_z_k(self, x, mask):
        """Compute kinetic embedding z_k from G2 difference maps.

        Returns (z_k, g2_present) where z_k is None if G2 is fully absent.
        Inputs are NOT detached – seg loss gradients flow through to
        diff_encoder via the gated residual in decoder_fuse.

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

    def forward(self, x, mask, return_logits=False):
        """
        x:    (B, M, H, W, D)
        mask: (B, M) bool
        return_logits: if True, also return raw logits for CA distillation
        """
        # ── Kinetic embedding (computed early, injected into decoder) ──
        z_k, g2_present = self._compute_z_k(x, mask)

        # ── encode each modality SEQUENTIALLY ──
        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))

        # stack at each scale: (B, M, C, …)
        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        # fusion decoder
        decoder_out = self.decoder_fuse(
            x1, x2, x3, x4, mask, z_k=z_k, g2_present=g2_present,
            return_logits=return_logits)

        if return_logits:
            fuse_pred, prm_preds, fuse_x4, logits = decoder_out
        else:
            fuse_pred, prm_preds, fuse_x4 = decoder_out

        if not self.is_training:
            if return_logits:
                return fuse_pred, logits
            return fuse_pred

        # ── per-modality seg (regulariser) — SEQUENTIAL ──
        sep_preds = []
        for m in range(self.num_modal):
            sep_preds.append(self.decoder_sep(*feats[m]))

        # ── style + reconstruction — SEQUENTIAL ──
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

        out = (fuse_pred, sep_preds, prm_preds, recon_out,
               mu_list, sigma_list, contents, styles,
               z_k, g2_present)
        if return_logits:
            out = out + (logits,)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

def dice_loss(output, target, num_cls=2, eps=1e-7):
    target = target.float()
    loss = 0.0
    for i in range(num_cls):
        num = torch.sum(output[:, i] * target[:, i])
        den = torch.sum(output[:, i]) + torch.sum(target[:, i]) + eps
        loss += 2.0 * num / den
    return 1.0 - loss / num_cls


def softmax_weighted_loss(output, target, num_cls=2):
    target = target.float()
    B, _, H, W, Z = output.size()
    loss = torch.zeros(1, device=output.device, dtype=output.dtype)
    for i in range(num_cls):
        w = 1.0 - (target[:, i].sum((1, 2, 3)) / (target.sum((1, 2, 3, 4)) + 1e-8))
        w = w.view(-1, 1, 1, 1)
        loss = loss + (-w * target[:, i] * torch.log(output[:, i].clamp(1e-5, 1.0))).mean()
    return loss


def KL_divergence(mu, logvar):
    return 0.5 * torch.mean(-1 + logvar.exp() + mu.pow(2) - logvar)


class Anatomy_Contrastive_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, features):
        B, M = features.size(0), features.size(1)
        self.logit_scale.data.clamp_(0, 4.6052)
        scale = self.logit_scale.exp()
        f = F.normalize(features.mean([3, 4, 5]).view(-1, features.size(2)), 2, 1)
        logits = (f @ f.t()) * scale
        target = torch.zeros_like(logits)
        for i in range(B):
            target[M * i: M * (i + 1), M * i: M * (i + 1)] = 1
        return F.binary_cross_entropy_with_logits(logits, target)


class Modality_Contrastive_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, features):
        B, M = features.size(0), features.size(1)
        self.logit_scale.data.clamp_(0, 4.6052)
        scale = self.logit_scale.exp()
        f = F.normalize(features.permute(1, 0, 2).reshape(-1, features.size(2)), 2, 1)
        logits = (f @ f.t()) * scale
        target = torch.zeros_like(logits)
        for m in range(M):
            target[B * m: B * (m + 1), B * m: B * (m + 1)] = 1
        return F.binary_cross_entropy_with_logits(logits, target)


class AnchorGuidedContrastiveLoss(nn.Module):
    """Asymmetric Anchor-Guided Anatomical Contrastive Learning (Eq. AGCL).

    Treats T2WI (modality index 0) as a fixed anatomical anchor and aligns
    every other *present* modality toward it via one-way InfoNCE with
    stop-gradient on the anchor embeddings.

    Parameters
    ----------
    in_ch : int
        Channel dimension of the input feature maps (bottleneck).
    proj_ch : int
        Dimension of the projected anatomical embeddings z_a^m.
    tau : float
        Temperature for the softmax denominator.
    anchor_idx : int
        Modality index that serves as the anchor (default 0 = T2WI).
    """

    def __init__(self, in_ch: int = basic_dims * 8, proj_ch: int = 128,
                 tau: float = 0.07, anchor_idx: int = 0):
        super().__init__()
        self.tau = tau
        self.anchor_idx = anchor_idx
        # Projection head  g_a : R^C -> R^{proj_ch}
        self.projector = nn.Sequential(
            nn.Linear(in_ch, proj_ch),
            nn.ReLU(inplace=True),
            nn.Linear(proj_ch, proj_ch),
        )

    def forward(self, features, mask):
        """
        Parameters
        ----------
        features : Tensor (B, M, C, H, W, Z)
            Bottleneck encoder features for all modalities.
        mask : Tensor (B, M)  bool
            Which modalities are present per sample.

        Returns
        -------
        loss : scalar tensor   (0 if no non-anchor modalities are present)
        """
        B, M, C = features.size(0), features.size(1), features.size(2)

        # Global average pool over spatial dims -> (B, M, C)
        pooled = features.mean(dim=[3, 4, 5])  # (B, M, C)

        # Project -> (B, M, proj_ch), then L2-normalise
        z = F.normalize(self.projector(pooled), dim=-1)  # (B, M, D)

        # Anchor embeddings (T2WI) – detach to stop gradients
        z_anchor = z[:, self.anchor_idx].detach()  # (B, D)

        loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)
        count = 0

        for m in range(M):
            if m == self.anchor_idx:
                continue
            # Which samples in the batch have modality m present?
            present = mask[:, m]  # (B,) bool
            if not present.any():
                continue

            # z_a^m for present samples
            z_m = z[present, m]  # (N, D)

            # Cosine similarities with ALL anchor embeddings in the batch
            # numerator:  sim(z_a^m_i, sg(z_a^T2_i))  for the positive pair
            # denominator: sum_j sim(z_a^m_i, sg(z_a^T2_j)) over entire batch
            sim_matrix = z_m @ z_anchor.t() / self.tau  # (N, B)

            # Positive indices: each present sample i matches anchor i
            pos_indices = torch.where(present)[0]  # original batch indices
            # For each row r in sim_matrix, the positive column is pos_indices[r]
            labels = pos_indices.to(sim_matrix.device)

            loss = loss + F.cross_entropy(sim_matrix, labels)
            count += 1

        return loss / max(count, 1)


# ── Kinetic Disentanglement for Dynamic Contrast (KiD) ───────────────────────
# Computes phase-to-baseline difference maps  ΔI^q = I^q − I^{C-pre}  for
# q ∈ {C+A, C+V, C+Delay}, encodes them with a dedicated difference encoder
# E_Δ, aggregates via cross-phase attention pooling → z_k, which is:
#   (a) injected into the fusion decoder via gated residual (helps seg)
#   (b) supervised by L_KCL (sharpens kinetic boundary cues)
# ─────────────────────────────────────────────────────────────────────────────

G2_BASELINE_IDX = 1          # C-pre  (within the 8-modality channel dim)
G2_PHASE_INDICES = [2, 3, 4] # C+A, C+V, C+Delay
NUM_KINETIC_PHASES = len(G2_PHASE_INDICES)
KID_CH = basic_dims * 2      # Output channels of DifferenceEncoder


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


class KineticContrastiveLoss(nn.Module):
    """Supervised kinetic contrastive loss L_KCL (SupCon).

    Forces float32 for numerical stability under AMP.
    Stratified fg/bg sampling with minimum-count guards.
    """

    def __init__(self, in_ch: int = KID_CH, proj_ch: int = 64,
                 tau: float = 0.2, num_samples: int = 128):
        super().__init__()
        self.tau = tau
        self.num_samples = num_samples
        self.projector = nn.Sequential(
            nn.Linear(in_ch, proj_ch),
            nn.ReLU(inplace=True),
            nn.Linear(proj_ch, proj_ch),
        )

    @amp_custom_fwd(cast_inputs=torch.float32)
    def forward(self, z_k, target_onehot):
        """
        z_k :           (B, C, H', W', Z')   Kinetic embedding (1/8 res).
        target_onehot : (B, cls, H, W, Z)     Full-res one-hot GT.
        """
        z_k = z_k.float()
        B, C, H, W, Z = z_k.shape

        gt_down = F.interpolate(target_onehot.float(), size=(H, W, Z), mode="nearest")
        labels = gt_down.argmax(dim=1).reshape(B, -1)

        z_flat = z_k.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        S = z_flat.shape[1]
        N = min(self.num_samples, S)

        loss = torch.tensor(0.0, device=z_k.device, dtype=torch.float32)
        valid = 0

        for b in range(B):
            fg_idx = (labels[b] > 0).nonzero(as_tuple=False).view(-1)
            bg_idx = (labels[b] == 0).nonzero(as_tuple=False).view(-1)

            n_fg = min(N // 2, len(fg_idx))
            n_bg = min(N - n_fg, len(bg_idx))
            if n_fg < 2 or n_bg < 2:
                continue

            fg_sel = fg_idx[torch.randperm(len(fg_idx), device=z_k.device)[:n_fg]]
            bg_sel = bg_idx[torch.randperm(len(bg_idx), device=z_k.device)[:n_bg]]
            perm = torch.cat([fg_sel, bg_sel])
            actual_N = len(perm)

            z_proj = F.normalize(self.projector(z_flat[b, perm]), dim=-1)
            y_samp = labels[b, perm]

            sim = z_proj @ z_proj.t() / self.tau
            mask_self = ~torch.eye(actual_N, dtype=torch.bool, device=z_k.device)
            mask_pos = (y_samp.unsqueeze(0) == y_samp.unsqueeze(1)) & mask_self

            n_pos = mask_pos.sum(dim=1)
            has_pos = n_pos > 0
            if not has_pos.any():
                continue

            sim_max = sim.detach().max(dim=1, keepdim=True).values
            logits = sim - sim_max
            exp_logits = torch.exp(logits)
            denom = (exp_logits * mask_self.float()).sum(dim=1, keepdim=True)
            log_prob = logits - torch.log(denom + 1e-8)

            per_sample = (log_prob * mask_pos.float()).sum(dim=1) / n_pos.float().clamp(min=1)
            loss = loss + (-per_sample[has_pos].mean())
            valid += 1

        return loss / max(valid, 1)


# ── Completeness-Aware Self-Distillation ─────────────────

class CompletenessAwareDistillationLoss(nn.Module):
    """L_CA: KL divergence between teacher (full-modality) and student (incomplete).

    L_CA = KL( σ(sg(P_max) / T) || σ(P_S / T) )

    CRITICAL: Uses reduction='mean' (not 'batchmean') for dense segmentation.
    With 'batchmean', loss sums over H×W×Z×C elements but only divides by B,
    producing values ~500,000× too large for typical 3D volumes.

    Forces float32 for numerical stability under AMP.
    """

    def __init__(self, temperature: float = 2.0, max_loss: float = 10.0):
        super().__init__()
        self.T = temperature
        self.max_loss = max_loss  # safety clamp

    @amp_custom_fwd(cast_inputs=torch.float32)
    def forward(self, logits_teacher, logits_student):
        """
        logits_teacher : (B, C, H, W, Z) raw logits from S_max (will be sg'd).
        logits_student : (B, C, H, W, Z) raw logits from incomplete S pass.
        Returns scalar loss.
        """
        logits_teacher = logits_teacher.float().detach()  # stop gradient
        logits_student = logits_student.float()

        # Temperature-scaled softmax
        p_teacher = F.softmax(logits_teacher / self.T, dim=1)
        log_p_student = F.log_softmax(logits_student / self.T, dim=1)

        # KL(teacher || student) – reduction='mean' averages over ALL elements
        # (B × C × H × W × Z), keeping loss magnitude independent of volume size
        loss = F.kl_div(log_p_student, p_teacher, reduction="mean")

        # Scale by T² to match gradient magnitudes with hard labels
        loss = loss * (self.T ** 2)

        # Safety clamp: prevent any remaining outliers from destabilising training
        return loss.clamp(max=self.max_loss)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING / VALIDATION / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, epoch, args,
                    ana_cl, mod_cl, agcl, kcl, ca_loss_fn):
    model.train()
    model.is_training = True
    num_cls = args.num_cls
    losses = []

    # KCL ramp-up
    if epoch < args.kcl_start_epoch:
        kcl_scale = 0.0
    else:
        kcl_scale = min(1.0, (epoch - args.kcl_start_epoch) / max(1, args.kcl_ramp_epochs))

    # CA ramp-up (delayed start to let model learn basic segmentation first)
    if epoch < args.ca_start_epoch:
        ca_scale = 0.0
    else:
        ca_scale = min(1.0, (epoch - args.ca_start_epoch) / max(1, args.ca_ramp_epochs))

    for i, (x, target, mask, mask_max, _) in enumerate(loader):
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        mask_max = mask_max.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Teacher pass (all modalities, no grad) for CA distillation ──
        # Completeness-aware self-distillation is part of the HGF component
        # It is only active when HGF is enabled.
        teacher_logits = None
        if model.use_hgf and ca_scale > 0:
            with torch.no_grad():
                with make_autocast(args.amp):
                    model.is_training = False
                    _, teacher_logits = model(x, mask_max, return_logits=True)
                    model.is_training = True

        ca_active = model.use_hgf and ca_scale > 0
        with make_autocast(args.amp):
            out = model(x, mask, return_logits=ca_active)

            if ca_active:
                (fuse_pred, sep_preds, prm_preds, recon_out,
                 mu_list, sigma_list, contents, styles,
                 z_k, g2_present, student_logits) = out
            else:
                (fuse_pred, sep_preds, prm_preds, recon_out,
                 mu_list, sigma_list, contents, styles,
                 z_k, g2_present) = out

            fuse_loss = softmax_weighted_loss(fuse_pred, target, num_cls) + \
                        dice_loss(fuse_pred, target, num_cls)
            sep_loss = sum(softmax_weighted_loss(sp, target, num_cls) +
                           dice_loss(sp, target, num_cls) for sp in sep_preds)
            prm_loss = sum(softmax_weighted_loss(pp, target, num_cls) +
                           dice_loss(pp, target, num_cls) for pp in prm_preds)

            if epoch < args.region_fusion_start_epoch:
                loss = sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

            alpha = 1.0
            recon_l = F.mse_loss(recon_out, x)
            kl_l = sum(KL_divergence(mu_list[m], torch.log(sigma_list[m].square() + 1e-8))
                       for m in range(args.num_modal))
            ana_l = ana_cl(contents)
            mod_l = mod_cl(styles)

            # Anchor-Guided Contrastive Learning (AGCL) – ablation-gated
            agcl_l = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if model.use_agcl:
                agcl_l = agcl(contents, mask)

            # Kinetic Contrastive Loss (KCL) – ablation-gated (part of DKD)
            kcl_l = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if (model.use_dkd and kcl_scale > 0 and z_k is not None
                    and g2_present is not None and g2_present.any()):
                kcl_l = kcl(z_k.detach(), target)

            # Completeness-Aware Self-Distillation (L_CA, Eq. distill)
            ca_l = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if ca_active and teacher_logits is not None:
                # Only distill when student uses incomplete modalities
                is_incomplete = ~(mask == mask_max).all(dim=1)  # (B,)
                if is_incomplete.any():
                    ca_l = ca_loss_fn(
                        teacher_logits[is_incomplete],
                        student_logits[is_incomplete],
                    )

            loss = loss + alpha * (recon_l + ana_l + mod_l
                                   + args.agcl_weight * agcl_l
                                   + args.kcl_weight * kcl_scale * kcl_l
                                   + args.ca_weight * ca_scale * ca_l
                                   + kl_l)

        scaler.scale(loss).backward()

        # Gradient clipping for stability (prevents CA outliers from causing spikes)
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if (i + 1) % max(1, len(loader) // 3) == 0:
            kcl_str = f"  kcl={kcl_l.item():.4f}×{kcl_scale:.2f}" if kcl_scale > 0 else ""
            ca_str = f"  ca={ca_l.item():.4f}×{ca_scale:.2f}" if ca_scale > 0 else ""
            gate_str = ""
            if hasattr(model, 'decoder_fuse') and hasattr(model.decoder_fuse, 'kid_gate_logit'):
                gv = torch.sigmoid(model.decoder_fuse.kid_gate_logit).item()
                gate_str = f"  kid_g={gv:.4f}"
            hgf_str = ""
            if hasattr(model, 'decoder_fuse') and hasattr(model.decoder_fuse, 'hgf_gate4'):
                hg4 = torch.sigmoid(model.decoder_fuse.hgf_gate4).item()
                hgf_str = f"  hgf4={hg4:.4f}"
            logging.info(f"  Ep {epoch+1} it {i+1}/{len(loader)}  loss={loss.item():.4f}"
                         f"{kcl_str}{ca_str}{gate_str}{hgf_str}")

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, args):
    model.eval()
    model.is_training = False
    dices = []
    for x, target, mask, _mask_max, _ in loader:
        x, target, mask = x.cuda(), target.cuda(), mask.cuda()
        with make_autocast(args.amp):
            pred = model(x, mask)
        pl = pred.argmax(1)
        gl = target.argmax(1)
        eps = 1e-8
        for b in range(x.size(0)):
            p, g = (pl[b] == 1).float(), (gl[b] == 1).float()
            dices.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))
    return np.array([d.item() for d in dices])


@torch.no_grad()
def test_all_combinations(model, test_loader, args, save_dir):
    model.eval()
    model.is_training = False
    results = {}
    all_combo_dice = []

    for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
        mask_list = groups_to_mask(g2, g3, g4)
        present = [MODALITY_NAMES[i] for i, v in enumerate(mask_list) if v]
        logging.info(f"Testing {combo_name}: {present}")

        scores = []
        for x, y_int, _ in test_loader:
            x, y_int = x.cuda(), y_int.cuda()
            mask = torch.tensor([mask_list] * x.size(0), dtype=torch.bool, device=x.device)
            with make_autocast(args.amp):
                pred = model(x, mask)
            pl = pred.argmax(1)
            eps = 1e-8
            for b in range(x.size(0)):
                p, g = (pl[b] == 1).float(), (y_int[b] == 1).float()
                scores.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))

        arr = np.array([s.item() for s in scores])
        m, s, md = arr.mean(), arr.std(), np.median(arr)
        results[combo_name] = {"mean": m, "std": s, "median": md, "scores": arr.tolist()}
        all_combo_dice.append(arr)
        logging.info(f"  {combo_name}: ${m:.3f}\ +/- {s:.3f}({md:.3f})")

    flat = np.concatenate(all_combo_dice)
    results["Average"] = {"mean": flat.mean(), "std": flat.std(), "median": np.median(flat)}
    logging.info(f"  Average: ${flat.mean():.3f}\ +/- {flat.std():.3f}({np.median(flat):.3f})")

    sav = {k: {kk: float(vv) for kk, vv in v.items() if kk != "scores"} for k, v in results.items()}
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(sav, f, indent=2)
    return results


@torch.no_grad()
def visualize_test(model, test_loader, args, save_dir):
    model.eval()
    model.is_training = False
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    mask_all = groups_to_mask(True, True, True)

    for x, y_int, names in test_loader:
        x = x.cuda()
        mask = torch.tensor([mask_all] * x.size(0), dtype=torch.bool, device=x.device)
        with make_autocast(args.amp):
            pred = model(x, mask)
        pl = pred.argmax(1)

        for b in range(x.size(0)):
            name = names[b] if isinstance(names, (list, tuple)) else names
            t2 = x[b, 0].cpu().numpy()
            gt = y_int[b].cpu().numpy()
            pr = pl[b].cpu().numpy()
            z = t2.shape[2] // 2

            base = t2[:, :, z].astype(np.float32)
            base -= base.min()
            if base.max() > 0:
                base /= base.max()
            rgb = np.stack([base]*3, -1)

            gt_s, pr_s = (gt[:, :, z] > 0), (pr[:, :, z] > 0)
            go, po, bo = gt_s & ~pr_s, pr_s & ~gt_s, gt_s & pr_s
            a = 0.55
            ov = rgb.copy()
            ov[go, 0] *= (1-a);            ov[go, 1] = ov[go, 1]*(1-a)+a; ov[go, 2] *= (1-a)
            ov[po, 0] = ov[po, 0]*(1-a)+a; ov[po, 1] *= (1-a);           ov[po, 2] *= (1-a)
            ov[bo, 0] = ov[bo, 0]*(1-a)+a; ov[bo, 1] = ov[bo, 1]*(1-a)+a; ov[bo, 2] *= (1-a)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"{name} z={z}  (R=Pred G=GT Y=Overlap)")
            ax.axis("off"); fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"{name}_z{z}.png"), dpi=150)
            plt.close(fig)
    logging.info(f"Saved visualizations to {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class LR_Scheduler:
    """Cosine annealing: lr = base_lr * 0.5 * (1 + cos(pi * epoch / T))"""
    def __init__(self, base_lr, num_epochs):
        self.lr, self.T = base_lr, num_epochs

    def __call__(self, optimizer, epoch):
        lr = self.lr * 0.5 * (1 + math.cos(math.pi * epoch / self.T))
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr


class _RepeatSampler:
    def __init__(self, s):
        self.s = s
    def __iter__(self):
        while True:
            yield from iter(self.s)


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.s)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DC-Seg Liver – Memory-Efficient Training")
    p.add_argument("--datapath", default="./data/preprocess_nii_256x32")
    p.add_argument("--savepath", default=None,
                   help="Output dir. If omitted, defaults to "
                        "./output_dcseg_liver_a2cl_dkd_hac_<variant>")
    p.add_argument("--resume", default=None)

    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)
    p.add_argument("--crop_x", type=int, default=None)
    p.add_argument("--crop_y", type=int, default=None)
    p.add_argument("--crop_z", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--region_fusion_start_epoch", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main thread, safest on Windows)")

    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--num_modal", type=int, default=8)

    p.add_argument("--amp", action="store_true", default=True,
                   help="Mixed-precision training (default ON)")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--no_checkpoint", dest="use_checkpoint", action="store_false",
                   default=True, help="Disable gradient checkpointing")

    # Anchor-Guided Contrastive Learning (AGCL)
    p.add_argument("--agcl_tau", type=float, default=0.07,
                   help="Temperature for AGCL InfoNCE (default 0.07)")
    p.add_argument("--agcl_proj_ch", type=int, default=128,
                   help="Projection dimension for AGCL head")
    p.add_argument("--agcl_weight", type=float, default=1.0,
                   help="Weight for AGCL loss (multiplied by alpha)")

    # Kinetic Disentanglement (KiD) – kinetic contrastive loss
    p.add_argument("--kcl_tau", type=float, default=0.2,
                   help="Temperature for KCL SupCon (default 0.2)")
    p.add_argument("--kcl_proj_ch", type=int, default=64,
                   help="Projection dimension for KCL head")
    p.add_argument("--kcl_weight", type=float, default=1.0,
                   help="Weight for KCL loss (multiplied by alpha × ramp)")
    p.add_argument("--kcl_num_samples", type=int, default=128,
                   help="Number of voxels sampled per volume for KCL")
    p.add_argument("--kcl_start_epoch", type=int, default=10,
                   help="Epoch at which KCL begins (warmup)")
    p.add_argument("--kcl_ramp_epochs", type=int, default=20,
                   help="Epochs over which KCL linearly ramps 0→1")

    # Completeness-Aware Self-Distillation – CORRECTED DEFAULTS
    p.add_argument("--ca_temperature", type=float, default=2.0,
                   help="Temperature T for CA distillation (default 2.0)")
    p.add_argument("--ca_weight", type=float, default=1.0,
                   help="Weight for L_CA loss (multiplied by alpha × ramp)")
    p.add_argument("--ca_start_epoch", type=int, default=30,
                   help="Epoch at which CA distillation begins (must be late "
                        "enough for teacher to be meaningful)")
    p.add_argument("--ca_ramp_epochs", type=int, default=30,
                   help="Epochs over which CA loss linearly ramps 0→1")
    p.add_argument("--ca_max_loss", type=float, default=10.0,
                   help="Safety clamp on CA loss to prevent outlier explosions")

    # Structured Group Dropout
    p.add_argument("--sgd_p_full", type=float, default=0.3,
                   help="Probability of sampling full-modality combo (default 0.3)")

    # Gradient clipping for training stability
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm (0 = disabled, default 1.0)")

    # ── Ablation control ──────────────────────────────────────────
    # A single training script reproduces every row of the ablation table via
    # the --variant preset, with optional per-component overrides:
    #   baseline  → DC-Seg (no AGCL, no DKD, no HGF)
    #   agcl      → + AGCL
    #   agcl_dkd  → + AGCL + DKD
    #   full      → + AGCL + DKD + HGF  (Full KiD-Seg, default)
    p.add_argument("--variant", choices=["baseline", "agcl", "agcl_dkd", "full"],
                   default="full",
                   help="Ablation preset. Overridable by "
                        "--use_agcl/--use_dkd/--use_hgf.")
    p.add_argument("--use_agcl", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) AGCL (else follow --variant)")
    p.add_argument("--use_dkd", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) DKD/KiD (else follow --variant)")
    p.add_argument("--use_hgf", type=int, default=None, choices=[0, 1],
                   help="Override: enable(1)/disable(0) HGF (else follow --variant)")

    return resolve_ablation(p.parse_args())


def resolve_ablation(args):
    """Resolve --variant + per-component overrides into boolean ablation flags.

    Maps the ablation presets onto (use_agcl, use_dkd, use_hgf) and
    applies any explicit --use_* overrides. Mutates and returns ``args``.
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

    # Auto-append the variant to the default output dir so each ablation run
    # writes to its own folder (explicit --savepath is respected as-is).
    if args.savepath is None:
        args.savepath = f"./output_dcseg_liver_a2cl_dkd_hac_{args.variant}"
    return args


def setup_logging(path):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        filename=os.path.join(path, "train.log"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("").addHandler(ch)


def _print_results(results):
    print("\n" + "=" * 70)
    print("TEST RESULTS (Dice)")
    print("=" * 70)
    for name, _ in TEST_COMBINATIONS:
        r = results[name]
        print(f"  {name:25s}:  ${r['mean']:.3f}\ +/- {r['std']:.3f}({r['median']:.3f})")
    r = results["Average"]
    print(f"  {'Average':25s}:  ${r['mean']:.3f}\ +/- {r['std']:.3f}({r['median']:.3f})")
    print("=" * 70)


def main():
    args = parse_args()
    setup_logging(args.savepath)
    logging.info(f"Args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # ── patients ──
    patients = discover_patients(args.datapath)
    logging.info(f"Found {len(patients)} patients: {patients}")
    if len(patients) < 3:
        logging.warning("Very few patients – using all for train/val/test")
        train_p = val_p = test_p = patients
    else:
        train_p, val_p, test_p = split_patients(patients)
    logging.info(f"Train: {train_p}  Val: {val_p}  Test: {test_p}")

    # ── preprocess into RAM ──
    shape = (args.resize_x, args.resize_y, args.resize_z)
    logging.info(f"Preprocessing to {shape} ...")

    def load_set(pl):
        data = []
        for n in pl:
            logging.info(f"  Loading {n} ...")
            imgs, lbl = preprocess_patient(args.datapath, n, shape)
            data.append((imgs, lbl, n))
        return data

    train_data = load_set(train_p)
    val_data = load_set(val_p)
    test_data = load_set(test_p)
    logging.info("All data loaded into RAM.")

    crop = None
    if args.crop_x or args.crop_y or args.crop_z:
        crop = (args.crop_x or args.resize_x,
                args.crop_y or args.resize_y,
                args.crop_z or args.resize_z)

    train_set = LiverDataset(train_data, args.num_cls, is_train=True, crop_size=crop,
                             p_full=args.sgd_p_full)
    val_set = LiverDataset(val_data, args.num_cls, is_train=False)
    test_set = LiverTestDataset(test_data)

    train_loader = MultiEpochsDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)

    # ── model (single GPU, NO DataParallel) ──
    logging.info(f"Ablation variant='{args.variant}'  "
                 f"AGCL={args.use_agcl}  DKD={args.use_dkd}  HGF={args.use_hgf}")
    model = DC_Seg_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=args.use_checkpoint,
        use_agcl=args.use_agcl,
        use_dkd=args.use_dkd,
        use_hgf=args.use_hgf,
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model: {nparams:.2f}M params | AMP={args.amp} | GradCkpt={args.use_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = make_scaler(args.amp)
    lr_sched = LR_Scheduler(args.lr, args.num_epochs)

    ana_cl = Anatomy_Contrastive_Loss().cuda()
    mod_cl = Modality_Contrastive_Loss().cuda()
    agcl = AnchorGuidedContrastiveLoss(
        in_ch=basic_dims * 8,
        proj_ch=args.agcl_proj_ch,
        tau=args.agcl_tau,
        anchor_idx=0,  # T2WI
    ).cuda()
    kcl = KineticContrastiveLoss(
        in_ch=KID_CH,
        proj_ch=args.kcl_proj_ch,
        tau=args.kcl_tau,
        num_samples=args.kcl_num_samples,
    ).cuda()
    ca_loss_fn = CompletenessAwareDistillationLoss(
        temperature=args.ca_temperature,
        max_loss=args.ca_max_loss,
    ).cuda()
    optimizer.add_param_group({"params": ana_cl.parameters(), "lr": args.lr})
    optimizer.add_param_group({"params": mod_cl.parameters(), "lr": args.lr})
    optimizer.add_param_group({"params": agcl.parameters(), "lr": args.lr})
    optimizer.add_param_group({"params": kcl.parameters(), "lr": args.lr})

    # ── evaluate only ──
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # handle DataParallel → single-GPU state_dict
        sd = ck["state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded checkpoint epoch {ck.get('epoch', '?')} (strict=False for KiD compat)")
        results = test_all_combinations(model, test_loader, args, args.savepath)
        _print_results(results)
        visualize_test(model, test_loader, args, args.savepath)
        return

    # ── train ──
    best_dice = 0.0
    t0 = time.time()

    for epoch in range(args.num_epochs):
        lr = lr_sched(optimizer, epoch)
        logging.info(f"Epoch {epoch+1}/{args.num_epochs}  lr={lr:.6f}")
        avg = train_one_epoch(model, train_loader, optimizer, scaler, epoch, args,
                              ana_cl, mod_cl, agcl, kcl, ca_loss_fn)
        logging.info(f"  Train loss: {avg:.4f}")

        if (epoch + 1) % 10 == 0 or epoch >= args.num_epochs - 5:
            da = validate(model, val_loader, args)
            vd = da.mean()
            logging.info(f"  Val Dice: {vd:.4f}")
            if vd > best_dice:
                best_dice = vd
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                             "optim": optimizer.state_dict(), "best_dice": best_dice},
                            os.path.join(args.savepath, "model_best.pth"))
                logging.info(f"  ★ Best: {best_dice:.4f}")

        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                     "optim": optimizer.state_dict()},
                    os.path.join(args.savepath, "model_last.pth"))
        if (epoch + 1) % 50 == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                        os.path.join(args.savepath, f"model_epoch{epoch+1}.pth"))

    logging.info(f"Training done in {(time.time()-t0)/3600:.2f}h")

    # ── final test ──
    bp = os.path.join(args.savepath, "model_best.pth")
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location="cpu", weights_only=False)["state_dict"])
    results = test_all_combinations(model, test_loader, args, args.savepath)
    _print_results(results)
    visualize_test(model, test_loader, args, args.savepath)


if __name__ == "__main__":
    main()