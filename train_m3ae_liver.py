#!/usr/bin/env python3
# Copyright (c) 2026 Ant Group and the KiD-Seg authors.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file or
# https://creativecommons.org/licenses/by-nc/4.0/ for details.
# For research and non-commercial use only; not for clinical use.
"""
train_m3ae_liver.py – M3AE for multi-modal liver segmentation with missing modalities.

Implements the full M3AE framework (AAAI 2023, arXiv:2303.05302) adapted from
the DC-Seg codebase.  All data-loading, modality-group, evaluation, and CLI
infrastructure is preserved verbatim; only the model and training loops change.

Two-stage training
──────────────────
Stage 1 – M3AE Self-Supervised Pretraining (--pretrain_epochs)
  • Random modality dropout  +  random 3-D patch masking at combined 87.5 %
  • Masked/dropped content replaced by a learnable substitute image x_sub
  • Network minimises MSE reconstruction loss
  • x_sub is updated jointly via back-prop 

Stage 2 – Fine-Tuning with Heterogeneous Self-Distillation (--num_epochs)
  • Segmentation head (Dice + CE) with deep supervision at ½ and ¼ resolution
  • Two random missing-modal instantiations per sample → bottleneck consistency loss
  • x_sub frozen; used as substitute for missing modalities at inference

Modality groups:
  G1 = [T2WI]                   (always present)
  G2 = [C-pre, C+A, C+V, C+Delay]
  G3 = [DWI]
  G4 = [InPhase, OutPhase]

Usage:
  # Full two-stage training
  python train_m3ae_liver.py --datapath /path/to/data --savepath ./out

  # Stage-1 only
  python train_m3ae_liver.py --datapath /path/to/data --savepath ./out \\
      --pretrain_epochs 100 --num_epochs 0

  # Stage-2 from pretrained checkpoint
  python train_m3ae_liver.py --datapath /path/to/data --savepath ./out \\
      --pretrain_epochs 0 --num_epochs 200 --resume_pretrain ./out/pretrain_best.pth

  # Evaluate only
  python train_m3ae_liver.py --datapath /path/to/data --savepath ./out \\
      --resume ./out/model_best.pth

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — Two-stage M3AE training
══════════════════════════════════════════════════════════════════════════════

### train

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=0
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

# Full two-stage M3AE training (Stage 1: self-supervised pretraining + Stage 2: fine-tuning)
nohup python -u train_m3ae_liver.py --datapath ./data/preprocess_nii_256x32 --savepath ./output_m3ae_liver --pretrain_epochs 100 --num_epochs 200 > <PROJECT_ROOT>/logs/train_m3ae_liver.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_m3ae_liver.log
"""
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast('cuda', enabled=enabled)
    def make_scaler(enabled):   return _GradScaler('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast(enabled=enabled)
    def make_scaler(enabled):   return _GradScaler(enabled=enabled)

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
# 1.  CONSTANTS & GROUP DEFINITIONS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
MODALITY_NAMES    = ["T2WI", "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase"]
MODALITY_SUFFIXES = MODALITY_NAMES[:]
NUM_MODALITIES    = len(MODALITY_NAMES)  # 8

GROUP_INDICES = {
    "G1": [0],
    "G2": [1, 2, 3, 4],
    "G3": [5],
    "G4": [6, 7],
}

TEST_COMBINATIONS = [
    ("G1",          (False, False, False)),
    ("G1+G2",       (True,  False, False)),
    ("G1+G3",       (False, True,  False)),
    ("G1+G4",       (False, False, True )),
    ("G1+G2+G3",    (True,  True,  False)),
    ("G1+G2+G4",    (True,  False, True )),
    ("G1+G3+G4",    (False, True,  True )),
    ("G1+G2+G3+G4", (True,  True,  True )),
]

VALID_GROUP_COMBOS = [
    (False, False, False),
    (True,  False, False),
    (False, True,  False),
    (False, False, True ),
    (True,  True,  False),
    (True,  False, True ),
    (False, True,  True ),
    (True,  True,  True ),
]


def groups_to_mask(g2: bool, g3: bool, g4: bool):
    m = [False] * NUM_MODALITIES
    for i in GROUP_INDICES["G1"]: m[i] = True
    if g2:
        for i in GROUP_INDICES["G2"]: m[i] = True
    if g3:
        for i in GROUP_INDICES["G3"]: m[i] = True
    if g4:
        for i in GROUP_INDICES["G4"]: m[i] = True
    return m


def structured_group_dropout(p_full: float = 0.3):
    if random.random() < p_full:
        return groups_to_mask(True, True, True)
    combo = random.choice(VALID_GROUP_COMBOS[:-1])
    return groups_to_mask(*combo)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA  (unchanged)
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
            for s in MODALITY_SUFFIXES)
        if ok:
            patients.append(d.name)
    patients.sort()
    return patients


def split_patients(patients, train_ratio=0.7, val_ratio=0.2):
    n       = len(patients)
    n_train = max(1, int(round(n * train_ratio)))
    n_val   = max(1, int(round(n * val_ratio)))
    n_test  = max(1, n - n_train - n_val)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n, 0, 0
    return patients[:n_train], patients[n_train:n_train+n_val], patients[n_train+n_val:]


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


class LiverDataset(Dataset):
    def __init__(self, data_list, num_cls=2, is_train=True,
                 crop_size=None, p_full=0.3):
        self.data      = data_list
        self.num_cls   = num_cls
        self.is_train  = is_train
        self.crop_size = crop_size
        self.p_full    = p_full

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
        mask = np.array(
            structured_group_dropout(p_full=self.p_full) if self.is_train
            else [True] * NUM_MODALITIES, dtype=bool)
        mask_max = np.array([True] * NUM_MODALITIES, dtype=bool)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(yo)),
                torch.from_numpy(mask),
                torch.from_numpy(mask_max),
                name)


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
# 3.  BASIC BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════
basic_dims = 16


def normalization(planes, norm="gn"):
    if norm == "bn": return nn.BatchNorm3d(planes)
    if norm == "gn": return nn.GroupNorm(min(4, planes), planes)
    if norm == "in": return nn.InstanceNorm3d(planes)
    raise ValueError(f"Unknown norm: {norm}")


class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1,
                 pad_type="reflect", norm="gn", act_type="lrelu", relufactor=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride, padding,
                              padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        self.act  = (nn.ReLU(inplace=True) if act_type == "relu"
                     else nn.LeakyReLU(relufactor, inplace=True))

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3d(nn.Module):
    """Two-conv residual block with the same in/out channel count."""
    def __init__(self, ch, norm="gn"):
        super().__init__()
        self.c1 = general_conv3d(ch, ch, norm=norm)
        self.c2 = general_conv3d(ch, ch, norm=norm)

    def forward(self, x):
        return x + self.c2(self.c1(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  M3AE ENCODER / DECODER / HEADS
# ═══════════════════════════════════════════════════════════════════════════════

class M3AE_Encoder(nn.Module):
    """
    Single-stream 3-D U-Net encoder.

    Accepts the full N-channel multimodal volume (missing modalities already
    replaced by x_sub before this call).  Produces four skip-connection
    feature tensors at spatial scales 1/1, 1/2, 1/4, 1/8.

    This is the architecture described in the original M3AE work:
      "essentially a 3D U-Net comprising a single encoder and a single decoder
       employing residual blocks and group normalisation"
    """
    def __init__(self, in_ch: int = NUM_MODALITIES, norm: str = "gn"):
        super().__init__()
        c = basic_dims
        # scale 1/1
        self.e1_in  = general_conv3d(in_ch, c, pad_type="reflect", norm=norm)
        self.e1_res = ResBlock3d(c, norm=norm)
        # scale 1/2
        self.e2_down = general_conv3d(c,     c * 2, stride=2, pad_type="reflect", norm=norm)
        self.e2_res  = ResBlock3d(c * 2, norm=norm)
        # scale 1/4
        self.e3_down = general_conv3d(c * 2, c * 4, stride=2, pad_type="reflect", norm=norm)
        self.e3_res  = ResBlock3d(c * 4, norm=norm)
        # scale 1/8  (bottleneck)
        self.e4_down = general_conv3d(c * 4, c * 8, stride=2, pad_type="reflect", norm=norm)
        self.e4_res  = ResBlock3d(c * 8, norm=norm)

    def forward(self, x):
        """x : (B, N, H, W, Z)  →  (f1, f2, f3, f4)"""
        f1 = self.e1_res(self.e1_in(x))          # (B, C,   H,   W,   Z  )
        f2 = self.e2_res(self.e2_down(f1))        # (B, 2C,  H/2, W/2, Z/2)
        f3 = self.e3_res(self.e3_down(f2))        # (B, 4C,  H/4, W/4, Z/4)
        f4 = self.e4_res(self.e4_down(f3))        # (B, 8C,  H/8, W/8, Z/8) bottleneck
        return f1, f2, f3, f4


class M3AE_Decoder(nn.Module):
    """
    U-Net decoder with skip connections.  Returns features at three scales so
    that the caller can attach either the regression head (pretraining) or the
    segmentation head with deep supervision (fine-tuning).
    """
    def __init__(self, norm: str = "gn"):
        super().__init__()
        c = basic_dims
        # 1/8 → 1/4
        self.up3  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c = general_conv3d(c * 8 + c * 4, c * 4, pad_type="reflect", norm=norm)
        self.d3_r = ResBlock3d(c * 4, norm=norm)
        # 1/4 → 1/2
        self.up2  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c = general_conv3d(c * 4 + c * 2, c * 2, pad_type="reflect", norm=norm)
        self.d2_r = ResBlock3d(c * 2, norm=norm)
        # 1/2 → 1/1
        self.up1  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c = general_conv3d(c * 2 + c,     c,     pad_type="reflect", norm=norm)
        self.d1_r = ResBlock3d(c, norm=norm)

    def forward(self, f1, f2, f3, f4):
        """Returns (d1, d2, d3) at full / ½ / ¼ resolution."""
        d3 = self.d3_r(self.d3_c(torch.cat([self.up3(f4), f3], dim=1)))
        d2 = self.d2_r(self.d2_c(torch.cat([self.up2(d3), f2], dim=1)))
        d1 = self.d1_r(self.d1_c(torch.cat([self.up1(d2), f1], dim=1)))
        return d1, d2, d3


class RegressionHead(nn.Module):
    """
    1×1×1 convolution without nonlinearity.
    Used during Stage-1 pretraining to reconstruct the N-channel input.
    (Regression head: 1x1x1 convolution without sigmoid)
    """
    def __init__(self, in_ch: int = basic_dims, out_ch: int = NUM_MODALITIES):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(x)


class SegmentationHead(nn.Module):
    """
    Three 1×1×1 convolutions for the main prediction and two deep-supervision
    auxiliary outputs at ½ and ¼ resolution (upsampled to full resolution).

    Deep supervision:
        L_seg = Σ_{α ∈ {1, ½, ¼}} L(s^gt, ŝ^α)
    """
    def __init__(self, num_cls: int = 2):
        super().__init__()
        c              = basic_dims
        self.head1     = nn.Conv3d(c,     num_cls, kernel_size=1, bias=True)  # full res
        self.head_half = nn.Conv3d(c * 2, num_cls, kernel_size=1, bias=True)  # ½ res
        self.head_qtr  = nn.Conv3d(c * 4, num_cls, kernel_size=1, bias=True)  # ¼ res
        self.softmax   = nn.Softmax(dim=1)
        self.up2       = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4       = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

    def forward(self, d1, d2, d3):
        """Returns (pred_full, pred_half_up, pred_qtr_up) all at full resolution."""
        p1 = self.softmax(self.head1(d1))
        p2 = self.softmax(self.head_half(d2))
        p3 = self.softmax(self.head_qtr(d3))
        return p1, self.up2(p2), self.up4(p3)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  M3AE MASKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_substitute(x: torch.Tensor,
                     x_sub: torch.Tensor,
                     modality_mask: torch.Tensor) -> torch.Tensor:
    """
    Replace channels of x where the corresponding modality is absent with the
    matching channel of x_sub.  Implements S(x, x_sub): substitute missing modalities with x_sub.

    x             : (B, N, H, W, Z)
    x_sub         : (1, N, H, W, Z)  – learnable substitute image
    modality_mask : (B, N) bool       – True = modality is present
    Returns       : (B, N, H, W, Z)
    """
    B, N = x.shape[0], x.shape[1]
    mask_f = modality_mask.view(B, N, 1, 1, 1).float()           # (B,N,1,1,1)
    sub    = x_sub.expand(B, -1, -1, -1, -1)                     # (B,N,H,W,Z)
    return x * mask_f + sub * (1.0 - mask_f)


def mask_patches_3d(x: torch.Tensor,
                    x_sub: torch.Tensor,
                    patch_size: int = 16,
                    target_mask_ratio: float = 0.875) -> torch.Tensor:
    """
    Randomly replace a fraction of non-overlapping 3-D patches with x_sub.

    The M3AE paper uses side length P=16 and a combined masking ratio of 87.5 %
    (higher than the 75 % used in the natural-image MAE, because cross-modal
    information makes reconstruction easier).

    x                 : (B, N, H, W, Z)
    x_sub             : (1, N, H, W, Z)
    patch_size        : P – side length of cubic patches
    target_mask_ratio : fraction of patches to mask
    Returns           : (B, N, H, W, Z)
    """
    B, N, H, W, Z = x.shape
    P = patch_size

    nH = H // P
    nW = W // P
    nZ = Z // P
    n_patches = nH * nW * nZ

    if n_patches == 0:       # volume too small; skip patch masking
        return x

    n_masked = max(1, int(round(n_patches * target_mask_ratio)))
    out = x.clone()

    for b in range(B):
        perm = torch.randperm(n_patches, device=x.device)[:n_masked]
        for idx_t in perm:
            idx = idx_t.item()
            zi  = idx % nZ
            tmp = idx // nZ
            wi  = tmp % nW
            hi  = tmp // nW
            hs, ws, zs = hi * P, wi * P, zi * P
            out[b, :, hs:hs+P, ws:ws+P, zs:zs+P] = \
                x_sub[0, :, hs:hs+P, ws:ws+P, zs:zs+P]

    return out


def prepare_masked_input(x: torch.Tensor,
                         x_sub: torch.Tensor,
                         modality_mask: torch.Tensor,
                         patch_size: int = 16,
                         patch_mask_ratio: float = 0.875) -> torch.Tensor:
    """
    Full M3AE masking pipeline:
      1. Whole-modality dropout  →  replace absent channels with x_sub
      2. Patch masking of remaining modalities  →  replace patches with x_sub

    The resulting tensor S(x, x_sub) feeds directly into the encoder.
    """
    x_m = apply_substitute(x, x_sub, modality_mask)          # step 1
    x_m = mask_patches_3d(x_m, x_sub,                        # step 2
                           patch_size=patch_size,
                           target_mask_ratio=patch_mask_ratio)
    return x_m


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  M3AE_Liver — main model
# ═══════════════════════════════════════════════════════════════════════════════

class M3AE_Liver(nn.Module):
    """
    M3AE catch-all model for multi-modal liver segmentation.

    A single 3-D U-Net (one encoder, one decoder) handles all possible subsets
    of the 8 input modalities.  The learnable substitute image x_sub (shape
    (1, N, H, W, Z)) is initialised lazily on the first call and registered as
    a nn.Parameter so it is updated by the same optimiser as the network.

    Forward modes
    ─────────────
    forward_pretrain(x)
        Stage-1: returns (loss_mse, x_sub) — MSE reconstruction loss plus a
        reference to x_sub for the L2 regularisation term.

    forward_finetune(x, mask)
        Stage-2 training: returns (pred, (pred_half, pred_qtr), loss_con)
        — segmentation predictions at three scales plus the bottleneck
        self-distillation consistency loss.

    forward(x, mask)
        Inference: returns the full-resolution segmentation probability map.
    """

    def __init__(self,
                 num_cls: int       = 2,
                 num_modal: int     = NUM_MODALITIES,
                 patch_size: int    = 16,
                 mask_ratio: float  = 0.875,
                 sub_init_std: float = 0.1,
                 use_checkpoint: bool = True):
        super().__init__()
        self.num_cls      = num_cls
        self.num_modal    = num_modal
        self.patch_size   = patch_size
        self.mask_ratio   = mask_ratio
        self.use_ckpt     = use_checkpoint
        self.is_training  = False   # flag so forward() knows it's eval

        self.encoder  = M3AE_Encoder(in_ch=num_modal)
        self.decoder  = M3AE_Decoder()
        self.reg_head = RegressionHead(in_ch=basic_dims, out_ch=num_modal)
        self.seg_head = SegmentationHead(num_cls=num_cls)

        # x_sub – initialised lazily (needs input spatial dimensions)
        self._sub_init_std = sub_init_std
        self.x_sub         = None   # will become nn.Parameter on first call

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _init_x_sub(self, x: torch.Tensor):
        """Lazily register x_sub as a parameter matching the spatial size of x."""
        B, N, H, W, Z = x.shape
        data = torch.randn(1, N, H, W, Z, device=x.device) * self._sub_init_std
        self.x_sub = nn.Parameter(data)

    def _get_x_sub_resized(self, x: torch.Tensor) -> torch.Tensor:
        """Return x_sub, interpolated to match x if spatial dims differ."""
        if self.x_sub is None:
            self._init_x_sub(x)
        sub = self.x_sub
        H, W, Z = x.shape[2], x.shape[3], x.shape[4]
        if sub.shape[2:] != (H, W, Z):
            sub = F.interpolate(sub, size=(H, W, Z), mode="trilinear",
                                align_corners=True)
        return sub

    def _encode(self, x_in: torch.Tensor):
        """Encode with optional gradient checkpointing."""
        if self.use_ckpt and self.training:
            return grad_checkpoint(self.encoder, x_in, use_reentrant=False)
        return self.encoder(x_in)

    # ── Stage-1: pretraining forward ──────────────────────────────────────────

    def forward_pretrain(self, x: torch.Tensor):
        """
        M3AE self-supervised pretraining forward pass.

        Pipeline
        --------
        1. Sample a random per-sample modality-dropout mask.
        2. Build S(x, x_sub) — modality dropout + patch masking.
        3. Encode → decode → regression head → x_hat.
        4. Compute MSE(x_hat, x).

        Returns
        -------
        loss_mse : scalar tensor
        x_sub    : the current substitute image parameter (for L2 reg)
        """
        x_sub  = self._get_x_sub_resized(x)
        B      = x.size(0)

        # Random modality dropout (per sample)
        mod_masks = torch.tensor(
            [structured_group_dropout(p_full=0.5) for _ in range(B)],
            dtype=torch.bool, device=x.device)                   # (B, N)

        # Build masked input S(x, x_sub)
        x_masked = prepare_masked_input(x, x_sub, mod_masks,
                                        patch_size=self.patch_size,
                                        patch_mask_ratio=self.mask_ratio)

        # Forward through encoder + decoder + regression head
        f1, f2, f3, f4 = self._encode(x_masked)
        d1, _d2, _d3   = self.decoder(f1, f2, f3, f4)
        x_hat          = self.reg_head(d1)                        # (B, N, H, W, Z)

        loss_mse = F.mse_loss(x_hat, x)
        return loss_mse, x_sub

    # ── Stage-2: fine-tuning forward ──────────────────────────────────────────

    def forward_finetune(self, x: torch.Tensor,
                         mask: torch.Tensor):
        """
        Fine-tuning forward pass with heterogeneous self-distillation.

        Two random missing-modal instantiations are processed for each sample.
        The bottleneck features from both instantiations are compared via MSE
        (consistency loss L_con).

        Parameters
        ----------
        x    : (B, N, H, W, Z)
        mask : (B, N) bool  — primary missing-modal mask from dataset

        Returns
        -------
        pred_full : (B, C, H, W, Z)          main segmentation output
        (pred_half, pred_qtr)                 deep-supervision outputs (full res)
        loss_con  : scalar                    bottleneck consistency loss
        """
        x_sub = self._get_x_sub_resized(x)
        B     = x.size(0)

        # ── Instantiation 0: use the provided mask ──
        x0           = apply_substitute(x, x_sub, mask)
        f1_0, f2_0, f3_0, f4_0 = self._encode(x0)
        d1_0, d2_0, d3_0 = self.decoder(f1_0, f2_0, f3_0, f4_0)
        pred_full, pred_half, pred_qtr = self.seg_head(d1_0, d2_0, d3_0)

        # ── Instantiation 1: independently sampled mask ──
        mask1 = torch.tensor(
            [structured_group_dropout(p_full=0.3) for _ in range(B)],
            dtype=torch.bool, device=x.device)
        x1               = apply_substitute(x, x_sub, mask1)
        f1_1, f2_1, f3_1, f4_1 = self._encode(x1)

        # Two-way bottleneck consistency (both directions, averaged)
        # L_con(x0, x1, x_sub) = MSE(f0, f1)
        loss_con = 0.5 * (F.mse_loss(f4_0, f4_1.detach()) +
                           F.mse_loss(f4_1, f4_0.detach()))

        return pred_full, (pred_half, pred_qtr), loss_con

    # ── Inference ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Inference: substitute missing modalities with x_sub, then segment.

        Parameters
        ----------
        x    : (B, N, H, W, Z)
        mask : (B, N) bool

        Returns
        -------
        (B, C, H, W, Z) softmax probability maps
        """
        x_sub = self._get_x_sub_resized(x)
        x_in  = apply_substitute(x, x_sub, mask)
        f1, f2, f3, f4 = self.encoder(x_in)
        d1, d2, d3     = self.decoder(f1, f2, f3, f4)
        pred, _ph, _pq = self.seg_head(d1, d2, d3)
        return pred


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

def dice_loss(output, target, num_cls=2, eps=1e-7):
    target = target.float()
    loss   = 0.0
    for i in range(num_cls):
        num  = torch.sum(output[:, i] * target[:, i])
        den  = torch.sum(output[:, i]) + torch.sum(target[:, i]) + eps
        loss += 2.0 * num / den
    return 1.0 - loss / num_cls


def softmax_weighted_loss(output, target, num_cls=2):
    target = target.float()
    loss   = torch.zeros(1, device=output.device, dtype=output.dtype)
    for i in range(num_cls):
        w = 1.0 - (target[:, i].sum((1, 2, 3)) / (target.sum((1, 2, 3, 4)) + 1e-8))
        w = w.view(-1, 1, 1, 1)
        loss = loss + (-w * target[:, i] *
                       torch.log(output[:, i].clamp(1e-5, 1.0))).mean()
    return loss


def seg_loss_fn(pred, target, num_cls):
    """Dice + weighted cross-entropy for one prediction tensor."""
    return (softmax_weighted_loss(pred, target, num_cls) +
            dice_loss(pred, target, num_cls))


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def train_pretrain_epoch(model, loader, optimizer, scaler, epoch, args):
    """
    Stage-1 M3AE pretraining.

    Objective: min_{F, x_sub} L_mse(x, F(S(x, x_sub))) + γ * R(x_sub)
    where R(x_sub) = ||x_sub||^2 / n  (L2 regularisation, paper γ=0.005).
    """
    model.train()
    model.is_training = True
    gamma  = args.sub_reg_weight
    losses = []

    for i, (x, _target, _mask, _mask_max, _name) in enumerate(loader):
        x = x.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(args.amp):
            loss_mse, x_sub = model.forward_pretrain(x)
            reg  = gamma * x_sub.pow(2).mean()
            loss = loss_mse + reg

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        freq = max(1, len(loader) // 3)
        if (i + 1) % freq == 0:
            logging.info(
                f"  [Pretrain Ep {epoch+1}] it {i+1}/{len(loader)}  "
                f"loss={loss.item():.4f}  mse={loss_mse.item():.4f}  "
                f"reg={reg.item():.6f}")

    return float(np.mean(losses))


def train_finetune_epoch(model, loader, optimizer, scaler, epoch, args):
    """
    Stage-2 fine-tuning with heterogeneous self-distillation.

    Objective:
        min_{f, fs, {fd}} λ * L_con(x0, x1, x_sub) + Σ_i L_seg(s_gt, xi, x_sub)

    Deep supervision adds the ½ and ¼ scale losses (deep supervision).
    """
    model.train()
    model.is_training = True
    num_cls = args.num_cls
    losses  = []

    for i, (x, target, mask, _mask_max, _name) in enumerate(loader):
        x      = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mask   = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(args.amp):
            p_full, (p_half, p_qtr), loss_con = model.forward_finetune(x, mask)

            # Deep-supervised segmentation loss
            l_seg = (seg_loss_fn(p_full, target, num_cls)
                   + seg_loss_fn(p_half, target, num_cls)
                   + seg_loss_fn(p_qtr,  target, num_cls))

            loss = l_seg + args.con_weight * loss_con

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        freq = max(1, len(loader) // 3)
        if (i + 1) % freq == 0:
            logging.info(
                f"  [Finetune Ep {epoch+1}] it {i+1}/{len(loader)}  "
                f"loss={loss.item():.4f}  seg={l_seg.item():.4f}  "
                f"con={loss_con.item():.4f}")

    return float(np.mean(losses))


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  VALIDATION / TESTING / VISUALIZATION  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, args):
    model.eval()
    model.is_training = False
    dices = []
    for x, target, mask, _mask_max, _ in loader:
        x, target, mask = x.cuda(), target.cuda(), mask.cuda()
        with make_autocast(args.amp):
            pred = model(x, mask)
        pl  = pred.argmax(1)
        gl  = target.argmax(1)
        eps = 1e-8
        for b in range(x.size(0)):
            p, g = (pl[b] == 1).float(), (gl[b] == 1).float()
            dices.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))
    return np.array([d.item() for d in dices])


@torch.no_grad()
def test_all_combinations(model, test_loader, args, save_dir):
    model.eval()
    model.is_training = False
    results        = {}
    all_combo_dice = []

    for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
        mask_list = groups_to_mask(g2, g3, g4)
        present   = [MODALITY_NAMES[i] for i, v in enumerate(mask_list) if v]
        logging.info(f"Testing {combo_name}: {present}")

        scores = []
        for x, y_int, _ in test_loader:
            x, y_int = x.cuda(), y_int.cuda()
            mask = torch.tensor([mask_list] * x.size(0),
                                dtype=torch.bool, device=x.device)
            with make_autocast(args.amp):
                pred = model(x, mask)
            pl  = pred.argmax(1)
            eps = 1e-8
            for b in range(x.size(0)):
                p, g = (pl[b] == 1).float(), (y_int[b] == 1).float()
                scores.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))

        arr = np.array([s.item() for s in scores])
        m, s, md = arr.mean(), arr.std(), np.median(arr)
        results[combo_name] = {"mean": m, "std": s, "median": md, "scores": arr.tolist()}
        all_combo_dice.append(arr)
        logging.info(f"  {combo_name}: {m:.3f} ± {s:.3f} ({md:.3f})")

    flat = np.concatenate(all_combo_dice)
    results["Average"] = {
        "mean": flat.mean(), "std": flat.std(), "median": np.median(flat)}
    logging.info(f"  Average: {flat.mean():.3f} ± {flat.std():.3f}"
                 f" ({np.median(flat):.3f})")

    sav = {k: {kk: float(vv) for kk, vv in v.items() if kk != "scores"}
           for k, v in results.items()}
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(sav, f, indent=2)
    return results


@torch.no_grad()
def visualize_test(model, test_loader, args, save_dir):
    model.eval()
    model.is_training = False
    vis_dir  = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    mask_all = groups_to_mask(True, True, True)

    for x, y_int, names in test_loader:
        x    = x.cuda()
        mask = torch.tensor([mask_all] * x.size(0), dtype=torch.bool, device=x.device)
        with make_autocast(args.amp):
            pred = model(x, mask)
        pl = pred.argmax(1)

        for b in range(x.size(0)):
            name = names[b] if isinstance(names, (list, tuple)) else names
            t2   = x[b, 0].cpu().numpy()
            gt   = y_int[b].cpu().numpy()
            pr   = pl[b].cpu().numpy()
            z    = t2.shape[2] // 2

            base = t2[:, :, z].astype(np.float32)
            base -= base.min()
            if base.max() > 0:
                base /= base.max()
            rgb = np.stack([base] * 3, -1)

            gt_s, pr_s = (gt[:, :, z] > 0), (pr[:, :, z] > 0)
            go, po, bo = gt_s & ~pr_s, pr_s & ~gt_s, gt_s & pr_s
            a  = 0.55
            ov = rgb.copy()
            ov[go, 0] *= (1-a);              ov[go, 1] = ov[go, 1]*(1-a)+a; ov[go, 2] *= (1-a)
            ov[po, 0] = ov[po, 0]*(1-a)+a;  ov[po, 1] *= (1-a);            ov[po, 2] *= (1-a)
            ov[bo, 0] = ov[bo, 0]*(1-a)+a;  ov[bo, 1] = ov[bo, 1]*(1-a)+a; ov[bo, 2] *= (1-a)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"{name} z={z}  (R=Pred G=GT Y=Overlap)")
            ax.axis("off");  fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"{name}_z{z}.png"), dpi=150)
            plt.close(fig)
    logging.info(f"Saved visualizations to {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class LR_Scheduler:
    """Cosine annealing: lr = base_lr * 0.5 * (1 + cos(pi * epoch / T))"""
    def __init__(self, base_lr, num_epochs):
        self.lr, self.T = base_lr, num_epochs

    def __call__(self, optimizer, epoch):
        lr = self.lr * 0.5 * (1 + math.cos(math.pi * epoch / max(self.T, 1)))
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


def _print_results(results):
    print("\n" + "=" * 70)
    print("TEST RESULTS (Dice)")
    print("=" * 70)
    for name, _ in TEST_COMBINATIONS:
        r = results[name]
        print(f"  {name:25s}:  {r['mean']:.3f} ± {r['std']:.3f} ({r['median']:.3f})")
    r = results["Average"]
    print(f"  {'Average':25s}:  {r['mean']:.3f} ± {r['std']:.3f} ({r['median']:.3f})")
    print("=" * 70)


def setup_logging(path):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        filename=os.path.join(path, "train.log"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("").addHandler(ch)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  ARGUMENT PARSING & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="M3AE Liver – Two-Stage Missing-Modal Segmentation")

    # ── Paths ──
    p.add_argument("--datapath", default='./data/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--savepath",        default="./output_m3ae_liver")
    p.add_argument("--resume",          default=None,
                   help="Resume checkpoint for eval-only mode")
    p.add_argument("--resume_pretrain", default=None,
                   help="Load Stage-1 pretrained weights before fine-tuning")

    # ── Volume sizes ──
    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)
    p.add_argument("--crop_x",   type=int, default=None)
    p.add_argument("--crop_y",   type=int, default=None)
    p.add_argument("--crop_z",   type=int, default=None)

    # ── Training schedule ──
    p.add_argument("--pretrain_epochs", type=int, default=100,
                   help="Stage-1 M3AE pretraining epochs (original M3AE: 600; 0 = skip)")
    p.add_argument("--num_epochs",      type=int, default=200,
                   help="Stage-2 fine-tuning epochs (original M3AE: 300; 0 = skip)")
    p.add_argument("--batch_size",      type=int, default=2)
    p.add_argument("--lr",              type=float, default=2e-4,
                   help="Initial LR (2e-4 with cosine decay)")
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--seed",            type=int, default=1024)
    p.add_argument("--num_workers",     type=int, default=0)
    p.add_argument("--grad_clip",       type=float, default=1.0)

    # ── Model / data ──
    p.add_argument("--num_cls",   type=int, default=2)
    p.add_argument("--num_modal", type=int, default=NUM_MODALITIES)

    # ── M3AE hyperparameters ──
    p.add_argument("--mask_ratio",     type=float, default=0.875,
                   help="Combined masking ratio (0.875)")
    p.add_argument("--patch_size",     type=int,   default=16,
                   help="3-D patch side length P (16)")
    p.add_argument("--sub_reg_weight", type=float, default=0.005,
                   help="L2 regularisation weight γ on x_sub (0.005)")
    p.add_argument("--con_weight",     type=float, default=0.1,
                   help="Self-distillation weight λ (0.1)")
    p.add_argument("--sgd_p_full",     type=float, default=0.3,
                   help="Probability of sampling full-modality combo")

    # ── AMP / checkpointing ──
    p.add_argument("--amp",           action="store_true",  default=True)
    p.add_argument("--no_amp",        dest="amp",           action="store_false")
    p.add_argument("--no_checkpoint", dest="use_checkpoint", action="store_false",
                   default=True)
    p.add_argument("--region_fusion_start_epoch", type=int, default=0,
                   help="(unused; kept for CLI compatibility)")

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.savepath)
    logging.info(f"Args: {args}")

    # ── Reproducibility ──
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark     = False
    cudnn.deterministic = True

    # ── Data ──
    patients = discover_patients(args.datapath)
    logging.info(f"Found {len(patients)} patients: {patients}")
    if len(patients) < 3:
        logging.warning("Very few patients – using all for train/val/test")
        train_p = val_p = test_p = patients
    else:
        train_p, val_p, test_p = split_patients(patients)
    logging.info(f"Train: {train_p}  Val: {val_p}  Test: {test_p}")

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
    val_data   = load_set(val_p)
    test_data  = load_set(test_p)
    logging.info("All data loaded into RAM.")

    crop = None
    if args.crop_x or args.crop_y or args.crop_z:
        crop = (args.crop_x or args.resize_x,
                args.crop_y or args.resize_y,
                args.crop_z or args.resize_z)

    train_set = LiverDataset(train_data, args.num_cls, is_train=True,
                             crop_size=crop, p_full=args.sgd_p_full)
    val_set   = LiverDataset(val_data,   args.num_cls, is_train=False)
    test_set  = LiverTestDataset(test_data)

    train_loader = MultiEpochsDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,  batch_size=1, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)

    # ── Build model ──
    model = M3AE_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        sub_init_std=0.1,
        use_checkpoint=args.use_checkpoint,
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model: {nparams:.2f}M params | AMP={args.amp} | "
                 f"GradCkpt={args.use_checkpoint}")

    # ── Eval-only mode ──
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded checkpoint (epoch {ck.get('epoch', '?')})")
        results = test_all_combinations(model, test_loader, args, args.savepath)
        _print_results(results)
        visualize_test(model, test_loader, args, args.savepath)
        return

    t0 = time.time()

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STAGE 1 – M3AE Pretraining                                     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    if args.pretrain_epochs > 0:
        logging.info("=" * 60)
        logging.info("STAGE 1 – M3AE Self-Supervised Pretraining")
        logging.info("=" * 60)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        scaler     = make_scaler(args.amp)
        lr_sched   = LR_Scheduler(args.lr, args.pretrain_epochs)
        best_loss  = float("inf")

        for epoch in range(args.pretrain_epochs):
            lr  = lr_sched(optimizer, epoch)
            avg = train_pretrain_epoch(model, train_loader, optimizer,
                                       scaler, epoch, args)
            logging.info(f"Pretrain Ep {epoch+1}/{args.pretrain_epochs}  "
                         f"lr={lr:.6f}  loss={avg:.4f}")

            if avg < best_loss:
                best_loss = avg
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                             "optim": optimizer.state_dict(),
                             "best_loss": best_loss},
                            os.path.join(args.savepath, "pretrain_best.pth"))

            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                        os.path.join(args.savepath, "pretrain_last.pth"))

        logging.info(f"Stage-1 done in {(time.time()-t0)/60:.1f} min  "
                     f"best_loss={best_loss:.4f}")

    # ── Optionally load external pretrained weights ──
    if args.pretrain_epochs == 0 and args.resume_pretrain:
        ck = torch.load(args.resume_pretrain, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded pretrain checkpoint: {args.resume_pretrain}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STAGE 2 – Fine-Tuning with Heterogeneous Self-Distillation     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    if args.num_epochs > 0:
        logging.info("=" * 60)
        logging.info("STAGE 2 – Fine-Tuning with Self-Distillation")
        logging.info("=" * 60)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        scaler     = make_scaler(args.amp)
        lr_sched   = LR_Scheduler(args.lr, args.num_epochs)
        best_dice  = 0.0
        t1 = time.time()

        for epoch in range(args.num_epochs):
            lr  = lr_sched(optimizer, epoch)
            avg = train_finetune_epoch(model, train_loader, optimizer,
                                       scaler, epoch, args)
            logging.info(f"Finetune Ep {epoch+1}/{args.num_epochs}  "
                         f"lr={lr:.6f}  loss={avg:.4f}")

            if (epoch + 1) % 10 == 0 or epoch >= args.num_epochs - 5:
                da = validate(model, val_loader, args)
                vd = da.mean()
                logging.info(f"  Val Dice: {vd:.4f}")
                if vd > best_dice:
                    best_dice = vd
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                                 "optim": optimizer.state_dict(),
                                 "best_dice": best_dice},
                                os.path.join(args.savepath, "model_best.pth"))
                    logging.info(f"  ★ Best: {best_dice:.4f}")

            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                         "optim": optimizer.state_dict()},
                        os.path.join(args.savepath, "model_last.pth"))
            if (epoch + 1) % 50 == 0:
                torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                            os.path.join(args.savepath, f"model_epoch{epoch+1}.pth"))

        logging.info(f"Stage-2 done in {(time.time()-t1)/3600:.2f}h  "
                     f"best_dice={best_dice:.4f}")

    logging.info(f"Total time: {(time.time()-t0)/3600:.2f}h")

    # ── Final test ──
    bp = os.path.join(args.savepath, "model_best.pth")
    if os.path.exists(bp):
        model.load_state_dict(
            torch.load(bp, map_location="cpu", weights_only=False)["state_dict"])
    results = test_all_combinations(model, test_loader, args, args.savepath)
    _print_results(results)
    visualize_test(model, test_loader, args, args.savepath)


if __name__ == "__main__":
    main()