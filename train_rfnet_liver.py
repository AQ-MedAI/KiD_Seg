#!/usr/bin/env python3
"""
train_rfnet_liver.py – RFNet for multi-modal liver segmentation with missing modalities.

Architecture:
  Pure RFNet (Region-aware Fusion Network, ICCV 2021) generalised to 8 modalities.
  Retains:
    - Encoder × num_modal  (shared architecture, separate weights)
    - Decoder_fuse  with region-aware modal fusion (RFM) + PRM generators
    - Decoder_sep   for per-modality segmentation regularisation
  Removed from the DC-Seg baseline:
    - Kinetic Disentanglement (KiD / DifferenceEncoder / KAPool / KCL)
    - Anchor-Guided Contrastive Learning (AGCL)
    - Hierarchical Group Fusion (HGF / IntraGroup / InterGroup attention)
    - Style encoders / Image decoders / Reconstruction loss
    - Completeness-Aware Self-Distillation (CA)
    - Anatomy / Modality contrastive losses

Modality groups (mask logic unchanged):
  G1 = [T2WI]                          (always present)
  G2 = [C-pre, C+A, C+V, C+Delay]
  G3 = [DWI]
  G4 = [InPhase, OutPhase]

Usage:
  python train_rfnet_liver.py --datapath /path/to/preprocess_nii_256x32_1 \
                               --savepath ./output --resize_x 128 --resize_y 128 --resize_z 16
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import logging
import random
import json
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
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast('cuda', enabled=enabled)
    def make_scaler(enabled): return _GradScaler('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_autocast(enabled): return _autocast(enabled=enabled)
    def make_scaler(enabled): return _GradScaler(enabled=enabled)
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")
from scipy.ndimage import zoom as scipy_zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & GROUP DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
MODALITY_NAMES    = ["T2WI", "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase"]
MODALITY_SUFFIXES = MODALITY_NAMES[:]
NUM_MODALITIES    = len(MODALITY_NAMES)  # 8

GROUP_INDICES = {
    "G1": [0],            # T2WI – always present
    "G2": [1, 2, 3, 4],  # contrast phases
    "G3": [5],            # DWI
    "G4": [6, 7],         # InPhase / OutPhase
}

# All 8 valid test combinations (G1 always on; G2/G3/G4 on/off)
TEST_COMBINATIONS = [
    ("G1",           (False, False, False)),
    ("G1+G2",        (True,  False, False)),
    ("G1+G3",        (False, True,  False)),
    ("G1+G4",        (False, False, True )),
    ("G1+G2+G3",     (True,  True,  False)),
    ("G1+G2+G4",     (True,  False, True )),
    ("G1+G3+G4",     (False, True,  True )),
    ("G1+G2+G3+G4",  (True,  True,  True )),
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


# ── Structured Group Dropout ──────────────────────────────────────────────────
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
    n_val   = max(1, int(round(n * val_ratio)))
    n_test  = max(1, n - n_train - n_val)
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
    lbl    = load_nifti(pdir / "labels" / f"{name}_T2WI.nii.gz")
    lbl    = (resize_volume(lbl, shape, order=0) > 0.5).astype(np.int64)
    return images, lbl


class LiverDataset(Dataset):
    def __init__(self, data_list, num_cls=2, is_train=True, crop_size=None, p_full=0.3):
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
        yo   = np.eye(self.num_cls, dtype=np.float32)[y.ravel()].reshape(H, W, Z, self.num_cls)
        yo   = yo.transpose(3, 0, 1, 2)
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
# 3.  MODEL – RFNet layers (generalised to num_modal modalities)
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
        self.conv       = nn.Conv3d(in_ch, out_ch, k_size, stride, padding,
                                    padding_mode=pad_type, bias=True)
        self.norm       = normalization(out_ch, norm=norm)
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


# ── Modal / Region fusion (memory-efficient – no 7-D tensor) ─────────────────

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
        prm_avg  = torch.mean(prm, dim=(2, 3, 4), keepdim=False) + 1e-7    # (B, 1)
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg.unsqueeze(-1)
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg.view(B, 1, 1, 1, 1)), dim=1)
        weight   = self.weight_layer(feat_avg).view(B, K, 1, 1, 1, 1)
        weight   = self.sigmoid(weight)
        return torch.sum(x * weight, dim=1)                                  # (B, C, H, W, Z)


class region_fusion(nn.Module):
    """Fuse per-region features into a single feature map via 1×1→3×3→1×1 convs."""
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel, k_size=1, padding=0),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0))

    def forward(self, region_feats):
        """region_feats: list of num_cls tensors (B, C, H, W, Z)."""
        return self.fusion_layer(torch.cat(region_feats, dim=1))


class region_aware_modal_fusion(nn.Module):
    """Memory-efficient region-aware modal fusion (RFM).

    Iterates over classes to avoid materialising the (B, K, cls, C, H, W, Z)
    tensor of the original RFNet implementation.
    """
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.num_cls      = num_cls
        self.num_modal    = num_modal
        self.modal_fusions = nn.ModuleList(
            [modal_fusion(in_channel, num_modal) for _ in range(num_cls)])
        self.region_fuse  = region_fusion(in_channel, num_cls)
        self.short_cut    = nn.Sequential(
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
            prm_c        = prm[:, c:c+1, :, :, :]           # (B, 1, H, W, Z)
            region_modal = y * prm_c.unsqueeze(2)            # (B, K, C, H, W, Z) – no 7-D!
            region_feats.append(self.modal_fusions[c](region_modal, prm_c))

        return torch.cat((self.region_fuse(region_feats),
                          self.short_cut(y.view(B, -1, H, W, Z))), dim=1)


# ── Encoder ───────────────────────────────────────────────────────────────────

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


# ── Separate decoder (per-modality regularisation) ───────────────────────────

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


# ── Fusion decoder with RFM (pure RFNet, no HGF / KiD) ───────────────────────

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
        """
        # ── Scale 4 (bottleneck) ──────────────────────────────────────
        p4 = self.prm4(x4, mask)
        d4 = self.RFM4(x4, p4.detach(), mask)
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
        return pred, prm_preds


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  RFNet_Liver – main model
# ═══════════════════════════════════════════════════════════════════════════════

class RFNet_Liver(nn.Module):
    """
    RFNet generalised to num_modal modalities for liver segmentation.

    Training mode (is_training=True):
        Returns (fuse_pred, sep_preds, prm_preds)
        - fuse_pred : (B, num_cls, H, W, Z)         main fused segmentation
        - sep_preds : list[num_modal] of same shape  per-modality predictions
        - prm_preds : 4-tuple of same shape          PRM predictions per scale

    Inference mode (is_training=False):
        Returns fuse_pred only.
    """
    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=True):
        super().__init__()
        self.num_modal  = num_modal
        self.num_cls    = num_cls
        self.use_ckpt   = use_checkpoint

        self.encoders     = nn.ModuleList([Encoder() for _ in range(num_modal)])
        self.decoder_fuse = Decoder_fuse(num_cls, num_modal)
        self.decoder_sep  = Decoder_sep(num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def _encode_one(self, encoder, x_m):
        """Single-modality encode with optional gradient checkpointing."""
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def forward(self, x, mask):
        """
        x    : (B, M, H, W, D)
        mask : (B, M) bool
        """
        # ── Encode every modality sequentially (avoids peak-memory spike) ──
        feats = [self._encode_one(self.encoders[m], x[:, m:m+1])
                 for m in range(self.num_modal)]

        # Stack at each encoder scale: (B, M, C, …)
        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        # ── Fused decoder ──
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if not self.is_training:
            return fuse_pred

        # ── Per-modality decoders (seg regularisation) – sequential ──
        sep_preds = [self.decoder_sep(*feats[m]) for m in range(self.num_modal)]

        return fuse_pred, sep_preds, prm_preds


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

def dice_loss(output, target, num_cls=2, eps=1e-7):
    target = target.float()
    loss = 0.0
    for i in range(num_cls):
        num  = torch.sum(output[:, i] * target[:, i])
        den  = torch.sum(output[:, i]) + torch.sum(target[:, i]) + eps
        loss += 2.0 * num / den
    return 1.0 - loss / num_cls


def softmax_weighted_loss(output, target, num_cls=2):
    target = target.float()
    loss   = torch.zeros(1, device=output.device, dtype=output.dtype)
    for i in range(num_cls):
        w    = 1.0 - (target[:, i].sum((1, 2, 3)) / (target.sum((1, 2, 3, 4)) + 1e-8))
        w    = w.view(-1, 1, 1, 1)
        loss = loss + (-w * target[:, i] * torch.log(output[:, i].clamp(1e-5, 1.0))).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING / VALIDATION / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, epoch, args):
    """
    One epoch of RFNet training.

    Total loss = fuse_loss + sep_loss + prm_loss
      - fuse_loss   : weighted CE + Dice on the fused prediction.
                      Gated OFF for the first `region_fusion_start_epoch` epochs,
                      mirroring the original RFNet warm-up schedule.
      - sep_loss    : sum of (weighted CE + Dice) over all per-modality decoders.
      - prm_loss    : sum of (weighted CE + Dice) over all 4 PRM predictions.
    """
    model.train()
    model.is_training = True
    num_cls = args.num_cls
    losses  = []

    for i, (x, target, mask, _mask_max, _) in enumerate(loader):
        x      = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mask   = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with make_autocast(args.amp):
            fuse_pred, sep_preds, prm_preds = model(x, mask)

            fuse_loss = (softmax_weighted_loss(fuse_pred, target, num_cls) +
                         dice_loss(fuse_pred, target, num_cls))

            sep_loss  = sum(softmax_weighted_loss(sp, target, num_cls) +
                            dice_loss(sp, target, num_cls)
                            for sp in sep_preds)

            prm_loss  = sum(softmax_weighted_loss(pp, target, num_cls) +
                            dice_loss(pp, target, num_cls)
                            for pp in prm_preds)

            # Warm-up: train sep+prm only, then add fuse_loss
            if epoch < args.region_fusion_start_epoch:
                loss = sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if (i + 1) % max(1, len(loader) // 3) == 0:
            logging.info(
                f"  Ep {epoch+1} it {i+1}/{len(loader)}"
                f"  loss={loss.item():.4f}"
                f"  fuse={fuse_loss.item():.4f}"
                f"  sep={sep_loss.item():.4f}"
                f"  prm={prm_loss.item():.4f}")

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
        pl  = pred.argmax(1)
        gl  = target.argmax(1)
        eps = 1e-8
        for b in range(x.size(0)):
            p = (pl[b] == 1).float()
            g = (gl[b] == 1).float()
            dices.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))
    return np.array([d.item() for d in dices])


@torch.no_grad()
def test_all_combinations(model, test_loader, args, save_dir):
    model.eval()
    model.is_training = False
    results       = {}
    all_combo_dice = []

    for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
        mask_list = groups_to_mask(g2, g3, g4)
        present   = [MODALITY_NAMES[i] for i, v in enumerate(mask_list) if v]
        logging.info(f"Testing {combo_name}: {present}")

        scores = []
        for x, y_int, _ in test_loader:
            x, y_int = x.cuda(), y_int.cuda()
            mask = torch.tensor([mask_list] * x.size(0), dtype=torch.bool, device=x.device)
            with make_autocast(args.amp):
                pred = model(x, mask)
            pl  = pred.argmax(1)
            eps = 1e-8
            for b in range(x.size(0)):
                p = (pl[b] == 1).float()
                g = (y_int[b] == 1).float()
                scores.append((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))

        arr  = np.array([s.item() for s in scores])
        m, s, md = arr.mean(), arr.std(), np.median(arr)
        results[combo_name] = {"mean": m, "std": s, "median": md, "scores": arr.tolist()}
        all_combo_dice.append(arr)
        logging.info(f"  {combo_name}: ${m:.3f}\\pm{s:.3f}({md:.3f})^\\dagger$")

    flat = np.concatenate(all_combo_dice)
    results["Average"] = {"mean": flat.mean(), "std": flat.std(), "median": np.median(flat)}
    logging.info(f"  Average: ${flat.mean():.3f}\\pm{flat.std():.3f}({np.median(flat):.3f})^\\dagger$")

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
            ov[go, 0] *= (1-a);              ov[go, 1]  = ov[go, 1]*(1-a)+a;  ov[go, 2] *= (1-a)
            ov[po, 0]  = ov[po, 0]*(1-a)+a; ov[po, 1] *= (1-a);              ov[po, 2] *= (1-a)
            ov[bo, 0]  = ov[bo, 0]*(1-a)+a; ov[bo, 1]  = ov[bo, 1]*(1-a)+a; ov[bo, 2] *= (1-a)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"{name} z={z}  (R=Pred G=GT Y=Overlap)")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"{name}_z{z}.png"), dpi=150)
            plt.close(fig)
    logging.info(f"Saved visualizations to {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class LR_Scheduler:
    def __init__(self, base_lr, num_epochs):
        self.lr, self.T = base_lr, num_epochs

    def __call__(self, optimizer, epoch):
        lr = self.lr * (1 - epoch / self.T) ** 0.9
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
    p = argparse.ArgumentParser(description="RFNet Liver – Multi-Modal Missing-Modality Seg.")
    p.add_argument("--datapath", default='/mnt/amed-heyuan/common/data/khorheeguan.khg/data/metaseg_data_liver_mri/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--savepath",  default="./output_rfnet_liver")
    p.add_argument("--resume",    default=None,
                   help="Checkpoint path – activates evaluation-only mode")

    # ── Volume dimensions ──
    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)
    p.add_argument("--crop_x",   type=int, default=None)
    p.add_argument("--crop_y",   type=int, default=None)
    p.add_argument("--crop_z",   type=int, default=None)

    # ── Training hyper-parameters ──
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--num_epochs",   type=int,   default=200)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=1024)
    p.add_argument("--region_fusion_start_epoch", type=int, default=20,
                   help="Epoch from which the fused decoder loss is included "
                        "(mirrors the original RFNet warm-up schedule)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main thread, safest on Windows/macOS)")

    # ── Model ──
    p.add_argument("--num_cls",   type=int, default=2,
                   help="Number of segmentation classes")
    p.add_argument("--num_modal", type=int, default=8,
                   help="Total number of input modalities")

    # ── AMP / gradient checkpointing ──
    p.add_argument("--amp",           action="store_true",  default=True,
                   help="Mixed-precision training (default ON)")
    p.add_argument("--no_amp",        dest="amp",            action="store_false")
    p.add_argument("--no_checkpoint", dest="use_checkpoint", action="store_false",
                   default=True, help="Disable gradient checkpointing")

    # ── Structured Group Dropout ──
    p.add_argument("--sgd_p_full", type=float, default=0.3,
                   help="Probability of sampling full-modality combo (default 0.3)")

    # ── Gradient clipping ──
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm for clipping (0 = disabled)")

    return p.parse_args()


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
        print(f"  {name:25s}:  ${r['mean']:.3f}\\pm{r['std']:.3f}({r['median']:.3f})^\\dagger$")
    r = results["Average"]
    print(f"  {'Average':25s}:  ${r['mean']:.3f}\\pm{r['std']:.3f}({r['median']:.3f})^\\dagger$")
    print("=" * 70)


def main():
    args = parse_args()
    setup_logging(args.savepath)
    logging.info(f"Args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark    = False
    cudnn.deterministic = True

    # ── Discover and split patients ──
    patients = discover_patients(args.datapath)
    logging.info(f"Found {len(patients)} patients: {patients}")
    if len(patients) < 3:
        logging.warning("Very few patients – using all for train/val/test")
        train_p = val_p = test_p = patients
    else:
        train_p, val_p, test_p = split_patients(patients)
    logging.info(f"Train: {train_p}  Val: {val_p}  Test: {test_p}")

    # ── Preprocess into RAM ──
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
    val_loader  = DataLoader(val_set,  batch_size=1, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)

    # ── Build model ──
    model = RFNet_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=args.use_checkpoint,
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"RFNet_Liver: {nparams:.2f}M params"
                 f"  |  AMP={args.amp}"
                 f"  |  GradCkpt={args.use_checkpoint}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, amsgrad=True)
    scaler   = make_scaler(args.amp)
    lr_sched = LR_Scheduler(args.lr, args.num_epochs)

    # ── Evaluate only ──
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        if any(k.startswith("module.") for k in sd):           # strip DataParallel
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded checkpoint epoch {ck.get('epoch', '?')}")
        results = test_all_combinations(model, test_loader, args, args.savepath)
        _print_results(results)
        visualize_test(model, test_loader, args, args.savepath)
        return

    # ── Training loop ──
    best_dice = 0.0
    t0        = time.time()
    os.makedirs(args.savepath, exist_ok=True)

    for epoch in range(args.num_epochs):
        lr = lr_sched(optimizer, epoch)
        logging.info(f"Epoch {epoch+1}/{args.num_epochs}  lr={lr:.6f}")

        avg = train_one_epoch(model, train_loader, optimizer, scaler, epoch, args)
        logging.info(f"  Train loss: {avg:.4f}")

        if (epoch + 1) % 10 == 0 or epoch >= args.num_epochs - 5:
            da = validate(model, val_loader, args)
            vd = da.mean()
            logging.info(f"  Val Dice: {vd:.4f}")
            if vd > best_dice:
                best_dice = vd
                torch.save({"epoch":      epoch,
                            "state_dict": model.state_dict(),
                            "optim":      optimizer.state_dict(),
                            "best_dice":  best_dice},
                           os.path.join(args.savepath, "model_best.pth"))
                logging.info(f"  ★ Best: {best_dice:.4f}")

        torch.save({"epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "optim":      optimizer.state_dict()},
                   os.path.join(args.savepath, "model_last.pth"))

        if (epoch + 1) % 50 == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                       os.path.join(args.savepath, f"model_epoch{epoch+1}.pth"))

    logging.info(f"Training done in {(time.time()-t0)/3600:.2f}h")

    # ── Final test with best checkpoint ──
    bp = os.path.join(args.savepath, "model_best.pth")
    if os.path.exists(bp):
        model.load_state_dict(
            torch.load(bp, map_location="cpu", weights_only=False)["state_dict"])
    results = test_all_combinations(model, test_loader, args, args.savepath)
    _print_results(results)
    visualize_test(model, test_loader, args, args.savepath)


if __name__ == "__main__":
    main()
