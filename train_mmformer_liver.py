#!/usr/bin/env python3
"""
train_mmformer_liver.py – mmFormer backbone for multi-modal liver segmentation
with missing modalities.

Model backbone replaced with mmFormer (mmformer/mmformer.py) adapted for 8
modalities and a 5-level encoder, while preserving ALL training infrastructure:

  • AMP + gradient checkpointing
  • Hierarchical Group Fusion (HGF)
  • Kinetic Disentanglement (KiD)
  • Anchor-Guided Contrastive Learning (AGCL)
  • Kinetic Contrastive Loss (KCL)
  • Completeness-Aware Self-Distillation (CA)
  • Structured Group Dropout
  • Style encoder / image decoder (anatomy–style disentanglement)

mmFormer additions over DC-Seg:
  • Per-modality IntraFormer self-attention at the bottleneck (level 5)
  • Cross-modal InterFormer attention with missing-modality masking
  • 5-level encoder (adds one stride-2 down-sampling relative to original
    DC-Seg 4-level encoder)
  • HGF retained at levels 1-4; InterFormer replaces HGF at level 5

Modality groups (unchanged):
  G1 = [T2WI]                          (always present)
  G2 = [C-pre, C+A, C+V, C+Delay]
  G3 = [DWI]
  G4 = [InPhase, OutPhase]

Usage:
  python train_mmformer_liver.py \
      --datapath /path/to/preprocess_nii_256x32_1 \
      --savepath ./output --resize_x 128 --resize_y 128 --resize_z 16

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — mmFormer training
══════════════════════════════════════════════════════════════════════════════

### train

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=0
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

nohup python -u train_mmformer_liver.py --datapath ./data/preprocess_nii_256x32 --savepath ./output_mmformer_liver > <PROJECT_ROOT>/logs/train_mmformer_liver.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/train_mmformer_liver.log
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import logging
import random
import json
import math
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
# 1.  CONSTANTS & GROUP DEFINITIONS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
MODALITY_NAMES = ["T2WI", "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase"]
MODALITY_SUFFIXES = MODALITY_NAMES[:]
NUM_MODALITIES = len(MODALITY_NAMES)  # 8

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


def groups_to_mask(g2: bool, g3: bool, g4: bool):
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
    return groups_to_mask(random.random() < 0.5,
                          random.random() < 0.5,
                          random.random() < 0.5)


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
    lbl = load_nifti(pdir / "labels" / f"{name}_T2WI.nii.gz")
    lbl = (resize_volume(lbl, shape, order=0) > 0.5).astype(np.int64)
    return images, lbl


class LiverDataset(Dataset):
    def __init__(self, data_list, num_cls=2, is_train=True, crop_size=None, p_full=0.3):
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

# mmFormer transformer hyper-params
# bottleneck channels = basic_dims * 16 = 256 → transformer_basic_dims = 256
transformer_basic_dims = basic_dims * 16   # 256
mlp_dim_tf  = 512
num_heads_tf = 8                           # 256 / 8 = 32 per head
depth_tf     = 1


def normalization(planes, norm="in"):
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    elif norm == "gn":
        return nn.GroupNorm(4, planes)
    elif norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"Unsupported norm: {norm}")


class general_conv3d(nn.Module):
    """3-D conv + norm + activation with safe reflect-padding fallback.

    PyTorch reflect padding requires each spatial dimension > padding size.
    At encoder level 5 (after 4 stride-2 downs from Z=16), Z=1 violates this.
    We therefore apply padding manually in forward(), falling back to zero-
    padding on any axis where the dimension is too small for reflect.
    """
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1,
                 pad_type="reflect", norm="in", act_type="lrelu", relufactor=0.2):
        super().__init__()
        self._pad      = padding
        self._pad_type = pad_type
        # padding=0 here; we apply it manually in forward
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride, padding=0, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        self.activation = (nn.ReLU(inplace=True) if act_type == "relu"
                           else nn.LeakyReLU(relufactor, inplace=True))

    def forward(self, x):
        if self._pad > 0:
            p = self._pad
            if self._pad_type == "reflect":
                # reflect requires every spatial dim > p; fall back to zeros otherwise
                _, _, H, W, Z = x.shape
                if H > p and W > p and Z > p:
                    x = F.pad(x, (p, p, p, p, p, p), mode="reflect")
                else:
                    x = F.pad(x, (p, p, p, p, p, p), mode="constant", value=0)
            else:
                x = F.pad(x, (p, p, p, p, p, p), mode="constant", value=0)
        return self.activation(self.norm(self.conv(x)))


class BasicConv(nn.Module):
    def __init__(self, inp, out, ks, stride=1, padding=0, relu=True, norm=True, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(inp, out, ks, stride, padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None: x = self.norm(x)
        if self.relu is not None: x = self.relu(x)
        return x


class Adaptive_instance_norm(nn.Module):
    def forward(self, content, gamma, beta, epsilon=1e-5):
        c_mean = torch.mean(content, [2, 3, 4], keepdim=True)
        c_std  = torch.std(content,  [2, 3, 4], keepdim=True)
        return gamma * ((content - c_mean) / (c_std + epsilon)) + beta


class Adaptive_resblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1   = BasicConv(in_planes, out_planes, 3, 1, 1, relu=False, norm=False)
        self.i_norm1 = Adaptive_instance_norm()
        self.conv2   = BasicConv(in_planes, out_planes, 3, 1, 1, relu=False, norm=False)
        self.i_norm2 = Adaptive_instance_norm()

    def forward(self, x_init, mu, sigma):
        x = F.relu(self.i_norm1(self.conv1(x_init), sigma, mu), inplace=True)
        x = self.i_norm2(self.conv2(x), sigma, mu)
        return x + x_init


# ── PRM generators  ──────────────────────────────────────────────────────────

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0),
            general_conv3d(in_channel // 4,        in_channel // 4, k_size=3, padding=1),
            general_conv3d(in_channel // 4,        in_channel,      k_size=1, padding=0))
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
            general_conv3d(in_channel // 4,        in_channel // 4, k_size=3, padding=1),
            general_conv3d(in_channel // 4,        in_channel,      k_size=1, padding=0))
        self.prm_layer = nn.Sequential(
            general_conv3d(in_channel * 2, 16, k_size=1, padding=0),
            nn.Conv3d(16, num_cls, 1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        return self.prm_layer(torch.cat((x1, self.embedding_layer(y.view(B, -1, H, W, Z))), 1))


# ── Modal / region fusion  ────────────────────────────────────────────────────

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
        prm_avg  = torch.mean(prm, dim=(2, 3, 4), keepdim=False) + 1e-7
        feat_avg = torch.mean(x,   dim=(3, 4, 5), keepdim=False) / prm_avg.unsqueeze(-1)
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg.view(B, 1, 1, 1, 1)), dim=1)
        weight   = self.weight_layer(feat_avg).view(B, K, 1, 1, 1, 1)
        weight   = self.sigmoid(weight)
        return torch.sum(x * weight, dim=1)


class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel,     k_size=1, padding=0),
            general_conv3d(in_channel,           in_channel,     k_size=3, padding=1),
            general_conv3d(in_channel,           in_channel // 2, k_size=1, padding=0))

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
            general_conv3d(in_channel * num_modal, in_channel,     k_size=1, padding=0),
            general_conv3d(in_channel,             in_channel,     k_size=3, padding=1),
            general_conv3d(in_channel,             in_channel // 2, k_size=1, padding=0))

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
# 3b. HIERARCHICAL ATOMIC-GROUP FUSION  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class IntraGroupAttentionPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_ch, max(in_ch // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_ch // 4, 4), 1, 1, bias=False))

    def forward(self, feats: list):
        if len(feats) == 1:
            return feats[0]
        stacked = torch.stack(feats, dim=1)
        B, G, C, H, W, Z = stacked.shape
        scores  = self.attn(stacked.reshape(B * G, C, H, W, Z)).reshape(B, G, 1, H, W, Z)
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)


class InterGroupCrossAttention(nn.Module):
    NUM_GROUPS = 4

    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        assert in_ch % num_heads == 0
        self.num_heads = num_heads
        self.d_k   = in_ch // num_heads
        self.scale = self.d_k ** -0.5
        self.proj_q   = nn.Conv3d(in_ch, in_ch, 1, bias=False)
        self.proj_k   = nn.Conv3d(in_ch, in_ch, 1, bias=False)
        self.proj_v   = nn.Conv3d(in_ch, in_ch, 1, bias=False)
        self.proj_out = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1, bias=True),
            nn.InstanceNorm3d(in_ch))

    def forward(self, f_anchor, h_groups, r):
        B, C, H, W, Z = f_anchor.shape
        G  = self.NUM_GROUPS
        nh = self.num_heads
        dk = self.d_k
        N  = H * W * Z
        groups = torch.stack([f_anchor] + h_groups, dim=1)
        q = self.proj_q(f_anchor).view(B, nh, dk, N)
        groups_flat = groups.view(B * G, C, H, W, Z)
        k = self.proj_k(groups_flat).view(B, G, nh, dk, N)
        v = self.proj_v(groups_flat).view(B, G, nh, dk, N)
        scores = (q.unsqueeze(1) * k).sum(dim=3) * self.scale
        attn_mask   = r.view(B, G, 1, 1)
        scores      = scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=1).nan_to_num(0.0)
        out = (attn_weights.unsqueeze(3) * v).sum(dim=1)
        out = out.reshape(B, C, H, W, Z)
        return f_anchor + self.proj_out(out)


class HierarchicalGroupFusion(nn.Module):
    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        self.intra_g2 = IntraGroupAttentionPool(in_ch)
        self.intra_g4 = IntraGroupAttentionPool(in_ch)
        self.inter    = InterGroupCrossAttention(in_ch, num_heads=num_heads)

    def forward(self, x_stacked, mask):
        f_t2wi = x_stacked[:, 0]
        g2_feats   = [x_stacked[:, i] for i in GROUP_INDICES["G2"]]
        g2_present = mask[:, GROUP_INDICES["G2"][0]]
        h2 = self.intra_g2(g2_feats) * g2_present.float().view(-1, 1, 1, 1, 1)
        h3 = x_stacked[:, GROUP_INDICES["G3"][0]]
        g3_present = mask[:, GROUP_INDICES["G3"][0]]
        h3 = h3 * g3_present.float().view(-1, 1, 1, 1, 1)
        g4_feats   = [x_stacked[:, i] for i in GROUP_INDICES["G4"]]
        g4_present = mask[:, GROUP_INDICES["G4"][0]]
        h4 = self.intra_g4(g4_feats) * g4_present.float().view(-1, 1, 1, 1, 1)
        r = torch.stack([mask[:, 0], g2_present, g3_present, g4_present], dim=1)
        return self.inter(f_t2wi, [h2, h3, h4], r)


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. mmFORMER TRANSFORMER COMPONENTS
#     (ported from mmformer/mmformer.py, adapted for 3-D volumes)
# ═══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim   = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(dropout_rate)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn      = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate))

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """Standard ViT-style transformer (from mmFormer)."""

    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList([
            Residual(PreNormDrop(embedding_dim, dropout_rate,
                                 SelfAttention(embedding_dim, heads=heads,
                                               dropout_rate=dropout_rate)))
            for _ in range(depth)])
        self.cross_ffn_list = nn.ModuleList([
            Residual(PreNorm(embedding_dim,
                             FeedForward(embedding_dim, mlp_dim, dropout_rate)))
            for _ in range(depth)])

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ENCODERS / DECODERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5-Level Encoder (mmFormer style) ─────────────────────────────────────────

class Encoder5(nn.Module):
    """5-level residual encoder (adds one stride-2 level over original DC-Seg).

    Output levels:
        x1: (B, C,    H,    W,    Z)     basic_dims      =  16
        x2: (B, 2C,   H/2,  W/2,  Z/2)  basic_dims*2    =  32
        x3: (B, 4C,   H/4,  W/4,  Z/4)  basic_dims*4    =  64
        x4: (B, 8C,   H/8,  W/8,  Z/8)  basic_dims*8    = 128
        x5: (B, 16C,  H/16, W/16, Z/16) basic_dims*16   = 256  ← new
    """
    def __init__(self):
        super().__init__()
        c = basic_dims
        self.e1_c1 = general_conv3d(1,   c,    pad_type="reflect")
        self.e1_c2 = general_conv3d(c,   c,    pad_type="reflect")
        self.e1_c3 = general_conv3d(c,   c,    pad_type="reflect")

        self.e2_c1 = general_conv3d(c,   c*2,  stride=2, pad_type="reflect")
        self.e2_c2 = general_conv3d(c*2, c*2,  pad_type="reflect")
        self.e2_c3 = general_conv3d(c*2, c*2,  pad_type="reflect")

        self.e3_c1 = general_conv3d(c*2, c*4,  stride=2, pad_type="reflect")
        self.e3_c2 = general_conv3d(c*4, c*4,  pad_type="reflect")
        self.e3_c3 = general_conv3d(c*4, c*4,  pad_type="reflect")

        self.e4_c1 = general_conv3d(c*4, c*8,  stride=2, pad_type="reflect")
        self.e4_c2 = general_conv3d(c*8, c*8,  pad_type="reflect")
        self.e4_c3 = general_conv3d(c*8, c*8,  pad_type="reflect")

        # Level 5 – new bottleneck (mirrors mmFormer's e5)
        self.e5_c1 = general_conv3d(c*8,  c*16, stride=2, pad_type="reflect")
        self.e5_c2 = general_conv3d(c*16, c*16, pad_type="reflect")
        self.e5_c3 = general_conv3d(c*16, c*16, pad_type="reflect")

    def forward(self, x):
        x1 = self.e1_c1(x);  x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1); x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x3 = self.e3_c1(x2); x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3); x4 = x4 + self.e4_c3(self.e4_c2(x4))
        x5 = self.e5_c1(x4); x5 = x5 + self.e5_c3(self.e5_c2(x5))
        return x1, x2, x3, x4, x5


# ── 5-Level Separate Decoder (mmFormer Decoder_sep style) ────────────────────

class Decoder_sep5(nn.Module):
    """Per-modality regularisation decoder — 5 levels to match Encoder5."""
    def __init__(self, num_cls=2):
        super().__init__()
        c = basic_dims

        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = general_conv3d(c*16, c*8,  pad_type="reflect")
        self.d4_c2 = general_conv3d(c*16, c*8,  pad_type="reflect")
        self.d4_out = general_conv3d(c*8,  c*8,  k_size=1, padding=0, pad_type="reflect")

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = general_conv3d(c*8,  c*4,  pad_type="reflect")
        self.d3_c2 = general_conv3d(c*8,  c*4,  pad_type="reflect")
        self.d3_out = general_conv3d(c*4,  c*4,  k_size=1, padding=0, pad_type="reflect")

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = general_conv3d(c*4,  c*2,  pad_type="reflect")
        self.d2_c2 = general_conv3d(c*4,  c*2,  pad_type="reflect")
        self.d2_out = general_conv3d(c*2,  c*2,  k_size=1, padding=0, pad_type="reflect")

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = general_conv3d(c*2,  c,    pad_type="reflect")
        self.d1_c2 = general_conv3d(c*2,  c,    pad_type="reflect")
        self.d1_out = general_conv3d(c,    c,    k_size=1, padding=0, pad_type="reflect")

        self.seg     = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        d = self.d4_c1(self.d4(x5))
        d = self.d4_out(self.d4_c2(torch.cat((d, x4), 1)))
        d = self.d3_c1(self.d3(d))
        d = self.d3_out(self.d3_c2(torch.cat((d, x3), 1)))
        d = self.d2_c1(self.d2(d))
        d = self.d2_out(self.d2_c2(torch.cat((d, x2), 1)))
        d = self.d1_c1(self.d1(d))
        d = self.d1_out(self.d1_c2(torch.cat((d, x1), 1)))
        return self.softmax(self.seg(d))


# ── Fusion Decoder (mmFormer + HGF + KiD, 5-level) ───────────────────────────

class Decoder_fuse(nn.Module):
    """Region-aware fusion decoder with:
      • InterFormer bottleneck (x5_inter) at level 5 — no HGF here
      • HGF gated residual at levels 1-4
      • KiD gated injection at level 4 (matches DifferenceEncoder spatial scale)
      • Hierarchical PRM supervision at all 5 levels

    Signature change vs DC-Seg:
        forward(x1, x2, x3, x4, x5_inter, mask, ...)
            x1-x4  : (B, K, C_i, H_i, W_i, Z_i)  stacked encoder features
            x5_inter: (B, K, C5,  H5,  W5,  Z5)  InterFormer output,
                      already cross-modal; K = num_modal, C5 = basic_dims*16
    """
    def __init__(self, num_cls=2, num_modal=8, kid_ch=0):
        super().__init__()
        c = basic_dims

        # ── Level 5 → 4 bridge ──
        self.d4_c1  = general_conv3d(c*16, c*8, pad_type="reflect")

        # ── Level 4 convs ──
        self.d4_c2  = general_conv3d(c*16, c*8, pad_type="reflect")
        self.d4_out = general_conv3d(c*8,  c*8, k_size=1, padding=0, pad_type="reflect")

        # ── Level 3 convs ──
        self.d3_c1  = general_conv3d(c*8,  c*4, pad_type="reflect")
        self.d3_c2  = general_conv3d(c*8,  c*4, pad_type="reflect")
        self.d3_out = general_conv3d(c*4,  c*4, k_size=1, padding=0, pad_type="reflect")

        # ── Level 2 convs ──
        self.d2_c1  = general_conv3d(c*4,  c*2, pad_type="reflect")
        self.d2_c2  = general_conv3d(c*4,  c*2, pad_type="reflect")
        self.d2_out = general_conv3d(c*2,  c*2, k_size=1, padding=0, pad_type="reflect")

        # ── Level 1 convs ──
        self.d1_c1  = general_conv3d(c*2,  c,   pad_type="reflect")
        self.d1_c2  = general_conv3d(c*2,  c,   pad_type="reflect")
        self.d1_out = general_conv3d(c,    c,   k_size=1, padding=0, pad_type="reflect")

        self.seg     = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.up2  = nn.Upsample(scale_factor=2,  mode="trilinear", align_corners=True)
        self.up4  = nn.Upsample(scale_factor=4,  mode="trilinear", align_corners=True)
        self.up8  = nn.Upsample(scale_factor=8,  mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        # ── Region fusion modules at all 5 levels ──
        self.RFM5 = region_aware_modal_fusion_gen(c*16, num_cls, num_modal)
        self.RFM4 = region_aware_modal_fusion_gen(c*8,  num_cls, num_modal)
        self.RFM3 = region_aware_modal_fusion_gen(c*4,  num_cls, num_modal)
        self.RFM2 = region_aware_modal_fusion_gen(c*2,  num_cls, num_modal)
        self.RFM1 = region_aware_modal_fusion_gen(c*1,  num_cls, num_modal)

        # ── PRM generators ──
        # Level 5: laststage (no upstream decoder feature)
        self.prm5 = prm_generator_laststage(c*16, num_cls, num_modal)
        # Levels 4-1: standard (conditioned on upsampled decoder feature)
        self.prm4 = prm_generator(c*8, num_cls, num_modal)
        self.prm3 = prm_generator(c*4, num_cls, num_modal)
        self.prm2 = prm_generator(c*2, num_cls, num_modal)
        self.prm1 = prm_generator(c*1, num_cls, num_modal)

        # ── HGF at levels 1-4 (InterFormer handles level 5) ──
        self.hgf4 = HierarchicalGroupFusion(c*8)
        self.hgf3 = HierarchicalGroupFusion(c*4)
        self.hgf2 = HierarchicalGroupFusion(c*2)
        self.hgf1 = HierarchicalGroupFusion(c*1)
        # Near-zero gate init: sigmoid(-3) ≈ 0.047
        self.hgf_gate4 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate3 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate2 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate1 = nn.Parameter(torch.tensor(-3.0))

        # ── KiD gated injection at level 4 ──
        # DifferenceEncoder outputs at H/8 = level-4 spatial — perfect match.
        self.has_kid = kid_ch > 0
        if self.has_kid:
            self.kid_proj = nn.Sequential(
                general_conv3d(kid_ch, c*4, k_size=1, padding=0, pad_type="reflect"),
                general_conv3d(c*4,    c*8, k_size=1, padding=0, pad_type="reflect"),
            )
            self.kid_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x1, x2, x3, x4, x5_inter, mask, z_k=None, g2_present=None,
                return_logits=False):
        # ── Level 5: InterFormer output — no HGF (cross-modal already done) ──
        p5 = self.prm5(x5_inter, mask)
        d5 = self.RFM5(x5_inter, p5.detach(), mask)  # (B, c*16, H5, W5, Z5)

        fuse_feat = d5                                # saved for image reconstruction
        d5_up = self.d4_c1(self.up2(d5))             # (B, c*8, H4, W4, Z4)

        # ── Level 4 ──
        p4 = self.prm4(d5_up, x4, mask)
        d4 = self.RFM4(x4, p4.detach(), mask)        # (B, c*8, H4, W4, Z4)
        d4 = d4 + torch.sigmoid(self.hgf_gate4) * self.hgf4(x4, mask)

        # KiD injection at level-4 spatial resolution
        if self.has_kid and z_k is not None:
            gate     = torch.sigmoid(self.kid_gate_logit)
            z_k_proj = self.kid_proj(z_k)            # (B, c*8, ...)
            if z_k_proj.shape[2:] != d4.shape[2:]:
                z_k_proj = F.interpolate(z_k_proj, size=d4.shape[2:],
                                         mode='trilinear', align_corners=True)
            if g2_present is not None and not g2_present.all():
                z_k_proj = z_k_proj * g2_present.float().view(-1, 1, 1, 1, 1)
            d4 = d4 + gate * z_k_proj

        d4 = self.d4_out(self.d4_c2(torch.cat((d4, d5_up), 1)))
        d4_up = self.d3_c1(self.up2(d4))             # (B, c*4, H3, W3, Z3)

        # ── Level 3 ──
        p3 = self.prm3(d4_up, x3, mask)
        d3 = self.RFM3(x3, p3.detach(), mask)
        d3 = d3 + torch.sigmoid(self.hgf_gate3) * self.hgf3(x3, mask)
        d3 = self.d3_out(self.d3_c2(torch.cat((d3, d4_up), 1)))
        d3_up = self.d2_c1(self.up2(d3))

        # ── Level 2 ──
        p2 = self.prm2(d3_up, x2, mask)
        d2 = self.RFM2(x2, p2.detach(), mask)
        d2 = d2 + torch.sigmoid(self.hgf_gate2) * self.hgf2(x2, mask)
        d2 = self.d2_out(self.d2_c2(torch.cat((d2, d3_up), 1)))
        d2_up = self.d1_c1(self.up2(d2))

        # ── Level 1 ──
        p1 = self.prm1(d2_up, x1, mask)
        d1 = self.RFM1(x1, p1.detach(), mask)
        d1 = d1 + torch.sigmoid(self.hgf_gate1) * self.hgf1(x1, mask)
        d1 = self.d1_out(self.d1_c2(torch.cat((d1, d2_up), 1)))

        logits = self.seg(d1)
        pred   = self.softmax(logits)
        # PRM supervision at all scales (up-sampled to full resolution)
        prm_preds = (p1,
                     self.up2(p2),
                     self.up4(p3),
                     self.up8(p4),
                     self.up16(p5))

        if return_logits:
            return pred, prm_preds, fuse_feat, logits
        return pred, prm_preds, fuse_feat


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  STYLE ENCODER / IMAGE DECODER  (n_up extended to 4 for 5-level encoder)
# ═══════════════════════════════════════════════════════════════════════════════

class Style_encoder(nn.Module):
    def __init__(self, in_channels=1, n=32):
        super().__init__()
        self.encoder = nn.Sequential(
            BasicConv(in_channels, n,   7, 1, 3, relu=True, norm=False),
            BasicConv(n,    n * 2,  4, 2, 1, relu=True, norm=False),
            BasicConv(n*2,  n * 4,  4, 2, 1, relu=True, norm=False),
            BasicConv(n*4,  n * 4,  4, 2, 1, relu=True, norm=False),
            BasicConv(n*4,  n * 4,  4, 2, 1, relu=True, norm=False))
        self.final = BasicConv(n*4, n*4, 1, 2, 0, relu=False, norm=False)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, [2, 3, 4], keepdim=True)
        return self.final(x)


class MLP(nn.Module):
    def __init__(self, in_ch=128, mlp_ch=128):
        super().__init__()
        self.ch  = mlp_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch,   mlp_ch), nn.ReLU(True),
            nn.Linear(mlp_ch, mlp_ch),  nn.ReLU(True))
        self.l_mu    = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)

    def forward(self, s):
        x = self.net(s.view(s.size(0), -1))
        return (self.l_mu(x).view(-1, self.ch, 1, 1, 1),
                self.l_sigma(x).view(-1, self.ch, 1, 1, 1))


class Image_decoder(nn.Module):
    """AdaIN-based image reconstruction decoder.

    n_up=4 (vs original 3) because the 5-level encoder bottleneck
    (fuse_feat) is at spatial H/16 — needs 4 upsamplings to reach H.
    """
    def __init__(self, in_style=128, in_content=basic_dims*16,
                 mlp_ch=basic_dims*16, n_up=4):
        super().__init__()
        ch = mlp_ch
        self.mlp       = MLP(in_style, mlp_ch)
        self.res_blocks = nn.ModuleList(
            [Adaptive_resblock(in_content, ch) for _ in range(4)])
        dec, c = [], ch
        for _ in range(n_up):
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
# 6.  KINETIC DISENTANGLEMENT (KiD)  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
G2_BASELINE_IDX  = 1
G2_PHASE_INDICES = [2, 3, 4]
NUM_KINETIC_PHASES = len(G2_PHASE_INDICES)
KID_CH = basic_dims * 2   # 32


class DifferenceEncoder(nn.Module):
    """E_Δ: 1-ch difference maps → (B, KID_CH, H/8, W/8, Z/8)."""
    def __init__(self, out_ch: int = KID_CH):
        super().__init__()
        c = basic_dims
        self.net = nn.Sequential(
            general_conv3d(1,    c,     pad_type="reflect"),
            general_conv3d(c,    c*2,   stride=2, pad_type="reflect"),
            general_conv3d(c*2,  c*2,   pad_type="reflect"),
            general_conv3d(c*2,  out_ch, stride=2, pad_type="reflect"),
            general_conv3d(out_ch, out_ch, pad_type="reflect"),
            general_conv3d(out_ch, out_ch, stride=2, pad_type="reflect"))

    def forward(self, x):
        return self.net(x)


class KineticAttentionPool(nn.Module):
    def __init__(self, in_ch: int = KID_CH):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch // 4, 1, 1, bias=False))

    def forward(self, phase_feats: list):
        stacked = torch.stack(phase_feats, dim=1)      # (B, Q, C, H, W, Z)
        B, Q, C, H, W, Z = stacked.shape
        flat    = stacked.reshape(B * Q, C, H, W, Z)
        scores  = self.attn_fc(flat).reshape(B, Q, 1, H, W, Z)
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  mmFormer_Liver — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class mmFormer_Liver(nn.Module):
    """mmFormer backbone for missing-modality liver segmentation.

    Architecture (per forward pass):
    ┌─────────────────────────────────────────────────────────────┐
    │  Per-modality 5-level encoders  (Encoder5 × num_modal)     │
    │       ↓                                                     │
    │  IntraFormer  (per-modality self-attention at level 5)     │
    │       ↓                                                     │
    │  Masker (zero out absent modality tokens)                  │
    │       ↓                                                     │
    │  InterFormer  (cross-modal attention, concat all modals)   │
    │       ↓                                                     │
    │  multimodal_decode_conv  →  x5_inter (B, K, C5, H5, W5, Z5)│
    │       ↓                                                     │
    │  Decoder_fuse (RFM + HGF levels 1-4, KiD at level 4)      │
    │       ↓                                                     │
    │  Decoder_sep5 (per-modality regularisation)  [training]    │
    │  Style_encoder + Image_decoder               [training]    │
    └─────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    num_cls : int
        Segmentation classes (default 2 for liver).
    num_modal : int
        Number of input modalities (default 8).
    use_checkpoint : bool
        Gradient checkpointing on encoders to save memory.
    spatial_size : tuple (H, W, Z)
        Expected input spatial size — used to pre-compute the number of
        transformer tokens (N = H/16 × W/16 × max(1, Z/16)).
    """

    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=True,
                 spatial_size=(128, 128, 16)):
        super().__init__()
        self.num_modal = num_modal
        self.num_cls   = num_cls
        self.use_ckpt  = use_checkpoint

        # ── Spatial token count at level 5 ──
        h5 = max(1, spatial_size[0] // 16)
        w5 = max(1, spatial_size[1] // 16)
        z5 = max(1, spatial_size[2] // 16)
        self.n_tokens = h5 * w5 * z5    # e.g. 8×8×1 = 64 for 128×128×16
        self._h5, self._w5, self._z5 = h5, w5, z5

        # ── Per-modality encoders ──
        self.encoders = nn.ModuleList([Encoder5() for _ in range(num_modal)])

        # ── IntraFormer: per-modality projection + transformer ──
        tdim = transformer_basic_dims
        self.encode_convs = nn.ModuleList([
            nn.Conv3d(basic_dims*16, tdim, 1) for _ in range(num_modal)])
        self.decode_convs = nn.ModuleList([
            nn.Conv3d(tdim, basic_dims*16, 1) for _ in range(num_modal)])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.n_tokens, tdim))
            for _ in range(num_modal)])
        self.intra_transformers = nn.ModuleList([
            Transformer(tdim, depth_tf, num_heads_tf, mlp_dim_tf)
            for _ in range(num_modal)])

        # ── InterFormer: cross-modal transformer ──
        # Concatenates all modality tokens → (B, num_modal × n_tokens, tdim)
        self.inter_transformer = Transformer(tdim, depth_tf, num_heads_tf, mlp_dim_tf)
        # Project back to per-modal feature space
        # (tdim × num_modal) → (basic_dims*16 × num_modal) via 1×1×1 conv
        self.multimodal_decode_conv = nn.Conv3d(
            tdim * num_modal,
            basic_dims * 16 * num_modal,
            kernel_size=1, padding=0)

        # ── Decoders ──
        self.decoder_fuse = Decoder_fuse(num_cls, num_modal, kid_ch=KID_CH)
        self.decoder_sep  = Decoder_sep5(num_cls)

        # ── Style encoder + image decoder (anatomy–style disentanglement) ──
        # Image_decoder content = fuse_feat at level 5 (C=basic_dims*16, H/16)
        # → 4 upsamplings to reach full resolution H.
        self.style_encoders = nn.ModuleList(
            [Style_encoder() for _ in range(num_modal)])
        self.img_decoders = nn.ModuleList([
            Image_decoder(in_style=128,
                          in_content=basic_dims*16,
                          mlp_ch=basic_dims*16,
                          n_up=4)
            for _ in range(num_modal)])

        # ── Kinetic Disentanglement (KiD) ──
        self.diff_encoder  = DifferenceEncoder(out_ch=KID_CH)
        self.kinetic_pool  = KineticAttentionPool(in_ch=KID_CH)

        self.is_training = False
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _encode_one(self, encoder, x_m):
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def _compute_z_k(self, x, mask):
        """Kinetic embedding from G2 phase-difference maps."""
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

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, mask, return_logits=False):
        """
        x    : (B, M, H, W, D)
        mask : (B, M) bool
        """
        B = x.size(0)

        # ── KiD (computed before encoding to save activations) ──
        z_k, g2_present = self._compute_z_k(x, mask)

        # ── 1. Per-modality encoding (sequential for memory) ──
        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))
        # feats[m] = (x1_m, x2_m, x3_m, x4_m, x5_m)

        # Stack per scale: (B, K, C_i, H_i, W_i, Z_i)
        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)
        # x5 only used for transformer; shape (B, K, C5, H5, W5, Z5)
        x5_raw = torch.stack([f[4] for f in feats], 1)

        # ── 2. IntraFormer: per-modality self-attention at bottleneck ──
        intra_tokens = []
        for m in range(self.num_modal):
            x5_m = feats[m][4]                          # (B, C5, H5, W5, Z5)
            tok  = self.encode_convs[m](x5_m)           # (B, tdim, H5, W5, Z5)
            tok  = tok.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)
            tok  = self.intra_transformers[m](tok, self.pos_embeds[m])
            intra_tokens.append(tok)                    # (B, N, tdim)

        # ── 3. Mask absent modality tokens before InterFormer ──
        # Zero the token sequences for missing modalities
        masked_tokens = []
        for m in range(self.num_modal):
            tok_m = intra_tokens[m]                      # (B, N, tdim)
            # mask[:, m]: (B,) bool → broadcast zero
            keep = mask[:, m].float().view(B, 1, 1)      # (B, 1, 1)
            masked_tokens.append(tok_m * keep)

        # ── 4. InterFormer: cross-modal attention ──
        # Concat all modal tokens: (B, K*N, tdim)
        multimodal_tokens = torch.cat(masked_tokens, dim=1)
        multimodal_pos    = torch.cat(list(self.pos_embeds), dim=1)  # (1, K*N, tdim)
        inter_tokens = self.inter_transformer(multimodal_tokens, multimodal_pos)
        # (B, K*N, tdim)  →  reshape to (B, tdim*K, H5, W5, Z5) for 1×1×1 conv
        inter_spatial = inter_tokens.view(
            B, self.num_modal, self.n_tokens, transformer_basic_dims
        ).permute(0, 1, 3, 2).contiguous()              # (B, K, tdim, N)
        inter_spatial = inter_spatial.view(
            B, self.num_modal * transformer_basic_dims,
            self._h5, self._w5, self._z5)               # (B, tdim*K, H5, W5, Z5)

        # Project back to feature space: (B, basic_dims*16*K, H5, W5, Z5)
        x5_inter_flat = self.multimodal_decode_conv(inter_spatial)

        # Reshape to stacked per-modal: (B, K, basic_dims*16, H5, W5, Z5)
        x5_inter = x5_inter_flat.view(
            B, self.num_modal, basic_dims * 16,
            self._h5, self._w5, self._z5)

        # ── 5. Fusion decoder ──
        decoder_out = self.decoder_fuse(
            x1, x2, x3, x4, x5_inter, mask,
            z_k=z_k, g2_present=g2_present,
            return_logits=return_logits)

        if return_logits:
            fuse_pred, prm_preds, fuse_feat, logits = decoder_out
        else:
            fuse_pred, prm_preds, fuse_feat = decoder_out

        if not self.is_training:
            if return_logits:
                return fuse_pred, logits
            return fuse_pred

        # ── 6. Per-modality sep decoder (regulariser) ──
        sep_preds = []
        for m in range(self.num_modal):
            sep_preds.append(self.decoder_sep(*feats[m]))

        # ── 7. Style encoder + image decoder ──
        # fuse_feat: (B, basic_dims*16, H5, W5, Z5) — used as "content"
        recon_list, mu_list, sigma_list, styles_raw = [], [], [], []
        for m in range(self.num_modal):
            st = self.style_encoders[m](x[:, m:m+1])
            styles_raw.append(st)
            rec, mu, sig = self.img_decoders[m](st, fuse_feat)
            recon_list.append(rec)
            mu_list.append(mu)
            sigma_list.append(sig)

        recon_out = torch.cat(recon_list, 1)
        # contents for anatomy contrastive / AGCL: (B, K, C5, H5, W5, Z5)
        # Use raw x5 (pre-transformer encoder features, cleanest anatomy signal)
        contents = x5_raw
        styles   = torch.stack(styles_raw, 1).squeeze(-1).squeeze(-1).squeeze(-1)

        out = (fuse_pred, sep_preds, prm_preds, recon_out,
               mu_list, sigma_list, contents, styles,
               z_k, g2_present)
        if return_logits:
            out = out + (logits,)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  LOSSES  (unchanged)
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
        loss = loss + (-w * target[:, i] *
                       torch.log(output[:, i].clamp(1e-5, 1.0))).mean()
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
    """Asymmetric Anchor-Guided Anatomical Contrastive Learning (AGCL).

    T2WI (index 0) serves as the fixed anatomical anchor.
    One-way InfoNCE with stop-gradient on anchor embeddings.
    """
    def __init__(self, in_ch: int = basic_dims * 16, proj_ch: int = 128,
                 tau: float = 0.07, anchor_idx: int = 0):
        super().__init__()
        self.tau        = tau
        self.anchor_idx = anchor_idx
        self.projector  = nn.Sequential(
            nn.Linear(in_ch, proj_ch),
            nn.ReLU(inplace=True),
            nn.Linear(proj_ch, proj_ch))

    def forward(self, features, mask):
        B, M, C = features.size(0), features.size(1), features.size(2)
        pooled   = features.mean(dim=[3, 4, 5])               # (B, M, C)
        z        = F.normalize(self.projector(pooled), dim=-1) # (B, M, D)
        z_anchor = z[:, self.anchor_idx].detach()              # (B, D)

        loss  = torch.tensor(0.0, device=features.device, dtype=features.dtype)
        count = 0
        for m in range(M):
            if m == self.anchor_idx:
                continue
            present = mask[:, m]
            if not present.any():
                continue
            z_m        = z[present, m]
            sim_matrix = z_m @ z_anchor.t() / self.tau
            labels     = torch.where(present)[0].to(sim_matrix.device)
            loss       = loss + F.cross_entropy(sim_matrix, labels)
            count += 1
        return loss / max(count, 1)


# ── Kinetic Contrastive Loss (KCL) ───────────────────────────────────────────

class KineticContrastiveLoss(nn.Module):
    def __init__(self, in_ch: int = KID_CH, proj_ch: int = 64,
                 tau: float = 0.2, num_samples: int = 128):
        super().__init__()
        self.tau         = tau
        self.num_samples = num_samples
        self.projector   = nn.Sequential(
            nn.Linear(in_ch, proj_ch), nn.ReLU(inplace=True),
            nn.Linear(proj_ch, proj_ch))

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, z_k, target_onehot):
        z_k = z_k.float()
        B, C, H, W, Z = z_k.shape
        gt_down = F.interpolate(target_onehot.float(), size=(H, W, Z), mode="nearest")
        labels  = gt_down.argmax(dim=1).reshape(B, -1)
        z_flat  = z_k.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        S = z_flat.shape[1]
        N = min(self.num_samples, S)
        loss  = torch.tensor(0.0, device=z_k.device, dtype=torch.float32)
        valid = 0
        for b in range(B):
            fg_idx = (labels[b] > 0).nonzero(as_tuple=False).view(-1)
            bg_idx = (labels[b] == 0).nonzero(as_tuple=False).view(-1)
            n_fg = min(N // 2, len(fg_idx))
            n_bg = min(N - n_fg, len(bg_idx))
            if n_fg < 2 or n_bg < 2:
                continue
            fg_sel   = fg_idx[torch.randperm(len(fg_idx), device=z_k.device)[:n_fg]]
            bg_sel   = bg_idx[torch.randperm(len(bg_idx), device=z_k.device)[:n_bg]]
            perm     = torch.cat([fg_sel, bg_sel])
            actual_N = len(perm)
            z_proj   = F.normalize(self.projector(z_flat[b, perm]), dim=-1)
            y_samp   = labels[b, perm]
            sim      = z_proj @ z_proj.t() / self.tau
            mask_self = ~torch.eye(actual_N, dtype=torch.bool, device=z_k.device)
            mask_pos  = (y_samp.unsqueeze(0) == y_samp.unsqueeze(1)) & mask_self
            n_pos = mask_pos.sum(dim=1)
            has_pos = n_pos > 0
            if not has_pos.any():
                continue
            sim_max    = sim.detach().max(dim=1, keepdim=True).values
            logits_s   = sim - sim_max
            exp_logits = torch.exp(logits_s)
            denom      = (exp_logits * mask_self.float()).sum(dim=1, keepdim=True)
            log_prob   = logits_s - torch.log(denom + 1e-8)
            per_sample = (log_prob * mask_pos.float()).sum(dim=1) / n_pos.float().clamp(min=1)
            loss = loss + (-per_sample[has_pos].mean())
            valid += 1
        return loss / max(valid, 1)


# ── Completeness-Aware Self-Distillation ─────────────────────────────────────

class CompletenessAwareDistillationLoss(nn.Module):
    """L_CA: KL divergence teacher (full) → student (incomplete)."""
    def __init__(self, temperature: float = 2.0, max_loss: float = 10.0):
        super().__init__()
        self.T       = temperature
        self.max_loss = max_loss

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, logits_teacher, logits_student):
        logits_teacher = logits_teacher.float().detach()
        logits_student = logits_student.float()
        p_teacher    = F.softmax(logits_teacher / self.T, dim=1)
        log_p_student = F.log_softmax(logits_student / self.T, dim=1)
        loss = F.kl_div(log_p_student, p_teacher, reduction="mean") * (self.T ** 2)
        return loss.clamp(max=self.max_loss)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TRAINING / VALIDATION / TESTING  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, epoch, args,
                    ana_cl, mod_cl, agcl, kcl, ca_loss_fn):
    model.train()
    model.is_training = True
    num_cls = args.num_cls
    losses  = []

    kcl_scale = (0.0 if epoch < args.kcl_start_epoch
                 else min(1.0, (epoch - args.kcl_start_epoch) /
                          max(1, args.kcl_ramp_epochs)))
    ca_scale  = (0.0 if epoch < args.ca_start_epoch
                 else min(1.0, (epoch - args.ca_start_epoch) /
                          max(1, args.ca_ramp_epochs)))

    for i, (x, target, mask, mask_max, _) in enumerate(loader):
        x        = x.cuda(non_blocking=True)
        target   = target.cuda(non_blocking=True)
        mask     = mask.cuda(non_blocking=True)
        mask_max = mask_max.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Teacher pass (all modalities, no grad) for CA ──
        teacher_logits = None
        if ca_scale > 0:
            with torch.no_grad():
                with make_autocast(args.amp):
                    model.is_training = False
                    _, teacher_logits = model(x, mask_max, return_logits=True)
                    model.is_training = True

        with make_autocast(args.amp):
            out = model(x, mask, return_logits=(ca_scale > 0))

            if ca_scale > 0:
                (fuse_pred, sep_preds, prm_preds, recon_out,
                 mu_list, sigma_list, contents, styles,
                 z_k, g2_present, student_logits) = out
            else:
                (fuse_pred, sep_preds, prm_preds, recon_out,
                 mu_list, sigma_list, contents, styles,
                 z_k, g2_present) = out

            fuse_loss = (softmax_weighted_loss(fuse_pred, target, num_cls) +
                         dice_loss(fuse_pred, target, num_cls))
            sep_loss  = sum(softmax_weighted_loss(sp, target, num_cls) +
                            dice_loss(sp, target, num_cls) for sp in sep_preds)
            prm_loss  = sum(softmax_weighted_loss(pp, target, num_cls) +
                            dice_loss(pp, target, num_cls) for pp in prm_preds)

            loss = (sep_loss + prm_loss if epoch < args.region_fusion_start_epoch
                    else fuse_loss + sep_loss + prm_loss)

            alpha   = 1.0
            recon_l = F.mse_loss(recon_out, x)
            kl_l    = sum(KL_divergence(mu_list[m],
                                        torch.log(sigma_list[m].square() + 1e-8))
                          for m in range(args.num_modal))
            ana_l   = ana_cl(contents)
            mod_l   = mod_cl(styles)
            agcl_l  = agcl(contents, mask)

            kcl_l = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if kcl_scale > 0 and z_k is not None and g2_present.any():
                kcl_l = kcl(z_k.detach(), target)

            ca_l = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if ca_scale > 0 and teacher_logits is not None:
                is_incomplete = ~(mask == mask_max).all(dim=1)
                if is_incomplete.any():
                    ca_l = ca_loss_fn(teacher_logits[is_incomplete],
                                      student_logits[is_incomplete])

            loss = loss + alpha * (recon_l + ana_l + mod_l
                                   + args.agcl_weight * agcl_l
                                   + args.kcl_weight * kcl_scale * kcl_l
                                   + args.ca_weight  * ca_scale  * ca_l
                                   + kl_l)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if (i + 1) % max(1, len(loader) // 3) == 0:
            kcl_str  = (f"  kcl={kcl_l.item():.4f}×{kcl_scale:.2f}"
                        if kcl_scale > 0 else "")
            ca_str   = (f"  ca={ca_l.item():.4f}×{ca_scale:.2f}"
                        if ca_scale > 0 else "")
            gate_str = ""
            if hasattr(model, 'decoder_fuse') and \
               hasattr(model.decoder_fuse, 'kid_gate_logit'):
                gv = torch.sigmoid(model.decoder_fuse.kid_gate_logit).item()
                gate_str = f"  kid_g={gv:.4f}"
            hgf_str = ""
            if hasattr(model, 'decoder_fuse') and \
               hasattr(model.decoder_fuse, 'hgf_gate4'):
                hg4 = torch.sigmoid(model.decoder_fuse.hgf_gate4).item()
                hgf_str = f"  hgf4={hg4:.4f}"
            logging.info(f"  Ep {epoch+1} it {i+1}/{len(loader)}"
                         f"  loss={loss.item():.4f}"
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
    results = {}
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
        logging.info(f"  {combo_name}: ${m:.3f}\ +/- {s:.3f}({md:.3f})")

    flat = np.concatenate(all_combo_dice)
    results["Average"] = {"mean": flat.mean(), "std": flat.std(),
                          "median": np.median(flat)}
    logging.info(f"  Average: ${flat.mean():.3f}\ +/- {flat.std():.3f}"
                 f"({np.median(flat):.3f})")
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
        x = x.cuda()
        mask = torch.tensor([mask_all] * x.size(0),
                            dtype=torch.bool, device=x.device)
        with make_autocast(args.amp):
            pred = model(x, mask)
        pl = pred.argmax(1)

        for b in range(x.size(0)):
            name = names[b] if isinstance(names, (list, tuple)) else names
            t2   = x[b, 0].cpu().numpy()
            gt   = y_int[b].cpu().numpy()
            pr   = pl[b].cpu().numpy()
            z    = t2.shape[2] // 2

            base  = t2[:, :, z].astype(np.float32)
            base -= base.min()
            if base.max() > 0:
                base /= base.max()
            rgb = np.stack([base] * 3, -1)

            gt_s, pr_s = (gt[:, :, z] > 0), (pr[:, :, z] > 0)
            go, po, bo = gt_s & ~pr_s, pr_s & ~gt_s, gt_s & pr_s
            a  = 0.55
            ov = rgb.copy()
            ov[go, 0] *= (1-a); ov[go, 1] = ov[go, 1]*(1-a)+a; ov[go, 2] *= (1-a)
            ov[po, 0]  = ov[po, 0]*(1-a)+a; ov[po, 1] *= (1-a); ov[po, 2] *= (1-a)
            ov[bo, 0]  = ov[bo, 0]*(1-a)+a; ov[bo, 1] = ov[bo, 1]*(1-a)+a; ov[bo, 2] *= (1-a)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"{name} z={z}  (R=Pred G=GT Y=Overlap)")
            ax.axis("off"); fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"{name}_z{z}.png"), dpi=150)
            plt.close(fig)
    logging.info(f"Saved visualizations to {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. HELPERS  (unchanged)
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
# 11. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="mmFormer-Liver – Memory-Efficient Training")
    p.add_argument("--datapath", default='./data/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--savepath",  default="./output_mmformer_liver")
    p.add_argument("--resume",    default=None)

    p.add_argument("--resize_x",  type=int, default=256)
    p.add_argument("--resize_y",  type=int, default=256)
    p.add_argument("--resize_z",  type=int, default=32)
    p.add_argument("--crop_x",    type=int, default=None)
    p.add_argument("--crop_y",    type=int, default=None)
    p.add_argument("--crop_z",    type=int, default=None)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed",       type=int, default=1024)
    p.add_argument("--region_fusion_start_epoch", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--num_cls",   type=int, default=2)
    p.add_argument("--num_modal", type=int, default=8)

    p.add_argument("--amp",    action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--no_checkpoint", dest="use_checkpoint",
                   action="store_false", default=True)

    # AGCL
    p.add_argument("--agcl_tau",     type=float, default=0.07)
    p.add_argument("--agcl_proj_ch", type=int,   default=128)
    p.add_argument("--agcl_weight",  type=float, default=1.0)

    # KCL
    p.add_argument("--kcl_tau",         type=float, default=0.2)
    p.add_argument("--kcl_proj_ch",     type=int,   default=64)
    p.add_argument("--kcl_weight",      type=float, default=1.0)
    p.add_argument("--kcl_num_samples", type=int,   default=128)
    p.add_argument("--kcl_start_epoch", type=int,   default=10)
    p.add_argument("--kcl_ramp_epochs", type=int,   default=20)

    # CA distillation
    p.add_argument("--ca_temperature", type=float, default=2.0)
    p.add_argument("--ca_weight",      type=float, default=1.0)
    p.add_argument("--ca_start_epoch", type=int,   default=30)
    p.add_argument("--ca_ramp_epochs", type=int,   default=30)
    p.add_argument("--ca_max_loss",    type=float, default=10.0)

    # Structured Group Dropout
    p.add_argument("--sgd_p_full",  type=float, default=0.3)

    # Gradient clipping
    p.add_argument("--grad_clip",   type=float, default=1.0)

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
        print(f"  {name:25s}:  ${r['mean']:.3f}\ +/- {r['std']:.3f}"
              f"({r['median']:.3f})")
    r = results["Average"]
    print(f"  {'Average':25s}:  ${r['mean']:.3f}\ +/- {r['std']:.3f}"
          f"({r['median']:.3f})")
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

    # ── mmFormer_Liver model ──
    # spatial_size feeds the token-count computation for positional embeddings.
    # Use the crop size if provided, otherwise the full resize shape.
    model_spatial = crop if crop is not None else shape
    model = mmFormer_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=args.use_checkpoint,
        spatial_size=model_spatial,
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"mmFormer_Liver: {nparams:.2f}M params | "
                 f"AMP={args.amp} | GradCkpt={args.use_checkpoint}")
    logging.info(f"  Bottleneck tokens per modality: {model.n_tokens} "
                 f"({model._h5}×{model._w5}×{model._z5})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler   = make_scaler(args.amp)
    lr_sched = LR_Scheduler(args.lr, args.num_epochs)

    ana_cl    = Anatomy_Contrastive_Loss().cuda()
    mod_cl    = Modality_Contrastive_Loss().cuda()
    agcl      = AnchorGuidedContrastiveLoss(
        in_ch=basic_dims * 16,           # bottleneck channels at level 5
        proj_ch=args.agcl_proj_ch,
        tau=args.agcl_tau,
        anchor_idx=0).cuda()
    kcl       = KineticContrastiveLoss(
        in_ch=KID_CH, proj_ch=args.kcl_proj_ch,
        tau=args.kcl_tau, num_samples=args.kcl_num_samples).cuda()
    ca_loss_fn = CompletenessAwareDistillationLoss(
        temperature=args.ca_temperature, max_loss=args.ca_max_loss).cuda()

    optimizer.add_param_group({"params": ana_cl.parameters(),    "lr": args.lr})
    optimizer.add_param_group({"params": mod_cl.parameters(),    "lr": args.lr})
    optimizer.add_param_group({"params": agcl.parameters(),      "lr": args.lr})
    optimizer.add_param_group({"params": kcl.parameters(),       "lr": args.lr})

    # ── evaluate only ──
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logging.info(f"Loaded checkpoint epoch {ck.get('epoch', '?')} (strict=False)")
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
                             "optim": optimizer.state_dict(),
                             "best_dice": best_dice},
                            os.path.join(args.savepath, "model_best.pth"))
                logging.info(f"  ★ Best: {best_dice:.4f}")

        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                     "optim": optimizer.state_dict()},
                    os.path.join(args.savepath, "model_last.pth"))
        if (epoch + 1) % 200 == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                        os.path.join(args.savepath,
                                     f"model_epoch{epoch+1}.pth"))

    logging.info(f"Training done in {(time.time()-t0)/3600:.2f}h")

    # ── final test ──
    bp = os.path.join(args.savepath, "model_best.pth")
    if os.path.exists(bp):
        model.load_state_dict(
            torch.load(bp, map_location="cpu", weights_only=False)["state_dict"])
    results = test_all_combinations(model, test_loader, args, args.savepath)
    _print_results(results)
    visualize_test(model, test_loader, args, args.savepath)


if __name__ == "__main__":
    main()