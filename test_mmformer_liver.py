#!/usr/bin/env python3
"""
Standalone test.py – mmFormer for multi-modal liver segmentation with missing modalities.

Evaluates the best model on the TEST split and reports:
  • Per-combination Dice and HD95: $mean \ +/-  std\(median)
  • Average Dice and HD95 across all 8 TEST_COMBINATIONS

Generates a comprehensive visualization figure per patient:
  Row 1 : 8 MRI modalities (middle axial slice)
  Row 2 : 8 TEST_COMBINATION overlay maps (pred=green, GT=red, overlap=yellow) on T2WI
  Row 3-L: t-SNE – Class Separability under Severe Missingness (G1-only)
  Row 3-R: t-SNE – Feature Invariance to Input Permutations across combinations

Usage:
  python test_mmformer_liver.py --datapath /path/to/preprocess_nii_256x32_1 \\
                                --checkpoint ./output_mmformer_liver/model_best.pth \\
                                --savepath ./test_output \\
                                --resize_x 128 --resize_y 128 --resize_z 16

══════════════════════════════════════════════════════════════════════════════
TERMINAL COMMANDS — mmFormer testing
══════════════════════════════════════════════════════════════════════════════

### test

conda init bash
source ~/.bashrc
cd <PROJECT_ROOT>
export CUDA_VISIBLE_DEVICES=1
conda activate kidseg
clear
mkdir -p <PROJECT_ROOT>/logs

nohup python -u test_mmformer_liver.py --datapath ./data/preprocess_nii_256x32 --checkpoint ./output_mmformer_liver/model_best.pth --savepath ./output_mmformer_liver/test_output > <PROJECT_ROOT>/logs/test_mmformer_liver.log 2>&1 &
tail -f <PROJECT_ROOT>/logs/test_mmformer_liver.log
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import logging
import math
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
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & GROUP DEFINITIONS  (identical to train_mmformer_liver.py)
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
# 2.  DATA  (identical to train_mmformer_liver.py)
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
# 3.  MODEL LAYERS  (from train_mmformer_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════
basic_dims = 16

# mmFormer transformer hyper-params
transformer_basic_dims = basic_dims * 16   # 256
mlp_dim_tf  = 512
num_heads_tf = 8
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
    """3-D conv + norm + activation with safe reflect-padding fallback."""
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1,
                 pad_type="reflect", norm="in", act_type="lrelu", relufactor=0.2):
        super().__init__()
        self._pad      = padding
        self._pad_type = pad_type
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride, padding=0, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        self.activation = (nn.ReLU(inplace=True) if act_type == "relu"
                           else nn.LeakyReLU(relufactor, inplace=True))

    def forward(self, x):
        if self._pad > 0:
            p = self._pad
            if self._pad_type == "reflect":
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
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
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


# ── PRM generators ────────────────────────────────────────────────────────────

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


# ── Modal / region fusion ─────────────────────────────────────────────────────

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
            general_conv3d(in_channel * num_cls, in_channel,      k_size=1, padding=0),
            general_conv3d(in_channel,           in_channel,      k_size=3, padding=1),
            general_conv3d(in_channel,           in_channel // 2, k_size=1, padding=0))

    def forward(self, region_feats):
        return self.fusion_layer(torch.cat(region_feats, dim=1))


class region_aware_modal_fusion_gen(nn.Module):
    def __init__(self, in_channel=64, num_cls=4, num_modal=8):
        super().__init__()
        self.num_cls   = num_cls
        self.num_modal = num_modal
        self.modal_fusions = nn.ModuleList(
            [modal_fusion(in_channel, num_modal) for _ in range(num_cls)])
        self.region_fuse = region_fusion(in_channel, num_cls)
        self.short_cut = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel,      k_size=1, padding=0),
            general_conv3d(in_channel,             in_channel,      k_size=3, padding=1),
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
# 3b. HIERARCHICAL ATOMIC-GROUP FUSION
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

class Encoder5(nn.Module):
    """5-level residual encoder (adds one stride-2 level over DC-Seg 4-level)."""
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


class Decoder_sep5(nn.Module):
    """Per-modality regularisation decoder — 5 levels."""
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


# KiD constants
G2_BASELINE_IDX  = 1
G2_PHASE_INDICES = [2, 3, 4]
KID_CH = basic_dims * 2   # 32


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=2, num_modal=8, kid_ch=0):
        super().__init__()
        c = basic_dims
        self.d4_c1  = general_conv3d(c*16, c*8, pad_type="reflect")
        self.d4_c2  = general_conv3d(c*16, c*8, pad_type="reflect")
        self.d4_out = general_conv3d(c*8,  c*8, k_size=1, padding=0, pad_type="reflect")
        self.d3_c1  = general_conv3d(c*8,  c*4, pad_type="reflect")
        self.d3_c2  = general_conv3d(c*8,  c*4, pad_type="reflect")
        self.d3_out = general_conv3d(c*4,  c*4, k_size=1, padding=0, pad_type="reflect")
        self.d2_c1  = general_conv3d(c*4,  c*2, pad_type="reflect")
        self.d2_c2  = general_conv3d(c*4,  c*2, pad_type="reflect")
        self.d2_out = general_conv3d(c*2,  c*2, k_size=1, padding=0, pad_type="reflect")
        self.d1_c1  = general_conv3d(c*2,  c,   pad_type="reflect")
        self.d1_c2  = general_conv3d(c*2,  c,   pad_type="reflect")
        self.d1_out = general_conv3d(c,    c,   k_size=1, padding=0, pad_type="reflect")
        self.seg     = nn.Conv3d(c, num_cls, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.up2  = nn.Upsample(scale_factor=2,  mode="trilinear", align_corners=True)
        self.up4  = nn.Upsample(scale_factor=4,  mode="trilinear", align_corners=True)
        self.up8  = nn.Upsample(scale_factor=8,  mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)
        self.RFM5 = region_aware_modal_fusion_gen(c*16, num_cls, num_modal)
        self.RFM4 = region_aware_modal_fusion_gen(c*8,  num_cls, num_modal)
        self.RFM3 = region_aware_modal_fusion_gen(c*4,  num_cls, num_modal)
        self.RFM2 = region_aware_modal_fusion_gen(c*2,  num_cls, num_modal)
        self.RFM1 = region_aware_modal_fusion_gen(c*1,  num_cls, num_modal)
        self.prm5 = prm_generator_laststage(c*16, num_cls, num_modal)
        self.prm4 = prm_generator(c*8, num_cls, num_modal)
        self.prm3 = prm_generator(c*4, num_cls, num_modal)
        self.prm2 = prm_generator(c*2, num_cls, num_modal)
        self.prm1 = prm_generator(c*1, num_cls, num_modal)
        self.hgf4 = HierarchicalGroupFusion(c*8)
        self.hgf3 = HierarchicalGroupFusion(c*4)
        self.hgf2 = HierarchicalGroupFusion(c*2)
        self.hgf1 = HierarchicalGroupFusion(c*1)
        self.hgf_gate4 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate3 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate2 = nn.Parameter(torch.tensor(-3.0))
        self.hgf_gate1 = nn.Parameter(torch.tensor(-3.0))
        self.has_kid = kid_ch > 0
        if self.has_kid:
            self.kid_proj = nn.Sequential(
                general_conv3d(kid_ch, c*4, k_size=1, padding=0, pad_type="reflect"),
                general_conv3d(c*4,    c*8, k_size=1, padding=0, pad_type="reflect"),
            )
            self.kid_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x1, x2, x3, x4, x5_inter, mask, z_k=None, g2_present=None,
                return_logits=False):
        p5 = self.prm5(x5_inter, mask)
        d5 = self.RFM5(x5_inter, p5.detach(), mask)
        fuse_feat = d5
        d5_up = self.d4_c1(self.up2(d5))
        p4 = self.prm4(d5_up, x4, mask)
        d4 = self.RFM4(x4, p4.detach(), mask)
        d4 = d4 + torch.sigmoid(self.hgf_gate4) * self.hgf4(x4, mask)
        if self.has_kid and z_k is not None:
            gate     = torch.sigmoid(self.kid_gate_logit)
            z_k_proj = self.kid_proj(z_k)
            if z_k_proj.shape[2:] != d4.shape[2:]:
                z_k_proj = F.interpolate(z_k_proj, size=d4.shape[2:],
                                         mode='trilinear', align_corners=True)
            if g2_present is not None and not g2_present.all():
                z_k_proj = z_k_proj * g2_present.float().view(-1, 1, 1, 1, 1)
            d4 = d4 + gate * z_k_proj
        d4 = self.d4_out(self.d4_c2(torch.cat((d4, d5_up), 1)))
        d4_up = self.d3_c1(self.up2(d4))
        p3 = self.prm3(d4_up, x3, mask)
        d3 = self.RFM3(x3, p3.detach(), mask)
        d3 = d3 + torch.sigmoid(self.hgf_gate3) * self.hgf3(x3, mask)
        d3 = self.d3_out(self.d3_c2(torch.cat((d3, d4_up), 1)))
        d3_up = self.d2_c1(self.up2(d3))
        p2 = self.prm2(d3_up, x2, mask)
        d2 = self.RFM2(x2, p2.detach(), mask)
        d2 = d2 + torch.sigmoid(self.hgf_gate2) * self.hgf2(x2, mask)
        d2 = self.d2_out(self.d2_c2(torch.cat((d2, d3_up), 1)))
        d2_up = self.d1_c1(self.up2(d2))
        p1 = self.prm1(d2_up, x1, mask)
        d1 = self.RFM1(x1, p1.detach(), mask)
        d1 = d1 + torch.sigmoid(self.hgf_gate1) * self.hgf1(x1, mask)
        d1 = self.d1_out(self.d1_c2(torch.cat((d1, d2_up), 1)))
        logits = self.seg(d1)
        pred   = self.softmax(logits)
        prm_preds = (p1, self.up2(p2), self.up4(p3), self.up8(p4), self.up16(p5))
        if return_logits:
            return pred, prm_preds, fuse_feat, logits
        return pred, prm_preds, fuse_feat


# ── Style encoder / Image decoder ────────────────────────────────────────────

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
    def __init__(self, in_style=128, in_content=basic_dims*16,
                 mlp_ch=basic_dims*16, n_up=4):
        super().__init__()
        ch = mlp_ch
        self.mlp        = MLP(in_style, mlp_ch)
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


# ── Kinetic Disentanglement ───────────────────────────────────────────────────

class DifferenceEncoder(nn.Module):
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
        stacked = torch.stack(phase_feats, dim=1)
        B, Q, C, H, W, Z = stacked.shape
        flat    = stacked.reshape(B * Q, C, H, W, Z)
        scores  = self.attn_fc(flat).reshape(B, Q, 1, H, W, Z)
        weights = F.softmax(scores, dim=1)
        return (weights * stacked).sum(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  mmFormer_Liver — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class mmFormer_Liver(nn.Module):
    """mmFormer backbone for missing-modality liver segmentation."""

    def __init__(self, num_cls=2, num_modal=8, use_checkpoint=False,
                 spatial_size=(128, 128, 16)):
        super().__init__()
        self.num_modal = num_modal
        self.num_cls   = num_cls
        self.use_ckpt  = use_checkpoint

        h5 = max(1, spatial_size[0] // 16)
        w5 = max(1, spatial_size[1] // 16)
        z5 = max(1, spatial_size[2] // 16)
        self.n_tokens = h5 * w5 * z5
        self._h5, self._w5, self._z5 = h5, w5, z5

        self.encoders = nn.ModuleList([Encoder5() for _ in range(num_modal)])

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

        self.inter_transformer = Transformer(tdim, depth_tf, num_heads_tf, mlp_dim_tf)
        self.multimodal_decode_conv = nn.Conv3d(
            tdim * num_modal,
            basic_dims * 16 * num_modal,
            kernel_size=1, padding=0)

        self.decoder_fuse = Decoder_fuse(num_cls, num_modal, kid_ch=KID_CH)
        self.decoder_sep  = Decoder_sep5(num_cls)

        self.style_encoders = nn.ModuleList(
            [Style_encoder() for _ in range(num_modal)])
        self.img_decoders = nn.ModuleList([
            Image_decoder(in_style=128, in_content=basic_dims*16,
                          mlp_ch=basic_dims*16, n_up=4)
            for _ in range(num_modal)])

        self.diff_encoder  = DifferenceEncoder(out_ch=KID_CH)
        self.kinetic_pool  = KineticAttentionPool(in_ch=KID_CH)

        self.is_training = False
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def _encode_one(self, encoder, x_m):
        if self.use_ckpt and self.training:
            return grad_checkpoint(encoder, x_m, use_reentrant=False)
        return encoder(x_m)

    def _compute_z_k(self, x, mask):
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

    def _run_transformer(self, feats, mask):
        """Shared IntraFormer + InterFormer pass. Returns x5_inter (B, K, C5, H5, W5, Z5)."""
        B = feats[0][0].size(0)
        intra_tokens = []
        for m in range(self.num_modal):
            x5_m = feats[m][4]
            tok  = self.encode_convs[m](x5_m)
            tok  = tok.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, transformer_basic_dims)
            tok  = self.intra_transformers[m](tok, self.pos_embeds[m])
            intra_tokens.append(tok)

        masked_tokens = []
        for m in range(self.num_modal):
            keep = mask[:, m].float().view(B, 1, 1)
            masked_tokens.append(intra_tokens[m] * keep)

        multimodal_tokens = torch.cat(masked_tokens, dim=1)
        multimodal_pos    = torch.cat(list(self.pos_embeds), dim=1)
        inter_tokens = self.inter_transformer(multimodal_tokens, multimodal_pos)

        inter_spatial = inter_tokens.view(
            B, self.num_modal, self.n_tokens, transformer_basic_dims
        ).permute(0, 1, 3, 2).contiguous()
        inter_spatial = inter_spatial.view(
            B, self.num_modal * transformer_basic_dims,
            self._h5, self._w5, self._z5)

        x5_inter_flat = self.multimodal_decode_conv(inter_spatial)
        x5_inter = x5_inter_flat.view(
            B, self.num_modal, basic_dims * 16,
            self._h5, self._w5, self._z5)
        return x5_inter

    def forward(self, x, mask, return_logits=False):
        B = x.size(0)
        z_k, g2_present = self._compute_z_k(x, mask)

        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))

        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)
        x5_raw = torch.stack([f[4] for f in feats], 1)

        x5_inter = self._run_transformer(feats, mask)

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

        sep_preds = []
        for m in range(self.num_modal):
            sep_preds.append(self.decoder_sep(*feats[m]))

        recon_list, mu_list, sigma_list, styles_raw = [], [], [], []
        for m in range(self.num_modal):
            st = self.style_encoders[m](x[:, m:m+1])
            styles_raw.append(st)
            rec, mu, sig = self.img_decoders[m](st, fuse_feat)
            recon_list.append(rec)
            mu_list.append(mu)
            sigma_list.append(sig)

        recon_out = torch.cat(recon_list, 1)
        contents  = x5_raw
        styles    = torch.stack(styles_raw, 1).squeeze(-1).squeeze(-1).squeeze(-1)

        out = (fuse_pred, sep_preds, prm_preds, recon_out,
               mu_list, sigma_list, contents, styles,
               z_k, g2_present)
        if return_logits:
            out = out + (logits,)
        return out

    def forward_with_features(self, x, mask):
        """Forward pass returning prediction + multi-scale features for t-SNE.

        Returns:
            fuse_pred  : (B, num_cls, H, W, D)
            fuse_feat  : (B, C5, H/16, W/16, D/16)  bottleneck (GAP-pooled invariance)
            x1_stacked : (B, K, C1, H, W, D)         full-res encoder features (class sep.)
        """
        B = x.size(0)
        z_k, g2_present = self._compute_z_k(x, mask)

        feats = []
        for m in range(self.num_modal):
            feats.append(self._encode_one(self.encoders[m], x[:, m:m+1]))

        x1 = torch.stack([f[0] for f in feats], 1)
        x2 = torch.stack([f[1] for f in feats], 1)
        x3 = torch.stack([f[2] for f in feats], 1)
        x4 = torch.stack([f[3] for f in feats], 1)

        x5_inter = self._run_transformer(feats, mask)

        fuse_pred, prm_preds, fuse_feat = self.decoder_fuse(
            x1, x2, x3, x4, x5_inter, mask,
            z_k=z_k, g2_present=g2_present)

        return fuse_pred, fuse_feat, x1


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS – Dice & Hausdorff Distance 95
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    intersection = np.sum(p & g)
    return float((2.0 * intersection + eps) / (p.sum() + g.sum() + eps))


def compute_hd95(pred: np.ndarray, gt: np.ndarray, voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return np.inf
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_b & ~binary_erosion(pred_b, structure=struct)
    gt_surface   = gt_b   & ~binary_erosion(gt_b,   structure=struct)
    if pred_surface.sum() == 0:
        pred_surface = pred_b
    if gt_surface.sum() == 0:
        gt_surface = gt_b
    dt_pred = distance_transform_edt(~pred_b, sampling=voxel_spacing)
    dt_gt   = distance_transform_edt(~gt_b,   sampling=voxel_spacing)
    d_gt2pred  = dt_pred[gt_surface]
    d_pred2gt  = dt_gt[pred_surface]
    all_dist   = np.concatenate([d_gt2pred, d_pred2gt])
    return float(np.percentile(all_dist, 95))


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TEST ALL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_all_combinations(model, test_loader, args, save_dir):
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

        d_arr    = np.array(dice_scores)
        h_arr    = np.array(hd95_scores)
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

        logging.info(f"  {combo_name} Dice : ${d_arr.mean():.3f}\ +/- {d_arr.std():.3f}"
                     f"\({np.median(d_arr):.3f})")
        logging.info(f"  {combo_name} HD95 : ${np.nanmean(h_finite):.3f}\ +/- {np.nanstd(h_finite):.3f}"
                     f"\({np.nanmedian(h_finite):.3f})")

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
    print("\n" + "=" * 90)
    print(f"{'Combination':25s} | {'Dice':45s} | {'HD95':45s}")
    print("=" * 90)
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str = f"{d['mean']:.3f} \ +/-  {d['std']:.3f}\({d['median']:.3f})"
        h_str = f"{h['mean']:.3f} \ +/-  {h['std']:.3f}\({h['median']:.3f})"
        print(f"  {name:23s} | {d_str:43s} | {h_str:43s}")
    print("-" * 90)
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str = f"{d['mean']:.3f} \ +/-  {d['std']:.3f}\({d['median']:.3f})"
    h_str = f"{h['mean']:.3f} \ +/-  {h['std']:.3f}\({h['median']:.3f})"
    print(f"  {'Average':23s} | {d_str:43s} | {h_str:43s}")
    print("=" * 90)


def save_results_json(dice_results, hd95_results, save_dir):
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


def save_results_latex(dice_results, hd95_results, save_dir):
    lines = []
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"Combination & Dice & HD95 \\")
    lines.append(r"\midrule")
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str = f"{d['mean']:.3f} \ +/-  {d['std']:.3f}\({d['median']:.3f})"
        h_str = f"{h['mean']:.3f} \ +/-  {h['std']:.3f}\({h['median']:.3f})"
        lines.append(f"{name} & {d_str} & {h_str} \\\\")
    lines.append(r"\midrule")
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str = f"{d['mean']:.3f} \ +/-  {d['std']:.3f}\({d['median']:.3f})"
    h_str = f"{h['mean']:.3f} \ +/-  {h['std']:.3f}\({h['median']:.3f})"
    lines.append(f"Average & {d_str} & {h_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path = os.path.join(save_dir, "test_results_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"LaTeX table saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FEATURE EXTRACTION FOR t-SNE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features_for_tsne(model, test_loader, args):
    model.eval()
    model.is_training = False

    class_features_by_patient     = {}
    invariance_features_by_patient = {}

    # (A) Class Separability (G1 only)
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

            gt     = y_int[b].cpu().numpy()
            mask_b = mask[b]
            x1_b   = x1_full[b]
            x1_avail  = x1_b[mask_b]
            feat_full = x1_avail.mean(dim=0).cpu().numpy()

            tumor_coords = np.argwhere(gt > 0)
            bg_coords    = np.argwhere(gt == 0)
            n_sample     = min(500, len(tumor_coords), len(bg_coords))
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
                    (feat_full[:, tc[0], tc[1], tc[2]], 1))
            for bc in bg_coords:
                class_features_by_patient[name].append(
                    (feat_full[:, bc[0], bc[1], bc[2]], 0))

    # (B) Invariance to Input Permutations (per patient)
    N_PATCHES = 50

    for x, y_int, names in test_loader:
        x, y_int = x.cuda(), y_int.cuda()
        if isinstance(names, str):
            names = [names]
        B = x.size(0)

        for b in range(B):
            name = names[b]
            invariance_features_by_patient.setdefault(name, [])

            gt           = y_int[b].cpu().numpy()
            tumor_coords = np.argwhere(gt > 0)
            if len(tumor_coords) < 2:
                continue

            xb = x[b:b+1]
            for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
                mask_list = groups_to_mask(g2, g3, g4)
                mask_t    = torch.tensor([mask_list], dtype=torch.bool, device=x.device)

                with make_autocast(args.amp):
                    _, _, x1_full = model.forward_with_features(xb, mask_t)

                mask_b    = mask_t[0]
                x1_avail  = x1_full[0][mask_b]
                feat_full = x1_avail.mean(dim=0).cpu().numpy()

                n_samp = min(N_PATCHES, len(tumor_coords))
                idx    = np.random.choice(len(tumor_coords), n_samp, replace=False)
                for i in idx:
                    tc = tumor_coords[i]
                    invariance_features_by_patient[name].append(
                        (feat_full[:, tc[0], tc[1], tc[2]], combo_name))

    return class_features_by_patient, invariance_features_by_patient


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  COMPREHENSIVE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_slice(s):
    s = s.astype(np.float32)
    s -= s.min()
    mx = s.max()
    if mx > 0:
        s /= mx
    return s


def _orient_slice(s):
    return np.fliplr(np.rot90(s, k=-1))


def _overlay_pred_gt(base_rgb, pred_slice, gt_slice, alpha=0.55):
    ov     = base_rgb.copy()
    pred_b = pred_slice.astype(bool)
    gt_b   = gt_slice.astype(bool)
    gt_only   = gt_b & ~pred_b
    pred_only = pred_b & ~gt_b
    both      = gt_b & pred_b
    ov[gt_only,   0] = ov[gt_only,   0] * (1 - alpha) + alpha
    ov[gt_only,   1] *= (1 - alpha)
    ov[gt_only,   2] *= (1 - alpha)
    ov[pred_only, 0] *= (1 - alpha)
    ov[pred_only, 1] = ov[pred_only, 1] * (1 - alpha) + alpha
    ov[pred_only, 2] *= (1 - alpha)
    ov[both,      0] = ov[both, 0] * (1 - alpha) + alpha
    ov[both,      1] = ov[both, 1] * (1 - alpha) + alpha
    ov[both,      2] *= (1 - alpha)
    return np.clip(ov, 0, 1)


def _fit_tsne_2d(X: np.ndarray, random_state: int = 42):
    if len(X) < 4:
        return None
    perp = min(30, max(2, len(X) // 4))
    base_kwargs = dict(n_components=2, perplexity=perp,
                       random_state=random_state, init="pca")
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
            x_pt   = x[b]
            gt_pt  = y_int[b].numpy()
            name   = names[b]

            H, W, D = gt_pt.shape
            z_mid   = D // 2

            modality_slices = []
            for m in range(NUM_MODALITIES):
                sl = _orient_slice(_norm_slice(x_pt[m, :, :, z_mid].numpy()))
                modality_slices.append(sl)

            t2_base = modality_slices[0]
            t2_rgb  = np.stack([t2_base] * 3, axis=-1)

            combo_overlays = []
            xb = x[b:b+1].cuda()
            for combo_name, (g2, g3, g4) in TEST_COMBINATIONS:
                mask_list = groups_to_mask(g2, g3, g4)
                mask_t    = torch.tensor([mask_list], dtype=torch.bool, device=xb.device)
                with make_autocast(args.amp):
                    pred = model(xb, mask_t)

                pred_slice = _orient_slice(
                    (pred.argmax(1)[0].cpu().numpy()[:, :, z_mid] > 0).astype(np.uint8))
                gt_slice = _orient_slice((gt_pt[:, :, z_mid] > 0).astype(np.uint8))
                ov = _overlay_pred_gt(t2_rgb.copy(), pred_slice, gt_slice)
                combo_overlays.append((combo_name, ov))

            class_features     = class_features_by_patient.get(name, [])
            invariance_features = invariance_features_by_patient.get(name, [])

            cf_emb, cf_y = None, None
            if len(class_features) >= 4:
                cf_X = np.array([f[0] for f in class_features])
                cf_y = np.array([f[1] for f in class_features])
                if len(cf_X) > 4000:
                    idx  = np.random.choice(len(cf_X), 4000, replace=False)
                    cf_X, cf_y = cf_X[idx], cf_y[idx]
                cf_emb = _fit_tsne_2d(cf_X, random_state=42)

            inv_emb, inv_y = None, None
            if len(invariance_features) >= 4:
                inv_X      = np.array([f[0] for f in invariance_features])
                inv_labels = [f[1] for f in invariance_features]
                inv_y      = np.array([combo_names_ordered.index(l) for l in inv_labels])
                if len(inv_X) > 4000:
                    idx    = np.random.choice(len(inv_X), 4000, replace=False)
                    inv_X, inv_y = inv_X[idx], inv_y[idx]
                inv_emb = _fit_tsne_2d(inv_X, random_state=42)

            fig = plt.figure(figsize=(28, 14), dpi=150)
            gs  = gridspec.GridSpec(3, 8, figure=fig, hspace=0.35, wspace=0.15,
                                    height_ratios=[1, 1, 1.3])

            group_colors = {"G1": "#2196F3", "G2": "#FF9800", "G3": "#4CAF50", "G4": "#9C27B0"}
            for i in range(8):
                ax  = fig.add_subplot(gs[0, i])
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
                                 c='#3498db', alpha=0.5, s=12, label='Background',
                                 edgecolors='none')
                ax_tsne1.scatter(cf_emb[tm_mask, 0], cf_emb[tm_mask, 1],
                                 c='#e74c3c', alpha=0.5, s=12, label='Tumor',
                                 edgecolors='none')
                ax_tsne1.legend(fontsize=9, loc='upper left',
                                bbox_to_anchor=(1.02, 1.0), borderaxespad=0, framealpha=0.8)
            else:
                ax_tsne1.text(0.5, 0.5, "Insufficient samples for per-patient t-SNE",
                              ha='center', va='center', fontsize=10,
                              transform=ax_tsne1.transAxes)
            ax_tsne1.set_title("Per-Patient t-SNE: Class Separability (G1 only)",
                                fontsize=11, fontweight="bold")
            ax_tsne1.set_xlabel("t-SNE 1", fontsize=9)
            ax_tsne1.set_ylabel("t-SNE 2", fontsize=9)
            ax_tsne1.tick_params(labelsize=7)
            ax_tsne1.set_box_aspect(1)
            ax_tsne1.grid(True, alpha=0.2)

            ax_tsne2 = fig.add_subplot(gs[2, 4:])

            def _mathcal_combo(name_):
                parts     = name_.split("+")
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
                ax_tsne2.legend(fontsize=7, loc='upper left',
                                bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                                framealpha=0.8, ncol=1)
            else:
                ax_tsne2.text(0.5, 0.5, "Insufficient samples for per-patient t-SNE",
                              ha='center', va='center', fontsize=10,
                              transform=ax_tsne2.transAxes)
            ax_tsne2.set_title("Per-Patient t-SNE: Feature Invariance to Input Permutations\n"
                                "(Tumor features across modality availability)",
                                fontsize=11, fontweight="bold")
            ax_tsne2.set_xlabel("t-SNE 1", fontsize=9)
            ax_tsne2.set_ylabel("t-SNE 2", fontsize=9)
            ax_tsne2.tick_params(labelsize=7)
            ax_tsne2.set_box_aspect(1)
            ax_tsne2.grid(True, alpha=0.2)

            fig.suptitle(f"mmFormer Test Visualization — Patient: {name} — Axial Slice z={z_mid}",
                         fontsize=14, fontweight="bold", y=0.98)

            out_path = os.path.join(vis_dir, f"{name}_comprehensive_z{z_mid}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved_paths.append(out_path)
            logging.info(f"Comprehensive visualization saved to {out_path}")

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="mmFormer Liver – Standalone Test Script")
    p.add_argument("--datapath", default='./data/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--checkpoint", default="./output_mmformer_liver/model_best.pth",
                   help="Path to best model checkpoint")
    p.add_argument("--savepath",   default="./output_mmformer_liver/test_output")

    p.add_argument("--resize_x", type=int, default=256)
    p.add_argument("--resize_y", type=int, default=256)
    p.add_argument("--resize_z", type=int, default=32)

    p.add_argument("--seed",        type=int, default=1024)
    p.add_argument("--num_cls",     type=int, default=2)
    p.add_argument("--num_modal",   type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--amp",    action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark    = False
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

    test_set    = LiverTestDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=args.num_workers, pin_memory=True)

    # ── Load mmFormer_Liver model ──
    model = mmFormer_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        use_checkpoint=False,          # no grad-ckpt needed at test time
        spatial_size=(args.resize_x, args.resize_y, args.resize_z),
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"mmFormer_Liver: {nparams:.2f}M params")
    logging.info(f"  Bottleneck tokens per modality: {model.n_tokens} "
                 f"({model._h5}×{model._w5}×{model._z5})")

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
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
    save_results_latex(dice_results, hd95_results, args.savepath)

    # ══════════════════════════════════════════════════════════════════════
    # B.  EXTRACT FEATURES FOR t-SNE
    # ══════════════════════════════════════════════════════════════════════
    logging.info("Extracting features for t-SNE ...")
    class_features_by_patient, invariance_features_by_patient = extract_features_for_tsne(
        model, test_loader, args)
    total_class = sum(len(v) for v in class_features_by_patient.values())
    total_inv   = sum(len(v) for v in invariance_features_by_patient.values())
    logging.info(f"  Class separability samples (total): {total_class}")
    logging.info(f"  Invariance samples (total): {total_inv}")
    for pn in sorted(class_features_by_patient.keys()):
        logging.info(f"    {pn}: class={len(class_features_by_patient[pn])}, "
                     f"inv={len(invariance_features_by_patient.get(pn, []))}")

    # ══════════════════════════════════════════════════════════════════════
    # C.  COMPREHENSIVE VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════
    logging.info("Generating comprehensive visualization ...")
    fig_paths = generate_comprehensive_visualization(
        model, test_loader, args, args.savepath,
        class_features_by_patient, invariance_features_by_patient)
    logging.info(f"Done. Generated {len(fig_paths)} figures in "
                 f"{os.path.join(args.savepath, 'visualizations')}")


if __name__ == "__main__":
    main()
