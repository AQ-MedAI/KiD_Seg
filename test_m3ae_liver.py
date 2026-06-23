#!/usr/bin/env python3
"""
Standalone test.py – M3AE for multi-modal liver segmentation with missing modalities.

Evaluates the best M3AE model on the TEST split and reports:
  • Per-combination Dice and HD95: $mean \\pm std\\;(median)^\\dagger$
  • Average Dice and HD95 across all 8 TEST_COMBINATIONS

Generates a comprehensive visualization figure per patient:
  Row 1 : 8 MRI modalities (middle axial slice)
  Row 2 : 8 TEST_COMBINATION overlay maps (pred=green, GT=red, overlap=yellow) on T2WI
  Row 3-L: t-SNE – Class Separability under Severe Missingness (G1-only)
  Row 3-R: t-SNE – Feature Invariance to Input Permutations across combinations

Usage:
  python test_m3ae_liver.py --datapath /path/to/preprocess_nii_256x32_1 \\
                            --checkpoint ./output_m3ae_liver/model_best.pth \\
                            --savepath ./test_output \\
                            --resize_x 128 --resize_y 128 --resize_z 16
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
# 1.  CONSTANTS & GROUP DEFINITIONS  (identical to train_m3ae_liver.py)
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


def structured_group_dropout(p_full: float = 0.3):
    if random.random() < p_full:
        return groups_to_mask(True, True, True)
    combo = random.choice(VALID_GROUP_COMBOS[:-1])
    return groups_to_mask(*combo)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA  (identical to train_m3ae_liver.py)
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
# 3.  MODEL LAYERS  (M3AE – from train_m3ae_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════
basic_dims = 16


def normalization(planes, norm="gn"):
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(min(4, planes), planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
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
# 4.  M3AE ENCODER / DECODER / HEADS / MASKING UTILITIES / MODEL
#     (from train_m3ae_liver.py)
# ═══════════════════════════════════════════════════════════════════════════════

class M3AE_Encoder(nn.Module):
    """
    Single-stream 3-D U-Net encoder.

    Accepts the full N-channel multimodal volume (missing modalities already
    replaced by x_sub before this call).  Produces four skip-connection
    feature tensors at spatial scales 1/1, 1/2, 1/4, 1/8.
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
    """
    def __init__(self, num_cls: int = 2):
        super().__init__()
        c              = basic_dims
        self.head1     = nn.Conv3d(c,     num_cls, kernel_size=1, bias=True)
        self.head_half = nn.Conv3d(c * 2, num_cls, kernel_size=1, bias=True)
        self.head_qtr  = nn.Conv3d(c * 4, num_cls, kernel_size=1, bias=True)
        self.softmax   = nn.Softmax(dim=1)
        self.up2       = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4       = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

    def forward(self, d1, d2, d3):
        """Returns (pred_full, pred_half_up, pred_qtr_up) all at full resolution."""
        p1 = self.softmax(self.head1(d1))
        p2 = self.softmax(self.head_half(d2))
        p3 = self.softmax(self.head_qtr(d3))
        return p1, self.up2(p2), self.up4(p3)


# ── M3AE masking utilities ────────────────────────────────────────────────────

def apply_substitute(x: torch.Tensor,
                     x_sub: torch.Tensor,
                     modality_mask: torch.Tensor) -> torch.Tensor:
    """
    Replace channels of x where the corresponding modality is absent with the
    matching channel of x_sub.  Implements S(x, x_sub) from the paper.

    x             : (B, N, H, W, Z)
    x_sub         : (1, N, H, W, Z)  – learnable substitute image
    modality_mask : (B, N) bool       – True = modality is present
    Returns       : (B, N, H, W, Z)
    """
    B, N = x.shape[0], x.shape[1]
    mask_f = modality_mask.view(B, N, 1, 1, 1).float()
    sub    = x_sub.expand(B, -1, -1, -1, -1)
    return x * mask_f + sub * (1.0 - mask_f)


def mask_patches_3d(x: torch.Tensor,
                    x_sub: torch.Tensor,
                    patch_size: int = 16,
                    target_mask_ratio: float = 0.875) -> torch.Tensor:
    """
    Randomly replace a fraction of non-overlapping 3-D patches with x_sub.
    Used only during Stage-1 pretraining (not needed at test time).
    """
    B, N, H, W, Z = x.shape
    P = patch_size
    nH, nW, nZ = H // P, W // P, Z // P
    n_patches = nH * nW * nZ
    if n_patches == 0:
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
    """Full M3AE masking pipeline: modality dropout + patch masking."""
    x_m = apply_substitute(x, x_sub, modality_mask)
    x_m = mask_patches_3d(x_m, x_sub,
                           patch_size=patch_size,
                           target_mask_ratio=patch_mask_ratio)
    return x_m


# ── M3AE_Liver main model ─────────────────────────────────────────────────────

class M3AE_Liver(nn.Module):
    """
    M3AE model for multi-modal liver segmentation.

    A single 3-D U-Net (one encoder, one decoder) handles all possible subsets
    of the 8 input modalities.  The learnable substitute image x_sub (shape
    (1, N, H, W, Z)) is initialised lazily on the first call and registered as
    a nn.Parameter so it is updated by the same optimiser as the network.

    Forward modes
    ─────────────
    forward_pretrain(x)
        Stage-1: returns (loss_mse, x_sub) — MSE reconstruction loss.

    forward_finetune(x, mask)
        Stage-2 training: returns (pred, (pred_half, pred_qtr), loss_con).

    forward(x, mask)
        Inference: returns the full-resolution segmentation probability map.

    forward_with_features(x, mask)
        Inference + feature extraction for t-SNE analysis.
        Returns (pred, f4_bottleneck, f1_stacked) matching the interface
        expected by extract_features_for_tsne() and the visualisation code.
    """

    def __init__(self,
                 num_cls: int        = 2,
                 num_modal: int      = NUM_MODALITIES,
                 patch_size: int     = 16,
                 mask_ratio: float   = 0.875,
                 sub_init_std: float = 0.1,
                 use_checkpoint: bool = True):
        super().__init__()
        self.num_cls     = num_cls
        self.num_modal   = num_modal
        self.patch_size  = patch_size
        self.mask_ratio  = mask_ratio
        self.use_ckpt    = use_checkpoint
        self.is_training = False   # flag for external eval loops

        self.encoder  = M3AE_Encoder(in_ch=num_modal)
        self.decoder  = M3AE_Decoder()
        self.reg_head = RegressionHead(in_ch=basic_dims, out_ch=num_modal)
        self.seg_head = SegmentationHead(num_cls=num_cls)

        # x_sub initialised lazily (needs input spatial dimensions)
        self._sub_init_std = sub_init_std
        self.x_sub         = None   # becomes nn.Parameter on first call

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

        Returns (loss_mse, x_sub).
        """
        x_sub  = self._get_x_sub_resized(x)
        B      = x.size(0)
        mod_masks = torch.tensor(
            [structured_group_dropout(p_full=0.5) for _ in range(B)],
            dtype=torch.bool, device=x.device)
        x_masked = prepare_masked_input(x, x_sub, mod_masks,
                                        patch_size=self.patch_size,
                                        patch_mask_ratio=self.mask_ratio)
        f1, f2, f3, f4 = self._encode(x_masked)
        d1, _d2, _d3   = self.decoder(f1, f2, f3, f4)
        x_hat          = self.reg_head(d1)
        loss_mse = F.mse_loss(x_hat, x)
        return loss_mse, x_sub

    # ── Stage-2: fine-tuning forward ──────────────────────────────────────────

    def forward_finetune(self, x: torch.Tensor,
                         mask: torch.Tensor):
        """
        Fine-tuning forward pass with heterogeneous self-distillation.

        Returns (pred_full, (pred_half, pred_qtr), loss_con).
        """
        x_sub = self._get_x_sub_resized(x)
        B     = x.size(0)
        x0           = apply_substitute(x, x_sub, mask)
        f1_0, f2_0, f3_0, f4_0 = self._encode(x0)
        d1_0, d2_0, d3_0 = self.decoder(f1_0, f2_0, f3_0, f4_0)
        pred_full, pred_half, pred_qtr = self.seg_head(d1_0, d2_0, d3_0)
        mask1 = torch.tensor(
            [structured_group_dropout(p_full=0.3) for _ in range(B)],
            dtype=torch.bool, device=x.device)
        x1           = apply_substitute(x, x_sub, mask1)
        f1_1, f2_1, f3_1, f4_1 = self._encode(x1)
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

    # ── Feature extraction for t-SNE (matches DC_Seg_Liver interface) ─────────

    def forward_with_features(self, x: torch.Tensor,
                               mask: torch.Tensor):
        """
        Inference forward that also returns encoder features for t-SNE analysis.

        Since M3AE uses a single shared encoder (not per-modality encoders),
        the full-resolution features f1 are broadcast across a virtual "K"
        modality axis so that the downstream t-SNE code (which averages over
        available modalities) works without modification.

        Returns
        -------
        fuse_pred  : (B, num_cls, H, W, Z)  — segmentation output
        fuse_x4    : (B, C*8, H/8, W/8, Z/8) — bottleneck features
        f1_stacked : (B, K, C, H, W, Z)    — full-res features, K=num_modal
                     (each modality slot contains the same shared f1)
        """
        x_sub = self._get_x_sub_resized(x)
        x_in  = apply_substitute(x, x_sub, mask)
        f1, f2, f3, f4 = self.encoder(x_in)
        d1, d2, d3     = self.decoder(f1, f2, f3, f4)
        pred, _ph, _pq = self.seg_head(d1, d2, d3)

        # Expand f1 to (B, K, C, H, W, Z) to match DC_Seg_Liver's interface.
        # The t-SNE code selects available modality slots and averages; since
        # all slots are identical, the result is simply f1[b] regardless of mask.
        B = f1.shape[0]
        f1_stacked = f1.unsqueeze(1).expand(B, self.num_modal, *f1.shape[1:])

        return pred, f4, f1_stacked


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

    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_b & ~binary_erosion(pred_b, structure=struct)
    gt_surface = gt_b & ~binary_erosion(gt_b, structure=struct)

    if pred_surface.sum() == 0:
        pred_surface = pred_b
    if gt_surface.sum() == 0:
        gt_surface = gt_b

    dt_pred = distance_transform_edt(~pred_b, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt_b, sampling=voxel_spacing)

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

        logging.info(f"  {combo_name} Dice : ${d_arr.mean():.3f}\\pm{d_arr.std():.3f}"
                     f"\\;({np.median(d_arr):.3f})^\\dagger$")
        logging.info(f"  {combo_name} HD95 : ${np.nanmean(h_finite):.3f}\\pm{np.nanstd(h_finite):.3f}"
                     f"\\;({np.nanmedian(h_finite):.3f})^\\dagger$")

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
    """Print results in LaTeX format."""
    print("\n" + "=" * 90)
    print(f"{'Combination':25s} | {'Dice':45s} | {'HD95':45s}")
    print("=" * 90)
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str = f"${d['mean']:.3f} \\pm {d['std']:.3f}\\;({d['median']:.3f})^\\dagger$"
        h_str = f"${h['mean']:.3f} \\pm {h['std']:.3f}\\;({h['median']:.3f})^\\dagger$"
        print(f"  {name:23s} | {d_str:43s} | {h_str:43s}")
    print("-" * 90)
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str = f"${d['mean']:.3f} \\pm {d['std']:.3f}\\;({d['median']:.3f})^\\dagger$"
    h_str = f"${h['mean']:.3f} \\pm {h['std']:.3f}\\;({h['median']:.3f})^\\dagger$"
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


def save_results_latex(dice_results, hd95_results, save_dir):
    """Save a LaTeX table snippet."""
    lines = []
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"Combination & Dice & HD95 \\")
    lines.append(r"\midrule")
    for name, _ in TEST_COMBINATIONS:
        d = dice_results[name]
        h = hd95_results[name]
        d_str = f"${d['mean']:.3f} \\pm {d['std']:.3f}\\;({d['median']:.3f})^\\dagger$"
        h_str = f"${h['mean']:.3f} \\pm {h['std']:.3f}\\;({h['median']:.3f})^\\dagger$"
        lines.append(f"{name} & {d_str} & {h_str} \\\\")
    lines.append(r"\midrule")
    d = dice_results["Average"]
    h = hd95_results["Average"]
    d_str = f"${d['mean']:.3f} \\pm {d['std']:.3f}\\;({d['median']:.3f})^\\dagger$"
    h_str = f"${h['mean']:.3f} \\pm {h['std']:.3f}\\;({h['median']:.3f})^\\dagger$"
    lines.append(f"Average & {d_str} & {h_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path = os.path.join(save_dir, "test_results_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"LaTeX table saved to {path}")


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

    NOTE: M3AE uses a single shared encoder, so "per-modality" features do not
    exist independently.  forward_with_features() returns f1 broadcast across K
    modality slots; averaging over available slots still yields the shared f1,
    which captures the model's internal representation per voxel.
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

            fig.suptitle(f"M3AE Test Visualization — Patient: {name} — Axial Slice z={z_mid}",
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
    p = argparse.ArgumentParser(description="M3AE Liver – Standalone Test Script")
    p.add_argument("--datapath", default='/mnt/amed-heyuan/common/data/khorheeguan.khg/data/metaseg_data_liver_mri/preprocess_nii_256x32',
                    help='Path to preprocess_nii_256x32 directory')
    p.add_argument("--checkpoint", default="./output_m3ae_liver/model_best.pth",
                   help="Path to best model checkpoint")
    p.add_argument("--savepath",   default="./output_m3ae_liver/test_output")

    p.add_argument("--resize_x",   type=int, default=256)
    p.add_argument("--resize_y",   type=int, default=256)
    p.add_argument("--resize_z",   type=int, default=32)

    p.add_argument("--seed",       type=int, default=1024)
    p.add_argument("--num_cls",    type=int, default=2)
    p.add_argument("--num_modal",  type=int, default=NUM_MODALITIES)
    p.add_argument("--num_workers", type=int, default=0)

    # M3AE-specific hyperparameters (must match training configuration)
    p.add_argument("--patch_size",  type=int,   default=16,
                   help="3-D patch side length P used during pretraining (default: 16)")
    p.add_argument("--mask_ratio",  type=float, default=0.875,
                   help="Combined masking ratio used during pretraining (default: 0.875)")

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

    # Reproducibility (must match training seed for identical split)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # ── Discover & split patients (identical logic to train_m3ae_liver.py) ──
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

    # ── Load M3AE model ──
    model = M3AE_Liver(
        num_cls=args.num_cls,
        num_modal=args.num_modal,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        sub_init_std=0.1,
        use_checkpoint=False  # no gradient checkpointing needed at test time
    ).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model: M3AE_Liver  {nparams:.2f}M params")

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
    total_inv = sum(len(v) for v in invariance_features_by_patient.values())
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
