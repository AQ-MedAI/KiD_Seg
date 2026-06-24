# KiD-Seg: Kinetic-Disentangled Contrastive Learning with Anchor Guidance for Incomplete Multi-Modal Liver Tumor Segmentation

Official implementation of **KiD-Seg**, a unified framework for robust liver-tumor
segmentation under *group-wise* missing MRI modalities. KiD-Seg treats T2-weighted
imaging (`T2WI`) as a permanent anatomical anchor and models the remaining sequences
as atomic modality groups, achieving the best overall and full-modality median Dice on
the 498-case LLD-MMRI dataset.

This repository contains standalone training/testing scripts for the proposed model,
its ablations, and three state-of-the-art baselines.

---

## Highlights

KiD-Seg combines three components on top of a tri-branch DC-Seg backbone:

1. **AGCL — Asymmetric Anchor-Guided Contrastive Learning.** A one-way InfoNCE
   (with stop-gradient on the anchor) pulls auxiliary modalities toward the
   high-fidelity `T2WI` embedding, insulating shared anatomy from distortion-prone
   sequences.
2. **DKD — Differential Kinetic Disentanglement.** Phase-to-baseline difference maps
   (`ΔI = I^phase − I^C-pre`) are encoded and aggregated by cross-phase attention into a
   kinetic feature, supervised by a supervised contrastive loss (KCL).
3. **HGF — Hierarchical Atomic-Group Fusion.** Intra-group attention pooling followed
   by `T2WI`-queried masked cross-attention, with completeness-aware self-distillation
   (KL between a full-modality teacher pass and an incomplete student pass).

### Modality groups

| Group | Modalities | Notes |
|-------|------------|-------|
| `G1`  | `T2WI` | anatomical anchor — **always present** |
| `G2`  | `C-pre`, `C+A`, `C+V`, `C+Delay` | dynamic contrast-enhanced phases |
| `G3`  | `DWI` | diffusion-weighted imaging |
| `G4`  | `InPhase`, `OutPhase` | Dixon |

At inference, the same group-aware fusion handles any valid block set
(`G1`-only … all 8 modalities) **without retraining**.

---

## Repository structure

```
.
├── train_kidseg.py            # Proposed model + ablations (--variant baseline|agcl|agcl_dkd|full)
├── test_kidseg.py             # Evaluation for the proposed model / ablations
├── train_rfnet_liver.py       # SOTA baseline: RFNet
├── test_rfnet_liver.py
├── train_mmformer_liver.py    # SOTA baseline: mmFormer
├── test_mmformer_liver.py
├── train_m3ae_liver.py        # SOTA baseline: M3AE (two-stage: pretrain + finetune)
├── test_m3ae_liver.py
└── README.md
```

`train_kidseg.py` doubles as the **DC-Seg** baseline (`--variant baseline`) and the
**Full KiD-Seg** model (`--variant full`).

---

## Installation

Tested with Python 3.12, PyTorch 2.6 (CUDA 12.x), on Linux and Windows.

```bash
conda create -n kidseg python=3.12 -y
conda activate kidseg

# PyTorch (pick the build matching your CUDA — see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Remaining dependencies
pip install numpy nibabel scipy scikit-learn matplotlib
```

`requirements.txt`:

```
torch>=2.1
numpy
nibabel
scipy
scikit-learn
matplotlib
```

A CUDA GPU is required. The paper uses a single NVIDIA A100 (80 GB) at 256×256×32;
the code also runs on smaller GPUs at reduced resolution (see *Quick demo*).

---

## Dataset preparation

Experiments use the **LLD-MMRI** cohort (Lou et al.; 498 cases, 8 MRI modalities).
Volumes are resampled to **256×256×32**
(1.4×1.4×1.7 mm), SyN-registered to the common `C-pre` space, and min–max normalized
per modality (the loader applies the resize + normalization for you).

Organize each patient as follows. Every patient needs an `images/` and a `labels/`
sub-folder, each containing one NIfTI per modality named `<patient>_<modality>.nii.gz`:

```
<datapath>/
├── <patient_id>/
│   ├── images/
│   │   ├── <patient_id>_T2WI.nii.gz
│   │   ├── <patient_id>_C-pre.nii.gz
│   │   ├── <patient_id>_C+A.nii.gz
│   │   ├── <patient_id>_C+V.nii.gz
│   │   ├── <patient_id>_C+Delay.nii.gz
│   │   ├── <patient_id>_DWI.nii.gz
│   │   ├── <patient_id>_InPhase.nii.gz
│   │   └── <patient_id>_OutPhase.nii.gz
│   └── labels/
│       ├── <patient_id>_T2WI.nii.gz      # tumor mask used for supervision (binarized > 0)
│       └── ... (one per modality; registered to the same space)
└── ...
```

Notes:
- Segmentation is **binary** (tumor vs. background); the supervision mask is the
  `T2WI` label, binarized at `> 0.5`.
- A label file is expected for every modality (the patient is skipped during discovery
  otherwise); these are the same registered mask.
- Patients are split **7:2:1** (train/val/test) deterministically at the patient level;
  test-set results are reported.

---

## Quick demo (small GPU, synthetic-scale settings)

To verify the pipeline end-to-end without the full dataset/compute, train a few epochs
at reduced resolution and test:

```bash
python train_kidseg.py --variant full --datapath ./data/preprocess_nii_256x32 \
    --savepath ./out/kidseg_full --num_epochs 5 --resize_x 64 --resize_y 64 --resize_z 32

python test_kidseg.py  --variant full --datapath ./data/preprocess_nii_256x32 \
    --checkpoint ./out/kidseg_full/model_best.pth --savepath ./out/kidseg_full/test \
    --resize_x 64 --resize_y 64 --resize_z 32
```

For paper-faithful runs, drop the `--resize_*` overrides (defaults are 256×256×32) and
use the default `--num_epochs 200`.

---

## Training

Default hyperparameters already match the paper: **AdamW** (lr `2e-4`, weight decay
`1e-5`), **cosine annealing**, **200 epochs**, **batch size 2**, **256×256×32**, shared
encoder depth **L = 4**. Each run saves `model_best.pth` (best val Dice) and
`model_last.pth` to its `--savepath`.

### Proposed model & ablations (Table 1)

`train_kidseg.py --variant {baseline,agcl,agcl_dkd,full}`:

| Variant     | Modules enabled (AGCL, DKD, HGF) | Paper row |
|-------------|----------------------------------|-----------|
| `baseline`  | ✗ ✗ ✗ | Baseline (DC-Seg) |
| `agcl`      | ✓ ✗ ✗ | + AGCL |
| `agcl_dkd`  | ✓ ✓ ✗ | + AGCL + DKD |
| `full`      | ✓ ✓ ✓ | **Full KiD-Seg** |

```bash
for V in baseline agcl agcl_dkd full; do
  python train_kidseg.py --variant $V --datapath ./data/preprocess_nii_256x32
done
# Default output dir: ./output_dcseg_liver_a2cl_dkd_hac_<variant>/
```

### SOTA baselines (Table 2)

```bash
python train_rfnet_liver.py    --datapath ./data/preprocess_nii_256x32    # -> ./output_rfnet_liver/
python train_mmformer_liver.py --datapath ./data/preprocess_nii_256x32    # -> ./output_mmformer_liver/
python train_m3ae_liver.py     --datapath ./data/preprocess_nii_256x32 \
    --pretrain_epochs 100 --num_epochs 200                                # -> ./output_m3ae_liver/
```

M3AE is two-stage: self-supervised pretraining (`--pretrain_epochs`) followed by
fine-tuning (`--num_epochs`).

---

## Evaluation

Each test script loads the trained checkpoint, evaluates **all 8 group-wise modality
combinations** (plus the pooled Overall), and reports **DSC** and **HD₉₅ (mm)** as
`mean ± std (median)` over every test subject — no per-patient filtering, no daggers.
It also extracts t-SNE features and writes a per-patient qualitative figure
(Input | combinations | overlays).

```bash
# Proposed model / ablations (match --variant to the trained checkpoint)
python test_kidseg.py        --variant full --datapath ./data/preprocess_nii_256x32

# SOTA baselines
python test_rfnet_liver.py    --datapath ./data/preprocess_nii_256x32
python test_mmformer_liver.py --datapath ./data/preprocess_nii_256x32
python test_m3ae_liver.py     --datapath ./data/preprocess_nii_256x32
```

Checkpoints are auto-discovered from the matching default output dir; override with
`--checkpoint` / `--savepath` as needed.

### Outputs

Each test run writes to its `--savepath`:

| File | Contents |
|------|----------|
| `test_results.json` | Per-combination & Overall DSC/HD₉₅ summary |
| `per_subject_scores.json` | Per-subject DSC/HD₉₅ for every combination (for significance tests) |
| `test.log` | Full evaluation log |
| `visualizations/` | Per-patient qualitative + t-SNE figures |

---

## Statistical significance (Wilcoxon signed-rank)

Significance is assessed with a **two-sided Wilcoxon signed-rank test (p < 0.05)**,
reported as **p-values** (no dagger/star markers). Every test run prints, for the model
under test, a within-model analysis (each combination vs. full-modality). To reproduce
the paper's method-vs-method comparison, point one model at another's saved per-subject
scores via `--compare_json`:

```bash
# e.g. KiD-Seg (full) vs. RFNet, paired per patient, per combination + pooled Overall
python test_kidseg.py --variant full --datapath ./data/preprocess_nii_256x32 \
    --compare_json ./output_rfnet_liver/test_output/per_subject_scores.json
```

Example output:

```
STATISTICAL SIGNIFICANCE - two-sided Wilcoxon signed-rank test (p < 0.05)
(B) This model vs. reference '.../rfnet/.../per_subject_scores.json' (paired per patient):
  Combination      | DSC p-value                | HD95 p-value               | n
  G1+G2+G3+G4      | 0.0143 (significant)       | 0.21 (n.s.)                | 50
  ...
  Overall          | 0.0008 (significant)       | 0.03 (significant)         | 400
```

---

## Experimental settings (paper ↔ code)

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr 2×10⁻⁴, weight decay 1×10⁻⁵) |
| Schedule | Cosine annealing, 200 epochs |
| Batch size | 2 |
| Resolution | 256 × 256 × 32 (1.4 × 1.4 × 1.7 mm) |
| Encoder depth | L = 4 |
| Classes / modalities | 2 (binary tumor) / 8 |
| Split | 7 : 2 : 1 (patient-level) |
| Metrics | DSC, HD₉₅ — `mean ± std (median)` |
| Significance | Two-sided Wilcoxon signed-rank, p < 0.05 |

Model sizes (at 256×256×32): RFNet 14.91M · mmFormer ~250M · M3AE 2.20M ·
DC-Seg 76.79M · KiD-Seg 77.02M (only +0.23M over DC-Seg).

---

## Citation

KiD-Seg has been **accepted to MICCAI 2026**. The official proceedings citation
(volume, pages, DOI) is not yet available and will be added once published. In the
meantime, please cite the accepted version:

```bibtex
@inproceedings{khor2026kidseg,
  title     = {KiD-Seg: Kinetic-Disentangled Contrastive Learning with Anchor Guidance
               for Incomplete Multi-Modal Liver Tumor Segmentation},
  author    = {Khor, Hee Guan and Zhang, Dong and Guo, Siyan and Chu, Tianhao and
               Wang, Junyi and Bai, Xiaoyu and Shi, Yinghong and Lu, Le},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer},
  note      = {To appear}
}
```

---

## License

This project is released under the **Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0)** license — see [`LICENSE`](LICENSE) for the full text.

Copyright © 2026 Ant Group and the KiD-Seg authors.

- **Free for research and non-commercial use**, with attribution.
- **Commercial use requires a separate license** — please contact the authors / Ant Group.
- Third-party components (the DC-Seg backbone; the RFNet, mmFormer, and M3AE baselines)
  and the **LLD-MMRI** dataset remain subject to their own licenses and terms of use;
  this license covers only the original code in this repository.

> **Disclaimer.** This software is provided for research purposes only. It is **not a
> medical device and is not intended for clinical use, diagnosis, or treatment**, and is
> distributed "AS IS" without warranty of any kind (see Section 5 of the `LICENSE`).

---

## Acknowledgements

This work was supported by Ant Group, the Ant Group Research Intern Program, and
Zhongshan Hospital, Fudan University. The framework builds on **DC-Seg** (Li et al., 2025)
and is evaluated on the **LLD-MMRI** dataset (Lou et al., 2025); the baselines reimplement
**RFNet**, **mmFormer**, and **M3AE**. See the paper for full references.
