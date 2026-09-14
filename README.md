# KiD-Seg: Kinetic-Disentangled Contrastive Learning with Anchor Guidance for Incomplete Multi-Modal Liver Tumor Segmentation

Official implementation of **KiD-Seg** (MICCAI 2026), a unified framework for robust liver-tumor
segmentation under *group-wise* missing MRI modalities. KiD-Seg treats T2-weighted imaging (`T2WI`) as a
permanent anatomical anchor and models the remaining sequences as atomic modality groups, achieving the
best overall and full-modality median Dice on the 498-case LLD-MMRI dataset.

This repository contains the proposed model and its ablation study (Table 1 of the paper): one training
script and one test script cover the DC-Seg baseline, + AGCL, + AGCL + DKD and the full KiD-Seg model.

---

## Highlights

KiD-Seg combines three components on top of a tri-branch DC-Seg backbone:

1. **AGCL — Asymmetric Anchor-Guided Contrastive Learning.** A one-way InfoNCE (with stop-gradient on the
   anchor) pulls auxiliary modalities toward the high-fidelity `T2WI` embedding, insulating shared anatomy
   from distortion-prone sequences.
2. **DKD — Differential Kinetic Disentanglement.** Phase-to-baseline difference maps
   (`ΔI = I^phase − I^C-pre`) are encoded and aggregated by cross-phase attention into a kinetic feature,
   supervised by a supervised contrastive loss (KCL) and injected into the fused decoder through a gated residual.
3. **HGF — Hierarchical Atomic-Group Fusion.** Intra-group attention pooling followed by `T2WI`-queried
   masked cross-attention at every decoder scale, with completeness-aware self-distillation (KL between a
   full-modality teacher pass and an incomplete student pass).

### Modality groups

| Group | Modalities | Notes |
|-------|------------|-------|
| `G1`  | `T2WI` | anatomical anchor — **always present** |
| `G2`  | `C-pre`, `C+A`, `C+V`, `C+Delay` | dynamic contrast-enhanced phases |
| `G3`  | `DWI` | diffusion-weighted imaging |
| `G4`  | `InPhase`, `OutPhase` | Dixon |

At inference, the same group-aware fusion handles any valid block set (`G1`-only … all 8 modalities)
**without retraining**; every test run evaluates all 8 combinations.

---

## Repository structure

```
.
├── train_kidseg.py   # Proposed model + ablations (--variant baseline|agcl|agcl_dkd|full)
├── test_kidseg.py    # Evaluation: DSC + HD95 for all 8 combinations, Wilcoxon tests, t-SNE, figures
└── README.md
```

---

## Installation

Tested with Python 3.12 and PyTorch 2.6 (CUDA 12.x).

```bash
conda create -n kidseg python=3.12 -y
conda activate kidseg

# PyTorch (pick the build matching your CUDA — see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Remaining dependencies
pip install numpy nibabel scipy scikit-learn matplotlib
```

A CUDA GPU is required. The paper trains on a single NVIDIA A100 (80 GB).

---

## Dataset

Experiments use the **LLD-MMRI** cohort (Lou et al.; 498 cases, 8 MRI modalities). Volumes are resampled
to **256×256×32** (1.4×1.4×1.7 mm), SyN-registered with their annotations to the common `C-pre` space, and
min–max normalised per modality (the scripts apply the resize and normalisation when loading).

Organise each patient as follows — an `images/` and a `labels/` sub-folder, each containing one NIfTI per
modality named `<patient>_<modality>.nii.gz`:

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
│       ├── <patient_id>_T2WI.nii.gz      # tumour mask used for supervision (binarised > 0.5)
│       └── ...                           # one per modality (the same registered mask)
└── ...
```

* Segmentation is **binary** (tumour vs. background).
* A label file is expected for every modality (patients missing any file are skipped during discovery).
* Patients are split **7:2:1** (train / val / test) deterministically at the patient level; test-set
  results are reported.

---

## Training

Default hyper-parameters are those of the paper: **AdamW** (lr `2e-4`, weight decay `1e-5`),
**cosine annealing**, **200 epochs**, **batch size 2**, **256×256×32**, shared encoder depth **L = 4**,
structured group dropout with `G1` always present. Each run saves `model_best.pth` (best validation Dice)
and `model_last.pth` to `--savepath`.

### Proposed model and ablations (Table 1)

`train_kidseg.py --variant {baseline,agcl,agcl_dkd,full}`:

| Variant     | AGCL | DKD | HGF | Paper row |
|-------------|------|-----|-----|-----------|
| `baseline`  | ✗ | ✗ | ✗ | Baseline (DC-Seg) |
| `agcl`      | ✓ | ✗ | ✗ | + AGCL |
| `agcl_dkd`  | ✓ | ✓ | ✗ | + AGCL + DKD |
| `full`      | ✓ | ✓ | ✓ | **Full KiD-Seg** |

```bash
for V in baseline agcl agcl_dkd full; do
  python train_kidseg.py --variant $V --datapath ./data/preprocess_nii_256x32
done
# Default output dir: ./output_dcseg_liver_a2cl_dkd_hac_<variant>/
```

Useful options: `--savepath DIR`, `--use_agcl/--use_dkd/--use_hgf {0,1}` (override the preset for custom
combinations), `--kcl_detach {0,1}` (default `1`, the setting used for the paper: the kinetic feature is
detached inside `L_KCL`), `--no_amp`, `--no_checkpoint`. Batch size must be ≥ 2 because the AGCL InfoNCE
denominator is taken over the batch.

---

## Evaluation

`test_kidseg.py` loads the trained checkpoint, evaluates **all 8 group-wise modality combinations** (plus
the pooled Overall) and reports **DSC** and **HD95 (mm)** as `mean ± std (median)` over every test subject.
It also extracts t-SNE features and writes one qualitative figure per patient (8 modalities |
8 combination overlays | t-SNE) on the axial slice with the largest tumour area.

```bash
# match --variant to the trained checkpoint (a mismatch raises an error)
python test_kidseg.py --variant full --datapath ./data/preprocess_nii_256x32
```

Checkpoints are auto-discovered from the matching default output dir; override with `--checkpoint` /
`--savepath`. `--max_vis_patients N` limits the per-patient t-SNE and figures to the first N test patients
(metrics always use every test patient).

### Outputs

| File | Contents |
|------|----------|
| `test_results.json` | Per-combination & Overall DSC/HD95 summary |
| `per_subject_scores.json` | Per-subject DSC/HD95 for every combination (for significance tests) |
| `test.log` | Full evaluation log including the significance tables |
| `visualizations/` | Per-patient qualitative + t-SNE figures |

### Statistical significance (Wilcoxon signed-rank)

Significance is assessed with a **two-sided Wilcoxon signed-rank test (p < 0.05)**. Every test run prints a
within-model analysis (each combination vs. full-modality). To compare two models paired per patient, point
one run at the other's saved per-subject scores:

```bash
python test_kidseg.py --variant baseline --datapath ./data/preprocess_nii_256x32 \
    --compare_json ./output_dcseg_liver_a2cl_dkd_hac_full/test_output/per_subject_scores.json
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
| Classes / modalities | 2 (binary tumour) / 8 |
| Split | 7 : 2 : 1 (patient-level) |
| Metrics | DSC, HD₉₅ (mm) — `mean ± std (median)` |
| Significance | Two-sided Wilcoxon signed-rank, p < 0.05 |

Model sizes at 256×256×32: DC-Seg 76.79 M · + AGCL 76.79 M · + AGCL + DKD 76.92 M · KiD-Seg 77.02 M
(only +0.23 M over DC-Seg).

### Ablation results (Table 1, Overall = pooled over all 8 combinations)

| Method variant | DSC ↑ | HD₉₅ (mm) ↓ |
|---|---|---|
| Baseline (DC-Seg) | 0.648 ± 0.304 (0.764) | 11.559 ± 21.085 (5.000) |
| + AGCL | 0.677 ± 0.290 (0.769) | 10.447 ± 20.188 (4.899) |
| + AGCL + DKD | 0.681 ± 0.278 (0.773) | 15.200 ± 28.369 (5.099) |
| + AGCL + DKD + HGF (Full KiD-Seg) | **0.670 ± 0.291 (0.781)** | **8.737 ± 16.189 (4.123)** |

---

## Citation

```bibtex
@inproceedings{khor2026kidseg,
  title     = {KiD-Seg: Kinetic-Disentangled Contrastive Learning with Anchor Guidance
               for Incomplete Multi-Modal Liver Tumor Segmentation},
  author    = {Khor, Hee Guan and Zhang, Dong and Guo, Siyan and Chu, Tianhao and
               Wang, Junyi and Bai, Xiaoyu and Shi, Yinghong and Lu, Le},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer}
}
```

---

## License

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)** license (see the `LICENSE` file of the repository).

Copyright © 2026 Ant Group and the KiD-Seg authors.

* **Free for research and non-commercial use**, with attribution.
* **Commercial use requires a separate license** — please contact the authors / Ant Group.
* The DC-Seg backbone and the **LLD-MMRI** dataset remain subject to their own licenses and terms of use;
  this license covers only the original code in this repository.

> **Disclaimer.** This software is provided for research purposes only. It is **not a medical device and is
> not intended for clinical use, diagnosis, or treatment**, and is distributed "AS IS" without warranty of any kind.

---

## Acknowledgements

This work was supported by Ant Group, the Ant Group Research Intern Program, and Zhongshan Hospital, Fudan
University. The framework builds on **DC-Seg** (Li et al., 2025) and is evaluated on the **LLD-MMRI** dataset
(Lou et al., 2025).
