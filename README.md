# Automated Ultrastructural Analysis of Platelet Activation in Electron Microscopy

A **segmentation-centric deep-learning framework** for the automated, quantitative
analysis of platelet activation from high-resolution Electron Microscopy (EM)
images. The project pairs a robust **instance segmentation** pipeline for
delineating individual platelet bodies and internal organelles with downstream,
clinically meaningful **multi-class activation grading** (Grade 0: resting →
Grade 3: fully activated).

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-blue">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-green">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-orange">
</p>

> **Collaboration.** This work is a joint effort between the **Department of
> Medical Sciences and Technology (DMST), IIT Madras** and **AIIMS Delhi**.

---

## Table of Contents

- [Motivation](#motivation)
- [Framework Overview](#framework-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretrained Model](#pretrained-model)
- [Phase 1 — Pre-annotation & Feature Extraction](#phase-1--pre-annotation--feature-extraction)
- [Phase 2 — Deep-Learning Segmentation & Grading](#phase-2--deep-learning-segmentation--grading)
- [Reference Notebook](#reference-notebook)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation

Platelets progress through a morphological continuum upon activation — from
discoid resting shapes to spreading forms with filopodia/pseudopodia and
centralization of dense granules. These ultrastructural changes are
diagnostically important for hemostasis, thrombosis, inflammation, and the
assessment of antiplatelet therapies, yet they are reliably visible only under
EM.

Objective, high-throughput quantification is impeded by three bottlenecks:

1. **Resolution** — light microscopy cannot resolve early/internal ultrastructure.
2. **Manual effort & subjectivity** — EM annotation is slow and reader-dependent.
3. **Segmentation difficulty** — platelets occur in dense clusters with poorly
   defined boundaries and high intra-class variability.

This framework targets the principal bottleneck — **reliable delineation of
individual platelet instances** — and builds quantitative grading on top of it.

---

## Framework Overview

The pipeline is organized into two sequential phases.

```
              ┌─────────────────────────────────────────────────────────────┐
  Raw EM      │  PHASE 1 — Automated Pre-annotation & Feature Extraction      │
  image  ───► │  (deterministic computer vision)                             │
              │                                                              │
              │  1. Plasma-membrane ROI   (adaptive threshold, GREEN)        │
              │  2. Dense granules        (stringent dark threshold, BLUE)   │
              │  3. Open Canalicular Sys. (bright threshold, RED)            │
              │  4. Morphology: fractal dim · GLCM texture · clustering      │
              │  5. Per-image CSV → consolidated feature table (.xlsx)       │
              └───────────────┬─────────────────────────────────────────────┘
                              │  composite masks + morphological feature table
                              ▼
              ┌─────────────────────────────────────────────────────────────┐
  Grade 0–3   │  PHASE 2 — Deep-Learning Segmentation & Grading              │
  + masks ◄── │  (DeepLabV3+ · Deformable ResNet-101 · VGG fusion · grading) │
              │  trained model shared via Google Drive — see Pretrained Model │
              └─────────────────────────────────────────────────────────────┘
```

- **Phase 1** is a fully implemented, deterministic computer-vision pipeline.
  The **primary, automated** path (`detection.py` → `batch_processing.py` →
  `consolidate.py`) produces composite ground-truth masks and a consolidated
  morphological **feature table**, fully unattended given per-case thresholds.
  A set of **interactive** tools is also provided for tuning those thresholds and
  for manual annotation. A transparent, **rule-based** analyzer offers an
  interpretable activation-grading baseline.
- **Phase 2** is the learned framework from the accompanying paper. Its module
  scaffolding and documented architecture are included here; the **trained model
  is distributed via Google Drive** (see [Pretrained Model](#pretrained-model))
  because of its size, and the full training scripts will follow.

---

## Repository Structure

```
IITM_AIIMSD_Platelet/
├── README.md
├── requirements.txt
├── pyproject.toml                  # installable package + `platelet-em` CLI
├── .gitignore
├── notebooks/
│   └── Annotation_&_Feature_Extraction.ipynb   # main reference notebook (full pipeline)
├── docs/
│   └── figures/                    # (place pipeline figures / example results here)
└── src/
    └── platelet_em/
        ├── __init__.py
        ├── cli.py                  # unified command-line interface
        │
        │   # ── Primary automated Phase-1 pipeline ──
        ├── detection.py            # automated detection + feature extraction
        ├── batch_processing.py     # Excel-driven batch over a dataset
        ├── consolidate.py          # per-image CSVs → feature-table workbook
        │
        │   # ── Interactive tools (threshold tuning / manual annotation) ──
        ├── plasma_membrane.py      # interactive ROI + dense-granule segmentation
        ├── open_canalicular.py     # interactive OCS detection
        ├── batch_annotate.py       # interactive batch driver
        ├── feature_analysis.py     # spatial stats + rule-based grading
        │
        └── deep_learning/          # Phase 2: placeholders (model shared via Drive)
            ├── __init__.py
            ├── segmentation.py     # DeepLabV3+ / Deformable ResNet-101 / VGG fusion
            └── classification.py   # ResNet-101 / MobileNet activation grader
```

---

## Installation

Requires **Python 3.8+**.

```bash
# 1. Clone
git clone https://github.com/venkateshs014/IITM_AIIMSD_Platelet.git
cd IITM_AIIMSD_Platelet

# 2. Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3a. Install dependencies only
pip install -r requirements.txt

# 3b. ...or install the package (provides the `platelet-em` command)
pip install -e .
```

> The **interactive** Phase-1 tools open OpenCV windows and use mouse/keyboard
> input, so they require a desktop (GUI) session. The **automated** pipeline
> (`detect` / `process` / `consolidate`) runs headless.

---

## Quick Start

Using the installed console script (`pip install -e .`):

```bash
# 1) Automated detection + feature extraction on a single image
platelet-em detect path/to/image.png --output-dir out/

# 2) Excel-driven batch processing of a whole dataset (case subfolders)
#    thresholds.xlsx columns: case_name, plasma_threshold, blue_threshold, white_threshold
platelet-em process thresholds.xlsx path/to/dataset/ output_batch/

# 3) Consolidate all per-image CSVs into one feature-table workbook
platelet-em consolidate output_batch/04_csv_measurements/ features.xlsx
```

Interactive tools (for threshold tuning / manual annotation):

```bash
platelet-em annotate path/to/image.png --output-dir out/   # interactive ROI + granules
platelet-em ocs out/..._final.png out/image_annotated.png  # interactive OCS
platelet-em analyze path/to/annotated.png --save fig.png   # rule-based grading
```

Equivalent invocations without installing (run from the `src/` directory):

```bash
cd src
python -m platelet_em.cli detect path/to/image.png --output-dir ../out/
```

Or import the API directly:

```python
from platelet_em.detection import detect_and_process_platelet
from platelet_em.consolidate import consolidate_measurements_main

results = detect_and_process_platelet("image.png", output_dir="out/")
consolidate_measurements_main("output_batch/04_csv_measurements/", "features.xlsx")
```

---

## Pretrained Model

The final trained model is large and is therefore hosted on Google Drive rather
than committed to the repository.

| Item | Details |
|------|---------|
| File | `model.pkl` (final trained model) |
| Download | **[⬇️ Download from Google Drive](https://drive.google.com/file/d/1fa4nY7B-01kkolyKh9Rmg-gDesg0CqC0/view?usp=sharing)** |
| Suggested location | place under `models/model.pkl` in the repo root |

Download from the command line (optional, requires `pip install gdown`):

```bash
mkdir -p models
gdown 1fa4nY7B-01kkolyKh9Rmg-gDesg0CqC0 -O models/model.pkl
```

Load and use the model:

```python
import pickle

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# `model` consumes the morphological feature table produced by Phase 1
# (see consolidate.py) and predicts the activation grade.
```

> If the file was saved with `joblib`, load it with `joblib.load("models/model.pkl")`
> instead. Keep large model files out of git (already covered by `.gitignore`).

---

## Phase 1 — Pre-annotation & Feature Extraction

A deterministic image-processing pipeline that automates the isolation of the
platelet plasma membrane and internal organelles, then quantifies morphology.

### Primary (automated) pipeline

**`detection.py` — `detect_and_process_platelet`.** Given an image and intensity
thresholds, it runs the entire analysis in one pass:

1. **Plasma membrane (green).** Inverse-binary thresholding (default
   `T_plasma = 135`) → area filtering → a composite *size + centrality* score
   selects the platelet contour within the field of view.
2. **Dense granules (blue).** A stringent dark threshold (default `70`) inside
   the platelet ROI.
3. **Open Canalicular System (red).** A high-intensity threshold (default `155`)
   inside the ROI, capturing electron-lucent vacuoles.
4. **Texture & morphology.** GLCM **homogeneity** (dense granules) and
   **contrast** (OCS); plasma-membrane **fractal dimension** (box-counting;
   `D_f > 1.2` indicates pseudopodial complexity).
5. **Spatial statistics.** Per-class pairwise-distance metrics and a clustering
   index (`std / mean`).

Per-image outputs: plasma outline, plasma mask, final overlay, **labeled
composite mask**, a measurements **CSV**, and a text summary.

**`batch_processing.py` — `process_batch_platelets`.** Applies the detector
across a dataset using **per-case thresholds** from an Excel workbook
(`case_name, plasma_threshold, blue_threshold, white_threshold`). The dataset is
organized as first-level **case subfolders** of images. Outputs are written into
organized folders:

```
output_batch/
├── 00_plasma_outlines/      ├── 03_labeled_composites/
├── 01_plasma_masks/         ├── 04_csv_measurements/
├── 02_final_overlays/       ├── 05_summary_reports/
└── BATCH_SUMMARY.txt
```

**`consolidate.py` — `consolidate_csv_to_excel`.** Merges every
`*_measurements.csv` into a single workbook with **one sheet per case**; each row
is one image with columns for platelet area, fractal dimension, dense-granule and
OCS counts/areas/clustering indices, GLCM homogeneity, and GLCM contrast. This is
the **feature table** consumed by the grading model.

### Interactive tools (complementary)

Useful for determining the per-case thresholds that feed the automated batch, and
for manual annotation:

- **`plasma_membrane.py`** — interactive dual-threshold ROI + dense-granule tool
  (polygon ROI selection, zoom/pan, live trackbars).
- **`open_canalicular.py`** — interactive OCS detection.
- **`batch_annotate.py`** — interactive batch driver.
- **`feature_analysis.py`** — DBSCAN/Clark-Evans spatial statistics with a
  transparent, rule-based grade (`UNACTIVATED` / `PARTIALLY_ACTIVATED` /
  `ACTIVATED`), a six-panel diagnostic figure, and a text report.

| Key (interactive tools) | Action |
|-------------------------|--------|
| Left-click | add ROI point |
| `t` / `n` | advance threshold phase / save & next |
| `+` / `-` | zoom in / out · right-drag to pan · `u` undo |
| `s` / `Esc` | save & process · cancel |

---

## Phase 2 — Deep-Learning Segmentation & Grading

> The **trained model** is available now via [Pretrained Model](#pretrained-model).
> The architecture below is documented in `src/platelet_em/deep_learning/`; the
> full training scripts will be added in a later release. Calling the placeholder
> builders raises `NotImplementedError`.

The learned stage extends **DeepLabV3+** with two task-specific modifications:

1. **Deformable ResNet-101 backbone.** The 3×3 convolutions in the `Conv2_x` and
   `Conv5_x` blocks are replaced with deformable convolutions so the receptive
   field adapts to the irregular, curved boundaries of activated platelets:

   ```
   y(p0) = Σ_{pn ∈ R}  w(pn) · x(p0 + pn + Δpn)
   ```

   where `Δpn` are learnable offsets predicted by a parallel conv layer.

2. **Hybrid feature-fusion decoder.** A semantic path (ResNet C5 → ASPP) is fused
   with a texture path (a frozen VGG16-BN encoder; `Conv2_2` features projected
   to 48 channels):

   ```
   F_fused = Concat( Upsample_4x(F_ASPP), F_VGG )
   ```

**Multi-task optimization:**

```
L_total = L_seg + λ · L_cls          (λ = 1.0)
L_seg   = BCE + Dice                  (handles platelet/background imbalance)
L_cls   = − α_t (1 − p_t)^γ log(p_t)  (Focal Loss, γ = 2, inverse-frequency α_t)
```

The classification head (Global Average Pooling → 2048 → 512 → 4) predicts the
activation grade (0–3) and fuses morphological descriptors (from Phase 1) with
CNN embeddings. Augmentation (Shift-Scale-Rotate, Random Gamma, Gaussian noise,
synthetic morphological variation) expands the limited EM dataset while
preserving ultrastructural realism.

**Reference training setup (from the paper):** PyTorch on an NVIDIA H100,
AdamW (`lr = 5e-5`, `weight_decay = 1e-4`), VGG encoder and early ResNet layers
frozen during initial epochs.

---

## Reference Notebook

`notebooks/Annotation_&_Feature_Extraction.ipynb` is the original, end-to-end
notebook for the full Phase-1 pipeline (detection → per-image CSV/summary →
Excel threshold handling → consolidated feature table). The `src/platelet_em/`
modules are the cleaned, importable refactor of this notebook; the notebook is
kept as the canonical reference and for exploratory runs.

---

## Results

Quantitative results on the curated EM dataset:

| Task | Metric | Value |
|------|--------|-------|
| Segmentation | mean Dice | _to be reported_ |
| Activation grading (4-class) | accuracy | _to be reported_ |

---

## Citation

If you use this code, please cite the accompanying paper (full reference to be
added upon publication):

```bibtex
@inproceedings{plateletem,
  title     = {Automated Ultrastructural Analysis of Platelet Activation via
               a Segmentation-Centric Deep Learning Framework},
  author    = {DMST, IIT Madras and AIIMS Delhi},
  booktitle = {Proceedings (to appear)},
  year      = {2026}
}
```

---

## Acknowledgements

This project is conducted in collaboration with **AIIMS Delhi** and the
**Department of Medical Sciences and Technology (DMST), IIT Madras**.

---

## License

Released under the **MIT License**. See `pyproject.toml` for the declared
license metadata; add a `LICENSE` file with the full text before public release.
