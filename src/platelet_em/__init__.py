"""platelet_em: Automated ultrastructural analysis of platelet activation in EM images.

A segmentation-centric framework for quantitative analysis of platelet
electron-microscopy (EM) images. The package is organized into two phases:

Phase 1 - Automated pre-annotation and quantitative feature extraction
    Primary (automated) pipeline:
    * :mod:`platelet_em.detection`        - automated, threshold-driven detection
      of plasma membrane / dense granules / OCS plus GLCM texture, fractal
      dimension, and clustering metrics (the main pipeline).
    * :mod:`platelet_em.batch_processing` - Excel-driven batch processing of a
      dataset of case folders.
    * :mod:`platelet_em.consolidate`      - merge per-image CSVs into a single
      feature-table workbook (one sheet per case).

    Interactive tools (complementary, for per-case threshold tuning):
    * :mod:`platelet_em.plasma_membrane`  - interactive plasma-membrane (ROI) +
      dense-granule segmentation.
    * :mod:`platelet_em.open_canalicular` - interactive Open Canalicular System
      (OCS) detection.
    * :mod:`platelet_em.batch_annotate`   - interactive batch driver.
    * :mod:`platelet_em.feature_analysis` - spatial-statistics descriptors and a
      transparent, rule-based activation grading baseline.

Phase 2 - Deep-learning segmentation and grading (released separately)
    * :mod:`platelet_em.deep_learning`    - placeholders for the DeepLabV3+-based
      segmentation network and the activation-grade classifier.

Submodules are imported lazily; import the specific module you need, e.g.::

    from platelet_em.feature_analysis import analyze_platelet_activation
"""

__version__ = "0.1.0"

__all__ = [
    "detection",
    "batch_processing",
    "consolidate",
    "plasma_membrane",
    "open_canalicular",
    "batch_annotate",
    "feature_analysis",
    "deep_learning",
]
