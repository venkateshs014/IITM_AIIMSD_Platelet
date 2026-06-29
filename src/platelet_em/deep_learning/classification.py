"""Phase-2 activation-grade classifier (placeholder).

Planned design (see the accompanying paper, Section II-B):

* A classification branch attached to the ResNet backbone (C5): Global Average
  Pooling followed by a dense stack (2048 -> 512 -> 4) predicting the
  morphological grade (0: resting -> 3: fully activated).
* Transfer learning on ResNet-101 / MobileNet feature extractors, fine-tuned on
  segmented, normalized crops.
* Morphological descriptors (area, convexity, pseudopod count, electron-dense
  core localization, and the spatial statistics from
  :mod:`platelet_em.feature_analysis`) are fused with the CNN embeddings.
* **Loss** - Focal Loss to focus training on hard-to-classify grades::

      L_cls = -alpha_t (1 - p_t)^gamma log(p_t)    (gamma = 2)

  with class weights ``alpha_t`` set by inverse frequency to counter imbalance.
* Augmentation (Shift-Scale-Rotate, Random Gamma, Gaussian noise, and synthetic
  morphological variation) expands the limited platelet EM dataset while
  preserving ultrastructural realism.

The full network is trained jointly with the segmentation head via
``L_total = L_seg + lambda * L_cls`` (``lambda = 1.0``).

This module is intentionally not implemented in this code release.
"""

GRADES = {
    0: "resting",
    1: "early activation",
    2: "intermediate activation",
    3: "fully activated",
}


def build_classifier(*args, **kwargs):
    """Construct the transfer-learning activation-grade classifier.

    Not implemented in this release; see the module docstring for the planned
    design.
    """
    raise NotImplementedError(
        "The Phase-2 deep-learning activation classifier is not part of this code "
        "release. See the module docstring and the paper for the planned design."
    )
