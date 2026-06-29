"""Phase-2 segmentation network (placeholder).

Planned architecture (see the accompanying paper, Section II-B):

* **Backbone** - Deformable ResNet-101. The standard 3x3 convolutions in the
  ``Conv2_x`` and ``Conv5_x`` blocks are replaced with deformable convolutions
  so the receptive field can adapt to the irregular, highly curved boundaries of
  activated platelets::

      y(p0) = sum_{pn in R} w(pn) . x(p0 + pn + dpn)

  where ``dpn`` are learnable offsets predicted by a parallel conv layer.

* **Hybrid feature-fusion decoder** - fuses two encoders:
    - Semantic path: high-level ResNet features (C5) processed by Atrous Spatial
      Pyramid Pooling (ASPP) for multi-scale context.
    - Texture path: a frozen VGG16-BN encoder; low-level ``Conv2_2`` features are
      projected to 48 channels via a 1x1 conv.
  Fusion concatenates the 4x-upsampled ASPP features with the VGG features::

      F_fused = Concat(Upsample_4x(F_ASPP), F_VGG)

* **Loss** - composite BCE + Dice to handle the platelet/background imbalance.

This module is intentionally not implemented in this code release.
"""


def build_segmentation_model(*args, **kwargs):
    """Construct the Deformable-ResNet101 / VGG-fusion DeepLabV3+ model.

    Not implemented in this release; see the module docstring for the planned
    architecture.
    """
    raise NotImplementedError(
        "The Phase-2 deep-learning segmentation model is not part of this code "
        "release. See the module docstring and the paper for the planned design."
    )
