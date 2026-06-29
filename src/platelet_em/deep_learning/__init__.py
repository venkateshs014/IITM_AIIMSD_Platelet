"""Phase 2: Deep-learning segmentation and activation grading.

This subpackage hosts the learned components of the framework described in the
accompanying paper:

* :mod:`platelet_em.deep_learning.segmentation` - a DeepLabV3+ encoder-decoder
  with a Deformable ResNet-101 backbone and a hybrid (ResNet + frozen VGG16)
  fusion decoder for pixel-wise platelet segmentation.
* :mod:`platelet_em.deep_learning.classification` - a transfer-learning
  classifier (ResNet-101 / MobileNet) that grades activation state (Grades 0-3)
  by fusing morphological descriptors with CNN embeddings.

.. warning::
   These modules are placeholders. The trained models and training scripts are
   being prepared for release and are **not** part of this code drop. Calling
   the placeholder entry points raises :class:`NotImplementedError`.
"""

__all__ = ["segmentation", "classification"]
