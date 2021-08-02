from .config import add_distill_cfg

# # import all the meta_arch, so they will be registered
from .fpn_distill import (
    build_resnet_fpn_backbone_distil,
    build_retinanet_resnet_fpn_backbone_distill,
)
from .retinanet_distill import RetinaNetDistill
from .distill_gid import DistillGID

__all__ = list(globals().keys())