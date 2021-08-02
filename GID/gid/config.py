from detectron2.config import CfgNode, get_cfg


def add_distill_cfg(cfg: CfgNode) -> CfgNode:
    cfg.DISTILL = CfgNode()

    cfg.DISTILL.TEACHER_YAML = ""
    cfg.DISTILL.TEACHER_CFG = get_cfg()
    cfg.DISTILL.TEACHER_CFG.MODEL.OUTPUTS = []

    cfg.DISTILL.STUDENT_YAML = ""
    cfg.DISTILL.STUDENT_CFG = get_cfg()
    cfg.DISTILL.STUDENT_CFG.MODEL.OUTPUTS = []

    cfg.DISTILL.GID = CfgNode()

    cfg.DISTILL.GID.GI_MODE = "ALL"              # in ["all", "pos", "semi_pos", "neg"]
    cfg.DISTILL.GID.GI_NMS_THRESH = 0.3
    cfg.DISTILL.GID.GI_PRE_NMS_TOPK = 3000
    cfg.DISTILL.GID.GI_POST_NMS_TOPK = 10

    cfg.DISTILL.GID.IOU_THRESHOLDS = [0.3, 0.7]  # For Response Distillation mask
    cfg.DISTILL.GID.IOU_LABELS = [0, -1, 1]      # For Response Distillation mask

    cfg.DISTILL.GID.ROI = CfgNode()
    cfg.DISTILL.GID.ROI.POOLER_RESOLUTION = 10
    cfg.DISTILL.GID.ROI.POOLER_SAMPLING_RATIO = 2
    cfg.DISTILL.GID.ROI.POOLER_TYPE = "ROIAlignV2"

    cfg.DISTILL.GID.ADAPTER = CfgNode()
    cfg.DISTILL.GID.ADAPTER.TE_NAMES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.DISTILL.GID.ADAPTER.ST_NAMES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.DISTILL.GID.ADAPTER.SHARE_WEIGHTS = True
    cfg.DISTILL.GID.ADAPTER.KERNEL_SIZE = 3

    cfg.DISTILL.LOSS = CfgNode()
    cfg.DISTILL.LOSS.GT_WEIGHT = 1.0

    cfg.DISTILL.LOSS.FEATURE = CfgNode()
    cfg.DISTILL.LOSS.FEATURE.TE_NAMES = ["p3", "p4", "p5", "p6", "p7"]  # in ["stem", "res2", ..., "res5", "p3", ..., "p7"]
    cfg.DISTILL.LOSS.FEATURE.ST_NAMES = ["p3", "p4", "p5", "p6", "p7"]  # in ["stem", "res2", ..., "res5", "p3", ..., "p7"]
    cfg.DISTILL.LOSS.FEATURE.MODE = "L2"                                # in ["L1", "L2"]
    cfg.DISTILL.LOSS.FEATURE.SMOOTH_L1_BETA = 0.0                       # only valid when MODE is "L1"
    cfg.DISTILL.LOSS.FEATURE.WEIGHT = 0.005

    cfg.DISTILL.LOSS.RELATION = CfgNode()
    cfg.DISTILL.LOSS.RELATION.TE_NAMES = ["p3", "p4", "p5", "p6", "p7"]  # in ["stem", "res2", ..., "res5", "p3", ..., "p7"]
    cfg.DISTILL.LOSS.RELATION.ST_NAMES = ["p3", "p4", "p5", "p6", "p7"]  # in ["stem", "res2", ..., "res5", "p3", ..., "p7"]
    cfg.DISTILL.LOSS.RELATION.MODE = "IRKD"                              # in ["IRKD"]
    cfg.DISTILL.LOSS.RELATION.DIST_MODE = "L2"                           # in ["L1", "L2"]
    cfg.DISTILL.LOSS.RELATION.SMOOTH_L1_BETA = 1.0                       # L1 loss param of relation distance, not depend on DIST_MODE
    cfg.DISTILL.LOSS.RELATION.WEIGHT = 40

    cfg.DISTILL.LOSS.RESPONSE_CLS = CfgNode()
    cfg.DISTILL.LOSS.RESPONSE_CLS.TE_NAMES = ["cls_logits"]              # in ["cls_logits"]
    cfg.DISTILL.LOSS.RESPONSE_CLS.ST_NAMES = ["cls_logits"]              # in ["cls_logits"]
    cfg.DISTILL.LOSS.RESPONSE_CLS.MODE = "BCE"                           # in ["BCE", "L1", "L2"]
    cfg.DISTILL.LOSS.RESPONSE_CLS.SMOOTH_L1_BETA = 0.0                   # only valid when MODE is "L1"
    cfg.DISTILL.LOSS.RESPONSE_CLS.WEIGHT = 0.1

    cfg.DISTILL.LOSS.RESPONSE_REG = CfgNode()
    cfg.DISTILL.LOSS.RESPONSE_REG.TE_NAMES = ["reg_deltas"]              # in ["reg_deltas"]
    cfg.DISTILL.LOSS.RESPONSE_REG.ST_NAMES = ["reg_deltas"]              # in ["reg_deltas"]
    cfg.DISTILL.LOSS.RESPONSE_REG.MODE = "L1"                            # in ["bce", "L1", "L2"]
    cfg.DISTILL.LOSS.RESPONSE_REG.SMOOTH_L1_BETA = 0.0                   # only valid when MODE is "L1"
    cfg.DISTILL.LOSS.RESPONSE_REG.WEIGHT = 0.1

    return cfg
