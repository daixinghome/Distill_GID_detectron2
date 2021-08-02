import logging
import torch
from torch import Tensor, nn
from typing import Dict, Tuple, List

from detectron2.config import configurable
from detectron2.layers import cat
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from detectron2.structures import Boxes, Instances, pairwise_ioa, pairwise_iou
from detectron2.utils.events import get_event_storage

from .proposal_utils import find_top_rpn_proposals
from .loss import distill_loss_feature, loss_RkdDistance, distill_loss_relation, distill_loss_response

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class DistillGID(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        teacher_net: nn.Module,
        student_net: nn.Module,
        distill_gi_mode,
        distill_gi_nms_thresh,
        distill_gi_pre_nms_topk,
        distill_gi_post_nms_topk,
        distill_gi_matcher,
        distill_gi_pooler_params,
        distill_adapter_feature_names_te,
        distill_adapter_feature_names_st,
        distill_adapter_feature_share,
        distill_adapter_feature_ksize,
        distill_loss_gt_weight,
        distill_loss_feature_param,
        distill_loss_relation_param,
        distill_loss_response_cls_param,
        distill_loss_response_reg_param,
    ):
        # ======== parameter init ======== #
        super().__init__()

        self.teacher = teacher_net
        self.student = student_net

        assert self.teacher.num_classes == self.student.num_classes
        self.num_classes = self.student.num_classes
        te_hint_outshape = self.teacher.backbone.output_shape()
        st_hint_outshape = self.student.backbone.output_shape()
        assert set(distill_adapter_feature_names_te).issubset(te_hint_outshape.keys()), \
            "{}, {}".format(distill_adapter_feature_names_te, list(te_hint_outshape.keys()))
        assert set(distill_adapter_feature_names_st).issubset(st_hint_outshape.keys()), \
            "{}, {}".format(distill_adapter_feature_names_st, list(st_hint_outshape.keys()))

        # ======== state init ======== #
        self.teacher.freeze()
        self.teacher.eval()

        # ======== distill init ======== #
        # distill loss target-weight dict
        self.distill_loss_gt_weight = distill_loss_gt_weight

        # Dict[feature_names: List[str], loss_mode: str, loss_weight: float]
        self.distill_feature = distill_loss_feature_param
        self.distill_relation = distill_loss_relation_param
        self.distill_response_cls = distill_loss_response_cls_param
        self.distill_response_reg = distill_loss_response_reg_param

        # distill operator init
        self.distill_gi_mode = distill_gi_mode
        self.distill_gi_nms_thresh = distill_gi_nms_thresh
        self.distill_gi_pre_nms_topk = distill_gi_pre_nms_topk
        self.distill_gi_post_nms_topk = distill_gi_post_nms_topk
        self.distill_gi_matcher = distill_gi_matcher

        # pooler is used by both teacher and student network
        gi_pooler_te_strides = tuple(1.0 / te_hint_outshape[k].stride
                                     for k in self.distill_feature["te_feature_names"])
        gi_pooler_st_strides = tuple(1.0 / st_hint_outshape[k].stride
                                     for k in self.distill_feature["st_feature_names"])
        # strides should be the same
        assert gi_pooler_te_strides == gi_pooler_st_strides, \
            "T:{}, S:{}".format(gi_pooler_te_strides, gi_pooler_st_strides)
        self.distill_gi_pooler_feature = ROIPooler(
            output_size=distill_gi_pooler_params["output_size"],
            scales=gi_pooler_st_strides,
            sampling_ratio=distill_gi_pooler_params["sampling_ratio"],
            pooler_type=distill_gi_pooler_params["pooler_type"],
        )
        self.distill_gi_pooler_relation = ROIPooler(
            output_size=distill_gi_pooler_params["output_size"],
            scales=gi_pooler_st_strides,
            sampling_ratio=distill_gi_pooler_params["sampling_ratio"],
            pooler_type=distill_gi_pooler_params["pooler_type"],
        )

        # distill layer init
        # Notice that adapter here is only for student network
        self.distill_adapter_feature_names_te = distill_adapter_feature_names_te
        self.distill_adapter_feature_names_st = distill_adapter_feature_names_st
        self.distill_adapter_feature_share = distill_adapter_feature_share
        self.distill_adapter_feature_ksize = distill_adapter_feature_ksize
        self.distill_adapt_layers = []
        last_te_channels = None
        last_st_channels = None
        for te_layer_name, st_layer_name in zip(
            self.distill_adapter_feature_names_te,
            self.distill_adapter_feature_names_st
        ):
            assert te_layer_name in self.teacher.outputs and st_layer_name in self.student.outputs
            te_channels = te_hint_outshape[te_layer_name].channels
            st_channels = st_hint_outshape[st_layer_name].channels

            if self.distill_adapter_feature_share and \
                len(self.distill_adapt_layers) > 0:
                assert last_te_channels == te_channels and last_st_channels == st_channels
                self.distill_adapt_layers.append(self.distill_adapt_layers[-1])
            else:
                adapt_layer = nn.Sequential(
                    nn.Conv2d(
                        st_channels,
                        te_channels,
                        kernel_size=self.distill_adapter_feature_ksize,
                        stride=1,
                        padding=1
                    ),
                )
                self.distill_adapt_layers.append(adapt_layer)

            last_te_channels = te_channels
            last_st_channels = st_channels


    @classmethod
    def from_config(cls, cfg):
        teacher_net = build_model(cfg.DISTILL.TEACHER_CFG)
        student_net = build_model(cfg.DISTILL.STUDENT_CFG)

        distill_gi_nms_thresh = cfg.DISTILL.GID.GI_NMS_THRESH
        distill_gi_pre_nms_topk = cfg.DISTILL.GID.GI_PRE_NMS_TOPK
        distill_gi_post_nms_topk = cfg.DISTILL.GID.GI_POST_NMS_TOPK

        distill_gi_mode = cfg.DISTILL.GID.GI_MODE
        distill_gi_matcher = Matcher(
            cfg.DISTILL.GID.IOU_THRESHOLDS,
            cfg.DISTILL.GID.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        distill_gi_pooler_params = {
            "output_size": (cfg.DISTILL.GID.ROI.POOLER_RESOLUTION,
                            cfg.DISTILL.GID.ROI.POOLER_RESOLUTION),
            "sampling_ratio": cfg.DISTILL.GID.ROI.POOLER_SAMPLING_RATIO,
            "pooler_type": cfg.DISTILL.GID.ROI.POOLER_TYPE,
        }

        distill_adapter_feature_names_te = cfg.DISTILL.GID.ADAPTER.TE_NAMES
        distill_adapter_feature_names_st = cfg.DISTILL.GID.ADAPTER.ST_NAMES
        distill_adapter_feature_share = cfg.DISTILL.GID.ADAPTER.SHARE_WEIGHTS
        distill_adapter_feature_ksize = cfg.DISTILL.GID.ADAPTER.KERNEL_SIZE

        # distill method and corresponding weight
        distill_loss_gt_weight = cfg.DISTILL.LOSS.GT_WEIGHT
        distill_loss_feature_param = {
            "te_feature_names": cfg.DISTILL.LOSS.FEATURE.TE_NAMES,
            "st_feature_names": cfg.DISTILL.LOSS.FEATURE.ST_NAMES,
            "loss_mode": cfg.DISTILL.LOSS.FEATURE.MODE,
            "loss_weight": cfg.DISTILL.LOSS.FEATURE.WEIGHT,
            "smooth_l1_beta": cfg.DISTILL.LOSS.FEATURE.SMOOTH_L1_BETA,
        }
        distill_loss_relation_param = {
            "te_feature_names": cfg.DISTILL.LOSS.RELATION.TE_NAMES,
            "st_feature_names": cfg.DISTILL.LOSS.RELATION.ST_NAMES,
            "loss_mode": cfg.DISTILL.LOSS.RELATION.MODE,
            "loss_weight": cfg.DISTILL.LOSS.RELATION.WEIGHT,
            "dist_mode": cfg.DISTILL.LOSS.RELATION.DIST_MODE,
            "smooth_l1_beta": cfg.DISTILL.LOSS.RELATION.SMOOTH_L1_BETA,
        }
        distill_loss_response_cls_param = {
            "te_feature_names": cfg.DISTILL.LOSS.RESPONSE_CLS.TE_NAMES,
            "st_feature_names": cfg.DISTILL.LOSS.RESPONSE_CLS.ST_NAMES,
            "loss_mode": cfg.DISTILL.LOSS.RESPONSE_CLS.MODE,
            "loss_weight": cfg.DISTILL.LOSS.RESPONSE_CLS.WEIGHT,
            "smooth_l1_beta": cfg.DISTILL.LOSS.RESPONSE_CLS.SMOOTH_L1_BETA,
        }
        distill_loss_response_reg_param = {
            "te_feature_names": cfg.DISTILL.LOSS.RESPONSE_REG.TE_NAMES,
            "st_feature_names": cfg.DISTILL.LOSS.RESPONSE_REG.ST_NAMES,
            "loss_mode": cfg.DISTILL.LOSS.RESPONSE_REG.MODE,
            "loss_weight": cfg.DISTILL.LOSS.RESPONSE_REG.WEIGHT,
            "smooth_l1_beta": cfg.DISTILL.LOSS.RESPONSE_REG.SMOOTH_L1_BETA,
        }

        cfg_dict = {
            "teacher_net": teacher_net,
            "student_net": student_net,
            "distill_gi_mode": distill_gi_mode,
            "distill_gi_nms_thresh": distill_gi_nms_thresh,
            "distill_gi_pre_nms_topk": distill_gi_pre_nms_topk,
            "distill_gi_post_nms_topk": distill_gi_post_nms_topk,
            "distill_gi_matcher": distill_gi_matcher,
            "distill_gi_pooler_params": distill_gi_pooler_params,
            "distill_adapter_feature_names_te": distill_adapter_feature_names_te,
            "distill_adapter_feature_names_st": distill_adapter_feature_names_st,
            "distill_adapter_feature_share": distill_adapter_feature_share,
            "distill_adapter_feature_ksize": distill_adapter_feature_ksize,
            "distill_loss_gt_weight": distill_loss_gt_weight,
            "distill_loss_feature_param": distill_loss_feature_param,
            "distill_loss_relation_param": distill_loss_relation_param,
            "distill_loss_response_cls_param": distill_loss_response_cls_param,
            "distill_loss_response_reg_param": distill_loss_response_reg_param,
        }
        return cfg_dict


    @property
    def device(self):
        return self.student.pixel_mean.device


    def data_to_device(self, batched_inputs: List[Dict[str, Tensor]], is_train=True):
        for x_i, x in enumerate(batched_inputs):
            batched_inputs[x_i]["image"] = x["image"].to(self.device)
            if is_train:
                batched_inputs[x_i]["instances"] = x["instances"].to(self.device)
        return batched_inputs


    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        batched_inputs = self.data_to_device(batched_inputs, self.training)
        if self.training:
            total_losses = {}

            ########## Sub module forward ##########
            # Teacher forward part
            with torch.no_grad():
                te_outputs = self.teacher(batched_inputs)

            # Student forward part
            st_outputs = self.student(batched_inputs)
            st_losses = st_outputs.get("losses")
            if st_losses is None:
                logger.warning("[Distill Net] Student model has no gt losses.")
            else:
                total_losses.update(st_losses)

            ########## Merge GT Loss and Distill Loss ##########
            # GT loss with weight
            for gt_loss_name, gt_loss in total_losses.items():
                total_losses[gt_loss_name] = gt_loss * self.distill_loss_gt_weight

            # Distillation loss with weight
            d_losses = self.distill_losses(batched_inputs, te_outputs, st_outputs)
            total_losses.update(d_losses)

            return total_losses
        else:
            return self.student(batched_inputs).get("results")


    def gism_method(
        self,
        teacher_proposal,
        teacher_proposal_logits,
        student_proposal,
        student_proposal_logits,
        gt_labels,
        image_sizes,
        *,
        nms_thresh: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        min_box_side_len: int,
    ):
        cat_te_objectness = cat(teacher_proposal_logits, dim=1)
        cat_te_proposal = cat(teacher_proposal, dim=1)
        cat_st_objectness = cat(student_proposal_logits, dim=1)
        cat_st_proposal = cat(student_proposal, dim=1)
        gt_labels = torch.stack(gt_labels)

        sub_score = torch.sigmoid(cat_te_objectness) - torch.sigmoid(cat_st_objectness)
        gi_score = torch.abs(sub_score)
        gi_box = cat_st_proposal
        gi_box[sub_score > 0, :] = cat_te_proposal[sub_score > 0, :]

        proposals_gi = find_top_rpn_proposals(
            [gi_box],
            [gi_score],
            [gt_labels],
            image_sizes,
            nms_thresh,
            pre_nms_topk,
            post_nms_topk,
            min_box_side_len,
            training=True,
        )
        return proposals_gi


    def select_gi(
        self,
        proposals_gi,
        gt_instances,
        gi_mode = "ALL",
    ):
        gi_mode = gi_mode.lower()
        assert gi_mode in ["all", "pos", "semi_pos", "neg"], gi_mode

        gi_list = []
        storage = get_event_storage()
        def mask_gi_instance(gi_instances, mask):
            gi_instances.proposal_boxes = gi_instances.proposal_boxes[mask]
            gi_instances.objectness_logits = gi_instances.objectness_logits[mask]
            gi_instances.gt_class = gi_instances.gt_class[mask]
            gi_instances.gi_labels = gi_instances.gi_labels[mask]
            gi_instances.IOP = gi_instances.IOP[mask]
            return gi_instances

        for proposals_per_image, gt_per_image in zip(proposals_gi, gt_instances):
            if len(gt_per_image) > 0:
                # intersection over proposal boxes
                match_iop_matrix = pairwise_ioa(gt_per_image.gt_boxes, proposals_per_image.proposal_boxes)

                gt_matched_idxs, gi_labels = self.distill_gi_matcher(match_iop_matrix)
                gi_new = Instances(proposals_per_image.image_size)
                gi_new.proposal_boxes = proposals_per_image.proposal_boxes
                gi_new.objectness_logits = proposals_per_image.objectness_logits
                gi_new.gt_class = proposals_per_image.gt_class
                gi_new.gi_labels = gi_labels

                if match_iop_matrix.shape[1] > 0:
                    max_IOP_foreach_proposal, _ = match_iop_matrix.max(dim=0)
                else:  # No proposal_boxes
                    max_IOP_foreach_proposal = torch.zeros_like(gi_new.gt_class, dtype=torch.float)
                gi_new.IOP = max_IOP_foreach_proposal
            else:
                gi_new = Instances(proposals_per_image.image_size)
                gi_new.proposal_boxes = proposals_per_image.proposal_boxes
                gi_new.objectness_logits = proposals_per_image.objectness_logits
                gi_new.gt_class = proposals_per_image.gt_class
                gi_new.gi_labels = torch.zeros_like(gi_new.gt_class)
                gi_new.IOP = torch.zeros_like(gi_new.gt_class)

            gi_pos_mask = ((gi_new.gt_class != self.num_classes) & \
                           (gi_new.gt_class >= 0))
            gi_semi_mask = (((gi_new.gt_class == self.num_classes) | (gi_new.gt_class == -1)) & \
                            (gi_new.gi_labels == 1))
            gi_neg_mask = (((gi_new.gt_class == self.num_classes) | (gi_new.gt_class == -1)) & \
                           (gi_new.gi_labels == 0))
            gi_neg_iop_mean = gi_new.IOP[gi_neg_mask].mean()

            if gi_mode == "pos":
                gi_new = mask_gi_instance(gi_new, gi_pos_mask)
            elif gi_mode == "semi_pos":
                gi_new = mask_gi_instance(gi_new, gi_semi_mask)
            elif gi_mode == "gi_neg_mask":
                gi_new = mask_gi_instance(gi_new, gi_neg_mask)
            else:  # "all"
                pass
            gi_list.append(gi_new)

            storage.put_scalar("Neg_IOP", gi_neg_iop_mean)
            storage.put_scalar("Num_pos", gi_pos_mask.sum())
            storage.put_scalar("Num_semi_pos", gi_semi_mask.sum())
            storage.put_scalar("Num_neg", gi_neg_mask.sum())
            
        return gi_list


    def get_GI_instance_labels(self, anchors, proposals, gt_matcher):
        distill_mask = []
        anchors = Boxes.cat(anchors)
        for proposals_per_image in proposals:
            instance_boxes = proposals_per_image.proposal_boxes
            match_quality_matrix = pairwise_iou(instance_boxes, anchors)
            gt_matched_idxs, anchor_labels = gt_matcher(match_quality_matrix)
            anchor_labels[anchor_labels == -1] = 0
            distill_mask.append(anchor_labels)
        return torch.stack(distill_mask)


    def get_outputs_pooler_features(self, outputs, feature_names, box_pooler, proposals):
        feat_list = []
        for name_feat in feature_names:
            feat_list.append(outputs[name_feat])
        feats = box_pooler(
            feat_list,
            [x.proposal_boxes for x in proposals]
        )
        return feats


    @classmethod
    def cat_response(cls, response: List[Tensor]) -> Tensor:
        """
        Rearrange the tensor layout from the network output, i.e.:
        list[Tensor]: #lvl tensors of shape (N, Hi*Wi*A, K)
        to per-image predictions, i.e.:
        Tensor: of shape (N x sum(Hi x Wi x A), K)
        """
        response = cat(response, dim=1)
        response = response.view(-1, response.shape[2])
        return response
        

    def distill_losses(self, batched_inputs, teacher_outputs, student_outputs):
        image_sizes = student_outputs["images"].image_sizes
        gt_labels = student_outputs["gt_labels"]  # should be the same with teacher in GID method
        gt_instances = [x["instances"] for x in batched_inputs]

        teacher_outputs["cls_logits"] = self.cat_response(teacher_outputs["cls_logits"])
        teacher_outputs["reg_deltas"] = self.cat_response(teacher_outputs["reg_deltas"])
        student_outputs["cls_logits"] = self.cat_response(student_outputs["cls_logits"])
        student_outputs["reg_deltas"] = self.cat_response(student_outputs["reg_deltas"])

        te_proposal = teacher_outputs["proposal"]
        te_proposal_logits = teacher_outputs["proposal_logits"]
        st_proposal = student_outputs["proposal"]
        st_proposal_logits = student_outputs["proposal_logits"]

        st_anchors = student_outputs["anchors"]

        # ========== Get General Instance (GI) ========== #
        proposals_gi = self.gism_method(
            te_proposal,
            te_proposal_logits,
            st_proposal,
            st_proposal_logits,
            gt_labels,
            image_sizes,
            nms_thresh = self.distill_gi_nms_thresh,
            pre_nms_topk = self.distill_gi_pre_nms_topk,
            post_nms_topk = self.distill_gi_post_nms_topk,
            min_box_side_len = 10,
        )

        proposals_gi = self.select_gi(
            proposals_gi,
            gt_instances,
            self.distill_gi_mode,
        )

        if len(self.distill_adapt_layers) == 0:
            for layer_name, adapt_layer in zip(self.distill_adapter_feature_names_st,
                                               self.distill_adapt_layers):
                student_outputs["layer_name"] = adapt_layer(student_outputs["layer_name"])

        total_losses = {}
        # distill feature loss
        te_feature_features = self.get_outputs_pooler_features(
            teacher_outputs,
            self.distill_feature["te_feature_names"],
            self.distill_gi_pooler_feature,
            proposals_gi
        )
        st_feature_features = self.get_outputs_pooler_features(
            student_outputs,
            self.distill_feature["st_feature_names"],
            self.distill_gi_pooler_feature,
            proposals_gi
        )
        loss_feature = distill_loss_feature(
            te_feature_features,
            st_feature_features,
            loss_mode=self.distill_feature["loss_mode"],
            smooth_l1_beta=self.distill_feature["smooth_l1_beta"],
        ) * self.distill_feature["loss_weight"]
        total_losses["loss_distill_feature"] = loss_feature

        # distill relation loss
        te_relation_features = self.get_outputs_pooler_features(
            teacher_outputs,
            self.distill_relation["te_feature_names"],
            self.distill_gi_pooler_relation,
            proposals_gi
        )
        st_relation_features = self.get_outputs_pooler_features(
            student_outputs,
            self.distill_relation["st_feature_names"],
            self.distill_gi_pooler_relation,
            proposals_gi
        )
        loss_relation = distill_loss_relation(
            te_relation_features,
            st_relation_features,
            loss_mode = self.distill_relation["loss_mode"],
            dist_mode = self.distill_relation["dist_mode"],
            smooth_l1_beta = self.distill_relation["smooth_l1_beta"],
        ) * self.distill_relation["loss_weight"]
        total_losses["loss_distill_relation"] = loss_relation

        # distill response loss
        response_gi_labels = self.get_GI_instance_labels(
            st_anchors,
            proposals_gi,
            self.student.anchor_matcher,
        )
        response_gi_mask = (response_gi_labels.flatten() > 0)

        loss_response_cls = {}
        for te_cls_name, st_cls_name in zip(
            self.distill_response_cls["te_feature_names"],
            self.distill_response_cls["st_feature_names"]
        ):
            loss_response_cls["loss_distill_"+st_cls_name] = distill_loss_response(
                teacher_outputs[te_cls_name],
                student_outputs[st_cls_name],
                response_gi_mask,
                loss_mode = self.distill_response_cls["loss_mode"],
                smooth_l1_beta = self.distill_response_cls["smooth_l1_beta"],
            ) * self.distill_response_cls["loss_weight"]
        total_losses.update(loss_response_cls)

        loss_response_reg = {}
        for te_reg_name, st_reg_name in zip(
            self.distill_response_reg["te_feature_names"],
            self.distill_response_reg["st_feature_names"],
        ):
            loss_response_reg["loss_distill_"+st_reg_name] = distill_loss_response(
                teacher_outputs[te_reg_name],
                student_outputs[st_reg_name],
                response_gi_mask,
                loss_mode = self.distill_response_reg["loss_mode"],
                smooth_l1_beta = self.distill_response_reg["smooth_l1_beta"],
            ) * self.distill_response_reg["loss_weight"]
        total_losses.update(loss_response_reg)

        return total_losses
