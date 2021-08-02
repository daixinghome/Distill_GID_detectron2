import logging
import copy
import torch
from torch import Tensor, nn
from typing import Dict, List

from detectron2.config import configurable
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNDistill(GeneralizedRCNN):

    @configurable
    def __init__(self, outputs=None, **kwarg):
        super().__init__(**kwarg)

        # outputs often in ["images", "stem", "res2", ..., "res5", "p2", "p3", ..., "p7",
        #                   "proposal", "cls_logits", "reg_deltas", "anchors",
        #                   "gt_labels", "gt_boxes", "losses", "results"]
        if outputs is None:
            self.outputs = ["losses", "results"]
        else:
            self.outputs = outputs


    @classmethod
    def from_config(cls, cfg):
        param_dict = super().from_config(cfg)
        param_dict["outputs"] = cfg.MODEL.OUTPUTS
        
        return param_dict


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        result_output = {}
        if not self.training or "results" in self.outputs:
            result_output["results"] = self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "images" in self.outputs:
            result_output["images"] = images  # Input ImageList

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"] for x in batched_inputs]
        else:
            gt_instances = None

        features_dict = self.backbone(images.tensor)
        # append ["stem", "res2", ..., "res5", "p2", "p3", ..., "p7"]
        for name, feat in features_dict.items():
            if name in self.outputs:
                result_output[name] = feat

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features_dict, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features_dict, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"] for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class RetinaNetDistill(RetinaNet):

    @configurable
    def __init__(self, outputs=None, **kwarg):
        super().__init__(**kwarg)

        # outputs often in ["images", "stem", "res2", ..., "res5", "p2", "p3", ..., "p7",
        #                   "proposal", "cls_logits", "reg_deltas", "anchors",
        #                   "gt_labels", "gt_boxes", "losses", "results"]
        if outputs is None:
            self.outputs = ["losses", "results"]
        else:
            self.outputs = outputs


    @classmethod
    def from_config(cls, cfg):
        param_dict = super().from_config(cfg)
        param_dict["outputs"] = cfg.MODEL.OUTPUTS
        
        return param_dict


    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        images = self.preprocess_image(batched_inputs)
        features_dict = self.backbone(images.tensor)
        features = [features_dict[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        # pred_logits: (N, (Ai x class_num), H, W)
        # pred_anchor_deltas: (N, (Ai x 4), H, W)
        pred_logits, pred_anchor_deltas = self.head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        result_output = {}
        if "images" in self.outputs:
            result_output["images"] = images  # Input ImageList

        if "anchors" in self.outputs:
            result_output["anchors"] = anchors  # (Hi*wi*A, B)

        # append ["stem", "res2", ..., "res5", "p2", "p3", ..., "p7"]
        for name, feat in features_dict.items():
            if name in self.outputs:
                result_output[name] = feat

        if "cls_logits" in self.outputs:
            result_output["cls_logits"] = pred_logits  # (N, Hi*Wi*A, K), K=class_num

        if "reg_deltas" in self.outputs:
            result_output["reg_deltas"] = pred_anchor_deltas  # (N, Hi*Wi*A, B)

        if "proposal" in self.outputs:
            # (N, Hi*Wi*A)
            result_output["proposal_logits"] = self.predict_objectness_logits(pred_logits)
            # (N, Hi*Wi*A, B), B=4
            result_output["proposal"] = self.predict_proposals(anchors, pred_anchor_deltas)

        if "losses" in self.outputs or "gt_labels" in self.outputs or "gt_boxes" in self.outputs \
            or self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"] for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            if "gt_labels" in self.outputs:
                result_output["gt_labels"] = gt_labels
            if "gt_boxes" in self.outputs:
                result_output["gt_boxes"] = gt_boxes

            if "losses" in self.outputs:
                losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
                result_output["losses"] = losses

        if "results" in self.outputs or not self.training:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                result_output["results"] = results
            else:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                result_output["results"] = processed_results

        return result_output


    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"] for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def predict_objectness_logits(self, pred_instance_logits):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, Hi*Wi*A, K) -> (N, Hi*Wi*A)
            score.max(dim=2)[0]
            for score in pred_instance_logits
        ]
        return pred_objectness_logits


    def predict_proposals(self, anchors, pred_anchor_deltas):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        batch_anchors = [copy.deepcopy(anchors) for _ in range(pred_anchor_deltas[0].shape[0])]
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        batch_anchors = list(zip(*batch_anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(batch_anchors, pred_anchor_deltas):
            N, _, B = pred_anchor_deltas_i.shape
            # Reshape: (N, Hi*Wi*A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


    def freeze(self):
        """
        Freeze all parameters. Including both items in conv and bn.
        Only work for BatchNorm2d method in different normalize methods.
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
