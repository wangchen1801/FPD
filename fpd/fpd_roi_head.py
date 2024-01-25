# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor

from mmfewshot.detection.models.utils.aggregation_layer import build_aggregator
from mmdet.models.builder import build_head
import torch.nn as nn


@HEADS.register_module()
class FPDRoIHead(StandardRoIHead):
    """Roi head for `FPD <https://arxiv.org/tbd>`.
    Args:
        aggregation_layer (ConfigDict): Config of `aggregation_layer`.
    """

    def __init__(self,
                 aggregation_layer: Optional[ConfigDict] = None,
                 prototypes_distillation: Optional[ConfigDict] = None,
                 prototypes_assignment: Optional[ConfigDict] = None,
                 num_novel=0,
                 novel_class=None,
                 rpn_head_: Optional[ConfigDict] = None,
                 meta_cls_ratio=1.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert prototypes_distillation is not None, "missing config of `prototypes_distillation`"
        self.prototypes_distillation = build_aggregator(copy.deepcopy(prototypes_distillation))
        assert prototypes_assignment is not None, "missing config of `prototypes_assignment`"
        self.prototypes_assignment = build_aggregator(copy.deepcopy(prototypes_assignment))
        self.num_novel = num_novel
        self.novel_class = novel_class
        self.meta_cls_ratio = meta_cls_ratio

        # RPN after FFA
        self.with_rpn = False
        if rpn_head_ is not None:
            self.with_rpn = True
            self.rpn_with_support = False
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head_ = rpn_head_

        # Non-Linear Fusion (NLF)
        d_model = 2048
        self.linear1 = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Linear(d_model * 2, d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(int(d_model * 2.5), d_model)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """Forward function for training.
        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        sampling_results = []
        if not self.with_rpn:
            if self.with_bbox:
                num_imgs = len(query_img_metas)
                if query_gt_bboxes_ignore is None:
                    query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for i in range(num_imgs):  # dense detector, bbox assign task
                    assign_result = self.bbox_assigner.assign(
                        proposals[i], query_gt_bboxes[i],
                        query_gt_bboxes_ignore[i], query_gt_labels[i])

                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposals[i],
                        query_gt_bboxes[i],
                        query_gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in
                               query_feats])
                    sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self._optimized_bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels, query_gt_bboxes_ignore)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def _optimized_bbox_forward_train(self, query_feats: List[Tensor],
                                      support_feats: List[Tensor],
                                      sampling_results: object,
                                      query_img_metas: List[Dict],
                                      query_gt_bboxes: List[Tensor],
                                      query_gt_labels: List[Tensor],
                                      support_gt_labels: List[Tensor],
                                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None, ) -> Dict:
        """Forward function and calculate loss for box head in training.
        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.
        Returns:
            dict: Predicted results and losses.
        """
        if not self.with_rpn:
            query_rois = bbox2roi(
                [res.bboxes for res in sampling_results])
            len_query_rois = [res.bboxes.size(0) for res in sampling_results]

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      query_gt_bboxes,
                                                      query_gt_labels,
                                                      self.train_cfg)
            (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        support_gt_labels_ = torch.cat(support_gt_labels)
        unique = set(support_gt_labels_.cpu().numpy())
        num_classes = len(unique)
        num_supp_shots = support_feats[0].size(0) // num_classes
        support_feat = self.extract_support_feats(support_feats)[0]

        # base & novel classes processed separately
        if self.num_novel != 0:
            nc = torch.tensor(self.novel_class, device='cuda')
            n_ids = torch.isin(support_gt_labels_, nc)
            b_ids = torch.logical_not(n_ids)
            b_gts = support_gt_labels_[b_ids]
            n_gts = support_gt_labels_[n_ids]
            # bc = torch.tensor(list(set(list(range(num_classes))) - set(nc)), device='cuda')  # nc is a tensor!
            bc = torch.tensor(list(set(list(range(num_classes))) - set(self.novel_class)), device='cuda')
            r_b_gts = torch.cat([torch.argwhere(bc == item)[0] for item in b_gts], dim=0)  # relative gt_labels
            r_n_gts = torch.cat([torch.argwhere(nc == item)[0] for item in n_gts], dim=0)  # relative gt_labels
            base_support_feats = support_feats[0][b_ids]
            novel_support_feats = support_feats[0][n_ids]

            # prototypes distillation
            weight_base, prototypes_base = self.prototypes_distillation(base_support_feats, support_gt_labels=r_b_gts)
            weight_novel, prototypes_novel = self.prototypes_distillation(novel_support_feats, forward_novel=True,
                                                                          support_gt_labels=r_n_gts)
            prototypes = torch.cat([prototypes_base, prototypes_novel], 0)
            weight = torch.cat([weight_base, weight_novel], 0)
        else:
            weight, prototypes = self.prototypes_distillation(support_feats[0], support_gt_labels=support_gt_labels_)

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        bbox_results = None

        # sampling positive & negative samples (B-CAS)
        supp_ids = []
        num_supp_per_im = num_supp_shots + 1
        for img_id in range(batch_size):
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            supp_id = []
            for i in range(support_feats[0].size(0)):
                if support_gt_labels[i] == random_query_label:
                    supp_id.append(i)
            while len(supp_id) < num_supp_per_im:
                supp_id.append(np.random.choice(range(support_feats[0].size(0))))
            supp_ids.append(supp_id)

        # prototypes assignment
        supp_order = []
        for k in range(num_supp_per_im):
            supp_order += [supp_ids[img_id][k] for img_id in range(batch_size)]
        fused_feats = self.prototypes_assignment(query_feats[0], prototypes)

        # ************ POST RPN *****************
        if self.with_rpn:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal', self.rpn_head_.test_cfg)
                if self.rpn_with_support:  # False
                    raise NotImplementedError
                else:
                    rpn_losses, proposal_list = self.rpn_head.forward_train(
                        [fused_feats],
                        copy.deepcopy(query_img_metas),
                        copy.deepcopy(query_gt_bboxes),
                        gt_labels=None,
                        gt_bboxes_ignore=query_gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg)
                proposals = proposal_list
                if self.with_bbox:
                    num_imgs = len(query_img_metas)
                    if query_gt_bboxes_ignore is None:
                        query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    for i in range(num_imgs):
                        assign_result = self.bbox_assigner.assign(
                            proposals[i], query_gt_bboxes[i],
                            query_gt_bboxes_ignore[i], query_gt_labels[i])
                        sampling_result = self.bbox_sampler.sample(
                            assign_result,
                            proposals[i],
                            query_gt_bboxes[i],
                            query_gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in
                                   query_feats])
                        sampling_results.append(sampling_result)
            query_rois = bbox2roi(
                [res.bboxes for res in sampling_results])
            len_query_rois = [res.bboxes.size(0) for res in sampling_results]
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      query_gt_bboxes,
                                                      query_gt_labels,
                                                      self.train_cfg)
            (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        # **************************************

        query_roi_feats = self.extract_query_roi_feat([fused_feats], query_rois)
        rpt_prototype = support_feat[supp_order]

        # roi_align, reslayer4, bbox_head, loss_func
        for k in range(num_supp_per_im):
            start, end = k * batch_size, (k + 1) * batch_size
            prototype = [item.unsqueeze(0).expand(len_query_roi, -1) for item, len_query_roi in
                         zip(rpt_prototype[start:end], len_query_rois)]
            prototype = torch.concat(prototype)  # (512, 2048)

            # Non-Linear Fusion (NLF)
            agg1 = self.linear1(query_roi_feats * prototype)
            agg2 = self.linear2(query_roi_feats - prototype)
            agg3 = self.linear4(torch.cat([query_roi_feats, prototype], dim=-1))
            agg = self.linear3(
                torch.cat([agg1, agg2, agg3, query_roi_feats], dim=-1)
            )
            bbox_results = self._bbox_forward_without_agg(agg)

            single_loss_bbox = self.bbox_head.loss(
                bbox_results['cls_score'], bbox_results['bbox_pred'],
                query_rois, labels,
                label_weights, bbox_targets,
                bbox_weights)
            for key in single_loss_bbox.keys():
                loss_bbox[key].append(single_loss_bbox[key])
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / (num_supp_per_im / 2)  # / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_labels = torch.cat(support_gt_labels)
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_meta_cls['loss_meta_cls'] = loss_meta_cls['loss_meta_cls'] * self.meta_cls_ratio
            loss_bbox.update(loss_meta_cls)

        bbox_results.update(loss_bbox=loss_bbox)
        if self.with_rpn:
            bbox_results['loss_bbox'].update(rpn_losses)
        return bbox_results

    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.
        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (bs*128, 5).
        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.
        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for lvl in range(len(feats)):
                out.append(self.shared_head.forward_support(feats[lvl]))
        else:
            out = feats
        return out

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]

        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1))
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _bbox_forward_without_agg(self, query_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.
        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
        Returns:
             dict: A dictionary of predicted results.
        """
        cls_score, bbox_pred = self.bbox_head(query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    new_support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.
        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            new_support_feats_dict: {'prototype': {cls_id: }, ..}.
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            new_support_feats_dict,
            query_img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            new_support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.
        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            new_support_feats_dict:
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        if not self.with_rpn:
            rois = bbox2roi(proposals)
            num_rois = rois.size(0)

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = len(support_feats_dict)
        support_feat = torch.cat([support_feats_dict[i] for i in range(num_classes)])
        prototypes = new_support_feats_dict['prototypes'][0]
        if self.num_novel != 0:
            prototypes_novel = new_support_feats_dict['prototypes_novel'][0]  # (5, 5, 1024)
            prototypes = torch.cat([prototypes, prototypes_novel], 0)

        # prototypes assignment
        fused_feats = self.prototypes_assignment(query_feats[0], prototypes, query_img_metas)

        # **** POST RPN ****
        if self.with_rpn:
            proposals = self.rpn_head.simple_test([fused_feats], query_img_metas)
            rois = bbox2roi(proposals)
            num_rois = rois.size(0)
        # *************

        query_roi_feats = self.extract_query_roi_feat([fused_feats], rois)
        query_roi_feats = query_roi_feats.repeat(support_feat.size(0), 1)
        rpt_support_feat = torch.cat([item.unsqueeze(0).expand(num_rois, -1) for item in support_feat])

        # Non-Linear Fusion (NLF)
        agg1 = self.linear1(query_roi_feats * rpt_support_feat)
        agg2 = self.linear2(query_roi_feats - rpt_support_feat)
        agg3 = self.linear4(torch.cat([query_roi_feats, rpt_support_feat], dim=-1))
        agg = self.linear3(
            torch.cat([agg1, agg2, agg3, query_roi_feats], dim=-1)
        )
        bbox_results = self._bbox_forward_without_agg(agg)

        for class_id in support_feats_dict.keys():
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][class_id * num_rois:(class_id + 1) * num_rois, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][class_id * num_rois:(class_id + 1) * num_rois,
                class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][class_id * num_rois:(class_id + 1) * num_rois, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][class_id * num_rois:(class_id + 1) * num_rois, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())  #

        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)  # tensor(141,21)
        bbox_pred = torch.cat(bbox_preds, dim=1)  # tensor(141,80)

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(
            len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
