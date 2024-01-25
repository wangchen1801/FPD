# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import mmcv
import numpy as np
from mmdet.datasets import PIPELINES
# from mmdet.datasets.pipelines import (Normalize, Pad, RandomCrop, RandomFlip,
#                                       Resize)



# new
@PIPELINES.register_module()
class CropResizeInstanceByRatio:
    """Crop and resize instance according to bbox form image.

    Args:
        num_context_pixels (int): Padding pixel around instance. Default: 16.
        target_size (tuple[int, int]): Resize cropped instance to target size.
            Default: (320, 320).
    """

    def __init__(
        self,
        num_context_pixels: int = 16,
        context_ratio: float = None,
        target_size: Tuple[int] = (320, 320)
    ) -> None:
        assert isinstance(num_context_pixels, int)
        assert len(target_size) == 2, 'target_size'
        self.num_context_pixels = num_context_pixels
        self.target_size = target_size

        self.context_ratio = context_ratio

    def __call__(self, results: Dict) -> Dict:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped and resized instance results.
        """
        img = results['img']
        gt_bbox = results['gt_bboxes']
        img_h, img_w = img.shape[:2]  # h, w
        x1, y1, x2, y2 = list(map(int, gt_bbox.tolist()[0]))

        # new
        if self.context_ratio is not None:
            gt_w, gt_h = x2 - x1, y2 - y1
            delta_w = max(0, self.target_size[1] - gt_w)
            delta_h = max(0, self.target_size[0] - gt_h)
            self.num_context_pixels = (gt_w + gt_h) * self.context_ratio + 0.04 * (delta_h + delta_w)
            self.num_context_pixels = int(self.num_context_pixels)

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        t_x1, t_y1, t_x2, t_y2 = 0, 0, bbox_w, bbox_h

        if bbox_w >= bbox_h:
            crop_x1 = x1 - self.num_context_pixels
            crop_x2 = x2 + self.num_context_pixels
            # t_x1 and t_x2 will change when crop context or overflow
            t_x1 = t_x1 + self.num_context_pixels
            t_x2 = t_x1 + bbox_w
            if crop_x1 < 0:
                t_x1 = t_x1 + crop_x1
                t_x2 = t_x1 + bbox_w
                crop_x1 = 0
            if crop_x2 > img_w:
                crop_x2 = img_w

            short_size = bbox_h
            long_size = crop_x2 - crop_x1
            y_center = int((y2 + y1) / 2)  # math.ceil((y2 + y1) / 2)
            crop_y1 = int(
                y_center -
                (long_size / 2))  # int(y_center - math.ceil(long_size / 2))
            crop_y2 = int(
                y_center +
                (long_size / 2))  # int(y_center + math.floor(long_size / 2))

            # t_y1 and t_y2 will change when crop context or overflow
            t_y1 = t_y1 + math.ceil((long_size - short_size) / 2)
            t_y2 = t_y1 + bbox_h

            if crop_y1 < 0:
                t_y1 = t_y1 + crop_y1
                t_y2 = t_y1 + bbox_h
                crop_y1 = 0
            if crop_y2 > img_h:
                crop_y2 = img_h

            crop_short_size = crop_y2 - crop_y1
            crop_long_size = crop_x2 - crop_x1

            square = np.zeros((crop_long_size, crop_long_size, 3),
                              dtype=np.uint8)
            delta = int(
                (crop_long_size - crop_short_size) /
                2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))
            square_y1 = delta
            square_y2 = delta + crop_short_size

            t_y1 = t_y1 + delta
            t_y2 = t_y2 + delta

            crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            square[square_y1:square_y2, :, :] = crop_box
        else:
            crop_y1 = y1 - self.num_context_pixels
            crop_y2 = y2 + self.num_context_pixels

            # t_y1 and t_y2 will change when crop context or overflow
            t_y1 = t_y1 + self.num_context_pixels
            t_y2 = t_y1 + bbox_h
            if crop_y1 < 0:
                t_y1 = t_y1 + crop_y1
                t_y2 = t_y1 + bbox_h
                crop_y1 = 0
            if crop_y2 > img_h:
                crop_y2 = img_h

            short_size = bbox_w
            long_size = crop_y2 - crop_y1
            x_center = int((x2 + x1) / 2)  # math.ceil((x2 + x1) / 2)
            crop_x1 = int(
                x_center -
                (long_size / 2))  # int(x_center - math.ceil(long_size / 2))
            crop_x2 = int(
                x_center +
                (long_size / 2))  # int(x_center + math.floor(long_size / 2))

            # t_x1 and t_x2 will change when crop context or overflow
            t_x1 = t_x1 + math.ceil((long_size - short_size) / 2)
            t_x2 = t_x1 + bbox_w
            if crop_x1 < 0:
                t_x1 = t_x1 + crop_x1
                t_x2 = t_x1 + bbox_w
                crop_x1 = 0
            if crop_x2 > img_w:
                crop_x2 = img_w

            crop_short_size = crop_x2 - crop_x1
            crop_long_size = crop_y2 - crop_y1
            square = np.zeros((crop_long_size, crop_long_size, 3),
                              dtype=np.uint8)
            delta = int(
                (crop_long_size - crop_short_size) /
                2)  # int(math.ceil((crop_long_size - crop_short_size) / 2))
            square_x1 = delta
            square_x2 = delta + crop_short_size

            t_x1 = t_x1 + delta
            t_x2 = t_x2 + delta
            crop_box = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            square[:, square_x1:square_x2, :] = crop_box        # 这才是padding的正确方法

        square = square.astype(np.float32, copy=False)
        square, square_scale = mmcv.imrescale(
            square, self.target_size, return_scale=True, backend='cv2')

        square = square.astype(np.uint8)

        t_x1 = int(t_x1 * square_scale)
        t_y1 = int(t_y1 * square_scale)
        t_x2 = int(t_x2 * square_scale)
        t_y2 = int(t_y2 * square_scale)
        results['img'] = square
        results['img_shape'] = square.shape
        results['gt_bboxes'] = np.array([[t_x1, t_y1, t_x2,
                                          t_y2]]).astype(np.float32)

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(num_context_pixels={self.num_context_pixels},' \
               f' target_size={self.target_size})'
