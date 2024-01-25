_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../fpd_r101_c4.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=30,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='30SHOT')],
            num_novel_shots=30,
            num_base_shots=30,
        )),
    model_init=dict(num_novel_shots=30, num_base_shots=30))

evaluation = dict(interval=8000)
checkpoint_config = dict(interval=8000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=18000)

# load_from = 'path of base training model'
load_from = \
    'work_dirs/fpd_r101_c4_2xb4_coco_base-training/latest.pth'

model = dict(
    with_refine=True,
    frozen_parameters=['backbone', 'shared_head'],
    roi_head=dict(
        bbox_head=dict(num_classes=80, num_meta_classes=80),
        novel_class=(0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62),
        num_novel=20,
        meta_cls_ratio=1.0,
        prototypes_distillation=dict(num_base_cls=60, num_novel=20)),
)

