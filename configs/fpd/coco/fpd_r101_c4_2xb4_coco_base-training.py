_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../fpd_r101_c4.py',
    '../../_base_/default_runtime.py'
]

lr_config = dict(warmup_iters=1000, step=[92000])
evaluation = dict(interval=110000)
checkpoint_config = dict(interval=55000)
runner = dict(max_iters=110000)
optimizer = dict(lr=0.004)

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60),
                  prototypes_distillation=dict(num_base_cls=60),
                  num_novel=0,
                  meta_cls_ratio=1.0),
)
