_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../fpd_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT2'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'),
    model_init=dict(classes='BASE_CLASSES_SPLIT2'))
lr_config = dict(warmup_iters=500, step=[17000])
evaluation = dict(interval=20000)
checkpoint_config = dict(interval=20000)
runner = dict(max_iters=20000)
optimizer = dict(lr=0.005)

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15),
                  num_novel=0, meta_cls_ratio=1.0))