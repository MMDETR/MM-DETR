_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
]
load_from = "path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth"

data_root = "path/to/vindrmammo dataset/"
class_name = ('Mass')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(30, 144, 255)])

model = dict(
    bbox_head=dict(num_classes=num_classes),
    backbone=dict(
        frozen_stages=4
    )
)

train_dataloader = dict(
    dataset=dict(
        type='MultiViewMammoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='path/to/annotations/multi_view_train_mass_calc_no_duplicates.json',
        data_prefix=dict(img='mammo/')
    )
)

val_dataloader = dict(
    dataset=dict(
        type='MultiViewMammoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='path/to/annotations/multi_view_test_mass_calc_no_duplicates.json',
        data_prefix=dict(img='mammo/')
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'path/toannotations/multi_view_test_mass_calc_no_duplicates.json', classwise=True)
test_evaluator = val_evaluator

max_epoch = 40

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1)
)

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[7],
        gamma=0.1
    )
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.),  
            'language_model': dict(lr_mult=0),
        }
    )
)
batch_size = 1
auto_scale_lr = dict(base_batch_size=batch_size)


train_dataloader['batch_size'] = batch_size
val_dataloader['batch_size'] = batch_size
test_dataloader['batch_size'] = batch_size
