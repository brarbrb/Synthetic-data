_base_ = [
    'mmpose::_base_/default_runtime.py',
    'mmpose::_base_/datasets/coco.py',  # brings in coco dataset defaults
    'mmpose::_base_/schedules/sgd_schedule_210e.py',
]

# ---- Model ----
# Use HRNet-W32 head for heatmaps
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1,
                        block='BOTTLENECK', num_blocks=(4,),
                        num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2,
                        block='BASIC', num_blocks=(4,4),
                        num_channels=(32,64)),
            stage3=dict(num_modules=4, num_branches=3,
                        block='BASIC', num_blocks=(4,4,4),
                        num_channels=(32,64,128)),
            stage4=dict(num_modules=3, num_branches=4,
                        block='BASIC', num_blocks=(4,4,4,4),
                        num_channels=(32,64,128,256))
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=5,             # <--- 5 keypoints
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        # OKS-based NMS not needed for single-instance per image, but harmless
        nms=None,
        max_per_img=20),
    train_cfg=dict()
)

# ---- Dataset meta ----
dataset_type = 'CocoDataset'
data_root = '/path/to/project/output/'   # <--- CHANGE THIS
image_size = (192, 256)   # (w,h)
heatmap_size = (48, 64)   # typically image_size / 4
num_joints = 5

# Load dataset_info we wrote
dataset_info = dict(_delete_=True)
from configs.tool_dataset_info import dataset_info as _tool_info
dataset_info = _tool_info

# ---- Pipeline ----
# Top-down expects a bbox; your JSON has bbox per object => use_gt_bbox=True
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody', num_joints_half_body=3, prob=0.0),  # optional: off
    dict(type='TopdownAffine', input_size=image_size),
    dict(type='GenerateTarget', target_type='GaussianHeatmap', sigma=2),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='TopdownAffine', input_size=image_size),
    dict(type='PackPoseInputs')
]

test_pipeline = val_pipeline

# ---- Dataloader ----
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root + 'train/',
        ann_file='coco_keypoints.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root + 'val/',
        ann_file='coco_keypoints.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = val_dataloader

# ---- Eval ----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/coco_keypoints.json',
    metric=['keypoints'],
    outfile_prefix='keypoints_results',
    nms_mode=None,  # not using detector
    oks_sigmas=dataset_info['sigmas']
)

test_evaluator = val_evaluator

# ---- Optim & schedule ----
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-3, momentum=0.9, weight_decay=1e-4),
)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[170, 200], gamma=0.1)
]

# ---- Runtime ----
train_cfg = dict(max_epochs=210, val_interval=5)
default_hooks = dict(checkpoint=dict(save_best='coco/AP', interval=5))
work_dir = '/path/to/project/work_dirs/hrnet_tool_256x192'  # <--- CHANGE
load_from = 'https://download.openmmlab.com/mmpose/v1/projects/hrnet/hrnet_w32_coco_256x192-6d0f1a',
# ^ optional: if mmpose warns about remote weight, you can remove or update to a local checkpoint
