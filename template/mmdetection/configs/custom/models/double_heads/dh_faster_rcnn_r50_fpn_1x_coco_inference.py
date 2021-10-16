# _base_ = 
_base_ = [
    "../../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]
data_root = "../../dataset/"


model = dict(
    roi_head=dict(
        type="DoubleHeadRoIHead",
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type="DoubleConvFCBBoxHead",
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=2.0),
        ),
    )
)


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1024, 1024)

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=128.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.1, 2), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=128.0),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=128.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline2 = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

data_root = "../../../dataset/"
dataset_type = "CocoDataset"
classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
data = dict(
    # samples_per_gpu=3,
    train=[
        dict(
            type="MultiImageMixDataset",
            dataset=dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + "train.json",
                img_prefix=data_root,
                pipeline=[
                    dict(type="LoadImageFromFile", to_float32=True),
                    dict(type="LoadAnnotations", with_bbox=True),
                ],
                filter_empty_gt=False,
            ),
            pipeline=train_pipeline,
            dynamic_scale=img_scale,
        ),
        dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + "train.json",
            img_prefix=data_root,
            pipeline=train_pipeline2,
        ),
    ]
)


# work_dir, wandb exp name
exp = "double_heads_mosaic"
work_dir = f"./work_dirs/{exp}"

# Wandb Log
log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="object-detection-recycling-trash", entity="boostcamp-2th-cv-02team", name=f"{exp}"
            ),
        ),
    ]
)

# learning policy
lr_config = dict(policy="step", warmup="linear", warmup_ratio=0.001, warmup_iters=1000, step=[7, 12])
runner = dict(type="EpochBasedRunner", max_epochs=30)
