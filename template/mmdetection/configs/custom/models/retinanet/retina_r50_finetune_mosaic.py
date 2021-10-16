# python tools/train.py configs/custom/models/retinanet/retina_r50_finetune_mosaic.py
# python tools/train_cv.py configs/custom/models/retinanet/retina_r50_finetune_mosaic.py

# python tools/inference.py configs/custom/models/retinanet/retina_r50_finetune_mosaic.py work_dirs/retinanet --epoch best_bbox_mAP_50_epoch_14


_base_ = [
    "retinanet_r50_fpn.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
# 0. base : ResNet50

# runtime
# work_dir, wandb exp name
exp = "retinanet_resnet_finetune_mosaic"
work_dir = f"./work_dirs/{exp}"

# Wandb Log
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='object-detection-recycling-trash',
                entity = 'boostcamp-2th-cv-02team',
                name = f"{exp}"
            ),
            )
    ])

workflow = [('train', 1), ('val', 1)] # random validation 기준 평가
# workflow = [('train', 1)]

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

dtrain_pipeline = [
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


# optimizer ----- 
# optimizer = dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# ------------------

# learning policy
lr_config = dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
runner = dict(type="EpochBasedRunner", max_epochs=12)

# dataset
evaluation = dict(classwise = True) # class 별 ap 확인 
data = dict(samples_per_gpu=4, workers_per_gpu=2)
