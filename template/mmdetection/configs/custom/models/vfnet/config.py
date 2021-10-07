# python tools/train.py configs/custom/models/vfnet/config.py
# python tools/inference.py configs/custom/models/vfnet/config.py work_dirs/vfnet_r50fpn_base --epoch best_bbox_mAP_50_epoch_14

# model settings
_base_ = [
    "detr_r50_8x2_150e_coco.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
# 0. R50


# runtime
# work_dir, wandb exp name
exp = "detr_r50_base"
work_dir = f"./work_dirs/{exp}"

# Wandb Log
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=500),
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project='object-detection-recycling-trash',
                entity = 'boostcamp-2th-cv-02team',
                name = f"{exp}"
            ),
            )
    ])

workflow = [('train', 1), ('val', 1)]

# optimizer -----
optimizer = dict(
    type="AdamW",
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}),
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy="step", step=[100])
runner = dict(type="EpochBasedRunner", max_epochs=150)
# ------------------

# dataset
data_root = "../../../dataset/"
evaluation = dict(_delete_ =True, interval=1, metric='bbox')


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]