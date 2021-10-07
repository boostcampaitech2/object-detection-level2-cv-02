# python tools/train.py configs/custom/config.py
# python tools/inference.py configs/custom/htc.py work_dirs/HTC_SwinTransformer --epoch best_bbox_mAP_50_epoch_14

# model settings
_base_ = [
    "htc_without_semantic_r50_fpn_1x_coco.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
model = dict(
    backbone=dict(
        _delete_=True,
        type="PyramidVisionTransformerV2",
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        mlp_ratios=(4, 4, 4, 4),
        init_cfg=dict(checkpoint="https://github.com/whai362/PVT/" "releases/download/v2/pvt_v2_b5.pth"),
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
)
# optimizer
optimizer = dict(type="AdamW", lr=0.0001 / 1.4, weight_decay=0.0001)
# dataset settings
data = dict(samples_per_gpu=1, workers_per_gpu=1)

# work_dir, wandb exp name
exp = "HTC_PyramidVisionTransformerV2_SoftNMS"
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

lr_config = dict(step=[8, 12])

runner = dict(type="EpochBasedRunner", max_epochs=15)
