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
pretrained = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa
)
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)


# work_dir, wandb exp name
exp = "HTC_SwinTransformer_SoftNMS"
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
