# python tools/train.py configs/custom/models/htc/htc_swin_b_sweeps.py
# python tools/inference.py configs/custom/models/htc/htc_swin_b.py --epoch best_bbox_mAP_50_epoch_13

# model settings
_base_ = [
    "htc_without_semantic_r50_fpn_1x_coco.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
pretrained = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"  # noqa
)
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
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
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

# work_dir, wandb exp name
exp = "htc_swin_b_sweeps"
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
lr_config = dict(policy="step", warmup="linear", warmup_ratio=0.001, warmup_iters=1099, step=[7, 12])
# learning policy
# lr_config = dict(policy="step", warmup="linear", warmup_ratio=0.001, warmup_iters=1000, step=[7, 12])
runner = dict(type="EpochBasedRunner", max_epochs=16)
