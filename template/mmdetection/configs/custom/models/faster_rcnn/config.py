# python tools/train.py configs/custom/models/faster_rcnn/config.py
# python tools/train_cv.py configs/custom/models/faster_rcnn/config.py

# python tools/inference.py configs/custom/models/faster_rcnn/config.py work_dirs/faster_rcnn --epoch best_bbox_mAP_50_epoch_14

# model settings
_base_ = [
    "faster_rcnn_r50_fpn.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
# 0. R50
# 1. swim Transformer : mask_rcnn_swin-t-p4-w7_fpn_1x_coco
# '''
pretrained = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa
)
model = dict(
    # type="MaskRCNN",
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
# '''

# 2. retinanet_pvtv2-b0_fpn_1x_coco
'''
model = dict(
    # type="RetinaNet",
    backbone=dict(
        _delete_=True,
        type="PyramidVisionTransformerV2",
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint="https://github.com/whai362/PVT/" "releases/download/v2/pvt_v2_b0.pth"),
    ),
    neck=dict(in_channels=[32, 64, 160, 256]),
)
'''

# 3.  retinanet_pvtv2-b5_fpn_1x_coco
'''
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
'''
# runtime
# work_dir, wandb exp name
exp = "faster_rcnn_pvtv2_b5_backbone"
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

# optimizer -----
# optimizer = dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# ------------------

# learning policy
lr_config = dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
runner = dict(type="EpochBasedRunner", max_epochs=12)

# dataset
# evaluation = dict(interval=1, metric='bbox', save_)
data = dict(samples_per_gpu=1, workers_per_gpu=1)
