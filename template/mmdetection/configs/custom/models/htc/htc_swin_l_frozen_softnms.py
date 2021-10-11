_base_ = "htc_swin_l_frozen.py"

# model settings
model = dict(
    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=2000, nms=dict(type="soft_nms", iou_threshold=0.7), min_bbox_size=0
        ),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="soft_nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.001, nms=dict(type="soft_nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5),
    ),
)

# work_dir, wandb exp name
exp = "htc_swin_l_frozen_softnms"
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
