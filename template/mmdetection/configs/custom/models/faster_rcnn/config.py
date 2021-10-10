# python tools/train.py configs/custom/models/faster_rcnn/config.py
# python tools/train_cv.py configs/custom/models/faster_rcnn/config.py

# python tools/inference.py configs/custom/models/faster_rcnn/config.py work_dirs/vfnet_r50fpn_base --epoch best_bbox_mAP_50_epoch_14

# model settings
_base_ = [
    "faster_rcnn_r50_fpn.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]

# backbone
# 0. R50


# runtime
# work_dir, wandb exp name
exp = "kfold_test"
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
optimizer = dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(type="EpochBasedRunner", max_epochs=1)
# ------------------

# learning policy
lr_config = dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
runner = dict(type="EpochBasedRunner", max_epochs=1)

# dataset
# evaluation = dict(interval=1, metric='bbox', save_)
