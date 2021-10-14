# python tools/train.py configs/custom/models/retinanet/config.py
# python tools/train_cv.py configs/custom/models/retinanet/config.py

# python tools/inference.py configs/custom/models/retinanet/config.py work_dirs/retinanet --epoch best_bbox_mAP_50_epoch_14


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
exp = "retinanet_resnet_base"
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
evaluation = dict(classwise = True) # class 별 ap 확인 
data = dict(samples_per_gpu=4, workers_per_gpu=2)
