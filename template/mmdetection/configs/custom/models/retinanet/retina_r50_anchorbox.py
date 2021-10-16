# python tools/train.py configs/custom/models/retinanet/retina_r50_anchorbox.py
# python tools/train_cv.py configs/custom/models/retinanet/retina_r50_anchorbox.py

# python tools/inference.py configs/custom/models/retinanet/retina_r50_anchorbox.py work_dirs/retinanet --epoch best_bbox_mAP_50_epoch_14


_base_ = [
    "retinanet_r50_fpn.py",
    "../../helper/dataset.py",
    "../../helper/runtime.py",
    "../../helper/schedule.py",
]


# model settings
# anchor box 작게
model = dict(
    bbox_head=dict(
        _delete_ = True,
        type="RetinaHead",
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            # octave_base_scale=4, # scale base
            # octave_base_scale=2, # scale 작게 
            octave_base_scale=8, # scale 크게 -> retina_r50_
        
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8,16, 32, 64, 128],
        ),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    )
)




# runtime
# work_dir, wandb exp name
# exp = "retina_r50_anchorbox"
exp = "retina_r50_anchorbox_big"

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
