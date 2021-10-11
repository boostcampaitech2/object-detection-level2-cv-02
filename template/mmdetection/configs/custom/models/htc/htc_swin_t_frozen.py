# model settings
_base_ = "htc_swin_t.py"

# freeze backbone
model = dict(backbone=dict(frozen_stages=4))

# work_dir, wandb exp name
exp = "htc_swin_t_frozen"
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
