# optimizer
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

# learning policy
lr_config = dict(policy="step", warmup="linear", warmup_ratio=0.001, warmup_iters=1000, step=[8, 12])
