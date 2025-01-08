_base_ = [
    '_base_/models/upernet_transnext.py',
    '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
# optimizer
# find_unused_parameters = True
model = dict(
    backbone=dict(
        pretrained='pretrained/transnext_tiny_224_1k.pth',
        # pretrained='pretrained/iter_48k.pth',
        type='transnext_tiny',
        pretrain_size=224,
        img_size=512,
        is_extrapolation=False,
    ),
    decode_head=dict(
        in_channels=[72, 144, 288, 576],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=288,
        num_classes=150
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# optimizer
# 动态生成 paramwise_cfg
paramwise_cfg = {
    'custom_keys': {
        'query_embedding': {'decay_mult': 0.},
        'relative_pos_bias_local': {'decay_mult': 0.},
        'cpb': {'decay_mult': 0.},
        'temperature': {'decay_mult': 0.},
        'norm': {'decay_mult': 0.},
        'head': {'lr_mult': 10.0}, # 新添加的，为了让decoder训练更快
        'mlp': {'lr_mult': 10.0}
    }
}
# 为每个 block1 和 block2 中的 attn 层设置 lr_mult
# for i in range(block1_depth):
#     paramwise_cfg['custom_keys'][f'block1.{i}.attn'] = {'lr_mult': 10.0}
# for i in range(block2_depth):
#     paramwise_cfg['custom_keys'][f'block2.{i}.attn'] = {'lr_mult': 10.0}


optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=paramwise_cfg)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
