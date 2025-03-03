_base_ = [
    '_base_/models/moenet_transnext.py',
    '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
# optimizer
model = dict(
    backbone=dict(
        pretrained='pretrained/transnext_tiny_224_1k.pth',
        type='transnext_tiny',
        pretrain_size=224,
        img_size=512,
        is_extrapolation=False,
    ),
    decode_head=dict(
        type='MoEHead',
        in_channels=[72, 144, 288, 576],
        # feature_strides=[4, 8, 16, 32],
        # channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True),
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'query_embedding': dict(decay_mult=0.),
                                                 'relative_pos_bias_local': dict(decay_mult=0.),
                                                 'cpb': dict(decay_mult=0.),
                                                 'temperature': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                #  new
                                                 'head': dict(lr_mult=10.0)
                                                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
