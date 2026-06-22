custom_imports = dict(
    imports=[
        "fallguard.training.mmaction_hooks",
        "fallguard.training.mmaction_metrics",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="RecognizerGCN",
    backbone=dict(type="STGCN", graph_cfg=dict(layout="coco", mode="stgcn_spatial")),
    cls_head=dict(
        type="GCNHead",
        num_classes=2,
        in_channels=256,
        loss_cls=dict(type="CrossEntropyLoss", class_weight=[1.0, 1.0]),
    ),
)

dataset_type = "PoseDataset"
ann_file = "data/processed/le2i_stgcn.pkl"
pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split="fold_0_train"),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=pipeline,
        split="fold_0_val",
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = [dict(type="BinaryClassificationMetric", threshold=0.5)]
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=45, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    paramwise_cfg=dict(custom_keys={"cls_head": dict(lr_mult=3.0)}),
)
param_scheduler = [dict(type="CosineAnnealingLR", T_max=45, eta_min=1e-6, by_epoch=True)]

custom_hooks = [
    dict(type="FreezeBackboneHook", freeze_epochs=5),
    dict(
        type="EarlyStoppingHook",
        monitor="binary/f1",
        rule="greater",
        patience=8,
        strict=True,
    ),
]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="binary/f1",
        rule="greater",
        max_keep_ckpts=3,
    ),
    logger=dict(type="LoggerHook", interval=20),
)
visualizer = dict(
    type="ActionVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
)
log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)
default_scope = "mmaction"
randomness = dict(seed=42, deterministic=True)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
