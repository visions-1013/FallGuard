# FallGuard

FallGuard 是一个面向本地视频的跌倒识别工程。系统使用
[Ultralytics YOLO26 Pose](https://docs.ultralytics.com/tasks/pose/) 提取 COCO-17
人体关键点，再使用 ST-GCN 对连续骨架序列进行 `non_fall / fall` 二分类。

本项目不再包含光流和手写宽高比、角度、下降速度规则。训练由 MMAction2
负责，本地视频应用只依赖轻量 PyTorch 模型，不需要安装 MMAction2。

## 处理流程

```text
Le2i AVI + 跌倒区间
  -> YOLO26n-pose骨架缓存
  -> 20 FPS / 32帧滑窗
  -> MMAction2 ST-GCN训练
  -> 运行包(best.pt + model_meta.json)
  -> 视频标注与事件JSON
```

## 目录

```text
configs/                 数据、MMAction2和推理配置
notebooks/               云端训练Notebook
src/fallguard/data/      Le2i清单、缓存、预处理和窗口
src/fallguard/pose/      YOLOPose和单人主目标选择
src/fallguard/models/    COCO图、ST-GCN和运行包
src/fallguard/training/  MMAction2训练、评估和报告
src/fallguard/inference/ 视频流水线、事件状态机和保存
src/fallguard/ui/        精简Tkinter界面
tests/                   自动测试
```

原始数据、骨架缓存、训练结果和权重均由 `.gitignore` 排除。

## 环境

推荐 Python 3.11。GPU训练：

```powershell
conda env create -f environment.yml
conda activate fallguard-stgcn
```

`environment.yml` 会以 editable 方式安装 MMAction2 官方 `v1.2.0` 源码。不要单独
安装 `mmaction2==1.2.0` 的 PyPI wheel：该 wheel 漏打包 DRN 目录，会在注册模块时
报 `ModuleNotFoundError`。ST-GCN 不依赖 MMCV 自定义算子，因此工程使用跨平台的
`mmcv-lite==2.1.0`。

只运行本地视频应用：

```powershell
python -m pip install -e .
```

Tkinter 来自 Python/Conda，不通过 pip 安装。

## Le2i数据范围

本项目识别原始发布包中的六个场景目录：

```text
Coffee_room_01  Coffee_room_02  Home_01  Home_02  Lecture_room  Office
```

Notebook启动时必须找到190个AVI，并应用三个内置标注修正。监督训练和评估只使用
存在真实TXT标注的130个视频：Coffee_room_01（48）、Coffee_room_02（22）、
Home_01（30）和Home_02（30）。TXT中的`0,0`表示整段视频为`non_fall`。

Lecture_room（27）和Office（33）共60个视频没有可信TXT区间标注，始终标记为
`unlabeled`并完全排除。工程不会根据视频编号推测其类别，也不会对它们提取训练骨架
或计算指标；排除记录写入`excluded_unlabeled_videos.csv`。

## 云端训练

AutoDL只需上传以下内容，无需上传或安装FallGuard项目源码：

```text
01_train_stgcn_cloud.ipynb
datasets/
```

将Notebook放在`datasets/`同级目录，打开
[01_train_stgcn_cloud.ipynb](notebooks/01_train_stgcn_cloud.ipynb)后执行`Run All`。
默认配置要求CUDA可用，并自动安装公开依赖、MMAction2官方v1.2.0源码以及下载公开
YOLOPose/ST-GCN权重。Notebook不会执行任何`from fallguard...`导入。

完整流程为：

1. 安装MMAction2 1.2训练环境；
2. 审核190个视频，只保留130个有TXT标注的视频；
3. 按`scene + video_label`联合分层，以视频为单位固定划分91/19/20个
   train/val/test视频，并保存和复用`split_manifest.csv`；
4. 只为这130个视频提取YOLO26n-pose骨架并校验缓存元数据；
5. 按真实FPS重采样到20 FPS，再生成32帧、步长4帧的窗口；
6. 加载NTU60 2D-joint官方ST-GCN骨干，替换为二分类头并微调；
7. 用validation选择最佳epoch和满足precision不低于85%的阈值；
8. 阈值锁定后只运行一次test，生成窗口级、视频事件级和分场景报告；
9. 导出MMAction checkpoint、本地运行包、完整报告和ZIP归档。

`FAST_DEV_RUN`只用于AutoDL环境烟雾测试；正式训练必须保持`False`。`AUTO_RESUME`
默认开启，已有`split_manifest.csv`和有效骨架缓存会被复用，不会重新随机划分。

官方预训练不能直接只保留 `standing up` 和 `falling` 两个NTU类别使用。项目会丢弃
60类分类头，换成 `non_fall / fall` 两类头，再用Le2i微调。

输出固定写入`outputs/autodl_training/`：

```text
manifest.csv
split_manifest.csv
excluded_unlabeled_videos.csv
poses/
processed/le2i_stgcn.pkl
work_dir/
reports/validation/
reports/test/
best_mmaction.pth
runtime_bundle/best.pt
runtime_bundle/model_meta.json
runtime_bundle/provenance.json
fallguard_training_artifacts.zip
```

训练集只计算loss；validation负责checkpoint、早停和阈值选择；test不参与类别权重、
训练、checkpoint、早停或阈值选择。由于同一场景会同时出现在三个集合中，该测试避免了
相邻窗口泄漏，但严格性仍低于真正的跨场景外部测试，报告会明确记录这一限制。

## 本地视频识别

命令行：

```powershell
fallguard infer-video input.avi outputs weights/runtime_bundle --device cuda:0
```

Tkinter：

```powershell
fallguard app
```

GUI 默认加载 `outputs/autodl_training/runtime_bundle` 中的 ST-GCN 权重和
`yolo26n-pose.pt`，自动选择 CPU/GPU。实时检测页显示监控画面、四项关键数据以及标记
跌倒区间的时间轴；历史记录页可以回放已检测视频。可通过 `--recordings-dir` 更改记录目录。

每次处理生成：

```text
<name>_fallguard.mp4   带骨架、概率、状态和时间戳的视频
<name>_events.json     跌倒事件起止、触发时刻、触发延迟和最大概率
<name>_summary.json    输入、模型版本、设备、耗时、帧数和事件数量
```

输出视频保持原分辨率和FPS，但不保留音频。取消或异常时保留 `.partial.mp4`，并在
`summary.json` 写明 `cancelled` 或 `failed`，成功后才原子重命名为正式MP4。
GUI 每次检测会在 `recordings/<检测时间>_<视频名>/` 下单独保存上述三个文件。

## 模型验收

只有同时满足以下保留视频测试指标，`model_meta.json` 才能标记为通过部署门槛：

- 事件级recall不低于90%；
- precision不低于85%；
- 误报不超过1次/小时；
- P95视频时间告警延迟不超过2秒。

没有通过门槛的权重仍可用于实验，但不能视为可靠的实际监护模型。

这些指标来自130个有标注视频中的固定test子集。由于尚无Lecture/Office可信标注，当前
结果不能替代跨场景验证；即使`passed_deployment_gate=true`，部署前仍应补充目标场景
数据验证。

## 测试

```powershell
python -m pytest
python -m ruff check .
python -m mypy src/fallguard
```

若已准备官方checkpoint和MMAction2源码环境，可额外运行非训练的数值一致性测试：

```powershell
$env:FALLGUARD_OFFICIAL_STGCN_CHECKPOINT = "weights/official_stgcn_ntu60_2d.pth"
python -m pytest tests/test_official_stgcn_parity.py
```

Windows下pytest临时目录已固定到项目内 `.pytest_tmp`，避免系统Temp目录权限问题。

## 参考

- Yan et al., Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.
- MMAction2 ST-GCN model zoo.
- Le2i fall detection dataset, Dijon UMR6306.
