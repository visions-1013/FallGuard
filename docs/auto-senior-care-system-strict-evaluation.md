# auto-senior-care-system 严格评价报告

日期：2026-06-13  
范围：视觉/骨架主线，包括 ST-GCN、骨架动作识别、视觉跌倒检测；不展开到 IMU、雷达、Wi-Fi、养老 IoT 全栈。

## 0. 一句话总评

这个仓库更像“基于 Le2i 骨架数据的 ST-GCN 跌倒检测研究原型/课程实验”，不是一个完整、可复现、可部署的“养老看护系统”。如果把它当作学习 ST-GCN 和骨架时序分类的材料，它有价值；如果把它当作可交付开源项目或真实养老场景系统，当前工程质量、评估严谨性和部署闭环都明显不足。

严格总分：**39 / 100**

| 维度 | 分数 | 严格评价 |
| --- | ---: | --- |
| 项目定位 | 6 / 10 | README 目标很大，但代码实际只覆盖骨架序列二分类实验，标题中的“Autonomous Senior Helper System”超过当前实现边界。 |
| 算法合理性 | 12 / 20 | ST-GCN 路线合理，45 帧骨架窗口也适合跌倒动作；但模型实现、训练损失、光流融合和实时管线都不够成熟。 |
| 数据与评估严谨性 | 7 / 20 | 有 Le2i 场景数据和 notebook 指标，但数据规模小，实验协议散落，跨场景泛化不稳定，复现实验未标准化。 |
| 工程可复现性 | 3 / 20 | 没有环境文件/包入口/标准 CLI，存在作者本机绝对路径，核心流程主要依赖 notebook。 |
| 与开源/SOTA 差距 | 4 / 15 | 只实现了较早期 ST-GCN 思路，离 PYSKL/MMAction2 这类成熟工具链的配置化、模型库、数据协议和测试体系差距很大。 |
| 部署、隐私、养老场景可用性 | 4 / 10 | 骨架表示有隐私优势，但仓库没有端到端视频到告警链路、延迟评估、误报处理、多人/遮挡策略。 |
| 文档维护性 | 3 / 5 | README 很短，已有 `docs/auto-senior-care-system-guide.md` 对学习有帮助，但项目自身仍缺运行说明和实验复现说明。 |

如果按“课程实验/研究探索”而不是“可复现开源系统”评分，可以放宽到约 **65 / 100**；但按用户要求的“严格全方位评价”和网上现有资源对比，39 分更接近真实水平。

## 1. 本地仓库事实

### 1.1 仓库结构

当前仓库核心文件如下：

| 路径 | 实际作用 | 评价 |
| --- | --- | --- |
| `README.md` | 简述项目目标、Berkeley 技术报告链接、视频 demo 链接 | 只有愿景，没有安装、训练、评估、数据准备、依赖说明。 |
| `st-gcn/model/` | ST-GCN 模型、空间图卷积、时间卷积、邻接矩阵 | 是项目真正核心。 |
| `st-gcn/train_utils.py` | PyTorch Dataset、BalancedBatchSampler、训练循环、KFoldCrossValidation | 有训练框架雏形，但不是独立 CLI，且有实现瑕疵。 |
| `st-gcn/eval.py` | 数据加载、45 帧 batch 构造、测试集评估、命令行权重参数 | 评估入口存在，但默认路径写死到作者本机。 |
| `st-gcn/*.ipynb` | 训练、测试、可视化实验记录 | 是主要实验载体，但不适合作为正式可复现接口。 |
| `st-gcn/model_weights/` | 多个 `.pth` 权重 | 有可用实验产物，但缺少权重对应配置和训练日志索引。 |
| `datasets/Le2i-train-test/` | Le2i 场景骨架 JSON/NPY | 本地包含数据，利于学习和复查。 |
| `denseOpticalFlow.py`、`sparseOpticalFlow.py`、`ash_optical_flow.py` | 光流实验脚本 | 辅助实验，不是主训练/部署链路。 |

只读核验结果：

- `datasets/Le2i-train-test` 当前包含 **190 个 `.json`** 和 **64 个 `.npy`**。
- 样本 `datasets/Le2i-train-test/Office/Skeletons_full/video (1).npy` 的键为 `boxes`、`filename`、`keypoints`、`scores`，其中 `keypoints.shape == (325, 17, 3)`。
- `st-gcn/model_weights/` 下有 `le2i_90acc_default.pth`、`le2i_94acc.pth`、`le2i_97acc.pth`、`le2i_97acc_default.pth`、`le2i_flow_97acc.pth`、`NTU_87acc.pth`。
- 根目录没有发现 `requirements.txt`、`environment.yml`、`setup.py`、`pyproject.toml`、`Dockerfile`、`Makefile` 等可复现工程入口。

### 1.2 代码证据

主要优点：

- 使用骨架关键点而不是直接 RGB 像素，有利于降低隐私暴露，也更贴近动作识别中的结构化时序建模。
- `st-gcn/model/st_graph.py` 提供了邻接矩阵构造和归一化，`st-gcn/eval.py` 定义了 17 个 COCO 风格关键点和 16 条骨架边。
- `st-gcn/eval.py` 的 `create_batch()` 使用 `segment_length=45`，把连续骨架序列切成 45 帧窗口，符合跌倒是“时间过程”而不是单帧姿态的判断逻辑。
- `st-gcn/train_utils.py` 有 `FallDataset`、`BalancedBatchSampler`、`Trainer` 和 `KFoldCrossValidation()`，说明作者尝试处理类别不均衡和交叉验证。

严重问题：

- `st-gcn/eval.py:11-13` 写死了 `D:\ASH\datasets\Le2i\...`，`st-gcn/utils.py:26` 和 `ash_optical_flow.py` 也有作者本机路径。别人 clone 仓库后无法直接跑通。
- `st-gcn/model/stgcn.py:117` 和 `st-gcn/model/stgcn.py:230` 在模型输出处调用 `F.softmax(x, dim=1)`，而训练代码使用 `nn.CrossEntropyLoss()`。PyTorch 的 `CrossEntropyLoss` 期望输入 logits，不期望已经 softmax 的概率。这不一定让训练完全失败，但属于典型实现错误，会影响数值稳定性和梯度质量。
- `st-gcn/train_utils.py:164` 返回字典里 `'train_loss': train_acc_list`，把训练准确率误写成训练损失。这会污染后续记录和调参判断。
- `BalancedBatchSampler` 每个 batch 都包含全部 fall 样本再追加 non-fall 样本，策略很粗，不是标准的 class-balanced mini-batch；数据稍大时会造成 batch 过大和训练动态异常。
- `eval.evaluate()` 对 fall 视频采用“任一窗口预测 fall 即视频 fall”，对 non-fall 视频采用“所有窗口都 non-fall 才算 non-fall”。这个策略会天然提高 non-fall 的误报风险，对真实监控很敏感，但报告中没有阈值、时间平滑或误报代价分析。

### 1.3 notebook 指标证据

以下结果来自仓库内 notebook 历史输出，不是本轮重新训练：

| 文件 | 记录结果 | 解读 |
| --- | --- | --- |
| `st-gcn/Le2i_train.ipynb` | train acc 0.9869，val acc 0.9474 | 单次训练/验证结果较高，但不能代表泛化。 |
| `st-gcn/Le2i_train_test.ipynb` | 3 折结果：train acc 0.9076±0.0602，val acc 0.8980±0.0700，test acc 0.8833±0.0624 | 有交叉验证意识，但数据规模小，波动不低。 |
| `st-gcn/Le2i_train_test.ipynb` | 另一次 unseen test accuracy 0.7333 | 跨场景/未见数据表现明显下降。 |
| `st-gcn/Le2i_flow_train.ipynb` | train acc 0.9768，val acc 0.9737 | 光流融合训练/验证看起来高，但需要看测试集。 |
| `st-gcn/Le2i_flow_train_test.ipynb` | test acc 0.8571，unseen test acc 0.7167 | 加光流后未见场景没有稳定提升，甚至可能更差。 |
| `st-gcn/stgcn_results.ipynb` | 多组 5 折记录，test acc 从约 0.684 到 0.832，sensitivity/std 波动大 | 说明作者尝试多配置，但稳定性和可读性不足。 |

最关键的评价是：**这个项目不是没有效果，而是效果证据还不够严谨。** 如果只看单次 val accuracy，会高估项目；如果看 unseen test 和多折波动，就会发现真实泛化能力还没有被证明。

## 2. 与网上开源项目和论文对比

### 2.1 与原始 ST-GCN 论文和官方仓库对比

ST-GCN 论文提出用图卷积自动学习人体骨架的空间和时间模式，目标是替代手工部件/遍历规则，并在 Kinetics、NTU-RGBD 等大数据集上验证。论文原始目标是通用 skeleton-based action recognition，不是专门面向 Le2i 小规模跌倒数据。

官方 `yysijie/st-gcn` 仓库有完整工程结构，包括 `config`、`feeder`、`net`、`processor`、`tools`、`requirements.txt` 等；同时 README 明确说该旧代码已转向 MMSkeleton，并将成为历史 artifact。

对比结论：

- 本项目算法思想来自 ST-GCN 主线，但工程结构没有官方仓库完整。
- 官方 ST-GCN 至少有配置、处理器、数据 feeder 和工具脚本；本项目主要靠 notebook 和少量 Python 文件串联。
- 本项目针对 Le2i 跌倒检测做了场景化实验，这是相对原始 ST-GCN 的应用价值；但从开源质量看，它更像“摘取 ST-GCN 思路后的实验代码”。

参考：

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- [yysijie/st-gcn](https://github.com/yysijie/st-gcn)

### 2.2 与 PYSKL 对比

PYSKL 是 PyTorch 骨架动作识别工具箱。论文说明它支持 GCN 和 CNN 类骨架动作识别算法，统一实现多个算法和多个 benchmark，并提供 ST-GCN++、训练测试配置、模型结果和数据准备说明。仓库 README 也列出 ST-GCN、ST-GCN++、PoseConv3D、AAGCN、MS-G3D、CTR-GCN 等算法，并给出 conda 环境和 editable install 流程。

对比结论：

- PYSKL 是“框架级工具箱”，本项目是“单任务实验仓库”。
- PYSKL 的优势是可比较、可复现、模型多、数据协议清晰；本项目优势是更小、更容易读懂，适合课程学习。
- PYSKL 当前 README 标注“not maintained by the developer”，所以它也不是完美活跃项目；但即使如此，它的工程成熟度仍显著高于本项目。

参考：

- [PYSKL paper](https://arxiv.org/abs/2205.09443)
- [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl)

### 2.3 与 MMAction2 / PoseC3D 对比

MMAction2 是 OpenMMLab 的视频理解工具箱，支持动作识别、动作定位、时空动作检测、骨架动作检测、视频检索等任务，并有文档、安装说明、模型库和单元测试。其模型库覆盖 ST-GCN、2s-AGCN、PoseC3D、STGCN++、CTRGCN、MSG3D 等骨架动作识别方法。

PoseC3D 论文指出，GCN 骨架方法在鲁棒性、互操作性和扩展性上有局限；PoseC3D 用 3D heatmap stack 表示骨架，强调对 pose estimation noise 更鲁棒、跨数据集泛化更好、多人场景更自然。

对比结论：

- 本项目仍停留在较早期的 ST-GCN 图序列输入，没有吸收 PoseC3D、ST-GCN++、CTR-GCN 等后续经验。
- 本项目没有 MMPose/MMAction2 式的“视频 -> 姿态估计 -> 数据格式 -> 训练 -> 测试 -> demo”完整链路。
- 如果目标是做严肃可复现研究，建议迁移到 MMAction2/PYSKL 这类框架做 baseline；如果目标是课程理解，当前轻量代码更容易读。

参考：

- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [Revisiting Skeleton-based Action Recognition / PoseC3D](https://arxiv.org/abs/2104.13586)

### 2.4 与视觉跌倒检测文献对比

视觉跌倒检测综述强调，跌倒检测对独居老人和辅助生活很重要，常用指标包括准确率、召回率、特异性等，也需要关注 benchmark dataset 和实际应用方向。

MUVIM 数据集论文明确指出真实跌倒检测难点：跌倒稀有且变化大，许多数据集缺少真实世界因素，例如光照变化、连续日常活动、摄像头位置变化。它还比较了 infra-red、depth、RGB、thermal 等视觉模态，并报告 RGB AUC ROC 低于 infra-red、thermal、depth。

近年骨架跌倒检测论文继续强调骨架 joint dynamics、时空依赖、模型轻量化和大数据集评估；2026 年解释性工作进一步强调临床/养老监测不仅要准确，还要有稳定、可解释的时序归因。

对比结论：

- 本项目选择骨架路线是合理的，因为骨架能减少 RGB 隐私暴露，也能表达跌倒过程中的关节动态。
- 但本项目的数据和评估还没有覆盖真实养老场景的关键问题：长时间日常活动、光照变化、摄像头位置变化、多人、遮挡、误报恢复、异常/未知动作拒识。
- 本项目没有解释性模块。对于养老场景，单纯输出 fall/non-fall 很难让护理人员或临床用户信任。

参考：

- [Vision-based Human Fall Detection Systems using Deep Learning: A Review](https://arxiv.org/abs/2207.10952)
- [Multi Visual Modality Fall Detection Dataset](https://arxiv.org/abs/2206.12740)
- [Modeling Human Skeleton Joint Dynamics for Fall Detection](https://arxiv.org/abs/2503.06938)
- [Explainable Fall Detection for Elderly Monitoring via Temporally Stable SHAP](https://arxiv.org/abs/2604.13279)

## 3. 优点

1. 方向选得对。  
   跌倒不是单帧分类，而是时序动作识别问题。用 45 帧骨架序列和 ST-GCN，比单纯用人体框高度、角度阈值更有研究价值。

2. 隐私思路比纯 RGB 更好。  
   骨架关键点只保留关节坐标和置信度，比保存原始视频更适合养老场景。当然，当前仓库仍没有真正的数据脱敏和部署隐私方案。

3. 仓库内有数据、权重和实验记录。  
   很多课程仓库只有空代码或 README，本项目至少包含 Le2i 骨架数据、模型权重、训练 notebook、测试 notebook，可以复查作者的实验思路。

4. 有跨场景意识。  
   notebook 中有 coffee room、home、lecture room、office 等场景拆分，也有 unseen test 的尝试。这比只做随机切分更接近真实泛化问题。

5. 代码量小，适合学习。  
   对初学者来说，PYSKL/MMAction2 太大；本项目的 `model/`、`train_utils.py`、`eval.py` 更容易完整读一遍。

## 4. 严重问题

### 4.1 不是完整系统

README 标题是“Autonomous Senior Helper System for Enhanced Safety and Well-Being”，但仓库里没有：

- 摄像头/视频输入到骨架提取的标准 pipeline；
- 实时推理服务；
- 告警、通知、日志、回放；
- 前端或护理人员界面；
- 误报处理和人工确认；
- 部署说明；
- 隐私和数据保留策略。

所以当前不能称为“养老看护系统”，最多称为“跌倒检测模型实验仓库”。

### 4.2 复现门槛高

核心问题：

- 缺少依赖文件。
- 缺少标准运行命令。
- 缺少数据准备脚本。
- 缺少训练配置文件。
- 评估默认路径指向 `D:\ASH\...`。
- notebook 和 `.py` 的逻辑重复，容易漂移。

一个严格开源项目至少应支持：

```powershell
conda env create -f environment.yml
conda activate auto-senior-care
python -m pip install -e .
python scripts/train_le2i.py --config configs/le2i_stgcn.yaml
python scripts/evaluate.py --weights checkpoints/le2i_stgcn.pth --split test
```

当前仓库做不到。

### 4.3 评估协议不够可靠

跌倒检测最怕漏报，也怕长期误报。当前项目主要看 accuracy，虽然 `eval.py` 能算 specificity 和 sensitivity，但没有形成正式报告、置信区间、ROC/PR 曲线、F1、混淆矩阵、每场景结果表、每视频错误分析。

更严重的是，notebook 中出现了高验证准确率与较低 unseen test accuracy 并存：

- `Le2i_train.ipynb` validation accuracy 约 0.947。
- `Le2i_train_test.ipynb` 3 折 test accuracy 约 0.883。
- `Le2i_train_test.ipynb` unseen test accuracy 约 0.733。
- `Le2i_flow_train_test.ipynb` unseen test accuracy 约 0.717。

这说明模型在训练/验证切分上看起来不错，但跨场景泛化仍不稳。严格评价时应以后者为重点。

### 4.4 模型实现有技术债

最需要修的点：

- `softmax + CrossEntropyLoss` 搭配不正确，应该让模型输出 logits，把 softmax 留给推理展示或 `torch.softmax()` 后处理。
- `Trainer.train()` 返回字段有明显 bug，`train_loss` 被写成 `train_acc_list`。
- `opt_method`、`momentum` 等参数传入但没有真正分支使用。
- `ST_Graph` 是空类。
- `sys.path.insert()` 和相对 import 说明包结构没有规范化。
- `eval.py` 同时承担数据加载、batch 构造、评估策略、CLI，边界不清。

### 4.5 光流融合没有证明有效

从 notebook 看，光流版训练/验证指标不差，但 unseen test 没有提升：

- skeleton-only unseen test：约 0.733。
- flow version unseen test：约 0.717。

这不代表光流无用，而是当前融合方法和实验协议没有证明它有用。`ash_optical_flow.py` 还把光流 PCA 处理路径写死为 `D:\ASH\datasets\Le2i\Office\...`，工程上也没有接入主流程。

## 5. 对比矩阵

| 对象 | 项目成熟度 | 算法覆盖 | 数据/评估 | 工程复现 | 对本项目的启示 |
| --- | --- | --- | --- | --- | --- |
| 本项目 | 研究原型/课程实验 | ST-GCN、简单光流融合 | Le2i 小规模场景，notebook 指标 | 弱，缺依赖和 CLI | 保留轻量学习价值，但必须补可复现性。 |
| ST-GCN 论文/旧官方仓库 | 经典基线 | 原始 ST-GCN | Kinetics、NTU-RGBD | 官方旧仓库较完整，但已历史化 | 本项目可作为 ST-GCN 应用改造，但不应停留在旧实现。 |
| PYSKL | 工具箱 | ST-GCN、ST-GCN++、PoseConv3D、AAGCN、MS-G3D、CTR-GCN 等 | 多 benchmark，模型库 | 明显强，有环境、配置、数据文档 | 如果做研究比较，应该迁移或至少对齐其配置/评估风格。 |
| MMAction2 | 大型视频理解框架 | 动作识别、检测、骨架动作识别等 | 模型库和文档完整 | 强，有安装、文档、测试 | 如果做长期项目，应借鉴其模块化结构和 model zoo 思路。 |
| 视觉跌倒检测文献 | 论文体系 | RGB、Depth、Thermal、Skeleton、多模态 | 关注稀有事件、日常活动、摄像头变化 | 不一定开源 | 本项目缺少真实世界泛化、误报、隐私、解释性评价。 |

## 6. 是否有创新性

严格说，**算法创新性较弱**。

已有部分：

- ST-GCN 是 2018 年经典方法。
- 骨架跌倒检测已有大量工作。
- 光流和骨架融合也不是新方向。
- PYSKL/MMAction2 已经覆盖更强的骨架动作识别 baseline。

本项目可能的价值不是“提出新算法”，而是：

- 把 ST-GCN 应用于 Le2i 跌倒检测场景；
- 提供小规模、可读的教学型实现；
- 为现有规则式 FallGuard 这类项目提供“可训练时序分类器”的迁移思路。

如果要提升为研究贡献，需要至少补一个明确创新点：

- 更严格的跨场景/跨数据集跌倒检测 benchmark；
- 面向隐私的骨架-only 养老监测 pipeline；
- 与规则系统融合的可解释 fall risk 输出；
- 对 pose noise、遮挡、多人场景的鲁棒性设计；
- 轻量模型在边缘设备上的延迟和误报控制。

## 7. 对真实养老场景的可用性

当前可用性很低。

真实养老场景需要回答：

- 摄像头画面如何转成骨架？用 AlphaPose、YOLO-Pose、MMPose，还是其他方法？
- 多人同屏怎么办？
- 老人坐下、躺床、弯腰捡东西、跪地、被家具遮挡怎么办？
- 误报后是否持续报警？
- 跌倒后如果老人起身，状态如何恢复？
- 夜间低光怎么办？
- 是否保存原视频？保存多久？谁能访问？
- 推理延迟是多少？CPU 能否实时？
- 断网、摄像头断开、骨架丢失如何处理？

本仓库基本没有这些答案。因此它不能直接用于养老院或家庭监控，只能作为模型研究/学习起点。

## 8. 改进路线

### 8.1 短期：让项目能被别人跑起来

优先级最高：

1. 增加 `environment.yml` 或 `requirements.txt`。
2. 增加包结构，例如 `src/auto_senior_care/`，移除 `sys.path.insert()`。
3. 把 `D:\ASH\...` 改成配置参数或相对路径。
4. 新增标准 CLI：
   - `scripts/prepare_le2i.py`
   - `scripts/train_le2i.py`
   - `scripts/evaluate_le2i.py`
5. 固化一个最小可复现实验：同一权重、同一 split、同一指标表。
6. 修复 `softmax + CrossEntropyLoss` 和 `train_loss` 返回 bug。

目标：别人 clone 后能在 30 分钟内复现实验指标。

### 8.2 中期：把评估做严谨

建议：

- 固定 train/val/test split 文件，避免 notebook 随机切分造成指标漂移。
- 每个场景单独报告 accuracy、precision、recall/sensitivity、specificity、F1、confusion matrix。
- 报告 video-level 和 window-level 两种指标。
- 明确 fall 视频聚合策略，例如 “任一窗口 fall”、连续 k 个窗口 fall、概率滑动平均。
- 增加错误样例表，列出 false positive/false negative 的视频名和可能原因。
- 做 ablation：
  - skeleton-only；
  - skeleton + confidence；
  - skeleton + optical flow；
  - learnable mask on/off；
  - shift window on/off。

目标：让指标足够可信，而不是只展示好看的准确率。

### 8.3 长期：对齐成熟开源工具链

两条路线：

1. 轻量教学路线：保留当前简洁代码，但补齐工程、文档、评估。
2. 研究/产品路线：迁移到 MMAction2/PYSKL 风格配置，接入 MMPose/YOLO-Pose，加入实时推理和告警闭环。

如果目标是毕业设计或课程展示，路线 1 足够。  
如果目标是论文或开源项目，路线 2 更合理。

## 9. 最终判断

这个项目“有研究方向价值，但没有工程完成度”。

值得肯定的是：它抓住了跌倒检测的核心问题，使用骨架时序模型而不是纯单帧规则；仓库内有真实数据、权重和 notebook 实验；从学习角度能帮助理解 ST-GCN。

必须直说的是：它离网上成熟开源项目和论文标准差距明显。官方 ST-GCN、PYSKL、MMAction2 都有更规范的工程结构和实验协议；视觉跌倒检测文献也已经把真实场景、隐私、多模态、解释性、延迟和泛化作为重点问题。本项目目前大多没有覆盖。

因此最准确的定位是：

> 一个可用于学习和二次开发的 ST-GCN 跌倒检测研究原型，不是可部署的养老看护系统，也不是达到现代骨架动作识别开源基线水平的完整项目。

如果后续继续做，第一优先级不是换更复杂模型，而是先把环境、路径、训练入口、评估协议和错误分析补齐。否则再高的 notebook 准确率都很难被别人信任。

## 10. 参考资料

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- [yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- [PYSKL: Towards Good Practices for Skeleton Action Recognition](https://arxiv.org/abs/2205.09443)
- [kennymckormick/pyskl](https://github.com/kennymckormick/pyskl)
- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [Revisiting Skeleton-based Action Recognition / PoseC3D](https://arxiv.org/abs/2104.13586)
- [Vision-based Human Fall Detection Systems using Deep Learning: A Review](https://arxiv.org/abs/2207.10952)
- [Multi Visual Modality Fall Detection Dataset](https://arxiv.org/abs/2206.12740)
- [Modeling Human Skeleton Joint Dynamics for Fall Detection](https://arxiv.org/abs/2503.06938)
- [Explainable Fall Detection for Elderly Monitoring via Temporally Stable SHAP](https://arxiv.org/abs/2604.13279)
