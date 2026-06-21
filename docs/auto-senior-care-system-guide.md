# auto-senior-care-system 项目学习文档

> 面向对象：视觉基础较少、准备做自己的 FallGuard 摔倒检测改进项目的同学。
>
> 阅读目标：看懂这个仓库在做什么、数据和模型怎样流动、训练如何发生、它和我们当前 FallGuard 规则判断方案有什么本质区别。

## 1. 项目一句话概括

`auto-senior-care-system` 不是一个完整的养老看护系统，而是一个以“人体骨架序列摔倒检测”为核心的研究/实验仓库。

它的主线是：

```text
视频
-> 人体姿态关键点
-> 45 帧左右的骨架时间序列
-> ST-GCN 时空图卷积模型
-> fall / non-fall 二分类
-> 用测试集指标和误报/漏报列表评估可靠性
```

这和我们当前 FallGuard 项目最重要的区别是：FallGuard 现在主要靠手写姿态规则判断状态，而本项目尝试用有标签数据训练一个时序分类模型。也就是说，本项目的可靠性可以通过训练集、验证集、测试集、准确率、敏感性、特异性、误报和漏报来讨论；纯规则方案则更依赖人工设定阈值，验证难度更高。

项目 README 中给出的研究定位是：基于计算机视觉做摔倒检测，并希望算法能推广到其他医疗紧急事件。README 也给出了 Berkeley EECS 技术报告链接：

- <http://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-115.pdf>

## 2. 给视觉小白的背景知识

### 2.1 人体关键点是什么

人体关键点是用坐标表示人的身体部位，例如鼻子、眼睛、肩膀、手肘、手腕、髋部、膝盖、脚踝。

这个项目主要使用 17 个关键点，接近 COCO 人体姿态格式：

```text
0 鼻子
1 左眼        2 右眼
3 左耳        4 右耳
5 左肩        6 右肩
7 左肘        8 右肘
9 左腕        10 右腕
11 左髋       12 右髋
13 左膝       14 右膝
15 左踝       16 右踝
```

在数据中，一个关键点通常有 2 或 3 个值：

```text
(x, y)        只包含坐标
(x, y, score) 包含坐标和置信度
```

例如一个视频整理成 `.npy` 后，`keypoints` 可能是：

```text
(325, 17, 3)
```

含义是：

- 325：视频中有 325 帧骨架结果
- 17：每帧有 17 个人体关键点
- 3：每个点有 x、y、置信度

### 2.2 骨架图是什么

ST-GCN 不只把 17 个点看作普通数字，还会把它们看成一张图：

```text
节点：人体关键点
边：身体连接关系，例如肩膀连手肘，髋部连膝盖
```

本项目在 `st-gcn/eval.py` 中定义了 16 条骨架边，例如：

```python
[5, 7]    # 左肩 - 左肘
[7, 9]    # 左肘 - 左腕
[11, 13]  # 左髋 - 左膝
[13, 15]  # 左膝 - 左踝
```

这样模型可以学习“相邻身体部位之间的关系”，而不是只看单个点的位置。

### 2.3 时间序列为什么重要

摔倒不是一个静态姿势，而是一个过程：

```text
站立/移动
-> 身体快速下降
-> 身体角度快速改变
-> 进入低姿态或水平姿态
-> 没有立即恢复
```

如果只看最后一帧，“正常躺下”和“摔倒后躺地上”可能都很像。可靠的摔倒检测需要看一段时间内的变化。因此本项目把连续 45 帧作为一个样本，让模型学习动作过程。

### 2.4 ST-GCN 是什么

ST-GCN 全称是 Spatial Temporal Graph Convolutional Network，中文可以理解为“时空图卷积网络”。

它做两件事：

- Spatial：在每一帧里，根据人体骨架连接关系学习身体部位之间的空间关系。
- Temporal：沿着时间轴学习多帧之间的动作变化。

在这个项目中，模型输入大致是：

```text
batch_size x channels x frames x joints
```

常见形状是：

```text
N x 2 x 45 x 17
```

含义是：

- N：一批样本数量
- 2：x、y 两个坐标通道
- 45：每个样本包含 45 帧
- 17：每帧 17 个人体关键点

输出是：

```text
fall / non-fall
```

也就是二分类。

### 2.5 光流是什么

光流是用来描述画面中像素运动方向和速度的视觉特征。简单说，它不是直接看“人长什么样”，而是看“画面哪里在动、怎么动”。

本项目有两类光流代码：

- `denseOpticalFlow.py`：实时显示稠密光流，可理解为每块区域都估计运动。
- `sparseOpticalFlow.py`：实时显示稀疏光流，只跟踪一部分特征点。
- `ash_optical_flow.py`：从视频批量计算 Farneback 光流，再用 PCA 降到 64 维并保存。

项目里还有 `flow_stgcn`，尝试把骨架序列和光流特征结合起来。不过从代码组织看，主线仍然是骨架 ST-GCN。

### 2.6 训练、验证、测试分别是什么

初学时最容易混淆这三个词：

| 名称 | 用途 | 类比 |
| --- | --- | --- |
| 训练集 | 给模型学习参数 | 做练习题 |
| 验证集 | 训练过程中挑参数、看是否过拟合 | 阶段小测 |
| 测试集 | 最后评估泛化能力 | 期末考试 |

如果只在训练集上表现好，不代表模型可靠。摔倒检测更关心没见过的视频上是否能稳定识别摔倒和非摔倒。

## 3. 仓库结构总览

当前仓库核心结构如下：

```text
auto-senior-care-system/
├── README.md
├── ash_optical_flow.py
├── denseOpticalFlow.py
├── sparseOpticalFlow.py
├── datasets/
│   └── Le2i-train-test/
│       ├── Coffee room/
│       ├── Home/
│       ├── Lecture room/
│       └── Office/
└── st-gcn/
    ├── eval.py
    ├── train_utils.py
    ├── utils.py
    ├── main.ipynb
    ├── Le2i_train.ipynb
    ├── Le2i_train_test.ipynb
    ├── Le2i_train_test_v2.ipynb
    ├── Le2i_flow_train.ipynb
    ├── Le2i_flow_train_test.ipynb
    ├── stgcn_results.ipynb
    ├── model/
    │   ├── st_graph.py
    │   ├── sgcn.py
    │   ├── tgcn.py
    │   └── stgcn.py
    ├── model_weights/
    └── runs/
```

### 3.1 顶层文件

| 路径 | 作用 |
| --- | --- |
| `README.md` | 项目简介、研究报告链接、demo 链接。没有完整运行或训练说明。 |
| `denseOpticalFlow.py` | 用摄像头实时计算 Farneback 稠密光流，并显示箭头图和 HSV 图。 |
| `sparseOpticalFlow.py` | 用摄像头实时跟踪角点轨迹，展示稀疏光流。 |
| `ash_optical_flow.py` | 对视频文件批量计算光流，用 PCA 降维到 64 维，并保存为 `.npy`。路径写死为作者本机 `D:\ASH\...`。 |

顶层三个光流脚本更像实验辅助代码，不是完整应用入口。

### 3.2 `datasets/`

`datasets/Le2i-train-test/` 是当前仓库里自带的数据目录，包含 4 个场景：

```text
Coffee room
Home
Lecture room
Office
```

每个场景通常包含：

```text
Skeletons/       原始关键点 JSON
Skeletons_full/  整理后的 numpy 数据
```

当前数据文件数量：

| 类型 | 数量 |
| --- | ---: |
| `.json` | 190 |
| `.npy` | 64 |

示例数据结构：

```text
datasets/Le2i-train-test/Home/Skeletons/video (1).json
```

JSON 中每帧有：

```text
image_id
category_id
keypoints
score
box
idx
```

整理后的 `.npy` 示例：

```text
datasets/Le2i-train-test/Office/Skeletons_full/video (1).npy
```

其中对象包含：

```text
filename
keypoints
scores
boxes
```

`keypoints` 示例形状：

```text
(325, 17, 3)
```

### 3.3 `st-gcn/model/`

这是模型实现目录。

| 路径 | 作用 |
| --- | --- |
| `st-gcn/model/st_graph.py` | 构建骨架图邻接矩阵，包括普通邻接矩阵、归一化邻接矩阵。 |
| `st-gcn/model/sgcn.py` | Spatial GCN，学习同一帧内身体关键点之间的空间关系。 |
| `st-gcn/model/tgcn.py` | Temporal GCN，学习连续帧之间的时间变化。 |
| `st-gcn/model/stgcn.py` | 把空间图卷积和时间卷积组合成完整 ST-GCN，并提供 `flow_stgcn` 光流融合版本。 |

### 3.4 `st-gcn/utils.py`

负责数据清洗和切片。

重点函数：

| 函数 | 作用 |
| --- | --- |
| `clean()` | 根据丢帧数量过滤质量差的骨架序列。 |
| `visualize()` | 把 AlphaPose 检测框画回视频中，用于检查关键点数据是否合理。 |
| `split_seq()` | 把普通非摔倒序列切成固定长度片段。 |
| `split_fall_seq()` | 根据 `fall_interval` 把摔倒片段和非摔倒片段分开。 |
| `split_skeletons()` | 对一批骨架数据生成 fall 样本和 non-fall 样本。 |
| `split_skeletons_and_flows()` | 同时处理骨架和光流，用于光流融合实验。 |

### 3.5 `st-gcn/train_utils.py`

封装训练流程。

重点类和函数：

| 名称 | 作用 |
| --- | --- |
| `FallDataset` | PyTorch 数据集，把 fall 样本标成 `1.0`，non-fall 样本标成 `0.0`。 |
| `BalancedBatchSampler` | 每个 batch 尽量平衡 fall 和 non-fall，缓解类别不均衡。 |
| `evaluate()` | 在 DataLoader 上计算 loss 和 accuracy。 |
| `Trainer` | 包装 Adam 优化器、训练循环、验证集评估、训练曲线。 |
| `KFoldCrossValidation()` | 对 fall/non-fall 分别做 K 折切分，重复训练和测试，输出平均指标。 |

### 3.6 `st-gcn/eval.py`

负责测试和命令行评估。

它定义了：

- Le2i 各测试集路径。
- Home、Office、Lecture room 的视频级标签。
- 17 关键点骨架连接边。
- `create_batch()`：把整段视频切成多个 45 帧滑窗。
- `evaluate()`：视频级评估。
- `evaluate_flow()`：骨架 + 光流版本评估。
- 命令行入口：加载模型权重并评估。

### 3.7 `st-gcn/*.ipynb`

这些 notebook 是实验记录和训练入口。

| 路径 | 作用 |
| --- | --- |
| `main.ipynb` | 较早的 ST-GCN 实验，使用 25 个点、85 帧窗口，可能来自 NTU 数据实验。 |
| `Le2i_train.ipynb` | Le2i 骨架训练实验，保存 `le2i_97acc_default.pth`。 |
| `Le2i_train_test.ipynb` | Le2i 训练 + 测试 + KFold 实验，包含 unseen data 测试。 |
| `Le2i_train_test_v2.ipynb` | Le2i 训练测试的另一个版本。 |
| `Le2i_flow_train.ipynb` | 骨架 + 光流融合训练实验。 |
| `Le2i_flow_train_test.ipynb` | 骨架 + 光流融合训练测试实验。 |
| `stgcn_results.ipynb` | 结果汇总、模型比较、边连接可视化和实验记录。 |

### 3.8 `st-gcn/model_weights/`

模型权重目录。

当前包含：

```text
NTU_87acc.pth
le2i_90acc_default.pth
le2i_94acc.pth
le2i_97acc.pth
le2i_97acc_default.pth
le2i_flow_97acc.pth
```

文件名里的 `90acc`、`94acc`、`97acc` 是作者记录的实验准确率含义，但具体可靠性仍要看测试集定义和评估方式，不能只看文件名。

## 4. 数据如何流动

### 4.1 总体流程

```mermaid
flowchart LR
    A["原始视频"] --> B["姿态估计器输出 JSON"]
    B --> C["整理为 .npy 骨架序列"]
    C --> D["按 45 帧切片"]
    D --> E["fall / non-fall 样本"]
    E --> F["ST-GCN 训练"]
    F --> G["模型权重 .pth"]
    G --> H["滑窗推理"]
    H --> I["视频级 fall / non-fall 结果"]
```

### 4.2 原始 JSON

`Skeletons/` 目录里是每个视频对应的关键点 JSON。

典型字段：

```text
image_id     帧或图像 ID
category_id  人体类别
keypoints    人体关键点列表
score        姿态估计置信度
box          人体框
idx          人物 ID 或检测序号
```

这些 JSON 看起来更像 AlphaPose 或类似姿态估计器的输出。也就是说，本仓库没有完整展示“从原始视频跑姿态估计”的步骤，而是把姿态估计结果作为已有输入。

### 4.3 整理后的 `.npy`

`Skeletons_full/` 里是更适合训练的 numpy 数据。

有两种形态：

一种是单个视频一个 `.npy`：

```text
Office/Skeletons_full/video (1).npy
```

内部包含：

```text
filename
keypoints
scores
boxes
```

另一种是一个 `.npy` 中包含多个样本：

```text
Coffee room/Skeletons_full/falls.npy
Coffee room/Skeletons_full/non_falls.npy
Home/Skeletons_full/falls.npy
Home/Skeletons_full/non_falls.npy
```

内部每个元素通常是字典：

```text
filename
keypoints
scores
boxes
offset
fall_interval
```

其中 `fall_interval` 很关键。它表示这个视频里摔倒事件从哪一帧开始、到哪一帧结束。

### 4.4 用 `fall_interval` 切出训练样本

`utils.split_skeletons()` 会遍历骨架序列。

如果：

```text
fall_interval == (0, 0)
```

说明这是非摔倒视频。代码会用 `split_seq()` 把它按 45 帧切成多个 non-fall 样本。

如果：

```text
fall_interval != (0, 0)
```

说明这个视频包含摔倒事件。代码会用 `split_fall_seq()`：

- 把摔倒发生附近的 45 帧切成 fall 样本。
- 把摔倒前后不属于摔倒事件的片段切成 non-fall 样本。

这一步非常重要：模型学到的不是“整段视频叫什么名字”，而是“这一小段时间序列是否包含摔倒动作”。

### 4.5 坐标归一化

Le2i 视频分辨率在代码中按：

```python
max_x = 320
max_y = 240
```

处理。

`eval.create_batch()` 会把坐标除以宽高：

```text
x = x / 320
y = y / 240
```

这样模型看到的是 0 到 1 左右的相对坐标，而不是原始像素值。归一化能让训练更稳定。

## 5. 核心代码解析

### 5.1 图构建：`st-gcn/model/st_graph.py`

这个文件把人体骨架连接关系变成邻接矩阵。

普通邻接矩阵可以理解为：

```text
A[i, j] = 1 代表第 i 个关键点和第 j 个关键点相连
A[i, j] = 0 代表它们不直接相连
```

核心函数：

```python
def get_adjacency(edges, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A
```

`get_distance_adjacency()` 会返回两类关系：

```text
I：自己和自己相连
N：身体骨架邻居
```

最终形状类似：

```text
2 x 17 x 17
```

### 5.2 空间图卷积：`st-gcn/model/sgcn.py`

`unit_sgcn` 负责同一帧内的人体结构学习。

它的大致逻辑是：

1. 根据邻接矩阵，把每个关键点和邻居关键点的信息聚合。
2. 对聚合后的特征做卷积。
3. 做 BatchNorm。
4. 做 ReLU 等非线性变换。

核心思想是：肩膀、手肘、手腕不是孤立点，模型应该看它们之间的组合关系。

文件里还有两个可学习选项：

```python
learnable_mask
learnable_edges
```

含义是让模型不完全死守固定骨架边，而是可以学习某些连接的重要程度。比如摔倒时，髋部、肩部、膝盖之间的关系可能更关键。

### 5.3 时间卷积：`st-gcn/model/tgcn.py`

`unit_tgcn` 负责沿时间方向学习动作变化。

它使用：

```python
nn.Conv2d(..., kernel_size=(kernel_size, 1))
```

这里 `kernel_size=(9, 1)` 的意思是：

- 在时间维度看 9 帧附近的信息。
- 在关键点维度不直接滑动。

直观理解：模型会学习“前几帧到后几帧身体怎样变化”。

### 5.4 完整 ST-GCN：`st-gcn/model/stgcn.py`

`stgcn` 类把空间图卷积和时间卷积串起来。

输入：

```text
N x C x T x V
```

其中：

```text
N batch size
C 输入通道，通常是 2，也就是 x/y
T 时间帧数，常用 45
V 关键点数量，常用 17
```

模型流程：

```text
输入骨架序列
-> 输入层 stgcn_in
-> 多个 stgcn_unit
-> 对关键点维度池化
-> 对时间维度池化
-> 1D 卷积输出类别
-> softmax 得到 fall/non-fall 概率
```

`stgcn_unit` 里面有残差连接：

```text
输出 = SGCN + TGCN + 原输入变换
```

残差连接的作用是让深层网络更容易训练。

### 5.5 光流融合模型：`flow_stgcn`

`flow_stgcn` 在普通 ST-GCN 基础上增加光流输入。

普通输入：

```text
skeleton: N x 2 x T x 17
```

光流输入：

```text
flows: N x 64 x T
```

模型会把光流当成额外节点拼到骨架特征上：

```python
x = torch.cat((x, flows.unsqueeze(-1)), dim=-1)
```

这表示它希望结合两类信息：

- 骨架：人身体结构怎么变。
- 光流：画面运动怎么变。

对 FallGuard 来说，光流不是第一优先级。更可控的路线是先把骨架序列分类器做出来，再考虑是否加入光流。

### 5.6 训练数据集：`st-gcn/train_utils.py`

`FallDataset` 把 fall 和 non-fall 样本组织成 PyTorch 数据集。

标签规则：

```text
fall     -> 1.0
non-fall -> 0.0
```

它还支持 `window_size` 随机截取：

```python
start_index = np.random.randint(0, self.seq_len - self.window_size + 1)
```

训练时随机截取可以减少过拟合；验证时固定取后段，保证评估稳定。

### 5.7 平衡采样：`BalancedBatchSampler`

摔倒检测通常类别不均衡：非摔倒片段比摔倒片段多。

如果直接训练，模型可能学会“总是预测 non-fall”，准确率看起来不低，但真正摔倒识别很差。

`BalancedBatchSampler` 的思路是：

```text
每个 batch 尽量放入所有 fall 样本
再配一批 non-fall 样本
```

这样训练时模型会更重视少数类 fall。

### 5.8 训练器：`Trainer`

`Trainer` 做标准 PyTorch 训练：

```text
前向传播
-> CrossEntropyLoss
-> optimizer.zero_grad()
-> backward()
-> optimizer.step()
-> 在验证集上 evaluate()
```

默认优化器实际使用的是 Adam：

```python
self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

虽然构造函数里有 `opt_method` 参数，但代码没有根据它切换优化器。

### 5.9 K 折交叉验证：`KFoldCrossValidation`

KFold 的目的不是训练一个最终模型，而是评估模型在不同数据划分下是否稳定。

大致流程：

```text
把 fall 样本分成 k 份
把 non-fall 样本分成 k 份
每轮拿其中 1 份做验证，其余做训练
训练一个新模型
在固定测试集上评估
最后输出平均训练/验证/测试指标
```

这比只随机切一次训练/验证更可靠，因为它能看到结果是否依赖某一次幸运切分。

## 6. 模型如何训练

### 6.1 训练入口在哪里

本项目没有一个统一的 `train.py`。训练主要在 notebook 中完成：

```text
st-gcn/Le2i_train.ipynb
st-gcn/Le2i_train_test.ipynb
st-gcn/Le2i_train_test_v2.ipynb
st-gcn/Le2i_flow_train.ipynb
st-gcn/Le2i_flow_train_test.ipynb
st-gcn/stgcn_results.ipynb
```

其中对我们最有参考价值的是：

- `Le2i_train_test.ipynb`：骨架 ST-GCN 的训练、KFold、测试。
- `stgcn_results.ipynb`：实验比较和结果记录。
- `Le2i_flow_train_test.ipynb`：加入光流后的实验。

### 6.2 标准训练流程

可以把 notebook 中的训练流程整理成下面几步：

```text
1. 加载 Coffee room / Home 的 falls.npy 和 non_falls.npy
2. 使用 utils.split_skeletons() 切出 45 帧 fall/non-fall 片段
3. 拼接所有 fall 样本，得到 X_falls
4. 拼接所有 non-fall 样本，得到 X_non_falls
5. 构造 17 关键点骨架边
6. 用 get_distance_adjacency() 构建邻接矩阵 A
7. 初始化 stgcn(num_class=2, window_size=45, num_point=17, graph=A)
8. 构造 FallDataset 和 BalancedBatchSampler
9. 使用 CrossEntropyLoss 和 Adam 训练
10. 每个 epoch 计算训练集和验证集指标
11. 用 eval.evaluate() 在测试视频上做视频级评估
12. 用 torch.save(model.state_dict(), "...pth") 保存权重
```

### 6.3 核心参数

| 参数 | 常见值 | 含义 |
| --- | ---: | --- |
| `num_class` | 2 | fall / non-fall 二分类。 |
| `window_size` | 45 | 每个样本使用 45 帧骨架序列。 |
| `num_point` | 17 | 每帧 17 个人体关键点。 |
| `in_channels` | 2 | 只使用 x/y 坐标，不使用置信度。 |
| `lr` | 0.001 | Adam 学习率。 |
| `epochs` | 30 或 35 | 训练轮数。 |
| `batch_size` | 100 或 128 | 普通 DataLoader 的 batch size；平衡采样时由 sampler 控制。 |
| `dropout` | 0.5 | ST-GCN 单元中的 dropout。 |
| `temporal_kernel_size` | 9 | 时间卷积看多少帧附近的信息。 |

### 6.4 模型结构参数

默认层配置：

```python
default_layer_config = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]
```

每个元组含义：

```text
(输入通道数, 输出通道数, 时间步长)
```

当 stride 为 2 时，时间维度会被下采样，模型逐步从细粒度帧变化提取更高层动作特征。

### 6.5 当前 notebook 中记录的结果

从 notebook 输出能看到一些实验结果，例如：

| notebook | 记录结果示例 |
| --- | --- |
| `Le2i_train.ipynb` | training accuracy 约 0.987，validation accuracy 约 0.947。 |
| `Le2i_train_test.ipynb` | 3 折实验中 test accuracy 约 0.80 到 0.95，平均约 0.883。 |
| `Le2i_flow_train.ipynb` | training accuracy 约 0.977，validation accuracy 约 0.974。 |
| `Le2i_flow_train_test.ipynb` | 某些测试记录 accuracy 约 0.857 或 0.717，说明光流融合不一定在所有划分下更稳。 |
| `main.ipynb` | 较早实验 training accuracy 约 0.907，validation accuracy 约 0.876。 |

注意：这些结果只能说明作者实验记录中的表现，不能直接证明模型在我们的摄像头、场景、老人动作、遮挡情况中可靠。我们要借鉴的是“可训练、可评估”的方法。

### 6.6 权重如何保存和加载

保存：

```python
torch.save(model.state_dict(), "le2i_97acc_default.pth")
```

加载：

```python
model = stgcn(
    num_class=2,
    window_size=45,
    num_point=17,
    graph=A,
    layer_config=default_layer_config,
)
model.load_state_dict(torch.load(args.model))
```

权重只保存参数，不保存完整训练配置。因此复用权重时，必须用同样的模型结构和输入格式。

## 7. 如何推理和评估可靠性

### 7.1 滑窗推理

`eval.create_batch()` 会把一个完整视频切成多个 45 帧窗口。

例如一个视频有 100 帧，`segment_length=45`、`stride=1` 时，会形成很多重叠窗口：

```text
第 0-44 帧
第 1-45 帧
第 2-46 帧
...
```

最后不足一段时，会从末尾再取一个 45 帧窗口，避免漏掉视频结尾。

### 7.2 视频级判断规则

窗口级结果要汇总成视频级结果。

`eval.evaluate()` 中的规则是：

| 真实标签 | 视频级预测规则 |
| --- | --- |
| fall 视频 | 只要任意窗口预测为 fall，就认为整段视频预测为 fall。 |
| non-fall 视频 | 必须所有窗口都预测为 non-fall，才认为整段视频预测为 non-fall。 |

这个规则对摔倒检测比较合理，因为一次摔倒可能只发生在整段视频的一小部分。但它也会提高误报风险：非摔倒视频中只要某一个窗口误判为 fall，整段视频就变成 false positive。

### 7.3 指标含义

`eval.evaluate()` 会统计：

```text
tp true positives    真实 fall，预测 fall
tn true negatives    真实 non-fall，预测 non-fall
fp false positives   真实 non-fall，预测 fall
fn false negatives   真实 fall，预测 non-fall
```

然后计算：

```text
accuracy = (tp + tn) / 全部视频
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
```

中文理解：

| 指标 | 关注点 |
| --- | --- |
| accuracy | 总体预测对了多少。 |
| specificity | 非摔倒视频中，有多少没有被误报成摔倒。 |
| sensitivity | 摔倒视频中，有多少被成功识别出来。 |

对摔倒检测来说，`sensitivity` 很重要，因为漏报摔倒风险大；`specificity` 也很重要，因为误报太多系统会没人信。

### 7.4 为什么误报/漏报列表很有价值

代码会打印：

```text
True positives
True negatives
False positives
False negatives
```

这些文件名比单个准确率更有用，因为它告诉我们：

- 哪些具体视频漏报了。
- 哪些具体视频误报了。
- 错误是否集中在某个场景，例如 Lecture room 或 Office。
- 错误是否集中在某类动作，例如坐下、弯腰、遮挡、多人场景。

做我们自己的 FallGuard 时，也应该保留类似机制：不要只显示“准确率 90%”，要能列出错在哪些视频上。

## 8. 与 FallGuard 的区别

当前 FallGuard 项目是一个 Python + Streamlit 演示系统，使用 YOLO-Pose 提取人体框和姿态关键点，再通过手写姿态特征和短时序特征判断状态。

核心链路大致是：

```text
视频/摄像头
-> YOLO-Pose
-> PoseFeatureExtractor
-> PoseHistory
-> PoseStateEstimator
-> unknown / standing / sitting / moving / lying / fall
-> Streamlit 展示
```

本项目链路是：

```text
视频
-> 姿态估计 JSON/.npy
-> 45 帧骨架序列
-> ST-GCN 训练/推理
-> fall / non-fall
-> 测试集指标
```

### 8.1 核心对比表

| 维度 | auto-senior-care-system | FallGuard 当前实现 |
| --- | --- | --- |
| 项目定位 | 研究型摔倒分类模型实验。 | 课程演示型本地监控应用。 |
| 输入 | 已整理好的骨架序列，部分实验加入光流。 | 摄像头或本地视频帧。 |
| 姿态来源 | 仓库中已有 JSON 和 `.npy`，推测来自 AlphaPose 或类似工具。 | 运行时用 Ultralytics YOLO-Pose。 |
| 判断方式 | ST-GCN 学习 45 帧骨架序列，输出 fall/non-fall。 | 手写阈值规则判断姿态状态。 |
| 输出类别 | 二分类：fall / non-fall。 | 多状态：unknown、standing、sitting、moving、lying、fall。 |
| 是否训练 | 是，需要有标签样本。 | 当前不训练自己的分类器。 |
| 可靠性验证 | 可以用测试集、KFold、sensitivity、specificity、误报/漏报列表验证。 | 主要靠规则是否合理和人工测试，系统化可靠性较弱。 |
| 可解释性 | 中等，可以看窗口预测和错误视频，但模型内部较黑盒。 | 较强，能解释为角度、水平程度、速度阈值。 |
| 实时部署 | 需要把训练好的模型接入实时滑窗推理。 | 已有 Streamlit 实时/视频演示链路。 |
| 迁移难点 | 数据格式、训练集、模型部署、实时滑窗缓存。 | 阈值泛化差，场景变化后可靠性难证明。 |

### 8.2 FallGuard 当前规则的局限

FallGuard 当前规则大致依赖：

- 人体框宽高比。
- 肩髋连线角度。
- 人体中心下降速度。
- 身体角度变化速度。
- 低姿态持续时间。

这种方案适合快速演示，也容易解释，但有明显问题：

- 阈值依赖画面角度、人物距离、摄像头高度。
- 正常躺下、弯腰、坐到地上可能和摔倒很像。
- 遮挡、关键点缺失时规则容易不稳定。
- 没有训练集/测试集时，很难证明“这个阈值可靠”。
- 一旦换场景，可能要重新手调阈值。

### 8.3 本项目能补上的能力

本项目提供的是另一种思路：

```text
不要只靠人工写 if 判断；
把连续骨架动作交给模型学习；
再用独立测试集证明效果。
```

它能补强 FallGuard 的地方：

- 从单帧/短历史规则升级为固定窗口时序分类。
- 从手调阈值升级为数据驱动训练。
- 从“看起来能用”升级为“能报告测试指标”。
- 从只输出当前状态升级为能分析误报/漏报样本。

它不能直接替代 FallGuard 的地方：

- 本项目没有完整实时 UI。
- 本项目没有完整摄像头输入到模型推理的工程链路。
- 本项目数据是 Le2i，未必覆盖我们的实际使用场景。
- ST-GCN 输出二分类，不能直接替代 FallGuard 的多状态展示。

## 9. 对我们 FallGuard 项目的改进建议

### 9.1 推荐总体方向

建议不要完全推翻 FallGuard，而是采用分阶段改进：

```text
阶段 1：保留 YOLO-Pose 和 Streamlit
阶段 2：新增骨架序列缓存
阶段 3：建立我们自己的标注评估集
阶段 4：训练一个 fall/non-fall 时序分类器
阶段 5：用模型结果替代或校正规则 fall 判断
阶段 6：保留规则特征作为解释信息
```

这样可以复用 FallGuard 已经有的摄像头、视频上传、可视化和本地隐私处理能力，同时引入本项目的可训练分类思想。

### 9.2 第一阶段：建立评估集

在改算法前，先解决“怎么验证可靠性”。

建议准备本地视频集：

```text
data/eval_videos/
├── falls/
│   ├── fall_001.mp4
│   ├── fall_002.mp4
│   └── ...
└── non_falls/
    ├── sit_001.mp4
    ├── lie_down_001.mp4
    ├── bend_001.mp4
    └── ...
```

每个视频至少记录：

```text
文件名
真实标签：fall 或 non_fall
动作说明：摔倒、坐下、弯腰、躺下、走动等
场景说明：摄像头角度、光照、遮挡、距离
```

如果可能，还要标注：

```text
摔倒开始帧
摔倒结束帧
```

这对应本项目里的 `fall_interval`。

### 9.3 第二阶段：把 YOLO-Pose 输出整理成训练样本

FallGuard 已经能从 YOLO-Pose 得到：

```text
box
confidence
keypoints: 17 x 3
```

可以把连续帧缓存成：

```text
frames x 17 x 3
```

然后按窗口切片：

```text
45 x 17 x 3
```

训练模型时先只用 x/y：

```text
45 x 17 x 2
```

置信度可以后续再加。

### 9.4 第三阶段：先做最小可训练分类器

不要一开始就做复杂光流融合。建议先做：

```text
输入：45 帧 x 17 点 x 2 坐标
输出：fall / non_fall
模型：ST-GCN 或更简单的 TCN/LSTM baseline
评估：accuracy、sensitivity、specificity、false positives、false negatives
```

目标不是马上追求最高准确率，而是建立一条可靠的训练和评估链路。

### 9.5 第四阶段：与现有规则融合

模型接入 FallGuard 后，可以先不要完全替代规则。

更稳的方式是：

```text
规则输出：standing/sitting/moving/lying/fall
模型输出：fall probability
最终决策：
    如果模型强烈认为 fall，则提示 fall
    如果规则认为 fall 但模型不支持，则标为 suspicious 或降低告警等级
    如果关键点质量差，则输出 unknown
```

这样既保留规则的解释性，又引入模型的时序判断能力。

### 9.6 第五阶段：记录错误样本并迭代

每次评估都保存：

```text
误报视频列表
漏报视频列表
对应窗口预测分数
关键点可视化结果
规则特征摘要
```

这会帮助我们回答：

- 是关键点检测错了，还是分类器错了？
- 是摄像头角度问题，还是动作本身容易混淆？
- 是 fall 漏报严重，还是 non-fall 误报严重？

没有这些记录，算法改进会变成盲调。

## 10. 这个项目的限制和注意事项

### 10.1 路径写死

很多 notebook 和脚本中有作者本机路径：

```text
D:\ASH\datasets\...
```

所以这个仓库不能直接无脑运行。要复现实验，需要把路径改成当前仓库相对路径，或重新整理数据目录。

### 10.2 训练入口不统一

项目没有标准化命令：

```text
python train.py
```

训练逻辑散落在多个 notebook 中。后续如果我们借鉴，应整理成脚本化流程：

```text
prepare_dataset.py
train_stgcn.py
evaluate_stgcn.py
```

### 10.3 数据集不等于真实部署场景

Le2i 是公开摔倒数据集，但它的视频场景、摄像头角度、人物动作和我们的实际应用不一定一致。

如果直接拿这里的权重用于 FallGuard，可能出现：

- 场景迁移失败。
- 摄像头角度不匹配。
- 老人动作和实验者动作不同。
- 遮挡和多人场景处理不好。

更合理的用法是借鉴方法，用我们自己的数据继续训练或微调。

### 10.4 模型输出只有二分类

本项目输出 fall/non-fall，不能直接给出：

```text
standing
sitting
moving
lying
unknown
```

如果 FallGuard 仍然需要多状态展示，建议继续保留规则状态模块，把 ST-GCN 作为摔倒风险判断器。

### 10.5 不能只看文件名里的准确率

例如：

```text
le2i_97acc_default.pth
```

看起来准确率很高，但要追问：

- 用哪个测试集？
- 是窗口级还是视频级？
- 是否有数据泄漏？
- 是否跨场景测试？
- sensitivity 和 specificity 分别是多少？
- 误报和漏报是谁？

这也是我们做 FallGuard 改进时要坚持的标准。

## 11. 建议阅读顺序

如果你是视觉小白，建议按这个顺序看代码：

1. `README.md`
   先了解项目目标和技术报告。

2. `datasets/Le2i-train-test/`
   看数据目录，理解 JSON 和 `.npy` 的区别。

3. `st-gcn/utils.py`
   看训练样本如何从完整视频切成 fall/non-fall 片段。

4. `st-gcn/model/st_graph.py`
   看骨架边如何变成邻接矩阵。

5. `st-gcn/model/sgcn.py` 和 `st-gcn/model/tgcn.py`
   分别理解空间关系和时间关系怎么学。

6. `st-gcn/model/stgcn.py`
   看完整模型如何组合。

7. `st-gcn/train_utils.py`
   看 PyTorch 数据集、采样器、训练器和 KFold。

8. `st-gcn/eval.py`
   看如何从窗口预测变成视频级结果，如何统计误报和漏报。

9. `st-gcn/Le2i_train_test.ipynb` 和 `st-gcn/stgcn_results.ipynb`
   看作者怎么做实验和记录结果。

## 12. 一句话总结给 FallGuard

当前 FallGuard 的问题不是“没有检测到人”，而是“摔倒状态由手写姿态规则判断，可靠性难以证明”。

这个项目提供的核心启发是：

```text
继续用姿态关键点作为输入，
但不要只靠单帧角度和速度阈值判断摔倒；
应该把连续骨架序列做成有标签样本，
训练一个可评估的时序分类器，
并用误报/漏报列表持续改进。
```

最适合我们的路线是：

```text
YOLO-Pose 负责提关键点
FallGuard 负责实时输入、展示和本地隐私
新增 ST-GCN/时序分类器负责 fall/non-fall 判断
规则模块继续负责可解释状态和兜底
评估集负责证明可靠性
```

