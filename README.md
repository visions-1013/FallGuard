# FallGuard

FallGuard 是一个 Python 课设演示项目，用预训练 YOLO-Pose 提取人体框和姿态关键点，再通过姿态特征和短时序特征判断基础状态。

## 功能

- 支持本地视频文件和电脑摄像头输入。
- 输出基础状态：`unknown`、`standing`、`sitting`、`moving`、`lying`、`fall`。
- 同屏绘制人体框、姿态关键点、骨架线和状态文字。
- 使用 Streamlit 展示标注画面、当前状态和历史事件。

## Conda 环境

创建并激活新环境：

```powershell
conda env create -f environment.yml
conda activate fallguard
```

如果环境已存在，更新依赖：

```powershell
conda env update -f environment.yml --prune
conda activate fallguard
```

安装当前项目为可编辑包，让 Streamlit 能直接导入 `fallguard`：

```powershell
python -m pip install -e .
```

如果临时不安装包，也可以在当前 PowerShell 会话中使用：

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## 运行测试

```powershell
python -m pytest tests -q
```

## 启动 Streamlit 演示

```powershell
streamlit run app/streamlit_app.py
```

页面中可以选择电脑摄像头，也可以上传本地视频文件。视频不会上传到外部服务，推理和状态判断都在本地完成。
项目会把 Ultralytics 配置写入本地 `.ultralytics/` 目录，避免写入用户 Roaming 目录导致权限问题。

## 项目结构

```text
FallGuard/
├── app/                  # Streamlit 演示页面
├── src/fallguard/        # 核心检测、特征、状态和可视化逻辑
├── tests/                # 姿态特征、时序历史和状态判断测试
└── data/samples/         # 本地测试视频说明
```
