from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_train_stgcn_cloud.ipynb"


def code(source: str, *tags: str):
    cell = new_code_cell(source.strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def markdown(source: str):
    return new_markdown_cell(source.strip())


cells = [
    markdown(
        """
# FallGuard：AutoDL独立YOLO26 Pose + ST-GCN训练

本Notebook不依赖FallGuard项目源码。上传本文件和`datasets/`后，点击“Run All”即可完成：

```text
审核190个视频 → 排除60个无标注视频 → 视频级70/15/15划分
→ YOLO26n-pose骨架缓存 → 官方NTU60 ST-GCN二分类微调
→ val选择best epoch与阈值 → test最终评估 → 导出runtime_bundle
```

Lecture_room和Office没有本地TXT标注，始终排除，绝不根据文件编号推测标签。
"""
    ),
    markdown("## 1. 安装公开依赖与MMAction2官方v1.2.0源码"),
    code(
        """
import subprocess
from pathlib import Path

MMACTION_SOURCE = Path.cwd() / ".deps" / "mmaction2"
if not (MMACTION_SOURCE / ".git").exists():
    MMACTION_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--branch", "v1.2.0",
            "https://github.com/open-mmlab/mmaction2.git", str(MMACTION_SOURCE),
        ],
        check=True,
    )

%pip install -q "mmcv-lite==2.1.0" "mmengine>=0.7.1,<1.0.0" \\
    "ultralytics>=8.4,<9" "scikit-learn>=1.3,<2" \\
    "matplotlib>=3.8,<4" "tensorboard>=2.15,<3" pandas future importlib-metadata
%pip install -q -e {MMACTION_SOURCE}
"""
    ),
    markdown("## 2. 唯一需要检查的配置单元"),
    code(
        """
import csv
import hashlib
import importlib.metadata
import json
import pickle
import platform
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

plt.switch_backend("Agg")

DATA_ROOT = Path.cwd() / "datasets"
OUTPUT_ROOT = Path.cwd() / "outputs/autodl_training"
YOLO_WEIGHT_PATH = None
STGCN_CHECKPOINT_PATH = None
DEVICE = "cuda:0"
SEED = 42
FAST_DEV_RUN = False
FORCE_REEXTRACT = False
AUTO_RESUME = True

if not torch.cuda.is_available():
    raise RuntimeError("正式训练需要CUDA GPU，请在AutoDL选择GPU实例后重新运行")
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"未找到datasets目录: {DATA_ROOT}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
"""
    ),
    markdown("## 3. 纯数据函数：标注、视频划分、重采样、窗口和阈值"),
    code(
        """
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ANNOTATED_SCENES = {"Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"}
UNLABELED_SCENES = {"Lecture_room", "Office"}
EXPECTED_SCENE_COUNTS = {
    "Coffee_room_01": 48,
    "Coffee_room_02": 22,
    "Home_01": 30,
    "Home_02": 30,
    "Lecture_room": 27,
    "Office": 33,
}
COCO17_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
ANNOTATION_OVERRIDES = {
    ("Coffee_room_01", "video (26)"): (197, 227),
    ("Coffee_room_02", "video (50)"): (1816, 1852),
    ("Coffee_room_02", "video (52)"): (87, 113),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotation_text(text: str, scene: str, video_id: str) -> tuple[int, int]:
    override = ANNOTATION_OVERRIDES.get((scene, video_id))
    if override is not None:
        return override
    lines = text.splitlines()
    try:
        start, end = int(lines[0].strip()), int(lines[1].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError(f"标注缺少起止帧: {scene}/{video_id}") from exc
    if start < 0 or end < start:
        raise ValueError(f"非法跌倒区间: {scene}/{video_id} {start}-{end}")
    return start, end


def video_label(start: int, end: int) -> str:
    return "non_fall" if (start, end) == (0, 0) else "fall"


def stratified_video_split(rows: list[dict], seed: int = 42) -> list[dict]:
    if len({row["video_key"] for row in rows}) != len(rows):
        raise ValueError("video_key必须唯一")
    indices = np.arange(len(rows))
    strata = np.asarray([f"{row['scene']}::{row['label']}" for row in rows])
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=strata,
    )
    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        stratify=strata[temporary_indices],
    )
    split_by_index = {
        **{int(index): "train" for index in train_indices},
        **{int(index): "val" for index in val_indices},
        **{int(index): "test" for index in test_indices},
    }
    output = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["split"] = split_by_index[index]
        output.append(item)
    return output


def resample_pose_sequence(
    keypoints: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"期望(T,17,3)，实际{keypoints.shape}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS必须为正数")
    if len(keypoints) <= 1 or source_fps == target_fps:
        return np.asarray(keypoints, dtype=np.float32).copy()
    duration = (len(keypoints) - 1) / source_fps
    output_frames = int(round(duration * target_fps)) + 1
    source_time = np.arange(len(keypoints), dtype=np.float64) / source_fps
    target_time = np.arange(output_frames, dtype=np.float64) / target_fps
    target_time[-1] = duration
    flat = np.asarray(keypoints, dtype=np.float32).reshape(len(keypoints), -1)
    output = np.empty((output_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target_time, source_time, flat[:, column])
    return output.reshape(output_frames, 17, 3)


def interpolate_missing_poses(
    keypoints: np.ndarray,
    max_gap: int = 3,
) -> np.ndarray:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"期望(T,17,3)，实际{keypoints.shape}")
    if max_gap < 0:
        raise ValueError("max_gap不能为负数")
    output = np.asarray(keypoints, dtype=np.float32).copy()
    present = np.any(output[..., 2] > 0, axis=1)
    index = 0
    while index < len(output):
        if present[index]:
            index += 1
            continue
        gap_start = index
        while index < len(output) and not present[index]:
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start
        bounded = gap_start > 0 and gap_end < len(output)
        if bounded and gap_length <= max_gap:
            left = output[gap_start - 1]
            right = output[gap_end]
            for offset in range(1, gap_length + 1):
                alpha = offset / (gap_length + 1)
                output[gap_start + offset - 1] = (1 - alpha) * left + alpha * right
    return output


def make_windows(
    keypoints: np.ndarray,
    window_frames: int = 32,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(
        0,
        max(len(keypoints) - window_frames + 1, 0),
        stride,
        dtype=np.int64,
    )
    if not len(starts):
        return np.empty((0, window_frames, 17, 3), dtype=np.float32), starts
    return np.stack([keypoints[start : start + window_frames] for start in starts]), starts


def build_labeled_windows(
    keypoints: np.ndarray,
    pose_fps: float,
    fall_start_seconds: float | None,
    fall_end_seconds: float | None,
    window_frames: int = 32,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, starts = make_windows(keypoints, window_frames, stride)
    if fall_start_seconds is None or fall_end_seconds is None:
        return windows, np.zeros(len(starts), dtype=np.int64), starts
    start_times = starts / pose_fps
    end_times = (starts + window_frames - 1) / pose_fps
    labels = np.full(len(starts), -1, dtype=np.int64)
    positive = (end_times >= fall_start_seconds + 0.4) & (
        end_times <= fall_end_seconds + 0.4
    )
    negative = (end_times < fall_start_seconds) | (
        start_times > fall_end_seconds + 1.0
    )
    labels[positive] = 1
    labels[negative] = 0
    keep = labels >= 0
    return windows[keep], labels[keep], starts[keep]


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_precision: float = 0.85,
) -> dict[str, float]:
    candidates = []
    for threshold in np.unique(probabilities):
        predictions = (probabilities >= threshold).astype(np.int64)
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
            }
        )
    eligible = [item for item in candidates if item["precision"] >= min_precision]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda item: (
            item["recall"], item["precision"], item["f1"], -item["threshold"]
        ),
    )


def compute_class_weights(labels: list[int] | np.ndarray) -> list[float]:
    values = np.asarray(labels, dtype=np.int64)
    counts = [int(np.sum(values == 0)), int(np.sum(values == 1))]
    if not all(counts):
        raise ValueError(f"训练集必须同时包含两类，实际counts={counts}")
    total = sum(counts)
    return [total / (2 * count) for count in counts]


def cache_metadata_matches(
    metadata: dict,
    row: dict,
    model_hash: str,
    source_hash: str,
    source_size: int,
    source_mtime_ns: int,
) -> bool:
    return bool(
        metadata.get("model_sha256") == model_hash
        and metadata.get("source_sha256") == source_hash
        and metadata.get("source_size") == source_size
        and metadata.get("source_mtime_ns") == source_mtime_ns
        and metadata.get("width") == int(row["width"])
        and metadata.get("height") == int(row["height"])
        and np.isclose(float(metadata.get("fps", -1.0)), float(row["fps"]))
    )
""",
        "unit-test",
    ),
    markdown("## 4. 审核190个视频、排除60个无标注视频并固定划分"),
    code(
        """
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
SPLIT_MANIFEST_PATH = OUTPUT_ROOT / "split_manifest.csv"
EXCLUDED_PATH = OUTPUT_ROOT / "excluded_unlabeled_videos.csv"
POSE_ROOT = OUTPUT_ROOT / "poses"
PROCESSED_PATH = OUTPUT_ROOT / "processed" / "le2i_stgcn.pkl"
WORK_DIR = OUTPUT_ROOT / "work_dir"
REPORT_ROOT = OUTPUT_ROOT / "reports"


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {path}")
    try:
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def build_manifest(dataset_root: Path) -> list[dict]:
    videos = sorted(
        [
            path for path in dataset_root.rglob("*.avi")
            if "Le2i-train-test" not in path.parts
        ],
        key=lambda path: str(path.relative_to(dataset_root)),
    )
    rows = []
    for video_path in videos:
        scene = video_path.relative_to(dataset_root).parts[0]
        video_id = video_path.stem
        annotations = list((dataset_root / scene).rglob(f"{video_id}.txt"))
        annotation_path = annotations[0] if annotations else None
        metadata = probe_video(video_path)
        if annotation_path is None:
            label = "unlabeled"
            start = end = None
        else:
            start, end = parse_annotation_text(
                annotation_path.read_text(encoding="utf-8", errors="replace"),
                scene,
                video_id,
            )
            label = video_label(start, end)
        rows.append(
            {
                "video_key": f"{scene}/{video_id}",
                "scene": scene,
                "video_id": video_id,
                "video_path": str(video_path.resolve()),
                "annotation_path": str(annotation_path.resolve()) if annotation_path else "",
                "label": label,
                "fall_start": start,
                "fall_end": end,
                **metadata,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不能写入空CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


manifest_rows = build_manifest(DATA_ROOT)
supervised_rows = [row for row in manifest_rows if row["label"] != "unlabeled"]
excluded_rows = [row for row in manifest_rows if row["label"] == "unlabeled"]
scene_counts = {
    scene: sum(row["scene"] == scene for row in manifest_rows)
    for scene in EXPECTED_SCENE_COUNTS
}

if len(manifest_rows) != 190:
    raise ValueError(f"期望190个AVI，实际{len(manifest_rows)}个")
if scene_counts != EXPECTED_SCENE_COUNTS:
    raise ValueError(f"六场景视频数量异常: {scene_counts}")
if len(supervised_rows) != 130 or len(excluded_rows) != 60:
    raise ValueError(
        f"期望130个有标注、60个无标注，实际{len(supervised_rows)}和{len(excluded_rows)}"
    )
if {row["scene"] for row in excluded_rows} != UNLABELED_SCENES:
    raise ValueError("无标注视频必须全部来自Lecture_room和Office")

write_csv(MANIFEST_PATH, manifest_rows)
write_csv(EXCLUDED_PATH, excluded_rows)

if SPLIT_MANIFEST_PATH.exists():
    with SPLIT_MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    saved_splits = {row["video_key"]: row["split"] for row in saved_rows}
    current_keys = {row["video_key"] for row in supervised_rows}
    if set(saved_splits) != current_keys:
        raise ValueError("现有split_manifest与当前130个标注视频不一致，请人工确认后删除重建")
    split_rows = [{**row, "split": saved_splits[row["video_key"]]} for row in supervised_rows]
else:
    split_rows = stratified_video_split(supervised_rows, seed=SEED)
    write_csv(SPLIT_MANIFEST_PATH, split_rows)

split_counts = {
    name: sum(row["split"] == name for row in split_rows)
    for name in ("train", "val", "test")
}
if split_counts != {"train": 91, "val": 19, "test": 20}:
    raise ValueError(f"固定划分数量异常: {split_counts}")

active_rows = split_rows
if FAST_DEV_RUN:
    active_rows = []
    for split_name in ("train", "val", "test"):
        active_rows.extend([row for row in split_rows if row["split"] == split_name][:2])

print("全部视频:", len(manifest_rows))
print("监督视频:", len(supervised_rows))
print("排除视频:", len(excluded_rows), UNLABELED_SCENES)
print("划分:", split_counts)
"""
    ),
    markdown("## 5. YOLO26n-pose单人骨架提取与可恢复缓存"),
    code(
        """
from ultralytics import YOLO


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)
    left_area = max(float(left[2] - left[0]), 0.0) * max(float(left[3] - left[1]), 0.0)
    right_area = max(float(right[2] - right[0]), 0.0) * max(float(right[3] - right[1]), 0.0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def select_primary_person(
    boxes: np.ndarray,
    scores: np.ndarray,
    previous_box: np.ndarray | None,
) -> int | None:
    if not len(boxes):
        return None
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    ranking = scores + 0.5 * areas / max(float(areas.max()), 1.0)
    if previous_box is not None:
        ranking += np.asarray([2.0 * box_iou(box, previous_box) for box in boxes])
    return int(np.argmax(ranking))


class YoloPoseExtractor:
    def __init__(self, model_ref: str, device: str) -> None:
        self.model = YOLO(model_ref)
        self.device = device
        self.previous_box = None

    def reset(self) -> None:
        self.previous_box = None

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        result = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,
            conf=0.25,
        )[0]
        if result.boxes is None or result.keypoints is None:
            return None
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()
        index = select_primary_person(boxes, scores, self.previous_box)
        if index is None:
            return None
        self.previous_box = boxes[index].astype(np.float32)
        return keypoints[index].astype(np.float32), self.previous_box


def load_cache_metadata(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keypoints = payload["keypoints"]
            boxes = payload["boxes"]
            metadata = json.loads(str(payload["metadata"].item()))
            if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
                return None
            if boxes.shape != (len(keypoints), 4):
                return None
            if int(metadata.get("frames", -1)) != len(keypoints):
                return None
            return metadata
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def cache_is_valid(path: Path, row: dict, model_hash: str) -> bool:
    if FORCE_REEXTRACT or not path.is_file():
        return False
    metadata = load_cache_metadata(path)
    if metadata is None:
        return False
    video_path = Path(row["video_path"])
    stat = video_path.stat()
    return cache_metadata_matches(
        metadata,
        row,
        model_hash=model_hash,
        source_hash=sha256_file(video_path),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def extract_video_to_cache(
    row: dict,
    output_path: Path,
    extractor: YoloPoseExtractor,
    model_name: str,
    model_hash: str,
) -> None:
    video_path = Path(row["video_path"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    extractor.reset()
    keypoints = []
    boxes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pose = extractor.extract(frame)
            if pose is None:
                keypoints.append(np.zeros((17, 3), dtype=np.float32))
                boxes.append(np.zeros(4, dtype=np.float32))
            else:
                points, box = pose
                keypoints.append(points)
                boxes.append(box)
    finally:
        capture.release()
    stat = video_path.stat()
    metadata = {
        "source_path": str(video_path),
        "source_sha256": sha256_file(video_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "model_name": model_name,
        "model_sha256": model_hash,
        "width": int(row["width"]),
        "height": int(row["height"]),
        "fps": float(row["fps"]),
        "frames": len(keypoints),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        keypoints=np.asarray(keypoints, dtype=np.float32),
        boxes=np.asarray(boxes, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


pose_model_ref = str(YOLO_WEIGHT_PATH or "yolo26n-pose.pt")
extractor = YoloPoseExtractor(pose_model_ref, DEVICE)
resolved_weight = Path(getattr(extractor.model, "ckpt_path", pose_model_ref))
pose_model_hash = (
    sha256_file(resolved_weight) if resolved_weight.is_file() else "ultralytics-managed"
)

for index, row in enumerate(active_rows, start=1):
    cache_path = POSE_ROOT / row["scene"] / f"{row['video_id']}.npz"
    video_path = Path(row["video_path"])
    if not cache_is_valid(cache_path, row, pose_model_hash):
        extract_video_to_cache(
            row,
            cache_path,
            extractor,
            "yolo26n-pose.pt",
            pose_model_hash,
        )
    print(f"[{index}/{len(active_rows)}] {cache_path}")
"""
    ),
    markdown("## 6. 生成MMAction2 PoseDataset；测试集保持只读"),
    code(
        """
POSE_FPS = 20.0
WINDOW_FRAMES = 32
WINDOW_STRIDE = 4


def load_pose_cache(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["keypoints"].astype(np.float32), json.loads(
            str(payload["metadata"].item())
        )


annotations = []
split_ids = {"train": [], "val": [], "test": []}
for row in active_rows:
    cache_path = POSE_ROOT / row["scene"] / f"{row['video_id']}.npz"
    poses, cache_metadata = load_pose_cache(cache_path)
    poses = interpolate_missing_poses(poses, max_gap=3)
    poses = resample_pose_sequence(poses, float(cache_metadata["fps"]), POSE_FPS)
    if row["label"] == "fall":
        fall_start_seconds = float(row["fall_start"]) / float(row["fps"])
        fall_end_seconds = float(row["fall_end"]) / float(row["fps"])
    else:
        fall_start_seconds = fall_end_seconds = None
    windows, labels, starts = build_labeled_windows(
        poses,
        POSE_FPS,
        fall_start_seconds,
        fall_end_seconds,
        WINDOW_FRAMES,
        WINDOW_STRIDE,
    )
    for window, label, start in zip(windows, labels, starts, strict=True):
        sample_id = f"{row['video_key']}/{int(start)}"
        split_ids[row["split"]].append(sample_id)
        annotations.append(
            {
                "frame_dir": sample_id,
                "video_key": row["video_key"],
                "scene": row["scene"],
                "label": int(label),
                "img_shape": (int(row["height"]), int(row["width"])),
                "original_shape": (int(row["height"]), int(row["width"])),
                "total_frames": WINDOW_FRAMES,
                "window_start": int(start),
                "keypoint": window[None, ..., :2].astype(np.float32),
                "keypoint_score": window[None, ..., 2].astype(np.float32),
            }
        )

video_sets = {
    name: {row["video_key"] for row in active_rows if row["split"] == name}
    for name in ("train", "val", "test")
}
split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
if any(video_sets[left] & video_sets[right] for left, right in split_pairs):
    raise RuntimeError("检测到视频跨集合泄漏")

train_id_set = set(split_ids["train"])
train_labels = [item["label"] for item in annotations if item["frame_dir"] in train_id_set]
CLASS_WEIGHTS = compute_class_weights(train_labels)

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROCESSED_PATH.open("wb") as handle:
    pickle.dump({"split": split_ids, "annotations": annotations}, handle)

print("窗口数:", {name: len(values) for name, values in split_ids.items()})
print("训练集类别权重:", CLASS_WEIGHTS)
print("PoseDataset:", PROCESSED_PATH)
"""
    ),
    markdown("## 7. 内置轻量COCO-17 ST-GCN，用于评估与本地runtime导出"),
    code(
        """
from torch import nn


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=0)
    inverse = np.zeros_like(adjacency)
    for index, value in enumerate(degree):
        if value > 0:
            inverse[index, index] = value**-1
    return adjacency @ inverse


def hop_distance(num_nodes: int, edges: list[tuple[int, int]], max_hop: int) -> np.ndarray:
    adjacency = np.eye(num_nodes)
    for left, right in edges:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    distance = np.full((num_nodes, num_nodes), np.inf)
    reachability = np.stack(
        [np.linalg.matrix_power(adjacency, hop) > 0 for hop in range(max_hop + 1)]
    )
    for hop in range(max_hop, -1, -1):
        distance[reachability[hop]] = hop
    return distance


class CocoGraph:
    num_nodes = 17
    center = 0
    inward = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
        (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
        (1, 0), (3, 1), (2, 0), (4, 2),
    ]

    def __init__(self, max_hop: int = 1) -> None:
        self.max_hop = max_hop
        self.hop_distance = hop_distance(self.num_nodes, self.inward, max_hop)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        adjacency[self.hop_distance <= max_hop] = 1
        normalized = normalize_digraph(adjacency)
        partitions = []
        for hop in range(max_hop + 1):
            close = np.zeros_like(normalized)
            further = np.zeros_like(normalized)
            for source in range(self.num_nodes):
                for target in range(self.num_nodes):
                    if self.hop_distance[target, source] != hop:
                        continue
                    target_hop = self.hop_distance[target, self.center]
                    source_hop = self.hop_distance[source, self.center]
                    if target_hop >= source_hop:
                        close[target, source] = normalized[target, source]
                    else:
                        further[target, source] = normalized[target, source]
            partitions.append(close)
            if hop > 0:
                partitions.append(further)
        self.A = np.stack(partitions).astype(np.float32)


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.num_subsets = adjacency.size(0)
        self.register_buffer("A", adjacency)
        self.PA = nn.Parameter(torch.ones_like(adjacency))
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, frames, vertices = inputs.shape
        adjacency = self.A * self.PA
        features = self.conv(inputs).view(batch, self.num_subsets, -1, frames, vertices)
        features = torch.einsum("nkctv,kvw->nctw", features, adjacency).contiguous()
        return self.act(self.bn(features))


class UnitTCN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int = 9, stride: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(padding, 0), stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(inputs))


class STGCNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, adjacency: torch.Tensor,
        stride: int = 1, residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(inputs)
        return self.relu(self.tcn(self.gcn(inputs)) + residual)


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        adjacency = torch.tensor(CocoGraph().A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        channels = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        blocks = []
        current = in_channels
        for stage, output in enumerate(channels, start=1):
            blocks.append(
                STGCNBlock(
                    current, output, adjacency.clone(),
                    stride=2 if stage in (5, 8) else 1,
                    residual=stage != 1,
                )
            )
            current = output
        self.gcn = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, people, frames, vertices, channels = inputs.shape
        features = inputs.permute(0, 1, 3, 4, 2).contiguous()
        features = self.data_bn(features.view(batch * people, vertices * channels, frames))
        features = (
            features.view(batch, people, vertices, channels, frames)
            .permute(0, 1, 3, 4, 2).contiguous()
            .view(batch * people, channels, frames, vertices)
        )
        for block in self.gcn:
            features = block(features)
        return features.reshape((batch, people) + features.shape[1:])


class STGCNClassifier(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = STGCNBackbone()
        self.head = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.head(features.mean(dim=(1, 3, 4)))


def load_mmaction_classifier(model: STGCNClassifier, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    result = model.backbone.load_state_dict(backbone_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"骨干权重不兼容: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    head_prefix = next(
        (
            prefix for prefix in ("cls_head.fc", "cls_head.fc_cls")
            if f"{prefix}.weight" in state and f"{prefix}.bias" in state
        ),
        None,
    )
    if head_prefix is None:
        raise KeyError("checkpoint缺少MMAction2 GCNHead二分类权重")
    model.head.load_state_dict(
        {
            "weight": state[f"{head_prefix}.weight"],
            "bias": state[f"{head_prefix}.bias"],
        }
    )


def pre_normalize_2d(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.asarray(keypoints, dtype=np.float32).copy()
    output[..., 0] = (output[..., 0] - width / 2) / (width / 2)
    output[..., 1] = (output[..., 1] - height / 2) / (height / 2)
    return output
""",
        "unit-test-model",
    ),
    markdown("## 8. 官方NTU60 ST-GCN微调：只训练pretrained路线"),
    code(
        """
from mmaction.registry import HOOKS, METRICS
from mmaction.utils import register_all_modules
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.runner import Runner


OFFICIAL_STGCN_2D_JOINT = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/"
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
    "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
)


@HOOKS.register_module(force=True)
class NotebookFreezeBackboneHook(Hook):
    def __init__(self, freeze_epochs: int = 5) -> None:
        self.freeze_epochs = freeze_epochs

    def before_train_epoch(self, runner) -> None:
        freeze = runner.epoch < self.freeze_epochs
        for parameter in runner.model.backbone.parameters():
            parameter.requires_grad = not freeze


@METRICS.register_module(force=True)
class NotebookBinaryMetric(BaseMetric):
    default_prefix = "binary"

    def __init__(self, threshold: float = 0.5, collect_device: str = "cpu") -> None:
        super().__init__(collect_device=collect_device)
        self.threshold = threshold

    def process(self, data_batch, data_samples) -> None:
        del data_batch
        for sample in data_samples:
            score = sample["pred_score"].detach().cpu().numpy()
            self.results.append(
                {
                    "fall_probability": float(score[1]),
                    "label": int(sample["gt_label"].item()),
                }
            )

    def compute_metrics(self, results) -> dict[str, float]:
        labels = np.asarray([item["label"] for item in results], dtype=np.int64)
        predictions = np.asarray(
            [item["fall_probability"] >= self.threshold for item in results],
            dtype=np.int64,
        )
        return {
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }


register_all_modules(init_default_scope=True)
pretrained_checkpoint = str(STGCN_CHECKPOINT_PATH or OFFICIAL_STGCN_2D_JOINT)
max_epochs = 2 if FAST_DEV_RUN else 45

dataset_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]

config = Config(
    dict(
        work_dir=str(WORK_DIR),
        default_scope="mmaction",
        model=dict(
            type="RecognizerGCN",
            backbone=dict(
                type="STGCN",
                graph_cfg=dict(layout="coco", mode="stgcn_spatial"),
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint=pretrained_checkpoint,
                    prefix="backbone.",
                ),
            ),
            cls_head=dict(
                type="GCNHead",
                num_classes=2,
                in_channels=256,
                loss_cls=dict(type="CrossEntropyLoss", class_weight=CLASS_WEIGHTS),
            ),
        ),
        train_dataloader=dict(
            batch_size=64,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=dict(
                type="PoseDataset",
                ann_file=str(PROCESSED_PATH),
                pipeline=dataset_pipeline,
                split="train",
            ),
        ),
        val_dataloader=dict(
            batch_size=64,
            num_workers=2,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                type="PoseDataset",
                ann_file=str(PROCESSED_PATH),
                pipeline=dataset_pipeline,
                split="val",
                test_mode=True,
            ),
        ),
        val_evaluator=[dict(type="NotebookBinaryMetric", threshold=0.5)],
        train_cfg=dict(
            type="EpochBasedTrainLoop", max_epochs=max_epochs,
            val_begin=1, val_interval=1,
        ),
        val_cfg=dict(type="ValLoop"),
        optim_wrapper=dict(
            type="AmpOptimWrapper",
            optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
            paramwise_cfg=dict(custom_keys={"cls_head": dict(lr_mult=3.0)}),
        ),
        param_scheduler=[
            dict(
                type="CosineAnnealingLR", T_max=max_epochs,
                eta_min=1e-6, by_epoch=True,
            )
        ],
        custom_hooks=[
            dict(type="NotebookFreezeBackboneHook", freeze_epochs=5),
            dict(
                type="EarlyStoppingHook", monitor="binary/f1",
                rule="greater", patience=8, strict=True,
            ),
        ],
        default_hooks=dict(
            checkpoint=dict(
                type="CheckpointHook", interval=1,
                save_best="binary/f1", rule="greater", max_keep_ckpts=3,
            ),
            logger=dict(type="LoggerHook", interval=20),
        ),
        visualizer=dict(
            type="ActionVisualizer",
            vis_backends=[
                dict(type="LocalVisBackend"),
                dict(type="TensorboardVisBackend"),
            ],
        ),
        log_processor=dict(type="LogProcessor", window_size=20, by_epoch=True),
        randomness=dict(seed=SEED, deterministic=True),
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
            dist_cfg=dict(backend="nccl"),
        ),
        resume=bool(AUTO_RESUME and (WORK_DIR / "last_checkpoint").is_file()),
    )
)

WORK_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "resolved_config.py").write_text(config.pretty_text, encoding="utf-8")
Runner.from_cfg(config).train()

best_candidates = sorted(
    WORK_DIR.glob("best_*.pth"), key=lambda path: path.stat().st_mtime_ns
)
if not best_candidates:
    best_candidates = sorted(
        WORK_DIR.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime_ns
    )
if not best_candidates:
    raise FileNotFoundError(f"训练结束但未找到checkpoint: {WORK_DIR}")
BEST_CHECKPOINT = best_candidates[-1]
checkpoint_meta = torch.load(
    BEST_CHECKPOINT, map_location="cpu", weights_only=True
).get("meta", {})
BEST_EPOCH = int(checkpoint_meta.get("epoch", -1))
BEST_MMACTION = OUTPUT_ROOT / "best_mmaction.pth"
shutil.copy2(BEST_CHECKPOINT, BEST_MMACTION)
print("Best checkpoint:", BEST_CHECKPOINT, "epoch:", BEST_EPOCH)
"""
    ),
    markdown("## 9. val锁定阈值；test只执行最终一次评估"),
    code(
        """
def predict_annotations(
    model: STGCNClassifier,
    selected_annotations: list[dict],
    device: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([item["label"] for item in selected_annotations], dtype=np.int64)
    inputs = []
    for item in selected_annotations:
        points = np.concatenate(
            [item["keypoint"], item["keypoint_score"][..., None]], axis=-1
        ).astype(np.float32)
        height, width = item["img_shape"]
        inputs.append(pre_normalize_2d(points, width, height))
    stacked = np.stack(inputs)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(stacked), batch_size):
            batch = torch.from_numpy(stacked[start : start + batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return labels, np.concatenate(probabilities)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def write_binary_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_rows: list[dict],
    threshold: float,
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = binary_metrics(labels, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(np.int64)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    prediction_rows = []
    for item, label, probability, prediction in zip(
        sample_rows, labels, probabilities, predictions, strict=True
    ):
        prediction_rows.append(
            {
                "sample_id": item["frame_dir"],
                "video_key": item["video_key"],
                "scene": item["scene"],
                "label": int(label),
                "prediction": int(prediction),
                "fall_probability": float(probability),
                "error_type": (
                    "" if label == prediction
                    else "false_positive" if prediction else "false_negative"
                ),
            }
        )
    write_csv(output_dir / "predictions.csv", prediction_rows)
    errors = [row for row in prediction_rows if row["error_type"]]
    if errors:
        write_csv(output_dir / "errors.csv", errors)
    else:
        (output_dir / "errors.csv").write_text(
            ",".join(prediction_rows[0]) + "\\n", encoding="utf-8"
        )

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=["non_fall", "fall"]).plot(
        cmap="Blues", colorbar=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    threshold_rows = []
    for value in np.linspace(0.0, 1.0, 101):
        prediction = (probabilities >= value).astype(np.int64)
        threshold_rows.append(
            {
                "threshold": float(value),
                "precision": float(precision_score(labels, prediction, zero_division=0)),
                "recall": float(recall_score(labels, prediction, zero_division=0)),
                "f1": float(f1_score(labels, prediction, zero_division=0)),
            }
        )
    thresholds = [row["threshold"] for row in threshold_rows]
    plt.plot(thresholds, [row["precision"] for row in threshold_rows], label="Precision")
    plt.plot(thresholds, [row["recall"] for row in threshold_rows], label="Recall")
    plt.plot(thresholds, [row["f1"] for row in threshold_rows], label="F1")
    plt.axvline(threshold, linestyle="--", color="black", label="Selected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curve.png", dpi=160)
    plt.close()
    return metrics


runtime_model = STGCNClassifier(num_classes=2).to(DEVICE)
load_mmaction_classifier(runtime_model, BEST_CHECKPOINT)

with PROCESSED_PATH.open("rb") as handle:
    dataset_payload = pickle.load(handle)
annotation_by_id = {item["frame_dir"]: item for item in dataset_payload["annotations"]}
val_rows = [annotation_by_id[item] for item in dataset_payload["split"]["val"]]
test_rows = [annotation_by_id[item] for item in dataset_payload["split"]["test"]]

val_labels, val_probabilities = predict_annotations(runtime_model, val_rows, DEVICE)
threshold_selection = select_threshold(val_labels, val_probabilities, min_precision=0.85)
LOCKED_THRESHOLD = threshold_selection["threshold"]
validation_metrics = write_binary_report(
    val_labels,
    val_probabilities,
    val_rows,
    LOCKED_THRESHOLD,
    REPORT_ROOT / "validation",
)
(REPORT_ROOT / "validation" / "threshold_selection.json").write_text(
    json.dumps(threshold_selection, indent=2), encoding="utf-8"
)

# test在阈值锁定后只执行这一次；不得回写训练配置、checkpoint或阈值。
test_labels, test_probabilities = predict_annotations(runtime_model, test_rows, DEVICE)
test_metrics = write_binary_report(
    test_labels,
    test_probabilities,
    test_rows,
    LOCKED_THRESHOLD,
    REPORT_ROOT / "test",
)

print("Validation:", validation_metrics)
print("Locked threshold:", LOCKED_THRESHOLD)
print("Test window metrics:", test_metrics)
"""
    ),
    markdown("## 10. test视频事件评估、分场景指标、误报/小时与P95延迟"),
    code(
        """
class FallEventStateMachine:
    def __init__(
        self,
        fall_threshold: float,
        recovery_threshold: float = 0.35,
        trigger_windows: int = 2,
        recovery_seconds: float = 2.0,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self.fall_threshold = fall_threshold
        self.recovery_threshold = recovery_threshold
        self.trigger_windows = trigger_windows
        self.recovery_seconds = recovery_seconds
        self.cooldown_seconds = cooldown_seconds
        self.active = None
        self.high_count = 0
        self.first_high_time = None
        self.recovery_started = None
        self.cooldown_until = float("-inf")

    def update(self, timestamp: float, probability: float) -> list[dict]:
        completed = []
        if self.active is not None:
            self.active["max_probability"] = max(
                self.active["max_probability"], probability
            )
            if probability < self.recovery_threshold:
                if self.recovery_started is None:
                    self.recovery_started = timestamp
                if timestamp - self.recovery_started >= self.recovery_seconds:
                    self.active["end_time"] = timestamp
                    completed.append(self.active)
                    self.active = None
                    self.cooldown_until = timestamp + self.cooldown_seconds
                    self.recovery_started = None
            else:
                self.recovery_started = None
            return completed
        if timestamp < self.cooldown_until:
            return completed
        if probability >= self.fall_threshold:
            if self.high_count == 0:
                self.first_high_time = timestamp
            self.high_count += 1
            if self.high_count >= self.trigger_windows:
                self.active = {
                    "start_time": self.first_high_time,
                    "trigger_time": timestamp,
                    "end_time": None,
                    "max_probability": probability,
                }
                self.high_count = 0
                self.first_high_time = None
        else:
            self.high_count = 0
            self.first_high_time = None
        return completed

    def finish(self, timestamp: float) -> list[dict]:
        if self.active is None:
            return []
        self.active["end_time"] = timestamp
        event = self.active
        self.active = None
        return [event]


def predict_video_probabilities(row: dict) -> tuple[np.ndarray, np.ndarray]:
    cache_path = POSE_ROOT / row["scene"] / f"{row['video_id']}.npz"
    poses, metadata = load_pose_cache(cache_path)
    poses = resample_pose_sequence(poses, float(metadata["fps"]), POSE_FPS)
    windows, starts = make_windows(poses, WINDOW_FRAMES, WINDOW_STRIDE)
    if not len(windows):
        return np.empty(0, dtype=np.float32), starts
    normalized = pre_normalize_2d(
        windows,
        int(row["width"]),
        int(row["height"]),
    )[:, None]
    probabilities = []
    runtime_model.eval()
    with torch.no_grad():
        for begin in range(0, len(normalized), 64):
            inputs = torch.from_numpy(normalized[begin : begin + 64]).to(DEVICE)
            probabilities.append(
                torch.softmax(runtime_model(inputs), dim=1)[:, 1].cpu().numpy()
            )
    return np.concatenate(probabilities), starts


test_video_rows = [row for row in active_rows if row["split"] == "test"]
video_predictions = []
alert_delays = []
true_positive_events = 0
false_alarm_events = 0
negative_seconds = 0.0

for row in test_video_rows:
    probabilities, starts = predict_video_probabilities(row)
    machine = FallEventStateMachine(LOCKED_THRESHOLD)
    events = []
    for start, probability in zip(starts, probabilities, strict=True):
        timestamp = (int(start) + WINDOW_FRAMES - 1) / POSE_FPS
        events.extend(machine.update(timestamp, float(probability)))
    duration = float(row["frames"]) / float(row["fps"])
    events.extend(machine.finish(duration))

    if row["label"] == "fall":
        fall_start = float(row["fall_start"]) / float(row["fps"])
        fall_end = float(row["fall_end"]) / float(row["fps"])
        matched = [
            event for event in events
            if fall_start <= event["trigger_time"] <= fall_end + 2.0
        ]
        prediction = int(bool(matched))
        true_positive_events += prediction
        false_alarm_events += len(events) - prediction
        negative_seconds += max(duration - (fall_end - fall_start), 0.0)
        first_trigger = min(
            (event["trigger_time"] for event in matched), default=fall_start,
        )
        delay = max(first_trigger - fall_start, 0.0)
        if matched:
            alert_delays.append(delay)
    else:
        prediction = int(bool(events))
        false_alarm_events += len(events)
        negative_seconds += duration
        delay = None

    video_predictions.append(
        {
            "video_key": row["video_key"],
            "scene": row["scene"],
            "label": int(row["label"] == "fall"),
            "prediction": prediction,
            "event_count": len(events),
            "max_fall_probability": float(probabilities.max(initial=0.0)),
            "alert_delay_seconds": delay if delay is not None else "",
            "duration_seconds": duration,
        }
    )

video_labels = np.asarray([row["label"] for row in video_predictions], dtype=np.int64)
video_outputs = np.asarray([row["prediction"] for row in video_predictions], dtype=np.int64)
fall_video_count = int(np.sum(video_labels == 1))
event_recall = true_positive_events / max(fall_video_count, 1)
event_precision = true_positive_events / max(true_positive_events + false_alarm_events, 1)
false_alarms_per_hour = false_alarm_events / max(negative_seconds / 3600, 1e-12)
p95_delay = float(np.percentile(alert_delays, 95)) if alert_delays else None

event_metrics = {
    "event_precision": event_precision,
    "event_recall": event_recall,
    "false_alarm_events": false_alarm_events,
    "false_alarms_per_hour": false_alarms_per_hour,
    "p95_alert_delay_seconds": p95_delay,
}

test_report_dir = REPORT_ROOT / "test"
write_csv(test_report_dir / "video_predictions.csv", video_predictions)
video_errors = [
    row for row in video_predictions if row["label"] != row["prediction"]
]
if video_errors:
    write_csv(test_report_dir / "video_errors.csv", video_errors)
else:
    (test_report_dir / "video_errors.csv").write_text(
        ",".join(video_predictions[0]) + "\\n", encoding="utf-8"
    )
(test_report_dir / "event_metrics.json").write_text(
    json.dumps(event_metrics, indent=2, sort_keys=True), encoding="utf-8"
)

scene_rows = []
for scene in sorted({row["scene"] for row in video_predictions}):
    selected = [row for row in video_predictions if row["scene"] == scene]
    labels = np.asarray([row["label"] for row in selected], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in selected], dtype=np.int64)
    scene_rows.append(
        {
            "scene": scene,
            "videos": len(selected),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
    )
write_csv(test_report_dir / "scene_metrics.csv", scene_rows)
(test_report_dir / "scene_metrics.json").write_text(
    json.dumps(scene_rows, indent=2, ensure_ascii=False), encoding="utf-8"
)

if alert_delays:
    plt.hist(alert_delays, bins=min(10, len(alert_delays)))
    plt.xlabel("Alert delay (seconds)")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(test_report_dir / "alert_delay_distribution.png", dpi=160)
    plt.close()

DEPLOYMENT_GATE_PASSED = bool(
    event_recall >= 0.90
    and event_precision >= 0.85
    and false_alarms_per_hour <= 1.0
    and p95_delay is not None
    and p95_delay <= 2.0
)
print("Test event metrics:", event_metrics)
print("Deployment gate passed:", DEPLOYMENT_GATE_PASSED)
"""
    ),
    markdown("## 11. 训练曲线、溯源信息、runtime bundle和最终压缩包"),
    code(
        """
def read_mmengine_history(path: Path) -> list[dict[str, float]]:
    epochs = {}
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        validation_line = any(
            key.endswith(("/precision", "/recall", "/f1")) for key in item
        )
        epoch = (
            int(item["epoch"])
            if "epoch" in item
            else int(item.get("step", 0)) + int(validation_line)
        )
        row = epochs.setdefault(epoch, {"epoch": float(epoch)})
        for source, target in {
            "train/loss": "train_loss",
            "train/lr": "lr",
            "lr": "lr",
        }.items():
            if source in item:
                row[target] = float(item[source])
        for metric in ("precision", "recall", "f1"):
            source = next((key for key in item if key.endswith(f"/{metric}")), None)
            if source:
                row[metric] = float(item[source])
    return [row for _, row in sorted(epochs.items())]


history = read_mmengine_history(WORK_DIR / "vis_data" / "scalars.json")
if history:
    fields = ["epoch"] + sorted({field for row in history for field in row if field != "epoch"})
    with (REPORT_ROOT / "validation" / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [row.get("train_loss", np.nan) for row in history])
    axes[0].set_title("Train loss")
    for metric in ("precision", "recall", "f1"):
        axes[1].plot(
            epochs,
            [row.get(metric, np.nan) for row in history],
            label=metric.title(),
        )
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[2].plot(epochs, [row.get("lr", np.nan) for row in history])
    axes[2].set_title("Learning rate")
    figure.tight_layout()
    figure.savefig(REPORT_ROOT / "validation" / "training_curves.png", dpi=160)
    plt.close(figure)

RUNTIME_DIR = OUTPUT_ROOT / "runtime_bundle"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
runtime_weights = RUNTIME_DIR / "best.pt"
torch.save({"state_dict": runtime_model.cpu().state_dict()}, runtime_weights)

package_versions = {}
for package in ("torch", "ultralytics", "mmaction2", "mmengine", "mmcv-lite"):
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = "not-installed"

metadata = {
    "architecture": "stgcn-coco17",
    "classes": ["non_fall", "fall"],
    "fall_threshold": LOCKED_THRESHOLD,
    "recovery_threshold": 0.35,
    "pose_fps": POSE_FPS,
    "window_frames": WINDOW_FRAMES,
    "window_stride": WINDOW_STRIDE,
    "trigger_windows": 2,
    "recovery_seconds": 2.0,
    "cooldown_seconds": 10.0,
    "passed_deployment_gate": DEPLOYMENT_GATE_PASSED,
    "weights_sha256": sha256_file(runtime_weights),
    "source_checkpoint": str(BEST_CHECKPOINT),
    "source_checkpoint_sha256": sha256_file(BEST_CHECKPOINT),
}
(RUNTIME_DIR / "model_meta.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
)

provenance = {
    "seed": SEED,
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "split_unit": "video",
    "stratified_by": ["scene", "video_label"],
    "supervised_videos": 130,
    "excluded_unlabeled_videos": 60,
    "unlabeled_scenes": sorted(UNLABELED_SCENES),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": package_versions,
    "best_epoch": BEST_EPOCH,
    "best_checkpoint": str(BEST_CHECKPOINT),
    "best_checkpoint_sha256": sha256_file(BEST_CHECKPOINT),
    "yolo_weight_sha256": pose_model_hash,
    "test_was_used_for_selection": False,
    "training_config": {
        "route": "official_ntu60_2d_joint_pretrained_only",
        "freeze_backbone_epochs": 5,
        "max_epochs": max_epochs,
        "early_stopping_patience": 8,
        "optimizer": "AdamW",
        "backbone_learning_rate": 1e-4,
        "classifier_learning_rate": 3e-4,
        "scheduler": "CosineAnnealingLR",
        "amp": True,
        "class_weights_from": "train_windows_only",
    },
}
(RUNTIME_DIR / "provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
)

package_dir = OUTPUT_ROOT / "artifact_package"
if package_dir.exists():
    shutil.rmtree(package_dir)
package_dir.mkdir()
for file_path in (
    MANIFEST_PATH,
    SPLIT_MANIFEST_PATH,
    EXCLUDED_PATH,
    BEST_MMACTION,
):
    shutil.copy2(file_path, package_dir / file_path.name)
shutil.copytree(REPORT_ROOT, package_dir / "reports")
shutil.copytree(RUNTIME_DIR, package_dir / "runtime_bundle")

archive_base = OUTPUT_ROOT / "fallguard_training_artifacts"
archive_path = Path(shutil.make_archive(str(archive_base), "zip", package_dir))
expected_archive_path = OUTPUT_ROOT / "fallguard_training_artifacts.zip"
if archive_path != expected_archive_path:
    raise RuntimeError(f"归档路径异常: {archive_path}")
print("训练与评估完成")
print("MMAction checkpoint:", BEST_MMACTION)
print("Runtime bundle:", RUNTIME_DIR)
print("最终压缩包:", archive_path)
"""
    ),
    markdown(
        """
## 12. 结果解释

- `train`只更新模型参数。
- `val`用于early stopping、best checkpoint和fall阈值选择。
- `test`只在阈值锁定后运行一次，不参与任何模型选择。
- 同一视频的窗口不会跨集合，但四个场景会按比例出现在train/val/test，因此结果不能表述为“跨场景泛化”。
- Lecture_room和Office没有TXT标注，已记录到`excluded_unlabeled_videos.csv`且从全流程排除。
- 若`passed_deployment_gate=false`，权重只能标记为实验模型。
"""
    ),
]


notebook = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
