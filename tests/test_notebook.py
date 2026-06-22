from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import torch

NOTEBOOK_PATH = Path("notebooks/01_train_stgcn_cloud.ipynb")


def _load_notebook() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    nbformat.validate(notebook)
    return notebook


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _unit_namespace(notebook: dict) -> dict:
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "unit-test" in cell.get("metadata", {}).get("tags", [])
    ]
    assert sources, "Notebook must expose its pure helpers in a unit-test tagged cell"
    namespace: dict = {}
    exec(compile("\n\n".join(sources), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def _tagged_source(notebook: dict, tag: str) -> str:
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    assert sources, f"Notebook must expose a {tag} tagged cell"
    return "\n\n".join(sources)


def test_notebook_is_standalone_autodl_workflow() -> None:
    notebook = _load_notebook()
    source = _source(notebook)

    assert "from fallguard" not in source
    assert "D:\\" not in source
    assert 'DATA_ROOT = Path.cwd() / "datasets"' in source
    assert 'OUTPUT_ROOT = Path.cwd() / "outputs/autodl_training"' in source
    assert "YOLO_WEIGHT_PATH = None" in source
    assert "STGCN_CHECKPOINT_PATH = None" in source
    assert "FAST_DEV_RUN = False" in source
    assert "FORCE_REEXTRACT = False" in source
    assert "AUTO_RESUME = True" in source
    assert "OFFICIAL_STGCN_2D_JOINT" in source
    assert "pretrained=False" not in source
    assert "excluded_unlabeled_videos.csv" in source
    assert "split_manifest.csv" in source
    assert "fallguard_training_artifacts.zip" in source
    assert "video_errors.csv" in source
    assert '"best_epoch"' in source
    assert '"training_config"' in source
    assert '"passed_deployment_gate"' in source
    assert "Lecture_room" in source and "Office" in source
    assert source.index("LOCKED_THRESHOLD =") < source.index("test_labels, test_probabilities")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_python_cells_have_valid_syntax() -> None:
    notebook = _load_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "%pip" in source or source.lstrip().startswith("!"):
            continue
        try:
            ast.parse(source, filename=f"{NOTEBOOK_PATH}:cell-{index}")
        except SyntaxError as error:
            raise AssertionError(f"Notebook cell {index} has invalid Python syntax") from error


def test_notebook_pure_helpers_parse_overrides_and_unlabeled_rows() -> None:
    helpers = _unit_namespace(_load_notebook())

    assert helpers["parse_annotation_text"]("0\n0\n", "Home_02", "video (8)") == (
        0,
        0,
    )
    assert helpers["parse_annotation_text"](
        "broken\nannotation\n", "Coffee_room_01", "video (26)"
    ) == (197, 227)
    assert helpers["video_label"](0, 0) == "non_fall"
    assert helpers["video_label"](10, 20) == "fall"
    assert helpers["UNLABELED_SCENES"] == {"Lecture_room", "Office"}
    assert helpers["EXPECTED_SCENE_COUNTS"] == {
        "Coffee_room_01": 48,
        "Coffee_room_02": 22,
        "Home_01": 30,
        "Home_02": 30,
        "Lecture_room": 27,
        "Office": 33,
    }
    assert helpers["COCO17_KEYPOINTS"] == (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )


def test_notebook_video_split_is_reproducible_and_leak_free() -> None:
    helpers = _unit_namespace(_load_notebook())
    rows = []
    for scene in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"):
        for label in ("fall", "non_fall"):
            for index in range(10):
                rows.append(
                    {
                        "video_key": f"{scene}/{label}/{index}",
                        "scene": scene,
                        "label": label,
                    }
                )

    first = helpers["stratified_video_split"](rows, seed=42)
    second = helpers["stratified_video_split"](rows, seed=42)

    assert [item["split"] for item in first] == [item["split"] for item in second]
    counts = {
        name: sum(item["split"] == name for item in first) for name in ("train", "val", "test")
    }
    assert counts == {"train": 56, "val": 12, "test": 12}
    assert len({item["video_key"] for item in first}) == len(first)
    assert all(item["split"] in {"train", "val", "test"} for item in first)


def test_notebook_window_labels_and_threshold_use_explicit_inputs() -> None:
    helpers = _unit_namespace(_load_notebook())
    poses = np.zeros((160, 17, 3), dtype=np.float32)
    windows, labels, starts = helpers["build_labeled_windows"](
        poses,
        pose_fps=20.0,
        fall_start_seconds=1.0,
        fall_end_seconds=2.5,
        window_frames=32,
        stride=4,
    )

    assert len(windows) == len(labels) == len(starts)
    assert set(labels.tolist()) == {0, 1}
    selection = helpers["select_threshold"](
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
        min_precision=0.85,
    )
    assert selection["threshold"] == 0.6
    assert selection["precision"] == 1.0
    assert selection["recall"] == 1.0


def test_notebook_interpolates_only_short_missing_pose_gaps() -> None:
    helpers = _unit_namespace(_load_notebook())
    poses = np.ones((12, 17, 3), dtype=np.float32)
    poses[:, :, 0] = np.arange(12, dtype=np.float32)[:, None]
    poses[3:5] = 0.0
    poses[7:11] = 0.0

    interpolated = helpers["interpolate_missing_poses"](poses, max_gap=3)

    np.testing.assert_allclose(interpolated[3, :, 0], 3.0)
    np.testing.assert_allclose(interpolated[4, :, 0], 4.0)
    np.testing.assert_allclose(interpolated[3:5, :, 2], 1.0)
    np.testing.assert_allclose(interpolated[7:11], 0.0)


def test_notebook_pose_cache_metadata_locks_video_and_yolo_inputs() -> None:
    helpers = _unit_namespace(_load_notebook())
    row = {"width": 320, "height": 240, "fps": 25.0}
    metadata = {
        "width": 320,
        "height": 240,
        "fps": 25.0,
        "model_sha256": "model-hash",
        "source_sha256": "video-hash",
        "source_size": 123,
        "source_mtime_ns": 456,
    }

    assert helpers["cache_metadata_matches"](
        metadata,
        row,
        model_hash="model-hash",
        source_hash="video-hash",
        source_size=123,
        source_mtime_ns=456,
    )
    assert not helpers["cache_metadata_matches"](
        {**metadata, "width": 321},
        row,
        model_hash="model-hash",
        source_hash="video-hash",
        source_size=123,
        source_mtime_ns=456,
    )
    assert not helpers["cache_metadata_matches"](
        metadata,
        row,
        model_hash="changed-model",
        source_hash="video-hash",
        source_size=123,
        source_mtime_ns=456,
    )


def test_notebook_runtime_model_returns_binary_logits_and_loads_mmaction_head(
    tmp_path: Path,
) -> None:
    notebook = _load_notebook()
    namespace = {"np": np, "torch": torch}
    exec(
        compile(
            _tagged_source(notebook, "unit-test-model"),
            str(NOTEBOOK_PATH),
            "exec",
        ),
        namespace,
    )
    source = namespace["STGCNClassifier"](num_classes=60)
    target = namespace["STGCNClassifier"](num_classes=2)
    state = {
        **{f"backbone.{key}": value for key, value in source.backbone.state_dict().items()},
        "cls_head.fc_cls.weight": torch.randn(2, 256),
        "cls_head.fc_cls.bias": torch.randn(2),
    }
    checkpoint = tmp_path / "mmaction_binary.pth"
    torch.save({"state_dict": state}, checkpoint)

    namespace["load_mmaction_classifier"](target, checkpoint)
    logits = target(torch.randn(2, 1, 32, 17, 3))

    assert logits.shape == (2, 2)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2), atol=1e-4)
