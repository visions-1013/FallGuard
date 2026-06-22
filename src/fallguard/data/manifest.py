from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnnotationOverride:
    scene: str
    video: str
    start: int
    end: int


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frames: int


@dataclass(frozen=True)
class ManifestRow:
    scene: str
    video_id: str
    video_path: str
    annotation_path: str
    label: str
    fall_start: int | None
    fall_end: int | None
    width: int
    height: int
    fps: float
    frames: int


def parse_annotation(path: Path, override: AnnotationOverride | None = None) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start, end = int(lines[0].strip()), int(lines[1].strip())
    except (IndexError, ValueError) as exc:
        if override is None:
            raise ValueError(f"missing fall interval in {path}") from exc
        start, end = override.start, override.end
    if start < 0 or end < start:
        raise ValueError(f"invalid fall interval {start}-{end} in {path}")
    return start, end


def probe_video(path: Path) -> VideoProbe:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        return VideoProbe(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def _video_number(stem: str) -> int:
    try:
        return int(stem.removeprefix("video (").removesuffix(")"))
    except ValueError as exc:
        raise ValueError(f"unsupported Le2i video name: {stem}") from exc


def build_manifest(
    dataset_root: Path,
    output_csv: Path,
    overrides: Iterable[AnnotationOverride],
    probe_video: Callable[[Path], VideoProbe] = probe_video,
) -> list[ManifestRow]:
    dataset_root = dataset_root.resolve()
    override_map = {(item.scene, item.video): item for item in overrides}
    rows: list[ManifestRow] = []
    videos = sorted(
        (path for path in dataset_root.rglob("*.avi") if "Le2i-train-test" not in path.parts),
        key=lambda path: (path.relative_to(dataset_root).parts[0], _video_number(path.stem)),
    )
    for video_path in videos:
        scene = video_path.relative_to(dataset_root).parts[0]
        annotation_candidates = list((dataset_root / scene).rglob(f"{video_path.stem}.txt"))
        annotation_path = annotation_candidates[0] if annotation_candidates else None
        if annotation_path:
            start, end = parse_annotation(
                annotation_path, override_map.get((scene, video_path.stem))
            )
            label = "non_fall" if (start, end) == (0, 0) else "fall"
            fall_start = None if label == "non_fall" else start
            fall_end = None if label == "non_fall" else end
        else:
            label = "unlabeled"
            fall_start = fall_end = None
        meta = probe_video(video_path)
        rows.append(
            ManifestRow(
                scene=scene,
                video_id=video_path.stem,
                video_path=str(video_path),
                annotation_path=str(annotation_path) if annotation_path else "",
                label=label,
                fall_start=fall_start,
                fall_end=fall_end,
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                frames=meta.frames,
            )
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else [field.name for field in ManifestRow.__dataclass_fields__.values()]
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return rows
