from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fallguard.data.manifest import (
    AnnotationOverride,
    VideoProbe,
    build_manifest,
    parse_annotation,
)


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_annotation_reads_interval_and_zero_means_non_fall(tmp_path: Path) -> None:
    fall = _touch(tmp_path / "fall.txt", "48\n80\n1,1,0,0,0,0\n")
    normal = _touch(tmp_path / "normal.txt", "0\n0\n1,1,0,0,0,0\n")

    assert parse_annotation(fall) == (48, 80)
    assert parse_annotation(normal) == (0, 0)


def test_parse_annotation_requires_override_for_missing_interval(tmp_path: Path) -> None:
    broken = _touch(tmp_path / "video (26).txt", "1,1,72,58,132,170\n")

    with pytest.raises(ValueError, match="missing fall interval"):
        parse_annotation(broken)

    override = AnnotationOverride(scene="Coffee_room_01", video="video (26)", start=197, end=227)
    assert parse_annotation(broken, override=override) == (197, 227)


def test_build_manifest_marks_scenes_without_txt_as_unlabeled(tmp_path: Path) -> None:
    _touch(tmp_path / "Coffee_room_01" / "Coffee_room_01" / "Videos" / "video (1).avi")
    _touch(
        tmp_path / "Coffee_room_01" / "Coffee_room_01" / "Annotation_files" / "video (1).txt",
        "10\n20\n",
    )
    _touch(tmp_path / "Lecture_room" / "Lecture room" / "video (1).avi")
    _touch(tmp_path / "Lecture_room" / "Lecture room" / "video (15).avi")
    _touch(tmp_path / "Office" / "Office" / "video (17).avi")
    _touch(tmp_path / "Office" / "Office" / "video (18).avi")

    manifest_path = tmp_path / "manifest.csv"
    rows = build_manifest(
        dataset_root=tmp_path,
        output_csv=manifest_path,
        overrides=[],
        probe_video=lambda _: VideoProbe(width=320, height=240, fps=25.0, frames=100),
    )

    by_key = {(row.scene, row.video_id): row for row in rows}
    assert by_key[("Coffee_room_01", "video (1)")].label == "fall"
    assert by_key[("Coffee_room_01", "video (1)")].fall_start == 10
    assert by_key[("Lecture_room", "video (1)")].label == "unlabeled"
    assert by_key[("Lecture_room", "video (15)")].label == "unlabeled"
    assert by_key[("Office", "video (17)")].label == "unlabeled"
    assert by_key[("Office", "video (18)")].label == "unlabeled"

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert {row["scene"] for row in written} == {"Coffee_room_01", "Lecture_room", "Office"}
