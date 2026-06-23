from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from fallguard.data.manifest import AnnotationOverride, build_manifest
from fallguard.models.runtime import export_runtime_bundle, load_runtime_bundle
from fallguard.pose.extract_video import extract_video_to_cache, sha256_file
from fallguard.pose.yolo_pose import YoloPoseExtractor
from fallguard.training.evaluate import evaluate_pose_dataset
from fallguard.training.mmaction_runner import run_mmaction_training
from fallguard.training.prepare import prepare_mmaction_dataset

if TYPE_CHECKING:
    from fallguard.inference.pipeline import FallGuardPipeline


def _add_path(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    parser.add_argument(name, type=Path, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fallguard", description="YOLOPose + ST-GCN")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="build the Le2i manifest")
    _add_path(manifest, "dataset_root")
    _add_path(manifest, "output_csv")
    _add_path(manifest, "--overrides", default=Path("configs/annotation_overrides.yaml"))
    extract = commands.add_parser("extract-poses", help="cache YOLO26 pose sequences")
    _add_path(extract, "manifest")
    _add_path(extract, "output_dir")
    extract.add_argument("--model", default="yolo26n-pose.pt")
    extract.add_argument("--device")
    prepare = commands.add_parser("prepare-training", help="export MMAction2 PoseDataset")
    _add_path(prepare, "manifest")
    _add_path(prepare, "pose_dir")
    _add_path(prepare, "output_pkl")
    train = commands.add_parser("train", help="run an MMAction2 fold")
    _add_path(train, "annotations")
    _add_path(train, "work_dir")
    train.add_argument("--config", type=Path, default=Path("configs/mmaction/stgcn_le2i.py"))
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--pretrained", action="store_true")
    evaluate = commands.add_parser("evaluate", help="generate binary metrics and plots")
    _add_path(evaluate, "checkpoint")
    _add_path(evaluate, "annotations")
    _add_path(evaluate, "output_dir")
    evaluate.add_argument("--split", default="fold_0_val")
    _add_path(evaluate, "--history-path")
    bundle = commands.add_parser("export-bundle", help="export lightweight runtime weights")
    _add_path(bundle, "checkpoint")
    _add_path(bundle, "output_dir")
    bundle.add_argument("--threshold", type=float, required=True)
    infer = commands.add_parser("infer-video", help="annotate and save one video")
    _add_path(infer, "input_video")
    _add_path(infer, "output_dir")
    _add_path(infer, "bundle_dir")
    infer.add_argument("--pose-model", default="yolo26n-pose.pt")
    infer.add_argument("--device")
    app = commands.add_parser("app", help="launch the minimal Tkinter application")
    _add_path(
        app,
        "bundle_dir",
        nargs="?",
        default=Path("outputs/autodl_training/runtime_bundle"),
    )
    app.add_argument("--pose-model", default="yolo26n-pose.pt")
    app.add_argument("--device")
    _add_path(app, "--recordings-dir", default=Path("recordings"))
    return parser


def _load_overrides(path: Path) -> list[AnnotationOverride]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [AnnotationOverride(**item) for item in payload["overrides"]]


def _build_pipeline(bundle_dir: Path, pose_model: str, device: str | None) -> FallGuardPipeline:
    from fallguard.inference.pipeline import FallGuardPipeline

    model, meta = load_runtime_bundle(bundle_dir, device)
    extractor = YoloPoseExtractor(pose_model, device)
    return FallGuardPipeline(
        extractor,
        model,
        device=device,
        pose_fps=float(meta["pose_fps"]),
        window_frames=int(meta["window_frames"]),
        window_stride=int(meta["window_stride"]),
        fall_threshold=float(meta["fall_threshold"]),
        recovery_threshold=float(meta["recovery_threshold"]),
        trigger_windows=int(meta["trigger_windows"]),
        recovery_seconds=float(meta["recovery_seconds"]),
        cooldown_seconds=float(meta["cooldown_seconds"]),
        model_version=meta["weights_sha256"],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        manifest_rows = build_manifest(
            args.dataset_root, args.output_csv, _load_overrides(args.overrides)
        )
        print(f"wrote {len(manifest_rows)} videos to {args.output_csv}")
    elif args.command == "extract-poses":
        extractor = YoloPoseExtractor(args.model, args.device)
        model_path = Path(args.model)
        model_hash = sha256_file(model_path) if model_path.exists() else "ultralytics-managed"
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            csv_rows = [row for row in csv.DictReader(handle) if row.get("label") != "unlabeled"]
        for index, row in enumerate(csv_rows, start=1):
            target = args.output_dir / row["scene"] / f"{row['video_id']}.npz"
            if not target.exists():
                extract_video_to_cache(
                    Path(row["video_path"]), target, extractor, args.model, model_hash
                )
            print(f"[{index}/{len(csv_rows)}] {target}")
    elif args.command == "prepare-training":
        print(prepare_mmaction_dataset(args.manifest, args.pose_dir, args.output_pkl))
    elif args.command == "train":
        run_mmaction_training(
            args.config, args.annotations, args.work_dir, args.fold, args.pretrained
        )
    elif args.command == "evaluate":
        print(
            evaluate_pose_dataset(
                args.checkpoint,
                args.annotations,
                args.split,
                args.output_dir,
                history_path=args.history_path,
            )
        )
    elif args.command == "export-bundle":
        export_runtime_bundle(args.checkpoint, args.output_dir, args.threshold)
    elif args.command == "infer-video":
        from fallguard.inference.video import process_video

        print(
            process_video(
                args.input_video,
                args.output_dir,
                _build_pipeline(args.bundle_dir, args.pose_model, args.device),
            )
        )
    elif args.command == "app":
        from fallguard.ui.tk_app import launch_app

        launch_app(args.bundle_dir, args.pose_model, args.device, args.recordings_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
