from __future__ import annotations

from fallguard.cli import build_parser


def test_cli_exposes_data_training_inference_and_app_commands() -> None:
    parser = build_parser()

    help_text = parser.format_help()
    assert "manifest" in help_text
    assert "extract-poses" in help_text
    assert "prepare-training" in help_text
    assert "evaluate-external" not in help_text
    assert "infer-video" in help_text
    assert "app" in help_text


def test_app_command_uses_project_model_and_recording_defaults() -> None:
    args = build_parser().parse_args(["app"])

    assert str(args.bundle_dir).replace("\\", "/") == (
        "outputs/autodl_training/runtime_bundle"
    )
    assert str(args.recordings_dir).replace("\\", "/") == "recordings"
