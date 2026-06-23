from __future__ import annotations

from fallguard.ui.timeline import intervals_to_pixels


def test_intervals_to_pixels_clamps_ranges_to_video_duration() -> None:
    pixels = intervals_to_pixels(
        [(-1.0, 2.0), (8.0, 15.0), (7.0, 5.0)],
        duration=10.0,
        width=200,
    )

    assert pixels == [(0, 40), (160, 200), (100, 140)]


def test_intervals_to_pixels_supports_live_fall_interval() -> None:
    assert intervals_to_pixels([(4.0, 6.5)], duration=10.0, width=100) == [(40, 65)]
