from __future__ import annotations

from fallguard.inference.events import FallEventStateMachine


def test_event_requires_two_hits_and_recovers_after_two_seconds() -> None:
    machine = FallEventStateMachine(
        fall_threshold=0.6,
        recovery_threshold=0.3,
        trigger_windows=2,
        recovery_seconds=2.0,
        cooldown_seconds=10.0,
    )

    assert machine.update(1.0, 0.8).state == "non_fall"
    triggered = machine.update(1.2, 0.9)
    assert triggered.state == "fall"
    assert triggered.active_event is not None
    assert triggered.active_event.start_time == 1.0
    assert machine.update(2.0, 0.2).state == "fall"
    completed = machine.update(4.0, 0.2)
    assert completed.state == "non_fall"
    assert len(completed.completed_events) == 1
    assert completed.completed_events[0].end_time == 4.0


def test_cooldown_prevents_duplicate_event() -> None:
    machine = FallEventStateMachine(trigger_windows=2, recovery_seconds=0.1, cooldown_seconds=10)
    machine.update(0.0, 0.9)
    machine.update(0.2, 0.9)
    machine.update(0.4, 0.0)
    machine.update(0.6, 0.0)

    assert machine.update(1.0, 0.9).state == "non_fall"
    assert machine.update(1.2, 0.9).state == "non_fall"
