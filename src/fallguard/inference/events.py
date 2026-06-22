from __future__ import annotations

from dataclasses import dataclass

from fallguard.types import FallEvent


@dataclass(frozen=True)
class EventUpdate:
    state: str
    active_event: FallEvent | None
    completed_events: tuple[FallEvent, ...] = ()


class FallEventStateMachine:
    def __init__(
        self,
        fall_threshold: float = 0.5,
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
        self.reset()

    def reset(self) -> None:
        self.state = "non_fall"
        self.active_event: FallEvent | None = None
        self._high_count = 0
        self._first_high_time: float | None = None
        self._recovery_started: float | None = None
        self._cooldown_until = float("-inf")

    def update(self, timestamp: float, probability: float) -> EventUpdate:
        if self.active_event is not None:
            self.active_event.max_probability = max(self.active_event.max_probability, probability)
            if probability < self.recovery_threshold:
                self._recovery_started = (
                    timestamp if self._recovery_started is None else self._recovery_started
                )
                if timestamp - self._recovery_started >= self.recovery_seconds:
                    self.active_event.end_time = timestamp
                    completed = self.active_event
                    self.active_event = None
                    self.state = "non_fall"
                    self._cooldown_until = timestamp + self.cooldown_seconds
                    self._recovery_started = None
                    return EventUpdate(self.state, None, (completed,))
            else:
                self._recovery_started = None
            return EventUpdate("fall", self.active_event)

        if timestamp < self._cooldown_until:
            return EventUpdate(self.state, None)
        if probability >= self.fall_threshold:
            if self._high_count == 0:
                self._first_high_time = timestamp
            self._high_count += 1
            if self._high_count >= self.trigger_windows:
                self.active_event = FallEvent(
                    start_time=self._first_high_time or timestamp,
                    trigger_time=timestamp,
                    max_probability=probability,
                )
                self.state = "fall"
                self._high_count = 0
                self._first_high_time = None
                return EventUpdate(self.state, self.active_event)
        else:
            self._high_count = 0
            self._first_high_time = None
        return EventUpdate(self.state, None)

    def finish(self, timestamp: float) -> tuple[FallEvent, ...]:
        if self.active_event is None:
            return ()
        self.active_event.end_time = timestamp
        completed = self.active_event
        self.active_event = None
        self.state = "non_fall"
        return (completed,)
