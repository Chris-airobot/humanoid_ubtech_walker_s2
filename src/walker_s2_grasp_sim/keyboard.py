from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class WalkerS2GraspKeyboardConfig:
    # Match the official GHRC Walker keyboard teleoperator.
    speed_levels: tuple[float, ...] = (0.010, 0.035)
    default_speed_index: int = 0
    initial_control_arm: str = "right"
    bimanual_mirror_signs: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    keymap: dict[str, str] = field(
        default_factory=lambda: {
            "w": "x_up", "s": "x_down", "a": "y_up", "d": "y_down",
            "r": "z_up", "f": "z_down", "y": "rx_up", "u": "rx_down",
            "v": "ry_up", "b": "ry_down", "n": "rz_up", "m": "rz_down",
            "k": "hand_open", "l": "hand_close",
        }
    )


@dataclass
class WalkerS2GraspTeleopCommand:
    arm_deltas: dict[str, np.ndarray]
    hand_delta: float = 0.0
    target_sides: tuple[str, ...] = ()
    go_home: bool = False
    assisted_grasp: bool = False
    save_episode: bool = False
    quit: bool = False


class WalkerS2GraspKeyboard:
    """Isaac Sim keyboard state for the standalone Walker grasp simulation."""

    def __init__(self, config: WalkerS2GraspKeyboardConfig | None = None):
        self.config = config or WalkerS2GraspKeyboardConfig()
        self.current_control_arm = self.config.initial_control_arm
        self.bimanual_control_enabled = False
        self._speed_index = self.config.default_speed_index
        self._pressed: set[str] = set()
        self._pending_presses: set[str] = set()
        self._events: set[str] = set()
        self._lock = Lock()
        self._input_interface = None
        self._keyboard = None
        self._subscription = None
        self._input_to_label: dict[object, str] = {}

    @property
    def speed(self) -> float:
        return self.config.speed_levels[self._speed_index]

    def connect(self) -> None:
        import carb
        import omni.appwindow

        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        keyboard_input = carb.input.KeyboardInput
        enum_names = {
            "W": "w", "S": "s", "A": "a", "D": "d", "R": "r", "F": "f",
            "Y": "y", "U": "u", "V": "v", "B": "b", "N": "n", "M": "m",
            "K": "k", "L": "l", "O": "o", "G": "g", "H": "h", "P": "p", "Q": "q",
            "KEY_0": "0", "NUMPAD_0": "0", "MINUS": "-", "EQUAL": "+",
        }
        self._input_to_label = {
            value: label
            for enum_name, label in enum_names.items()
            if (value := getattr(keyboard_input, enum_name, None)) is not None
        }
        self._subscription = self._input_interface.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def close(self) -> None:
        if self._subscription is not None:
            self._input_interface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def _on_keyboard_event(self, event, *args):
        import carb

        label = self._input_to_label.get(event.input)
        if label is None:
            return True
        with self._lock:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                first_press = label not in self._pressed
                self._pressed.add(label)
                if first_press:
                    self._handle_edge(label)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._pressed.discard(label)
        return True

    def _handle_edge(self, label: str) -> None:
        if label == "o":
            self.current_control_arm = "left" if self.current_control_arm == "right" else "right"
            print(f"[TELEOP] Selected arm: {self.current_control_arm}")
        elif label == "0":
            self.bimanual_control_enabled = not self.bimanual_control_enabled
            mode = "bimanual" if self.bimanual_control_enabled else "single arm"
            print(f"[TELEOP] Control mode: {mode}")
        elif label == "+":
            self._speed_index = min(self._speed_index + 1, len(self.config.speed_levels) - 1)
            print(f"[TELEOP] Motion step: {self.speed:.3f}")
        elif label == "-":
            self._speed_index = max(self._speed_index - 1, 0)
            print(f"[TELEOP] Motion step: {self.speed:.3f}")
        elif label == "h":
            self._events.add("home")
        elif label == "g":
            self._events.add("grasp")
        elif label == "p":
            self._events.add("save_episode")
        elif label == "q":
            self._events.add("quit")
        elif label in self.config.keymap:
            self._pending_presses.add(label)
            print(
                f"[TELEOP] Key {label.upper()}: {self.config.keymap[label]} "
                f"(arm={self.current_control_arm}, step={self.speed:.3f})"
            )

    def _poll_keyboard_state(self) -> None:
        if self._input_interface is None or self._keyboard is None:
            return
        down = {
            label
            for key, label in self._input_to_label.items()
            if self._input_interface.get_keyboard_value(self._keyboard, key) > 0.0
        }
        with self._lock:
            for label in down - self._pressed:
                self._handle_edge(label)
            self._pressed = down

    def sample(self) -> WalkerS2GraspTeleopCommand:
        self._poll_keyboard_state()
        with self._lock:
            pressed = self._pressed | self._pending_presses
            self._pending_presses.clear()
            events = set(self._events)
            self._events.clear()
            selected_arm = self.current_control_arm
            bimanual = self.bimanual_control_enabled
            speed = self.speed

        active_delta = np.zeros(6, dtype=float)
        axes = ("x", "y", "z", "rx", "ry", "rz")
        active_actions = {self.config.keymap[key] for key in pressed if key in self.config.keymap}
        for index, axis in enumerate(axes):
            if f"{axis}_up" in active_actions:
                active_delta[index] += speed
            if f"{axis}_down" in active_actions:
                active_delta[index] -= speed

        arm_deltas = {"left": np.zeros(6), "right": np.zeros(6)}
        arm_deltas[selected_arm] = active_delta
        target_sides = (selected_arm,)
        if bimanual:
            other_arm = "left" if selected_arm == "right" else "right"
            mirror = np.asarray(self.config.bimanual_mirror_signs, dtype=float)
            arm_deltas[other_arm] = active_delta * mirror
            target_sides = ("left", "right")

        hand_delta = 0.0
        if "hand_open" in active_actions:
            hand_delta -= 0.015
        if "hand_close" in active_actions:
            hand_delta += 0.015

        return WalkerS2GraspTeleopCommand(
            arm_deltas=arm_deltas,
            hand_delta=hand_delta,
            target_sides=target_sides,
            go_home="home" in events,
            assisted_grasp="grasp" in events,
            save_episode="save_episode" in events,
            quit="quit" in events,
        )
