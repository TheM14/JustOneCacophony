"""Project-owned image-source early reflections and shared unitary FDN."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ShoeboxRoomConfig:
    dimensions_m: tuple[float, float, float] = (18.0, 18.0, 14.0)
    listener_position_m: tuple[float, float, float] = (9.0, 9.0, 7.0)
    wall_reflection_gain: tuple[float, float, float, float, float, float] = (
        0.62, 0.60, 0.58, 0.61, 0.52, 0.56)
    speed_of_sound_m_s: float = 343.3

    def validate(self) -> None:
        dimensions = np.asarray(self.dimensions_m, dtype=np.float64)
        listener = np.asarray(self.listener_position_m, dtype=np.float64)
        gains = np.asarray(self.wall_reflection_gain, dtype=np.float64)
        if (dimensions.shape != (3,) or not np.isfinite(dimensions).all()
                or np.any(dimensions <= 0.0)):
            raise ValueError("room dimensions must be three positive finite values")
        if (listener.shape != (3,) or not np.isfinite(listener).all()
                or np.any(listener <= 0.0) or np.any(listener >= dimensions)):
            raise ValueError("listener must be strictly inside the shoebox")
        if (gains.shape != (6,) or not np.isfinite(gains).all()
                or np.any(np.abs(gains) >= 1.0)):
            raise ValueError("six finite wall gains must have magnitude below one")
        if not math.isfinite(self.speed_of_sound_m_s) or self.speed_of_sound_m_s <= 0.0:
            raise ValueError("speed of sound must be positive")


@dataclass(frozen=True)
class EarlyReflection:
    wall: str
    direction_adm: np.ndarray
    path_distance_m: float
    extra_delay_samples: float
    reflection_gain: float


_WALL_NAMES = ("left", "right", "back", "front", "floor", "ceiling")


def first_order_image_sources(direction_adm, source_distance_m: float,
                              sample_rate_hz: float,
                              config: ShoeboxRoomConfig = ShoeboxRoomConfig()
                              ) -> tuple[EarlyReflection, ...]:
    """Return six first-order image-source paths for one object."""
    config.validate()
    direction = np.asarray(direction_adm, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("reflection direction must contain three finite ADM values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-15:
        direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    distance = float(source_distance_m)
    rate = float(sample_rate_hz)
    if not math.isfinite(distance) or distance <= 0.0 or not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("source distance and sample rate must be positive")
    dimensions = np.asarray(config.dimensions_m, dtype=np.float64)
    listener = np.asarray(config.listener_position_m, dtype=np.float64)
    source = listener + direction * distance
    if np.any(source <= 0.0) or np.any(source >= dimensions):
        raise ValueError(
            "source lies outside the configured public shoebox; enlarge the room")
    images = []
    for axis in range(3):
        low = source.copy()
        low[axis] = -source[axis]
        high = source.copy()
        high[axis] = 2.0 * dimensions[axis] - source[axis]
        images.extend((low, high))
    result = []
    for wall, image, gain in zip(_WALL_NAMES, images, config.wall_reflection_gain):
        vector = image - listener
        path_distance = float(np.linalg.norm(vector))
        path_direction = vector / path_distance
        extra = max(0.0, (path_distance - distance)
                    * rate / config.speed_of_sound_m_s)
        air = math.exp(-0.002 * max(path_distance - distance, 0.0))
        result.append(EarlyReflection(
            wall=wall,
            direction_adm=np.asarray(path_direction, dtype=np.float64),
            path_distance_m=path_distance,
            extra_delay_samples=extra,
            reflection_gain=float(gain) * air,
        ))
    return tuple(result)


def normalized_hadamard4() -> np.ndarray:
    return 0.5 * np.asarray([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ], dtype=np.float64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = int(math.sqrt(value))
    return all(value % divisor for divisor in range(3, limit + 1, 2))


def _next_prime(value: int) -> int:
    candidate = max(2, int(value))
    while not _is_prime(candidate):
        candidate += 1
    return candidate


class SchroederAllpass:
    def __init__(self, delay_samples: int, gain: float):
        self.delay_samples = int(delay_samples)
        self.gain = float(gain)
        if self.delay_samples <= 0 or not 0.0 <= abs(self.gain) < 1.0:
            raise ValueError("all-pass delay must be positive and |gain| < 1")
        self.buffer = np.zeros(self.delay_samples, dtype=np.float64)
        self.position = 0

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.position = 0

    def process(self, values) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        output = np.empty_like(source)
        for index, value in enumerate(source):
            delayed = self.buffer[self.position]
            result = delayed - self.gain * value
            self.buffer[self.position] = value + self.gain * result
            self.position = (self.position + 1) % self.delay_samples
            output[index] = result
        return output


@dataclass(frozen=True)
class LateFdnConfig:
    sample_rate_hz: float = 48000.0
    rt60_seconds: float = 0.85
    damping: float = 0.32
    output_gain: float = 0.22
    delay_seconds: tuple[float, float, float, float] = (
        0.0297, 0.0371, 0.0411, 0.0437)
    allpass_seconds: tuple[float, float] = (0.0023, 0.0067)
    allpass_gain: tuple[float, float] = (0.63, 0.51)


class SharedUnitaryFdn:
    """One shared late room driven by the sum of all object room sends."""

    def __init__(self, config: LateFdnConfig = LateFdnConfig()):
        self.config = config
        self.sample_rate_hz = float(config.sample_rate_hz)
        self.rt60_seconds = float(config.rt60_seconds)
        self.damping = float(config.damping)
        self.output_gain = float(config.output_gain)
        if (not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0
                or not math.isfinite(self.rt60_seconds) or self.rt60_seconds <= 0.0):
            raise ValueError("FDN sample rate and RT60 must be positive and finite")
        if (not math.isfinite(self.damping) or not 0.0 <= self.damping < 1.0
                or not math.isfinite(self.output_gain)):
            raise ValueError("invalid FDN damping/output gain")
        delay_seconds = np.asarray(config.delay_seconds, dtype=np.float64)
        allpass_seconds = np.asarray(config.allpass_seconds, dtype=np.float64)
        allpass_gain = np.asarray(config.allpass_gain, dtype=np.float64)
        if (delay_seconds.shape != (4,) or not np.isfinite(delay_seconds).all()
                or np.any(delay_seconds <= 0.0)):
            raise ValueError("FDN requires four positive finite delay times")
        if (allpass_seconds.shape != (2,) or not np.isfinite(allpass_seconds).all()
                or np.any(allpass_seconds <= 0.0)):
            raise ValueError("FDN requires two positive finite all-pass delay times")
        if (allpass_gain.shape != (2,) or not np.isfinite(allpass_gain).all()
                or np.any(np.abs(allpass_gain) >= 1.0)):
            raise ValueError("FDN requires two finite all-pass gains with magnitude below one")
        self.matrix = normalized_hadamard4()
        self.delays = np.asarray([
            _next_prime(round(seconds * self.sample_rate_hz))
            for seconds in delay_seconds
        ], dtype=np.int32)
        self.feedback_gain = np.power(
            10.0, -3.0 * self.delays / (self.rt60_seconds * self.sample_rate_hz)
        ).astype(np.float64)
        self.buffers = [np.zeros(int(delay), dtype=np.float64) for delay in self.delays]
        self.positions = np.zeros(4, dtype=np.int32)
        self.damping_state = np.zeros(4, dtype=np.float64)
        self.input_vector = 0.5 * np.asarray([1.0, -1.0, 1.0, 1.0], dtype=np.float64)
        self.output_matrix = 0.5 * np.asarray([
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ], dtype=np.float64)
        self.diffusers = [
            SchroederAllpass(
                _next_prime(round(seconds * self.sample_rate_hz)), gain)
            for seconds, gain in zip(allpass_seconds, allpass_gain)
        ]

    @property
    def tail_samples(self) -> int:
        return int(math.ceil(1.5 * self.rt60_seconds * self.sample_rate_hz))

    def reset(self) -> None:
        for buffer in self.buffers:
            buffer.fill(0.0)
        self.positions.fill(0)
        self.damping_state.fill(0.0)
        for diffuser in self.diffusers:
            diffuser.reset()

    def process(self, mono) -> np.ndarray:
        values = np.asarray(mono, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("FDN input must be one finite mono vector")
        diffused = values
        for diffuser in self.diffusers:
            diffused = diffuser.process(diffused)
        output = np.empty((len(values), 2), dtype=np.float64)
        for sample, value in enumerate(diffused):
            delayed = np.asarray([
                self.buffers[line][int(self.positions[line])]
                for line in range(4)
            ], dtype=np.float64)
            self.damping_state = (
                self.damping * self.damping_state + (1.0 - self.damping) * delayed)
            output[sample] = self.output_gain * (self.output_matrix @ self.damping_state)
            feedback = self.matrix @ (self.damping_state * self.feedback_gain)
            write = self.input_vector * value + feedback
            for line in range(4):
                position = int(self.positions[line])
                self.buffers[line][position] = write[line]
                self.positions[line] = (position + 1) % int(self.delays[line])
        return output

    def info(self) -> dict:
        return {
            "name": "SharedUnitaryFdn",
            "sample_rate_hz": self.sample_rate_hz,
            "rt60_seconds": self.rt60_seconds,
            "delay_samples": [int(value) for value in self.delays],
            "feedback_gain": [float(value) for value in self.feedback_gain],
            "matrix_unitarity_max_error": float(
                np.max(np.abs(self.matrix.T @ self.matrix - np.eye(4)))),
            "allpass_delay_samples": [value.delay_samples for value in self.diffusers],
            "precision": "float64",
        }
