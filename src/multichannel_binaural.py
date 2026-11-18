"""Render ordinary 2.0 / 5.1 / 7.1 PCM WAV to binaural audio.

The binaural renderer is a JOC-object renderer: it expects sixteen source
channels (LFE plus fifteen objects) with a direction per object.  For ordinary
multichannel PCM there is no OAMD, so this module

  * maps the layout's channel order to fixed ADM directions,
  * turns each non-LFE channel into one synthetic object and the LFE channel
    into source 0,
  * feeds the same Rosella / jochrtf binaural adapter the JOC path uses.

The HRTF runtime is native 48 kHz.  Prefer letting ffmpeg/libsoxr produce a
48 kHz float32 WAV and feed that in directly:

    ffmpeg -i INPUT -af "aresample=resampler=soxr:precision=28:osr=48000" \
        -c:a pcm_f32le OUT.wav

The module then renders and writes 48 kHz stereo with no further sample-rate
conversion.  The supported families are 44.1k (44.1/88.2/176.4) and 48k
(48/96/192), up to 192 kHz; a non-48 kHz input is converted once on the way in,
and the output stays 48 kHz by default.  An explicit
non-48k output rate is possible but adds a second conversion.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from binaural_renderer import JochrtfBinauralRenderer
from rosella_binaural_renderer import (DEFAULT_PERSONALIZED_HEADPHONE,
                                       RosellaBinauralRenderer)
from rosella_bank import DEFAULT_FILTERBANK

TARGET_RATE = 48000
FRAME_SAMPLES = 1536
SOURCE_CHANNELS = 16

# WAV channel order follows the ffmpeg layout definitions:
#   2.0 = FL FR
#   5.1 = FL FR FC LFE BL BR
#   7.1 = FL FR FC LFE BL BR SL SR
# BL/BR are the rear pair, SL/SR the side pair.
LAYOUTS = {
    "2.0": (
        ("L", (-1.0, 1.0, 0.0)),
        ("R", (1.0, 1.0, 0.0)),
    ),
    "5.1": (
        ("L", (-1.0, 1.0, 0.0)),
        ("R", (1.0, 1.0, 0.0)),
        ("C", (0.0, 1.0, 0.0)),
        ("LFE", None),
        ("BL", (-1.0, -1.0, 0.0)),
        ("BR", (1.0, -1.0, 0.0)),
    ),
    "7.1": (
        ("L", (-1.0, 1.0, 0.0)),
        ("R", (1.0, 1.0, 0.0)),
        ("C", (0.0, 1.0, 0.0)),
        ("LFE", None),
        ("BL", (-1.0, -1.0, 0.0)),
        ("BR", (1.0, -1.0, 0.0)),
        ("SL", (-1.0, 0.0, 0.0)),
        ("SR", (1.0, 0.0, 0.0)),
    ),
}

LAYOUT_CHANNELS = {"2.0": 2, "5.1": 6, "7.1": 8}


def detect_layout(channels: int) -> str:
    for name, count in LAYOUT_CHANNELS.items():
        if count == int(channels):
            return name
    raise ValueError(
        f"cannot infer a binaural layout for {channels} channels; "
        f"supported layouts: 2.0 (2), 5.1 (6), 7.1 (8)")


def object_positions(layout: str) -> np.ndarray:
    """Return the 15 synthetic object directions for one layout."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    positions = np.zeros((15, 3), dtype=np.float64)
    slot = 0
    for label, position in LAYOUTS[layout]:
        if position is None:
            continue
        if slot >= 15:
            raise ValueError(f"layout {layout} needs more than 15 object slots")
        positions[slot] = position
        slot += 1
    return positions


def pcm_to_objects(pcm: np.ndarray, layout: str) -> np.ndarray:
    """Map interleaved layout PCM ``[samples, channels]`` to 16 JOC sources."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if pcm.ndim != 2 or pcm.shape[1] != LAYOUT_CHANNELS[layout]:
        raise ValueError(
            f"{layout} expects {LAYOUT_CHANNELS[layout]} channels, got {pcm.shape}")
    objects = np.zeros((len(pcm), SOURCE_CHANNELS), dtype=np.float64)
    slot = 0
    for channel, (label, position) in enumerate(LAYOUTS[layout]):
        if position is None:
            if label != "LFE":
                raise ValueError(f"only LFE may omit a direction, got {label}")
            objects[:, 0] = pcm[:, channel]
            continue
        objects[:, 1 + slot] = pcm[:, channel]
        slot += 1
    return objects


def _rate_ratio(source_rate: int, target_rate: int) -> tuple[int, int]:
    common = math.gcd(int(source_rate), int(target_rate))
    return int(target_rate) // common, int(source_rate) // common


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample ``[samples, channels]`` with a polyphase anti-alias filter."""
    if int(source_rate) == int(target_rate):
        return np.asarray(data, dtype=np.float64)
    if int(source_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError("sample rates must be positive")
    up, down = _rate_ratio(source_rate, target_rate)
    result = signal.resample_poly(
        np.asarray(data, dtype=np.float64), up, down, axis=0,
        window=("kaiser", 10.0))
    expected = int(round(len(data) * float(target_rate) / float(source_rate)))
    if len(result) > expected:
        result = result[:expected]
    elif len(result) < expected:
        pad = np.zeros((expected - len(result), result.shape[1]), dtype=result.dtype)
        result = np.concatenate((result, pad), axis=0)
    return result


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(str(path))
    if data.ndim == 1:
        data = data[:, None]
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / scale
    else:
        data = data.astype(np.float64)
    return int(rate), data


def write_wav(path: str | Path, rate: int, data: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(target), int(rate), np.asarray(data, dtype=np.float32))


def _make_renderer(*, backend: str, model, compiled_hrtf, mode: str,
                   filterbank: int, tail_seconds: float,
                   fixed_positions: np.ndarray):
    if backend == "rosella":
        return RosellaBinauralRenderer(
            model, mode=mode, filterbank=filterbank,
            tail_seconds=tail_seconds, fixed_positions=fixed_positions)
    if backend == "jochrtf":
        if not compiled_hrtf:
            raise ValueError("--backend jochrtf requires --compiled-hrtf PATH")
        return JochrtfBinauralRenderer.from_compiled_cache(
            compiled_hrtf, mode=mode, tail_seconds=tail_seconds,
            fixed_positions=fixed_positions)
    raise ValueError(f"unknown binaural backend {backend!r}")


def render_file(input_path, output_path, *, layout="auto",
                backend="rosella", model=None, compiled_hrtf=None,
                mode="mid", filterbank: int = DEFAULT_FILTERBANK,
                tail_seconds: float = 2.0, output_rate=48000) -> dict:
    """Decode a PCM WAV, synthesize objects and render it binaurally."""
    input_rate, pcm = read_wav(input_path)
    channels = pcm.shape[1]
    resolved = detect_layout(channels) if layout == "auto" else layout
    if resolved not in LAYOUTS:
        raise ValueError(f"unsupported layout {resolved!r}")
    if channels != LAYOUT_CHANNELS[resolved]:
        raise ValueError(
            f"layout {resolved} expects {LAYOUT_CHANNELS[resolved]} channels, "
            f"WAV has {channels}")

    input_resampled = input_rate != TARGET_RATE
    internal = pcm if not input_resampled else resample(
        pcm, input_rate, TARGET_RATE)
    objects = pcm_to_objects(internal, resolved)
    padded = np.zeros(
        ((len(objects) + FRAME_SAMPLES - 1) // FRAME_SAMPLES * FRAME_SAMPLES,
         SOURCE_CHANNELS), dtype=np.float64)
    padded[:len(objects)] = objects

    positions = object_positions(resolved)
    renderer = _make_renderer(
        backend=backend, model=model, compiled_hrtf=compiled_hrtf, mode=mode,
        filterbank=filterbank, tail_seconds=tail_seconds,
        fixed_positions=positions)
    chunks = []
    for start in range(0, len(padded), FRAME_SAMPLES):
        chunks.append(renderer.render_frame(padded[start:start + FRAME_SAMPLES]))
    chunks.append(renderer.finish())
    stereo = np.concatenate([chunk for chunk in chunks if len(chunk)], axis=0)

    if output_rate in (48000, "48000", "48k"):
        target_rate = TARGET_RATE
    elif output_rate == "input":
        target_rate = input_rate
    else:
        target_rate = int(output_rate)
    if target_rate != TARGET_RATE:
        stereo = resample(stereo, TARGET_RATE, target_rate)
    write_wav(output_path, target_rate, stereo)
    return {
        "input": str(Path(input_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "layout": resolved,
        "input_channels": channels,
        "input_rate": input_rate,
        "input_resampled": input_resampled,
        "render_rate": TARGET_RATE,
        "output_rate": target_rate,
        "input_samples": len(pcm),
        "output_samples": len(stereo),
        "backend": backend,
        "mode": mode,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render 2.0/5.1/7.1 PCM WAV files binaurally")
    parser.add_argument("input", type=Path, help="interleaved PCM WAV")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--layout", default="auto",
                        choices=("auto", "2.0", "5.1", "7.1"))
    parser.add_argument("--backend", default="rosella",
                        choices=("rosella", "jochrtf"))
    parser.add_argument("--model", type=Path, default=None,
                        help="Rosella .personalized_headphone model")
    parser.add_argument("--compiled-hrtf", type=Path, default=None,
                        help="JOC .jochrtf compiled cache")
    parser.add_argument("--binaural-mode", default="mid",
                        choices=("near", "mid", "far"))
    parser.add_argument("--filterbank", type=int, default=DEFAULT_FILTERBANK)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--output-rate", default="48000",
                        help="48k (default, native HRTF rate) or an explicit rate; "
                             "non-48k output adds a second resampling pass")
    parser.add_argument("--list-layouts", action="store_true")
    args = parser.parse_args(argv)
    if args.list_layouts:
        for name, channels in LAYOUT_CHANNELS.items():
            print(f"{name}: {channels} channels")
        return 0
    output = args.output or Path(str(args.input) + ".binaural.wav")
    model = args.model or DEFAULT_PERSONALIZED_HEADPHONE
    info = render_file(
        args.input, output, layout=args.layout, backend=args.backend,
        model=model, compiled_hrtf=args.compiled_hrtf,
        mode=args.binaural_mode, filterbank=args.filterbank,
        tail_seconds=args.tail_seconds, output_rate=args.output_rate)
    for key, value in info.items():
        print(f"{key}: {value}")
    if info["input_resampled"]:
        print(
            "warning: input was not 48 kHz and was converted once internally; "
            "prefer ffmpeg/libsoxr to produce a 48 kHz float32 WAV first, e.g. "
            "ffmpeg -i INPUT -af \"aresample=resampler=soxr:precision=28:osr=48000\" "
            "-c:a pcm_f32le OUT.wav",
            file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
