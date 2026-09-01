"""Unified auto/native/python entry point for float64 speaker rendering."""
from __future__ import annotations

import time

import numpy as np

from speaker_layouts import SpeakerLayout, get_speaker_layout
from speaker_native_renderer import NativeSpeakerRenderer
from speaker_renderer import PythonSpeakerRenderer, payload_events_from_index, render_objects16

FRAME_SAMPLES = 1536


def create_speaker_renderer(layout: str | SpeakerLayout, *, backend="auto", native_library=None):
    """Create a stateful speaker renderer and return ``(renderer, info)``."""
    if backend not in ("auto", "native", "python"):
        raise ValueError(f"unknown speaker backend: {backend}")
    target = get_speaker_layout(layout) if isinstance(layout, str) else layout
    fallback_reason = None
    if backend in ("auto", "native"):
        try:
            renderer = NativeSpeakerRenderer(target, native_library)
            return renderer, {
                "name": "native",
                "layout": target.name,
                "library": str(renderer.library_path),
                "fallback_reason": None,
            }
        except (OSError, RuntimeError) as exc:
            fallback_reason = str(exc)
    return PythonSpeakerRenderer(target), {
        "name": "python",
        "layout": target.name,
        "library": None,
        "fallback_reason": fallback_reason,
    }


def render_speaker_layout(objects16, index, layout: str | SpeakerLayout, *,
                          backend="auto", native_library=None,
                          metadata_offset=1473):
    """Render a complete indexed object stream and return ``(pcm64, info)``."""
    target = get_speaker_layout(layout) if isinstance(layout, str) else layout
    source = np.asarray(objects16)
    if source.ndim != 2 or source.shape[1] != 16:
        raise ValueError(f"objects16 must have shape [samples,16], got {source.shape}")
    if len(source) % FRAME_SAMPLES:
        raise ValueError("objects16 sample count must be divisible by 1536")
    frames = len(source) // FRAME_SAMPLES
    if frames > len(index.rows):
        raise ValueError(f"metadata has {len(index.rows)} frames, PCM needs {frames}")

    renderer, info = create_speaker_renderer(
        target, backend=backend, native_library=native_library)
    output = np.empty((len(source), target.channel_count), dtype=np.float64)
    started = time.perf_counter()
    try:
        for frame in range(frames):
            start = frame * FRAME_SAMPLES
            subs = index.subpayloads(index.rows[frame])
            output[start:start + FRAME_SAMPLES] = renderer.render_frame(
                source[start:start + FRAME_SAMPLES], subs.get(11), metadata_offset)
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
    info = dict(info)
    info["seconds"] = time.perf_counter() - started
    return output, info

