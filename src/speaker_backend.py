"""Float64 speaker rendering entry point."""
from __future__ import annotations

import time

import numpy as np

from speaker_layouts import SpeakerLayout, get_speaker_layout
from speaker_renderer import PythonSpeakerRenderer, payload_events_from_index, render_objects16

FRAME_SAMPLES = 1536


def create_speaker_renderer(layout: str | SpeakerLayout):
    """Create a stateful speaker renderer and return ``(renderer, info)``."""
    target = get_speaker_layout(layout) if isinstance(layout, str) else layout
    return PythonSpeakerRenderer(target), {
        "name": "python",
        "layout": target.name,
    }


def render_speaker_layout(objects16, index, layout: str | SpeakerLayout, *,
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

    renderer, info = create_speaker_renderer(target)
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

