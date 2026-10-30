"""Selectable Rosella 64-QMF / 77-hybrid filterbank.

The Rosella renderer can run either bank:

``PUBLIC_FILTERBANK`` (1)
    The shared public bank of ``rosella_filterbank.py``: a 13-tap hybrid
    analysis, a 6-slot high-band delay and 961 samples of analysis/synthesis
    latency.  This is the bank the reference implementation has always used.

``CAPTURED_FILTERBANK`` (2)
    The bank the reference headphone renderer actually runs, captured from it:
    an 11-tap hybrid analysis, a 3-slot high-band delay and 448 samples of
    latency.

Both banks expose the same four stateful stages, so the choice changes the
coefficients, the high-band delay, the history layout and the latency together.
Nothing is shared between them.
"""
from __future__ import annotations

from pathlib import Path

from rosella_captured_filterbank import (
    DEFAULT_CAPTURED_KERNEL_DATA,
    HybridAnalysis as CapturedHybridAnalysis,
    HybridSynthesis as CapturedHybridSynthesis,
    QmfAnalysis as CapturedQmfAnalysis,
    QmfSynthesis as CapturedQmfSynthesis,
)
from rosella_filterbank import (
    DEFAULT_KERNEL_DATA,
    HybridAnalysis as PublicHybridAnalysis,
    HybridSynthesis as PublicHybridSynthesis,
    QmfAnalysis as PublicQmfAnalysis,
    QmfSynthesis as PublicQmfSynthesis,
)

PUBLIC_FILTERBANK = 1
CAPTURED_FILTERBANK = 2
FILTERBANK_CHOICES = (PUBLIC_FILTERBANK, CAPTURED_FILTERBANK)
DEFAULT_FILTERBANK = PUBLIC_FILTERBANK

FILTERBANK_NAMES = {
    PUBLIC_FILTERBANK: "public",
    CAPTURED_FILTERBANK: "captured",
}

# analysis/synthesis latency of each bank, in samples at 48 kHz
FILTERBANK_LATENCY_SAMPLES = {
    PUBLIC_FILTERBANK: 961,
    CAPTURED_FILTERBANK: 448,
}


def latency_samples(filterbank: int) -> int:
    """Length of the startup transient the runtime removes for ``filterbank``."""
    try:
        return FILTERBANK_LATENCY_SAMPLES[int(filterbank)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"Rosella filterbank must be 1 (public) or 2 (captured), got {filterbank!r}"
        ) from None


class RosellaBank:
    """The four stages of one bank, each with its own state."""

    def __init__(self, filterbank: int = DEFAULT_FILTERBANK, *,
                 source_channels: int, output_channels: int,
                 kernel_data: str | Path | None = None):
        if filterbank not in FILTERBANK_CHOICES:
            raise ValueError(
                f"Rosella filterbank must be 1 (public) or 2 (captured), got {filterbank!r}")
        kind = int(filterbank)
        if source_channels <= 0 or output_channels <= 0:
            raise ValueError("channel counts must be positive")
        self.filterbank = kind
        self.name = FILTERBANK_NAMES[kind]
        self.latency_samples = latency_samples(kind)
        if kind == PUBLIC_FILTERBANK:
            self.table_path = Path(
                DEFAULT_KERNEL_DATA if kernel_data is None else kernel_data
            ).expanduser().resolve()
            self.qmf_analysis = PublicQmfAnalysis(source_channels, self.table_path)
            self.hybrid_analysis = PublicHybridAnalysis(source_channels, self.table_path)
            self.hybrid_synthesis = PublicHybridSynthesis(output_channels, self.table_path)
            self.qmf_synthesis = PublicQmfSynthesis(output_channels, self.table_path)
        else:
            self.table_path = Path(
                DEFAULT_CAPTURED_KERNEL_DATA if kernel_data is None else kernel_data
            ).expanduser().resolve()
            self.qmf_analysis = CapturedQmfAnalysis(source_channels, self.table_path)
            self.hybrid_analysis = CapturedHybridAnalysis(source_channels, self.table_path)
            self.hybrid_synthesis = CapturedHybridSynthesis(output_channels, self.table_path)
            self.qmf_synthesis = CapturedQmfSynthesis(output_channels, self.table_path)

    def reset(self):
        self.qmf_analysis.reset()
        self.hybrid_analysis.reset()
        self.hybrid_synthesis.reset()
        self.qmf_synthesis.reset()
