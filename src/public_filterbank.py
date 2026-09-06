"""Public 64-QMF and 77-band hybrid filterbank for binaural rendering.

The fixed resource is ``data/rosella_kernels.npz``: the fixed 64-QMF /
``3 -> 8+4+4`` 77-hybrid analysis tables and the causal synthesis tables
computed from that analysis bank.  The filter bank is publicly standardized:
the 64-QMF → 77-hybrid structure, the 13-tap low-band prototypes and their
half-bin complex modulation follow 3GPP TS 26.405 / ETSI TS 126 405 (Section
5.2.2, Table 1, ``Q=8``/``Q=4``); the 64-band QMF analysis is the MPEG-4
AAC/SBR 64 complex QMF analysis bank (ISO/IEC 14496-3/AMD1:2003, subclause
4.B.18.2), stored here as the polyphase form
``A[r,t] = ((-1)**t / 128) * c[63 - r + 64*t]`` of the public 640-tap SBR
prototype.  The QMF synthesis table is the causal left inverse of that
analysis polyphase matrix (``A @ W = P`` with the 577-sample delay
permutation; total latency ``961 = 577 + 6*64``), stored as the rank-4
factorization ``W[b,l] = sum_r taps[b,l,r] * basis[b,r,:]``; the hybrid
synthesis table is the 77->64 recombination (identity for the high bands,
signed summation of each 8+4+4 child group for the low bands), stored as a
154-entry sparse map.  The archive and every array inside it are
hash-validated before use, and those hashes participate in every
compiled-HRTF cache key.  Provenance and rights boundaries are documented in
``data/README.md`` and ``THIRD_PARTY_NOTICES.md``.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import zipfile

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILTERBANK_DATA = PROJECT_DIR / "data" / "rosella_kernels.npz"
FILTERBANK_TABLE_VERSION = "joc-public-64qmf-77hybrid-v1"
SAMPLE_RATE = 48000
QMF_HOP = 64
QMF_BANDS = 64
HYBRID_BANDS = 77
ANALYSIS_SYNTHESIS_LATENCY_SAMPLES = 961

_ARCHIVE_SHA256 = "C05BEF4D26E96ECBD4694E2572F05DA400255C777BA5047300B9D3B1F81081CD"
_TABLE_SPECS = {
    "format_version": (np.dtype("<i4"), (1,),
                       "67ABDD721024F0FF4E0B3F4C2FC13BC5BAD42D0B7851D456D88D203D15AAA450",
                       False, False),
    "qmf_analysis_coefficients": (
        np.dtype("<f4"), (64, 10),
        "AEFF6C7117D41664B9C4BF03BBF563F5319EC1B8C551F171ADBB90CF19D9D306",
        False, False),
    "hybrid_analysis_low_kernel": (
        np.dtype("<f4"), (3, 2, 13, 16, 2),
        "D00D36133B81BA699A7630C4DF1BE203FA1B7E371E595EAAEBBE8957DB322627",
        False, False),
    "hybrid_synthesis_indices": (
        np.dtype("<i2"), (154, 4),
        "F5BEB3220E4530FCF28E7F4DA7F07E821074265D118C911D61A590E00753A573",
        True, False),
    "hybrid_synthesis_values": (
        np.dtype("<f4"), (154,),
        "99409FDD9D20D1D7C2BE16BBC1E2159C8042487227C72160850745164C9CEE7F",
        False, False),
    "qmf_synthesis_basis": (
        np.dtype("<f8"), (64, 4, 128),
        "A0C4A55385F6D6C7C92D7615C83AD5FBDA51046D9EF785CAC0B9AC9A760DC527",
        False, False),
    "qmf_synthesis_taps": (
        np.dtype("<f8"), (64, 10, 4),
        "CD7756D060D51FBF02F44C1CE53CB6225B221099505C94C3F58D3BEE6F428150",
        False, False),
}

# Hybrid-band center frequencies measured from the public analysis bank at
# 48 kHz (positive-frequency response peaks).  They are part of the validated
# reference behavior: the runtime uses them only for the fractional-delay band
# phase and the project LFE low-pass, never as filterbank coefficients.
_BAND_CENTER_FREQUENCIES_HZ = np.asarray([
    53.19564095937407,
    26.3876219849709,
    140.9074183269806,
    98.55238901464415,
    234.09258166297573,
    344.8745321543293,
    321.80435900819805,
    401.38762185365727,
    476.19239745597804,
    473.552388917001,
    648.8076026837931,
    719.8745321201852,
    780.1254679168173,
    851.1923973714038,
    1023.8076024849751,
    1155.125467695814,
    1293.0325067063661,
    1668.032511377253,
    2043.0325074138086,
    2456.967490229666,
    2831.96748495363,
    3206.967488172826,
    3581.9675052705525,
    3918.0325089666067,
    4331.967500201844,
    4706.967500350216,
    5043.032510771545,
    5456.9674914391635,
    5831.967490225267,
    6168.0324885741875,
    6543.032508972284,
    6956.96748844665,
    7293.032503259869,
    7668.032503551393,
    8043.032504806491,
    8418.032498852166,
    8793.032502508235,
    9206.967488589786,
    9543.032511549152,
    9956.967486913867,
    10293.032513008677,
    10668.032507835102,
    11043.032513641429,
    11456.967482937946,
    11793.032513984212,
    12206.967486015788,
    12543.032517062376,
    12956.96748635822,
    13331.967492165066,
    13706.967486991198,
    14043.032513086031,
    14456.967488451,
    14793.032511410214,
    15206.967497492202,
    15581.967501147887,
    15956.967495193188,
    16331.96749644876,
    16706.967496740173,
    17043.03251155318,
    17456.96749102773,
    17831.967511425748,
    18168.032509774734,
    18543.032508560515,
    18956.967489228293,
    19293.032499649784,
    19668.032499798002,
    20081.967491033392,
    20418.0324947296,
    20793.032511827063,
    21168.032515046092,
    21543.032509770488,
    21956.967492586176,
    22331.96748862257,
    22706.967493293465,
    23043.03251375972,
    23418.03251061199,
    23831.96749768645,
], dtype=np.float64)


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest().upper()


def _validate_npy_member_header(
        archive: zipfile.ZipFile, member: zipfile.ZipInfo,
        *, name: str, dtype: np.dtype, shape: tuple[int, ...],
        allow_fortran: bool) -> None:
    try:
        with archive.open(member, "r") as payload:
            version = np.lib.format.read_magic(payload)
            if version == (1, 0):
                actual_shape, actual_fortran_order, actual_dtype = (
                    np.lib.format.read_array_header_1_0(
                        payload, max_header_size=4096))
            elif version == (2, 0):
                actual_shape, actual_fortran_order, actual_dtype = (
                    np.lib.format.read_array_header_2_0(
                        payload, max_header_size=4096))
            else:
                raise ValueError(f"unsupported .npy version {version!r}")
            header_size = payload.tell()
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(
            f"invalid public filterbank table .npy header: {member.filename}: "
            f"{exc}") from exc

    actual_shape = tuple(actual_shape)
    actual_dtype = np.dtype(actual_dtype)
    if (actual_shape != shape or actual_dtype != dtype
            or (bool(actual_fortran_order) and not allow_fortran)):
        expected_order = "C-order" if not allow_fortran else "C- or Fortran-order"
        actual_order = "Fortran-order" if actual_fortran_order else "C-order"
        raise ValueError(
            f"invalid public filterbank table .npy header for {name}: expected "
            f"{dtype}{shape} {expected_order}, got "
            f"{actual_dtype}{actual_shape} {actual_order}")
    expected_size = header_size + dtype.itemsize * int(np.prod(shape))
    if member.file_size != expected_size:
        raise ValueError(
            f"invalid public filterbank table .npy payload size for {name}: "
            f"expected {expected_size} bytes including the header, "
            f"got {member.file_size}")


def _validate_table_members(stream) -> None:
    expected = {name + ".npy" for name in _TABLE_SPECS}
    limits = {
        name + ".npy": dtype.itemsize * int(np.prod(shape)) + 4096
        for name, (dtype, shape, _, _, _) in _TABLE_SPECS.items()
    }
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream, "r") as archive:
            members = archive.infolist()
            if (len(members) != len(expected)
                    or {member.filename for member in members} != expected):
                raise ValueError(
                    "public filterbank table archive has an invalid member set")
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("encrypted public filterbank tables are unsupported")
                if member.compress_type not in (
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("unsupported public filterbank table compression")
                if member.file_size > limits[member.filename]:
                    raise ValueError(
                        f"public filterbank table member is unexpectedly large: "
                        f"{member.filename}")
            if sum(member.file_size for member in members) > sum(limits.values()):
                raise ValueError("public filterbank tables expand beyond their size limit")
            for member in members:
                name = member.filename[:-4]
                dtype, shape, _, allow_fortran, _ = _TABLE_SPECS[name]
                _validate_npy_member_header(
                    archive, member, name=name, dtype=dtype, shape=shape,
                    allow_fortran=allow_fortran)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid public filterbank table archive: {exc}") from exc


@lru_cache(maxsize=2)
def _load_tables(path_string: str) -> dict[str, np.ndarray]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"public filterbank table resource not found: {path}")
    with path.open("rb") as stream:
        archive_size = os.fstat(stream.fileno()).st_size
        if archive_size <= 0 or archive_size > 8 << 20:
            raise ValueError(
                f"public filterbank table resource is unexpectedly large: {path}")
        if path.resolve() == DEFAULT_FILTERBANK_DATA.resolve():
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
            actual_archive_hash = digest.hexdigest().upper()
            if actual_archive_hash != _ARCHIVE_SHA256:
                raise ValueError(
                    "public filterbank table archive hash mismatch: "
                    f"expected {_ARCHIVE_SHA256}, got {actual_archive_hash}")
        _validate_table_members(stream)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if set(archive.files) != set(_TABLE_SPECS):
                raise ValueError("public filterbank table archive has an invalid key set")
            result: dict[str, np.ndarray] = {}
            for name, (dtype, shape, expected_hash, _, _) in _TABLE_SPECS.items():
                value = np.asarray(archive[name])
                if value.dtype != dtype or value.shape != shape:
                    raise ValueError(
                        f"invalid public filterbank table {name}: "
                        f"expected {dtype}{shape}, got {value.dtype}{value.shape}")
                actual_hash = _sha256_bytes(value.tobytes(order="C"))
                if actual_hash != expected_hash:
                    raise ValueError(f"public filterbank table hash mismatch: {name}")
                result[name] = np.ascontiguousarray(value)
                result[name].setflags(write=False)
    if int(result["format_version"][0]) != 1:
        raise ValueError("unsupported public filterbank table format version")
    return result


def load_filterbank_tables(
        path: str | Path = DEFAULT_FILTERBANK_DATA) -> dict[str, np.ndarray]:
    """Load the validated project resource used by the public filterbank."""
    cached = _load_tables(str(Path(path).expanduser().resolve()))
    result = {name: value.copy() for name, value in cached.items()}
    for value in result.values():
        value.setflags(write=False)
    return result


def filterbank_fingerprint() -> dict:
    """Return stable identifiers used in compiled-HRTF cache keys."""
    centers = np.ascontiguousarray(_BAND_CENTER_FREQUENCIES_HZ, dtype="<f8")
    return {
        "table_version": FILTERBANK_TABLE_VERSION,
        "archive_sha256": _ARCHIVE_SHA256,
        "band_centers_sha256": _sha256_bytes(centers.tobytes(order="C")),
        "array_sha256": {
            name: spec[2] for name, spec in _TABLE_SPECS.items()
        },
    }


class QmfAnalysis:
    """Batchable 64-band analysis with float64 state and complex128 FFTs."""

    def __init__(self, channels: int,
                 table_data: str | Path = DEFAULT_FILTERBANK_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_filterbank_tables(table_data)
        self.coefficients = np.asarray(
            tables["qmf_analysis_coefficients"], dtype=np.float64)
        self.channels = int(channels)
        self.history = np.zeros((9, self.channels, 64), dtype=np.float64)
        phase = np.arange(64, dtype=np.float64)
        self.premod = np.exp(-1j * np.pi * phase / 128.0).astype(np.complex128)
        self.post = np.exp(
            -1j * 3.0 * (np.arange(64, dtype=np.float64) + 0.5) * np.pi / 128.0
        ).astype(np.complex128)
        self.even_post = (
            1j * ((-1.0) ** np.arange(64, dtype=np.float64))
        ).astype(np.complex128)

    def reset(self) -> None:
        self.history.fill(0.0)

    def process_chunk(self, hops) -> np.ndarray:
        values = np.asarray(hops, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("QMF input contains non-finite values")
        count = values.shape[0]
        joined = np.concatenate((self.history, values), axis=0)
        even = np.zeros_like(values)
        odd = np.zeros_like(values)
        for lag in range(10):
            source = joined[9 - lag:9 - lag + count]
            target = even if lag % 2 == 0 else odd
            target += source * self.coefficients[:, lag][None, None, :]
        self.history[:] = joined[-9:]

        def transform(block):
            prepared = block.astype(np.complex128, copy=False) * self.premod
            transformed = np.fft.fft(prepared, n=128, axis=-1)[..., :64]
            return transformed * self.post

        return np.asarray(transform(odd) + transform(even) * self.even_post,
                          dtype=np.complex128)


class HybridAnalysis:
    """Sparse 64-QMF to 77-hybrid analysis in float64/complex128."""

    def __init__(self, channels: int,
                 table_data: str | Path = DEFAULT_FILTERBANK_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_filterbank_tables(table_data)
        self.low_kernel = np.asarray(
            tables["hybrid_analysis_low_kernel"], dtype=np.float64)
        self.channels = int(channels)
        self.history = np.zeros((12, self.channels, 3, 2), dtype=np.float64)
        self.high_history = np.zeros(
            (6, self.channels, 61), dtype=np.complex128)

    def reset(self) -> None:
        self.history.fill(0.0)
        self.high_history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("hybrid-analysis input contains non-finite values")
        count = values.shape[0]
        low = np.stack((values[:, :, :3].real, values[:, :, :3].imag), axis=-1)
        joined = np.concatenate((self.history, low), axis=0)
        output = np.zeros((count, self.channels, 77, 2), dtype=np.float64)
        for lag in range(13):
            source = joined[12 - lag:12 - lag + count]
            output[:, :, :16] += np.einsum(
                "tcpi,pibo->tcbo", source, self.low_kernel[:, :, lag],
                dtype=np.float64, optimize=False)
        self.history[:] = joined[-12:]

        high_joined = np.concatenate((self.high_history, values[:, :, 3:]), axis=0)
        high = high_joined[:count]
        output[:, :, 16:, 0] = high.real
        output[:, :, 16:, 1] = high.imag
        self.high_history[:] = high_joined[-6:]
        return np.asarray(output[..., 0] + 1j * output[..., 1], dtype=np.complex128)


class HybridSynthesis:
    """Instantaneous sparse 77-hybrid to 64-QMF synthesis map."""

    def __init__(self, channels: int,
                 table_data: str | Path = DEFAULT_FILTERBANK_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_filterbank_tables(table_data)
        indices = np.asarray(tables["hybrid_synthesis_indices"], dtype=np.int64)
        values = np.asarray(tables["hybrid_synthesis_values"], dtype=np.float64)
        if indices.ndim != 2 or indices.shape[1] != 4 or len(indices) != len(values):
            raise ValueError("invalid hybrid synthesis sparse table")
        self.mapping = [
            (int(index[0]), int(index[1]), int(index[2]), int(index[3]), float(value))
            for index, value in zip(indices, values)
        ]
        self.channels = int(channels)

    def reset(self) -> None:
        return None

    def process_chunk(self, hybrid) -> np.ndarray:
        values = np.asarray(hybrid, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 77):
            raise ValueError(f"expected [slots,{self.channels},77], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("hybrid-synthesis input contains non-finite values")
        source = np.stack((values.real, values.imag), axis=-1)
        output = np.zeros((values.shape[0], self.channels, 64, 2), dtype=np.float64)
        for input_band, input_component, output_band, output_component, gain in self.mapping:
            output[:, :, output_band, output_component] += (
                source[:, :, input_band, input_component] * gain)
        return np.asarray(output[..., 0] + 1j * output[..., 1], dtype=np.complex128)


class QmfSynthesis:
    """Rank-4 64-band synthesis with float64 state and accumulation."""

    def __init__(self, channels: int,
                 table_data: str | Path = DEFAULT_FILTERBANK_DATA):
        if channels <= 0:
            raise ValueError("channels must be positive")
        tables = load_filterbank_tables(table_data)
        self.basis = np.asarray(tables["qmf_synthesis_basis"], dtype=np.float64)
        self.taps = np.asarray(tables["qmf_synthesis_taps"], dtype=np.float64)
        if self.basis.shape != (64, 4, 128) or self.taps.shape != (64, 10, 4):
            raise ValueError("invalid QMF synthesis factorization")
        self.channels = int(channels)
        self.rank = 4
        self.history = np.zeros(
            (9, self.channels, 64, self.rank), dtype=np.float64)

    def reset(self) -> None:
        self.history.fill(0.0)

    def process_chunk(self, qmf) -> np.ndarray:
        values = np.asarray(qmf, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, 64):
            raise ValueError(f"expected [slots,{self.channels},64], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("QMF-synthesis input contains non-finite values")
        count = values.shape[0]
        flat = np.stack((values.real, values.imag), axis=-1).reshape(
            count * self.channels, 128)
        modulation = self.basis.reshape(64 * self.rank, 128)
        features = (flat @ modulation.T).reshape(
            count, self.channels, 64, self.rank)
        joined = np.concatenate((self.history, features), axis=0)
        output = np.zeros((count, self.channels, 64), dtype=np.float64)
        for lag in range(10):
            output += np.sum(
                joined[9 - lag:9 - lag + count]
                * self.taps[:, lag, :][None, None, :, :],
                axis=-1, dtype=np.float64)
        self.history[:] = joined[-9:]
        return output


class PublicAnalysis77:
    """Full-rate PCM to the public 77-band hybrid representation."""

    def __init__(self, channels: int):
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        self.qmf = QmfAnalysis(self.channels)
        self.hybrid = HybridAnalysis(self.channels)

    def reset(self) -> None:
        self.qmf.reset()
        self.hybrid.reset()

    def process(self, samples) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim == 1 and self.channels == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(f"samples must have shape [N,{self.channels}]")
        if len(values) % QMF_HOP:
            raise ValueError("sample count must be divisible by the 64-sample QMF hop")
        if not np.isfinite(values).all():
            raise ValueError("samples contain non-finite values")
        hops = values.reshape(-1, QMF_HOP, self.channels).transpose(0, 2, 1)
        return np.asarray(
            self.hybrid.process_chunk(self.qmf.process_chunk(hops)),
            dtype=np.complex128)


class PublicSynthesis77:
    """Public 77-band hybrid representation to full-rate PCM."""

    def __init__(self, channels: int):
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        self.hybrid = HybridSynthesis(self.channels)
        self.qmf = QmfSynthesis(self.channels)

    def reset(self) -> None:
        self.hybrid.reset()
        self.qmf.reset()

    def process(self, hybrid) -> np.ndarray:
        values = np.asarray(hybrid, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.channels, HYBRID_BANDS):
            raise ValueError(
                f"hybrid must have shape [slots,{self.channels},{HYBRID_BANDS}]")
        if not np.isfinite(values).all():
            raise ValueError("hybrid input contains non-finite values")
        qmf = self.hybrid.process_chunk(values)
        time = self.qmf.process_chunk(qmf)
        return np.asarray(time.transpose(0, 2, 1).reshape(-1, self.channels),
                          dtype=np.float64)


def identity_impulse_response(sample_count: int = 4096) -> np.ndarray:
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    total = ((sample_count + QMF_HOP - 1) // QMF_HOP) * QMF_HOP
    impulse = np.zeros((total, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    analysis = PublicAnalysis77(1)
    synthesis = PublicSynthesis77(1)
    return synthesis.process(analysis.process(impulse))[:, 0]


@lru_cache(maxsize=2)
def _hybrid_band_center_frequencies_hz_cached(rate: float) -> np.ndarray:
    centers = np.asarray(
        _BAND_CENTER_FREQUENCIES_HZ * (rate / SAMPLE_RATE), dtype=np.float64)
    centers.setflags(write=False)
    return centers


def hybrid_band_center_frequencies_hz(
        sample_rate_hz: float = SAMPLE_RATE) -> np.ndarray:
    """Return the 77 hybrid-band reference center frequencies."""
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample rate must be positive and finite")
    centers = _hybrid_band_center_frequencies_hz_cached(rate).copy()
    centers.setflags(write=False)
    return centers


def table_info() -> dict:
    return {
        "resource": DEFAULT_FILTERBANK_DATA.name,
        "sample_rate_hz": SAMPLE_RATE,
        "qmf_bands": QMF_BANDS,
        "hybrid_bands": HYBRID_BANDS,
        "hop_samples": QMF_HOP,
        "analysis_synthesis_latency_samples": ANALYSIS_SYNTHESIS_LATENCY_SAMPLES,
        "precision": "float64/complex128",
        "fingerprint": filterbank_fingerprint(),
        "provenance": {
            "qmf": (
                "MPEG-4 AAC/SBR 64 complex QMF analysis (ISO/IEC "
                "14496-3/AMD1:2003 4.B.18.2), polyphase form of the public "
                "640-tap SBR prototype"),
            "hybrid": (
                "3GPP TS 26.405 / ETSI TS 126 405 5.2.2 Table 1 (Q=8/Q=4) "
                "with standard half-bin complex modulation"),
            "synthesis": (
                "causal left inverse of the public analysis bank "
                "(A·W = P, 577-sample QMF delay); 77→64 sparse recombination"),
            "resource": "data/rosella_kernels.npz",
        },
    }


@lru_cache(maxsize=8)
def _hybrid_gain_synthesis_dictionary_cached(count: int) -> np.ndarray:
    total = int(np.ceil(
        (ANALYSIS_SYNTHESIS_LATENCY_SAMPLES + count + 512) / QMF_HOP) * QMF_HOP)
    impulse = np.zeros((total, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    base = PublicAnalysis77(1).process(impulse)[:, 0, :]
    parameter_count = 2 * HYBRID_BANDS
    hybrid = np.zeros(
        (len(base), parameter_count, HYBRID_BANDS), dtype=np.complex128)
    for band in range(HYBRID_BANDS):
        hybrid[:, 2 * band, band] = base[:, band]
        hybrid[:, 2 * band + 1, band] = 1j * base[:, band]
    rendered = PublicSynthesis77(parameter_count).process(hybrid)
    start = ANALYSIS_SYNTHESIS_LATENCY_SAMPLES
    dictionary = np.asarray(
        rendered[start:start + count], dtype=np.float64).copy()
    dictionary.setflags(write=False)
    return dictionary


def hybrid_gain_synthesis_dictionary(sample_count: int) -> np.ndarray:
    """Return the 154-real-parameter analysis/gain/synthesis dictionary.

    Each hybrid band contributes one real-gain and one imaginary-gain column.
    The common 961-sample filterbank latency is removed from every column.
    """
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    dictionary = _hybrid_gain_synthesis_dictionary_cached(count).copy()
    dictionary.setflags(write=False)
    return dictionary


def project_hrir_to_hybrid_gains(
        hrir, *, embedded_delay_samples=None,
        sample_rate_hz: float = SAMPLE_RATE, ridge: float = 1.0e-3,
        ) -> tuple[np.ndarray, dict]:
    """Project FIRs and remove only a known embedded arrival delay.

    Non-zero SOFA ``Data.Delay`` is external and must be passed as zero here.
    A positive onset separated from ``Data.IR`` is de-rotated once, then restored
    once by the runtime field.  A zero-origin FIR keeps its authored complex
    phase and therefore also passes zero.
    """
    values = np.asarray(hrir, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2 or values.shape[2] <= 0:
        raise ValueError("hrir must have shape [M,2,N]")
    if not np.isfinite(values).all():
        raise ValueError("hrir contains non-finite values")
    regularization = float(ridge)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("projection ridge must be finite and non-negative")
    if embedded_delay_samples is None:
        delay = np.zeros(values.shape[:2], dtype=np.float64)
    else:
        delay = np.asarray(embedded_delay_samples, dtype=np.float64)
        if delay.shape != values.shape[:2] or not np.isfinite(delay).all():
            raise ValueError("embedded_delay_samples must have finite shape [M,2]")
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")

    dictionary = _hybrid_gain_synthesis_dictionary_cached(values.shape[2])
    gram = dictionary.T @ dictionary
    scale = float(np.trace(gram)) / gram.shape[0]
    system = gram + regularization * scale * np.eye(gram.shape[0], dtype=np.float64)
    target = values.reshape(-1, values.shape[2]).T
    parameters = np.linalg.solve(system, dictionary.T @ target).T
    parts = parameters.reshape(values.shape[0], 2, 2 * HYBRID_BANDS)
    transfer = np.asarray(parts[..., 0::2] + 1j * parts[..., 1::2],
                          dtype=np.complex128)
    centers = hybrid_band_center_frequencies_hz(rate)
    removal_phase = np.exp(
        2j * np.pi * delay[..., None] * centers[None, None, :] / rate)
    aligned = np.asarray(transfer * removal_phase, dtype=np.complex128)

    reconstructed = dictionary @ parameters.T
    error = target - reconstructed
    reference_energy = np.sum(target * target, axis=0, dtype=np.float64)
    error_energy = np.sum(error * error, axis=0, dtype=np.float64)
    snr = 10.0 * np.log10(
        np.maximum(reference_energy, 1.0e-300)
        / np.maximum(error_energy, 1.0e-300))
    report = {
        "method": "regularized public analysis/gain/synthesis dictionary",
        "dictionary_shape": list(dictionary.shape),
        "real_parameters": 2 * HYBRID_BANDS,
        "ridge": regularization,
        "embedded_delay_samples_min": float(np.min(delay)),
        "embedded_delay_samples_max": float(np.max(delay)),
        "fir_reconstruction_snr_db_median": float(np.median(snr)),
        "fir_reconstruction_snr_db_p05": float(np.percentile(snr, 5.0)),
        "fir_reconstruction_snr_db_min": float(np.min(snr)),
        "maximum_absolute_hybrid_gain": float(np.max(np.abs(aligned))),
        "precision": "float64/complex128",
    }
    return aligned, report


def project_aligned_hrir_to_hybrid_gains(
        aligned_hrir, *, ridge: float = 1.0e-3) -> tuple[np.ndarray, dict]:
    return project_hrir_to_hybrid_gains(aligned_hrir, ridge=ridge)
