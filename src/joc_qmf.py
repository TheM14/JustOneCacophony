"""实现 JOC QMF、参数带映射和矩阵时间插值。

静态表保存在 ``data`` 中；NumPy 批量函数保持各通道、各对象的状态彼此独立。
"""
from pathlib import Path

import numpy as np


N = 64
_TABLES = np.load(Path(__file__).resolve().parent.parent / "data" / "tables.npz")
ANALYSIS_WINDOW = np.asarray(_TABLES["analysis_window"], dtype=np.float64)
QMF5_WINDOW = np.asarray(_TABLES["qmf5_window"], dtype=np.float64)


# 子带到参数带的映射表。
_TABLE39 = {
    23: "0 1 2 3 4 5 6 7 8 9 10 11 12 12 13 13 14 14 15 15 16 16 16 17 17 17 18 18 18 18 19 19 19 19 19 20 20 20 20 20 20 21 21 21 21 21 21 21 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22",
    15: "0 1 2 3 4 5 6 7 8 9 9 10 10 11 11 11 12 12 12 12 13 13 13 13 13 13 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14",
    12: "0 1 2 3 4 4 5 5 6 6 6 7 7 7 8 8 8 8 9 9 9 9 9 10 10 10 10 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11",
    9: "0 1 2 3 3 3 4 5 5 6 6 6 7 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8",
    7: "0 1 2 2 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6",
    5: "0 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4",
    3: "0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2",
    1: "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
}
_PB_MAP = {k: np.fromstring(v, dtype=np.int64, sep=" ") for k, v in _TABLE39.items()}


def sb_to_pb_table(n_bands):
    try:
        return _PB_MAP[n_bands]
    except KeyError as exc:
        raise ValueError(f"不支持的 JOC 参数带数: {n_bands}") from exc


def interp_matrix(obj_info, dq, prev, n_ts=24):
    """按数据点和跨帧状态插值，返回 ``[channel,subband,timeslot]``。"""
    dq_sb = np.asarray(dq, dtype=np.float64)[:, :, sb_to_pb_table(dq.shape[2])]
    previous = np.asarray(prev, dtype=np.float64)
    n_dp = obj_info["n_dpoints"]
    slope = obj_info["slope_idx"]
    if slope == 0:
        if n_dp == 1:
            alpha = (np.arange(n_ts, dtype=np.float64) + 1.0) / n_ts
            return previous[:, :, None] * (1.0 - alpha) + dq_sb[0, :, :, None] * alpha
        half = n_ts // 2
        a0 = (np.arange(half, dtype=np.float64) + 1.0) / half
        a1 = (np.arange(n_ts - half, dtype=np.float64) + 1.0) / (n_ts - half)
        first = previous[:, :, None] * (1.0 - a0) + dq_sb[0, :, :, None] * a0
        second = dq_sb[0, :, :, None] * (1.0 - a1) + dq_sb[1, :, :, None] * a1
        return np.concatenate((first, second), axis=2)

    ts = np.arange(n_ts)
    offsets = obj_info.get("offset_ts", [])
    if n_dp == 1:
        return np.where(ts[None, None, :] < offsets[0], previous[:, :, None], dq_sb[0, :, :, None])
    out = np.where(ts[None, None, :] < offsets[0], previous[:, :, None], dq_sb[0, :, :, None])
    return np.where(ts[None, None, :] < offsets[1], out, dq_sb[1, :, :, None])


def qmf_analysis_step(fifo, ring):
    """分析 QMF 的单时隙入口；批量入口见 :func:`qmf_analysis_frame`。"""
    f = np.asarray(fifo, dtype=np.float64)
    r = np.asarray(ring, dtype=np.float64)
    single = f.ndim == 2
    if single:
        f, r = f[None, ...], r[None, ...]
    x, new = qmf_analysis_frame(f, r[:, None, :])
    interleaved = np.empty((len(f), 128), dtype=np.float64)
    interleaved[:, 0::2] = x[:, :, 0].real
    interleaved[:, 1::2] = x[:, :, 0].imag
    return (interleaved[0], new[0]) if single else (interleaved, new)


def qmf_analysis_frame(fifo, rings):
    """批量完成整帧时隙的分析窗、旋转和 64 点 FFT。"""
    f = np.asarray(fifo, dtype=np.float64)
    r = np.asarray(rings, dtype=np.float64)
    if f.ndim != 3 or f.shape[1:] != (9, 64) or r.ndim != 3 or r.shape[0] != f.shape[0] or r.shape[2] != 64:
        raise ValueError(f"analysis QMF shape 错误: fifo={f.shape}, rings={r.shape}")
    slots = r.shape[1]
    seq = np.concatenate((f[:, ::-1, :], r), axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(seq, 9, axis=1)
    history = windows[:, :slots].transpose(0, 1, 3, 2)[:, :, ::-1, :]

    t = ANALYSIS_WINDOW
    v36 = np.sum(history[:, :, 0::2, :] * t[[8, 6, 4, 2, 0]][None, None], axis=2)
    v40 = np.sum(history[:, :, 1::2, :] * t[[7, 5, 3, 1]][None, None], axis=2) + r * t[9]
    pairs = np.stack((v40, v36), axis=-1)
    kernel = pairs.reshape(len(f), slots, 16, 4, 2)[:, :, ::-1, ::-1, :].reshape(len(f), slots, 128)

    k = np.arange(64, dtype=np.float64)
    a = 0.5 * np.sin(np.pi * k / 128.0)
    b = 0.5 * np.cos(np.pi * k / 128.0)
    re, im = kernel[:, :, 0::2], kernel[:, :, 1::2]
    freq = np.fft.fft((im * a - re * b) + 1j * (im * b + re * a), axis=2) / 64.0
    src = np.empty((len(f), slots, 128), dtype=np.float64)
    src[:, :, 0::2], src[:, :, 1::2] = freq.real, freq.imag
    out = np.empty_like(src).reshape(len(f), slots, 32, 4)
    out[:, :, :, 0] = src[:, :, 0:64:2]
    out[:, :, :, 1] = -src[:, :, 1:64:2]
    out[:, :, :, 2] = src[:, :, 126:62:-2]
    out[:, :, :, 3] = src[:, :, 127:63:-2]
    flat = out.reshape(len(f), slots, 128)
    complex_qmf = (flat[:, :, 0::2] + 1j * flat[:, :, 1::2]).transpose(0, 2, 1)
    combined = np.concatenate((f[:, ::-1, :], r), axis=1)
    new_fifo = combined[:, -9:, :][:, ::-1, :].copy()
    return complex_qmf, new_fifo


_SURROUND_DC_A = np.array([
    -.0006242550443857908, -.0019234686624258757, -.0042654648423194885,
    -.008168308064341545, -.014327201060950756, -.023759860545396805,
    -.03757232800126076, -.05577569454908371, -.07568276673555374,
    -.09172472357749939, -.5979374051094055, -.09172472357749939,
    -.07568276673555374, -.05577569454908371, -.03757232800126076,
    -.023759860545396805, -.014327201060950756, -.008168308064341545,
    -.0042654648423194885, -.0019234686624258757, -.0006242550443857908,
], dtype=np.float64)
_SURROUND_DC_B = np.array([
    .0013996040215715766, .003839150769636035, .007512642536312342,
    .012419373728334904, .018367428332567215, .0249701626598835,
    .03167900815606117, .03785000368952751, .04283412545919418,
    .04607561603188515, .047200120985507965, .04607561603188515,
    .04283412545919418, .03785000368952751, .03167900815606117,
    .0249701626598835, .018367428332567215, .012419373728334904,
    .007512642536312342, .003839150769636035, .0013996040215715766,
], dtype=np.float64)
_SURROUND_DC_C = _SURROUND_DC_B + 1j * _SURROUND_DC_A


def surround_post_frame(x, delay, dc_hist):
    """处理 Ls/Rs 的 10 槽延迟、-j 旋转和 band-0 FIR。"""
    src = np.asarray(x, dtype=np.complex128)
    qdelay = np.asarray(delay, dtype=np.complex128).copy()
    hist = np.asarray(dc_hist, dtype=np.complex128).copy()
    if src.shape != (2, 64, 24):
        raise ValueError(f"surround QMF shape 错误: {src.shape}")
    out = np.empty_like(src)
    for group in range(0, 24, 4):
        current = src[:, :, group:group + 4].transpose(0, 2, 1)
        queued = np.concatenate((qdelay, current), axis=1)
        block = -1j * queued[:, :4, :]
        qdelay = queued[:, 4:, :]
        dc_buf = np.concatenate((hist, current[:, :, 0]), axis=1)
        windows = np.lib.stride_tricks.sliding_window_view(dc_buf, 21, axis=1)
        block[:, :, 0] = 2.0 * np.sum(windows * _SURROUND_DC_C[None, None, :], axis=2)
        hist = dc_buf[:, 4:]
        out[:, :, group:group + 4] = block.transpose(0, 2, 1)
    return out, qdelay, hist
