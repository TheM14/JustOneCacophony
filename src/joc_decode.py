"""解析 JOC 位流并生成对象混合矩阵。

范围：
  - EMDF ID14 的 joc_header、joc_info 和 Huffman joc_data；
  - 差分还原得到 joc_mix_mtx_q（dense MTX 与 sparse IDX/VEC 两条语法）；
  - 去量化得到 joc_mix_mtx_dq；
  - 位流自洽验证（joc_data 后剩余 = padding_bits 0..7 + 可能 joc_ext_data）
后续的时间插值、QMF/时域重建和 ``joc_clipgain`` 位于 ``renderer.py``。
公式与符号定义见 ``docs/math.md`` 第 2 节。
"""
from pathlib import Path

import numpy as np

# 格式：节点数组 [left, right]；正 = 内部节点索引，负 = 叶（值 = -node-1）
_TABLES_PATH = Path(__file__).resolve().parent.parent / "data" / "tables.npz"
_HUFF_NAMES = (
    "joc_huff_code_coarse_generic",
    "joc_huff_code_fine_generic",
    "joc_huff_code_coarse_coeff_sparse",
    "joc_huff_code_fine_coeff_sparse",
    "joc_huff_code_5ch_pos_index_sparse",
    "joc_huff_code_7ch_pos_index_sparse",
)


def _load_huff_tables():
    with np.load(_TABLES_PATH) as tables:
        return {
            name: np.asarray(tables[name], dtype=np.int64).tolist()
            for name in _HUFF_NAMES
        }


H = _load_huff_tables()

JOC_NUM_CHANNELS = {0: 5, 1: 7, 2: 7, 3: 5, 4: 7}          # Table 33
JOC_NUM_BANDS = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 12, 6: 15, 7: 23}  # Table 35
JOC_NUM_QUANT = {0: 96, 1: 192}                            # Table 51
# dense 的量化零点就是 nquant/2；sparse 的递推起点比它高 2 个量化步。
JOC_DENSE_OFFSET = {0: 48, 1: 96}
JOC_SPARSE_OFFSET = {0: 50, 1: 100}


class BR:
    """MSB-first 位读取器；位置以载荷内的 bit offset 表示。"""

    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.d[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v


def huff_decode(tree, br):
    node = 0
    while node >= 0:
        node = tree[node][br.bits(1)]
    return -node - 1


def get_huff_code(mode, typ, nch):
    if typ == "IDX":
        return H["joc_huff_code_5ch_pos_index_sparse" if nch == 5
                 else "joc_huff_code_7ch_pos_index_sparse"]
    if typ == "VEC":
        return H["joc_huff_code_coarse_coeff_sparse" if mode == 0
                 else "joc_huff_code_fine_coeff_sparse"]
    return H["joc_huff_code_coarse_generic" if mode == 0
             else "joc_huff_code_fine_generic"]           # MTX


def parse_joc(payload):
    """解析 id14 载荷（joc() 位流）。返回字段 dict + 解析后剩余位数。"""
    br = BR(payload)
    out = {}
    out["dmx_config_idx"] = br.bits(3)
    out["num_objects_bits"] = br.bits(6)
    out["ext_config_idx"] = br.bits(3)
    n_objects = out["num_objects_bits"] + 1
    n_channels = JOC_NUM_CHANNELS.get(out["dmx_config_idx"])
    if n_channels is None:
        raise ValueError(f"未知的 JOC downmix 配置 {out['dmx_config_idx']}")
    out["n_objects"], out["n_channels"] = n_objects, n_channels
    out["clipgain_x_bits"] = br.bits(3)
    out["clipgain_y_bits"] = br.bits(5)
    out["seq_count_bits"] = br.bits(10)
    # clipgain = 1 + (y/32)·2^(x−4)，值域为 [1, 8.75]。
    out["clipgain"] = 1 + out["clipgain_y_bits"] / 32.0 * 2 ** (out["clipgain_x_bits"] - 4)
    objs = []
    for obj in range(n_objects):
        o = {}
        o["present"] = br.bits(1)
        if o["present"]:
            o["num_bands_idx"] = br.bits(3)
            o["n_bands"] = JOC_NUM_BANDS[o["num_bands_idx"]]
            o["sparse"] = br.bits(1)
            o["quant_idx"] = br.bits(1)
            o["slope_idx"] = br.bits(1)
            o["num_dpoints_bits"] = br.bits(1)
            o["n_dpoints"] = o["num_dpoints_bits"] + 1
            if o["slope_idx"] == 1:
                o["offset_ts"] = [br.bits(5) + 1 for _ in range(o["n_dpoints"])]
        objs.append(o)
    out["objs"] = objs
    # joc_data（Huffman）：dense 逐声道逐带读 MTX；sparse 读 IDX 后读 VEC。
    for obj, o in enumerate(objs):
        if not o["present"]:
            continue
        o["channel_idx"] = []
        o["vec"] = []
        o["mtx"] = []
        for dp in range(o["n_dpoints"]):
            if o["sparse"] == 1:
                tree = get_huff_code(o["quant_idx"], "IDX", n_channels)
                idx = [br.bits(3)]
                idx += [huff_decode(tree, br) for _ in range(o["n_bands"] - 1)]
                tree = get_huff_code(o["quant_idx"], "VEC", n_channels)
                o["channel_idx"].append(idx)
                o["vec"].append([huff_decode(tree, br) for _ in range(o["n_bands"])])
                o["mtx"].append(None)
            else:
                tree = get_huff_code(o["quant_idx"], "MTX", n_channels)
                mtx = [[huff_decode(tree, br) for _ in range(o["n_bands"])]
                       for _ in range(n_channels)]
                o["mtx"].append(mtx)
                o["channel_idx"].append(None)
                o["vec"].append(None)
    out["data_end_bits"] = br.p
    out["remaining_bits"] = len(payload) * 8 - br.p
    out["tail_bytes"] = payload[br.p // 8:]
    return out


def reconstruct_dense(o, dp, n_ch):
    """Dense 差分还原 → 量化矩阵 ``[ch][pb]``。

    每个核心声道各自从 ``nquant/2``（去量化 0）出发，沿参数带累加 MTX 符号。
    """
    nquant = JOC_NUM_QUANT[o["quant_idx"]]
    offset = JOC_DENSE_OFFSET[o["quant_idx"]]
    mtx = o["mtx"][dp]
    if mtx is None:
        raise ValueError("Dense JOC 对象缺少 MTX 符号")
    q = np.zeros((n_ch, o["n_bands"]), dtype=np.int64)
    for ch in range(n_ch):
        q[ch][0] = (offset + mtx[ch][0]) % nquant
        for pb in range(1, o["n_bands"]):
            q[ch][pb] = (q[ch][pb - 1] + mtx[ch][pb]) % nquant
    return q


def reconstruct_sparse(o, dp, n_ch):
    """Sparse 差分还原 → 量化矩阵 ``[ch][pb]``。

    每个参数带只有一个 active 声道：
      - ``active[0]`` 是 3 bit 绝对声道号，其后由 IDX 符号累加得到，
        因此递推锚点是**已重建**的 active 声道，而不是编码器送的符号本身；
      - 系数是一个跨参数带连续的单累加器（active 声道切换时**不**重置），
        起点为 sparse offset，增量为 VEC 符号；
      - 非 active 项取 ``nquant/2``，即去量化后的 0。

    IDX 符号取值恒为 ``0..n_ch-1``（Huffman 叶数即声道数），
    故 ``(active + idx) % n_ch`` 与单次条件减等价。
    """
    if n_ch not in (5, 7):
        raise ValueError(f"Sparse JOC 需要 5 或 7 个核心声道，实际 {n_ch}")
    nquant = JOC_NUM_QUANT[o["quant_idx"]]
    offset = JOC_SPARSE_OFFSET[o["quant_idx"]]
    n_bands = o["n_bands"]
    idx = o["channel_idx"][dp]
    vec = o["vec"][dp]
    if idx is None or vec is None:
        raise ValueError("Sparse JOC 对象缺少 channel_idx/vec 符号")
    if len(idx) != n_bands or len(vec) != n_bands:
        raise ValueError(
            f"Sparse JOC 维度不符: bands={n_bands}, idx={len(idx)}, vec={len(vec)}")
    if not 0 <= idx[0] < n_ch:
        raise ValueError(f"Sparse JOC 初始声道 {idx[0]} 超出 {n_ch} 个声道")

    q = np.full((n_ch, n_bands), nquant // 2, dtype=np.int64)
    active = idx[0]
    coefficient = offset
    for pb in range(n_bands):
        if pb:
            active = (active + idx[pb]) % n_ch
        coefficient = (coefficient + vec[pb]) % nquant
        q[active][pb] = coefficient
    return q


def diff_decode(out):
    """差分还原 → joc_mix_mtx_q[obj][dp][ch][pb]。"""
    mix_q = {}
    n_ch = out["n_channels"]
    for obj, o in enumerate(out["objs"]):
        if not o["present"]:
            continue
        q = np.zeros((o["n_dpoints"], n_ch, o["n_bands"]), dtype=np.int64)
        for dp in range(o["n_dpoints"]):
            if o["sparse"] == 1:
                q[dp] = reconstruct_sparse(o, dp, n_ch)
            else:
                q[dp] = reconstruct_dense(o, dp, n_ch)
        mix_q[obj] = q
    return mix_q


def dequantize(out, mix_q):
    """去量化 → joc_mix_mtx_dq。

    Sparse 的非 active 项在 ``joc_mix_mtx_q`` 中取 ``nquant/2``，
    因此与 dense 共用同一条去量化公式即得到 0。
    """
    mix_dq = {}
    for obj, o in enumerate(out["objs"]):
        if not o["present"]:
            continue
        nquant = JOC_NUM_QUANT[o["quant_idx"]]
        q = mix_q[obj]
        dq = (q.astype(np.float64) - nquant / 2) * 820 / (4096 * (1 + o["quant_idx"]))
        mix_dq[obj] = dq
    return mix_dq
