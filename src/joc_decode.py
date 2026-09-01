"""解析 JOC 位流并生成对象混合矩阵。

范围：
  - EMDF ID14 的 joc_header、joc_info 和 Huffman joc_data；
  - 差分解码得到 joc_mix_mtx_q；
  - 去量化得到 joc_mix_mtx_dq；
  - 位流自洽验证（joc_data 后剩余 = padding_bits 0..7 + 可能 joc_ext_data）
后续的时间插值、QMF/时域重建和 ``joc_clipgain`` 位于 ``renderer.py``。
Sparse 分支仍缺少实际样本验证。
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
_PUBLIC_TABLE39_EXCERPT_UNUSED = {  # 仅保留作表格差异说明；渲染映射在 joc_qmf.py。
    23: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22],
}

class BR:
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
        b = br.bits(1)
        node = tree[node][b]
    return -node - 1


def get_huff_code(mode, typ, nch):
    if typ == "IDX":
        return H["joc_huff_code_5ch_pos_index_sparse" if nch == 5 else "joc_huff_code_7ch_pos_index_sparse"]
    if typ == "VEC":
        return H["joc_huff_code_coarse_coeff_sparse" if mode == 0 else "joc_huff_code_fine_coeff_sparse"]
    # MTX
    return H["joc_huff_code_coarse_generic" if mode == 0 else "joc_huff_code_fine_generic"]


def parse_joc(payload):
    """解析 id14 载荷（joc() 位流）。返回字段 dict + 解析后剩余位数。"""
    br = BR(payload)
    out = {}
    out["dmx_config_idx"] = br.bits(3)
    out["num_objects_bits"] = br.bits(6)
    out["ext_config_idx"] = br.bits(3)
    n_objects = out["num_objects_bits"] + 1
    n_channels = JOC_NUM_CHANNELS.get(out["dmx_config_idx"])
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
    # joc_data（Huffman）
    for obj, o in enumerate(objs):
        if not o["present"]:
            continue
        nquant = 96 if o["quant_idx"] == 0 else 192
        o["channel_idx"] = []
        o["vec"] = []
        o["mtx"] = []
        for dp in range(o["n_dpoints"]):
            if o["sparse"] == 1:
                # Sparse JOC 使用 VEC/IDX Huffman 树；此分支尚无真实码流验证。
                ci0 = br.bits(3)
                tree = get_huff_code(n_channels, "IDX", n_channels)
                ci = [ci0] + [huff_decode(tree, br) for _ in range(o["n_bands"] - 1)]
                o["channel_idx"].append(ci)
                tree = get_huff_code(o["quant_idx"], "VEC", n_channels)
                vec = [huff_decode(tree, br) for _ in range(o["n_bands"])]
                o["vec"].append(vec)
            else:
                tree = get_huff_code(o["quant_idx"], "MTX", n_channels)
                mtx = [[huff_decode(tree, br) for _ in range(o["n_bands"])]
                       for _ in range(n_channels)]
                o["mtx"].append(mtx)
    out["data_end_bits"] = br.p
    out["remaining_bits"] = len(payload) * 8 - br.p
    out["tail_bytes"] = payload[br.p // 8:]
    return out


def diff_decode(out):
    """6.6.2：差分解码 → joc_mix_mtx_q[obj][dp][ch][pb]。"""
    mix_q = {}
    n_ch = out["n_channels"]
    for obj, o in enumerate(out["objs"]):
        if not o["present"]:
            continue
        nquant = 96 if o["quant_idx"] == 0 else 192
        q = np.zeros((o["n_dpoints"], n_ch, o["n_bands"]), dtype=np.int64)
        for dp in range(o["n_dpoints"]):
            if o["sparse"] == 1:
                # Sparse 差分路径尚无真实码流验证。
                offset = 50 if o["quant_idx"] == 0 else 100
                ci = o["channel_idx"][dp]
                vec = o["vec"][dp]
                for pb in range(o["n_bands"]):
                    ci_mod = ci[0] if pb == 0 else (ci[pb - 1] + ci[pb]) % n_ch
                    for ch in range(n_ch):
                        if ch == ci_mod:
                            if pb == 0:
                                q[dp][ch][pb] = (offset + vec[pb]) % nquant
                            else:
                                q[dp][ch][pb] = (q[dp][ch][pb - 1] + vec[pb]) % nquant
                        else:
                            q[dp][ch][pb] = offset
            else:
                offset = 48 if o["quant_idx"] == 0 else 96
                mtx = o["mtx"][dp]
                for ch in range(n_ch):
                    q[dp][ch][0] = (offset + mtx[ch][0]) % nquant
                    for pb in range(1, o["n_bands"]):
                        q[dp][ch][pb] = (q[dp][ch][pb - 1] + mtx[ch][pb]) % nquant
        mix_q[obj] = q
    return mix_q


def dequantize(out, mix_q):
    """6.6.4：去量化 → joc_mix_mtx_dq。"""
    mix_dq = {}
    for obj, o in enumerate(out["objs"]):
        if not o["present"]:
            continue
        nquant = 96 if o["quant_idx"] == 0 else 192
        q = mix_q[obj]
        dq = (q.astype(np.float64) - nquant / 2) * 820 / (4096 * (1 + o["quant_idx"]))
        mix_dq[obj] = dq
    return mix_dq
