"""解包 EVO MD-set evolution 载荷，返回各 payload ID 的字节数据和位偏移。"""
_MARK = "1001001000000"

# 各子载荷字段: (id, 头部前缀, 同步标记, 后缀常量, 尺寸域位数, 是否有转义)
# 头部前缀 = 5 位 id 的 MSB 二进制（id11 字段前另有 5 位容器前导 00000）
# 尺寸域单位 = nibble（4 位）。id14 有转义：9 位值=0 → 再读 9 位 = 字节数。
_LAYOUT = [
    (11, "0000001011", "010000000000000", 8, False),
    (14, "01110",      "01000000000000",  9, True),
    (2,  "00010",      "000100",          7, False),
    (1,  "00001",      "1110000000000000000000000000", 4, False),
    (30, "11110",      "1110000000000000000000000000", 4, False),
]


def _msb_bits(data: bytes):
    return [(x >> (7 - i)) & 1 for x in data for i in range(8)]


def _val(bits, off, n):
    v = 0
    for b in bits[off:off + n]:
        v = (v << 1) | b
    return v


class _LooseSkip(Exception):
    def __init__(self, ident):
        self.ident = ident


def unpack_evolution(payload: bytes, loose=False):
    """解包 evolution 载荷 → (subs, offsets)。subs 键为 id 整数。
    loose=True 时对每个 id 的 (前缀+标记+后缀) 全模式做位流重同步扫描
    （不同编码流的子载荷次序/内部常量可有合法差异，如 kanata 的 id11）。"""
    bits = _msb_bits(payload)
    pos = 0
    subs = {}
    offsets = {}
    for ident, pref, suff, sbits, escape in _LAYOUT:
        pat = pref + _MARK + suff
        if loose:
            hit = -1
            for i in range(pos, len(bits) - len(pat)):
                if ''.join(map(str, bits[i:i + len(pat)])) == pat:
                    hit = i
                    break
            if hit < 0:
                continue
            pos = hit + len(pat)
        else:
            for name, const, expect in (("前缀", bits[pos:pos + len(pref)], pref),
                                        ("标记", bits[pos + len(pref):pos + len(pref) + len(_MARK)], _MARK),
                                        ("后缀", bits[pos + len(pref) + len(_MARK):
                                                     pos + len(pref) + len(_MARK) + len(suff)], suff)):
                got = ''.join(map(str, const))
                if got != expect:
                    raise ValueError(f"id={ident}: {name}常量不匹配 @bit{pos} got={got} want={expect}")
            pos += len(pref) + len(_MARK) + len(suff)
        n_nib = _val(bits, pos, sbits)
        pos += sbits
        if escape and n_nib == 1:
            n_nib = 512 + _val(bits, pos, sbits)
            pos += sbits
        n_bits = n_nib * 4
        body = bits[pos:pos + n_bits]
        pos += n_bits
        b = bytearray(len(body) // 8)
        for i in range(0, len(body) // 8 * 8, 8):
            v = 0
            for x in body[i:i + 8]:
                v = (v << 1) | x
            b[i // 8] = v
        subs[ident] = bytes(b)
        offsets[ident] = pos
    return subs, offsets
