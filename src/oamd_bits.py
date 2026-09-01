"""OAMD 位载荷 → 16 个对象槽的 q1/q2/q3 增量状态。

槽 0 是 bed/LFE；槽 1..15 对应输出 ch1..15 的对象元数据。
"""
import numpy as np

from variant_error import UnsupportedVariantError, bytes_descriptor

Q1_OFF = 192
N_Q12 = 62
N_Q3 = 15
SAMPLE_OFFSET_INDEX = (8, 16, 18, 24)
RAMP_DURATIONS = (0, 512, 1536)
RAMP_DURATION_INDEX = (
    32, 64, 128, 256, 320, 480, 1000, 1001,
    1024, 1600, 1601, 1602, 1920, 2000, 2002, 2048,
)


def q_of(k, n):
    q = int(np.floor(32768.0 * k / n + 0.5))
    return min(32767, q)


def _payload_bits(bits_one):
    if isinstance(bits_one, (bytes, bytearray, memoryview)):
        src = np.frombuffer(bits_one, dtype=np.uint8)
    else:
        src = np.asarray(bits_one, dtype=np.uint8)
    if src.ndim == 1 and src.shape[0] in (536, 552):
        bits = src
    elif src.size in (67, 69):
        bits = np.unpackbits(src.reshape(-1), bitorder="big")
    else:
        raw = (bytes(bits_one) if isinstance(bits_one, (bytes, bytearray, memoryview))
               else np.asarray(bits_one, dtype=np.uint8).tobytes())
        raise UnsupportedVariantError(
            "oamd", f"payload_length_{len(raw)}B",
            f"发现未覆盖的 OAMD 载荷长度 {len(raw)}B",
            details={
                "supported_payload_bytes": [67, 69],
                "payload": bytes_descriptor(raw),
                "repair_hint": "检查 OAMD header、element 数量及可选字段造成的位偏移变化",
            })
    raw_payload = np.packbits(bits, bitorder="big").tobytes()
    return bits, raw_payload


class _BitReader:
    def __init__(self, bits):
        self.bits = bits
        self.position = 0

    def read(self, count):
        end = self.position + count
        if end > len(self.bits):
            raise ValueError(f"OAMD 位流越界: bit={self.position}, need={count}")
        value = 0
        for bit in self.bits[self.position:end]:
            value = (value << 1) | int(bit)
        self.position = end
        return value

    def skip(self, count):
        self.read(count)


def _variable_bits(reader, width, max_groups=5):
    value = 0
    for _ in range(max_groups + 1):
        value += reader.read(width)
        if not reader.read(1):
            return value
        value = (value + 1) << width
    raise ValueError(f"OAMD variable_bits({width}) 延伸组过多")


def _update_timing(bits, alternate_object_present, element_count, raw_payload):
    """按 OAMD element/MDUpdateInfo 读取位置块的开始偏移和 ramp 时长。"""
    reader = _BitReader(bits)
    reader.position = 14
    for _ in range(element_count):
        element_index = reader.read(4)
        element_length = _variable_bits(reader, 4)
        element_end = reader.position + element_length + 1
        reader.skip(5 if alternate_object_present else 1)
        if element_index == 1:
            offset_code = reader.read(2)
            if offset_code == 0:
                sample_offset = 0
            elif offset_code == 1:
                sample_offset = SAMPLE_OFFSET_INDEX[reader.read(2)]
            elif offset_code == 2:
                sample_offset = reader.read(5)
            else:
                raise UnsupportedVariantError(
                    "oamd", "md_sample_offset_mode",
                    "OAMD 使用了当前未覆盖的 MD sample-offset 模式",
                    details={"payload": bytes_descriptor(raw_payload)})

            block_count = reader.read(3) + 1
            blocks = []
            for _block in range(block_count):
                block_offset_factor = reader.read(6)
                block_offset = sample_offset + block_offset_factor * 32
                ramp_code = reader.read(2)
                if ramp_code == 3:
                    if reader.read(1):
                        ramp_duration = RAMP_DURATION_INDEX[reader.read(4)]
                    else:
                        ramp_duration = reader.read(11)
                else:
                    ramp_duration = RAMP_DURATIONS[ramp_code]
                blocks.append((block_offset, ramp_duration))
            if len(blocks) != 1:
                raise UnsupportedVariantError(
                    "oamd", "multiple_position_blocks",
                    "OAMD 一帧含多个对象位置更新块，固定位置窗口不能安全套用",
                    details={
                        "block_count": len(blocks),
                        "blocks": blocks,
                        "payload": bytes_descriptor(raw_payload),
                        "repair_hint": "按 ObjectInfoBlock 顺序逐块解析坐标，再生成分段 ADM ramp",
                    })
            return blocks[0]
        reader.position = element_end
    raise UnsupportedVariantError(
        "oamd", "missing_object_element",
        "OAMD 中没有 object element (element_index=1)",
        details={"payload": bytes_descriptor(raw_payload)})


def frame_update(bits_one):
    """单帧 OAMD → 位置字段增量及其 sample offset/ramp duration。"""
    bits, raw_payload = _payload_bits(bits_one)
    header = {
        "version": int((bits[0] << 1) | bits[1]),
        "objects_minus_one": int(sum(int(bits[2 + i]) << (4 - i) for i in range(5))),
        "dynamic_object_only": int(bits[7]),
        "lfe_present": int(bits[8]),
        "alternate_object_present": int(bits[9]),
        "element_count": int(sum(int(bits[10 + i]) << (3 - i) for i in range(4))),
    }
    if raw_payload[:2] != b"\x1f\x88":
        raise UnsupportedVariantError(
            "oamd", f"header_signature_{raw_payload[:2].hex()}",
            "OAMD 长度已知，但 header 与当前位置字段布局不一致",
            details={
                "supported_header_prefix_hex": "1f88",
                "header_probe": header,
                "payload": bytes_descriptor(raw_payload),
                "repair_hint": "按新 header 的 program assignment 和 element 布局重新定位对象位置字段",
            })
    block_offset, ramp_duration = _update_timing(
        bits, bool(header["alternate_object_present"]), header["element_count"], raw_payload)
    weights = 1 << np.arange(7, -1, -1)
    out = {}
    for obj in range(16):
        start = 112 + 31 * (obj - 3)
        wq1 = int((bits[start:start + 8] * weights).sum())
        wq2 = int((bits[start + 8:start + 16] * weights).sum())
        wq3 = int((bits[start + 16:start + 24] * weights).sum())
        if obj and (wq1 >> 6 != 3 or wq2 & 2 != 2 or wq3 & 0x1F != 1):
            raise UnsupportedVariantError(
                "oamd", "position_layout_signature",
                "OAMD 长度和 header 已知，但对象位置字段标记或位偏移发生变化",
                details={
                    "object_slot": obj,
                    "position_start_bit": start,
                    "q1_window_hex": f"{wq1:02x}",
                    "q2_window_hex": f"{wq2:02x}",
                    "q3_window_hex": f"{wq3:02x}",
                    "header_probe": header,
                    "payload": bytes_descriptor(raw_payload),
                    "repair_hint": "解析 OAMD element 可选字段并更新每个对象的位置窗口偏移",
                })
        k1 = wq1 - Q1_OFF
        if obj == 0:
            out[(0, "q1")] = None
            out[(0, "q2")] = None
        else:
            out[(obj, "q1")] = q_of(k1, N_Q12) if 0 <= k1 <= N_Q12 else None
        k2 = wq2 >> 2
        if obj != 0:
            out[(obj, "q2")] = q_of(k2, N_Q12) if 0 <= k2 <= N_Q12 else None
        k3 = (wq2 & 1) * 8 + (wq3 >> 5)
        out[(obj, "q3")] = q_of(k3, N_Q3) if 0 <= k3 <= N_Q3 else None
    return {
        "values": out,
        "block_offset_samples": block_offset,
        "ramp_duration_samples": ramp_duration,
    }


def frame_update_values(bits_one):
    """兼容接口：只返回 ``{(slot, field): q|None}``。"""
    return frame_update(bits_one)["values"]


class JocFieldState:
    """未更新/非法窗保持旧值；slot0 q1/q2 的 DLL 初值为中心 16384。"""

    def __init__(self):
        self.q = {(obj, field): 0 for obj in range(16)
                  for field in ("q1", "q2", "q3")}
        self.q[(0, "q1")] = 16384
        self.q[(0, "q2")] = 16384

    def apply(self, updates):
        for key, value in updates.items():
            if value is not None:
                self.q[key] = value
        return self

    def snapshot(self):
        return dict(self.q)
