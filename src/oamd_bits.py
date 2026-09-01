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
OBJECT_ELEMENT_ID = 1
TRIM_ELEMENT_ID = 2
EXTENDED_OBJECT_ELEMENT_ID = 5
POSITION_WINDOW_END_BIT = 112 + 31 * (15 - 3) + 24


def q_of(k, n):
    q = int(np.floor(32768.0 * k / n + 0.5))
    return min(32767, q)


def _payload_bits(bits_one):
    if isinstance(bits_one, (bytes, bytearray, memoryview)):
        raw_payload = bytes(bits_one)
        bits = np.unpackbits(
            np.frombuffer(raw_payload, dtype=np.uint8), bitorder="big")
    else:
        src = np.asarray(bits_one)
        if src.ndim != 1:
            raw = np.asarray(bits_one, dtype=np.uint8).tobytes()
            raise UnsupportedVariantError(
                "oamd", "payload_shape",
                "OAMD 载荷必须是一维 byte 或 bit 序列",
                details={"shape": list(src.shape), "payload": bytes_descriptor(raw)})
        is_bit_vector = bool(src.size) and bool(np.all((src == 0) | (src == 1)))
        if is_bit_vector:
            if src.size % 8:
                raw = np.packbits(src.astype(np.uint8), bitorder="big").tobytes()
                raise UnsupportedVariantError(
                    "oamd", "payload_bit_alignment",
                    "OAMD bit 载荷没有按整字节对齐",
                    details={
                        "payload_bits": int(src.size),
                        "payload": bytes_descriptor(raw),
                    })
            bits = src.astype(np.uint8, copy=False)
            raw_payload = np.packbits(bits, bitorder="big").tobytes()
        else:
            try:
                values = src.astype(np.int64, copy=False)
            except (TypeError, ValueError, OverflowError) as exc:
                raise UnsupportedVariantError(
                    "oamd", "payload_type",
                    "OAMD 载荷不能转换为 byte 序列",
                    details={"dtype": str(src.dtype), "parser_error": str(exc)}) from exc
            if np.any(values < 0) or np.any(values > 255):
                raise UnsupportedVariantError(
                    "oamd", "payload_byte_range",
                    "OAMD byte 载荷包含 0..255 之外的值",
                    details={"dtype": str(src.dtype), "payload_values": int(src.size)})
            raw_payload = values.astype(np.uint8).tobytes()
            bits = np.unpackbits(
                np.frombuffer(raw_payload, dtype=np.uint8), bitorder="big")
    return bits, raw_payload


class _BitReader:
    def __init__(self, bits, position=0, limit=None):
        self.bits = bits
        self.position = int(position)
        self.limit = len(bits) if limit is None else int(limit)

    def read(self, count):
        end = self.position + count
        if end > self.limit:
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


def _element_details(elements):
    return [{
        "ordinal": element["ordinal"],
        "element_id": element["element_id"],
        "size_bytes": element["size_bytes"],
        "header_start_bit": element["header_start_bit"],
        "body_start_bit": element["body_start_bit"],
        "body_end_bit": element["body_end_bit"],
        "discard_unknown": element["discard_unknown"],
    } for element in elements]


def _parse_elements(bits, header, header_end, raw_payload):
    """建立 OAMD element 目录；element size 表示其后 body 的字节数。"""
    reader = _BitReader(bits, header_end)
    elements = []
    for ordinal in range(header["element_count"]):
        header_start = reader.position
        try:
            element_id = reader.read(4)
            size_bytes = _variable_bits(reader, 4) + 1
        except ValueError as exc:
            raise UnsupportedVariantError(
                "oamd", "element_header_truncated",
                "OAMD element header 不完整",
                details={
                    "element_ordinal": ordinal,
                    "header_start_bit": header_start,
                    "payload": bytes_descriptor(raw_payload),
                    "parser_error": str(exc),
                }) from exc

        body_start = reader.position
        body_end = body_start + size_bytes * 8
        if body_end > len(bits):
            raise UnsupportedVariantError(
                "oamd", "element_bounds",
                "OAMD element 声明长度超过 payload 边界",
                details={
                    "element_ordinal": ordinal,
                    "element_id": element_id,
                    "size_bytes": size_bytes,
                    "body_start_bit": body_start,
                    "body_end_bit": body_end,
                    "payload_bits": len(bits),
                    "payload": bytes_descriptor(raw_payload),
                })

        control_bits = 5 if header["alternate_object_present"] else 1
        if body_start + control_bits > body_end:
            raise UnsupportedVariantError(
                "oamd", "element_control_bounds",
                "OAMD element 太短，无法容纳控制字段",
                details={
                    "element_ordinal": ordinal,
                    "element_id": element_id,
                    "size_bytes": size_bytes,
                    "control_bits": control_bits,
                    "payload": bytes_descriptor(raw_payload),
                })
        control = _BitReader(bits, body_start, body_end)
        alternate_data_id = (control.read(4)
                             if header["alternate_object_present"] else None)
        discard_unknown = bool(control.read(1))
        elements.append({
            "ordinal": ordinal,
            "element_id": element_id,
            "size_bytes": size_bytes,
            "header_start_bit": header_start,
            "body_start_bit": body_start,
            "data_start_bit": control.position,
            "body_end_bit": body_end,
            "alternate_data_id": alternate_data_id,
            "discard_unknown": discard_unknown,
        })
        reader.position = body_end

    padding = bits[reader.position:]
    if len(padding) > 7:
        raise UnsupportedVariantError(
            "oamd", "trailing_payload_data",
            "OAMD element 结束后仍有超过一个字节的未声明数据",
            details={
                "elements": _element_details(elements),
                "trailing_bits": len(padding),
                "payload": bytes_descriptor(raw_payload),
            })
    if np.any(padding):
        raise UnsupportedVariantError(
            "oamd", "nonzero_padding",
            "OAMD payload 尾部 padding 含非零位",
            details={
                "elements": _element_details(elements),
                "padding_start_bit": reader.position,
                "padding_bits": "".join(str(int(bit)) for bit in padding),
                "payload": bytes_descriptor(raw_payload),
            })

    object_elements = [element for element in elements
                       if element["element_id"] == OBJECT_ELEMENT_ID]
    if len(object_elements) != 1:
        raise UnsupportedVariantError(
            "oamd", "object_element_count",
            "OAMD 必须包含且只能包含一个 object element",
            details={
                "object_element_count": len(object_elements),
                "elements": _element_details(elements),
                "payload": bytes_descriptor(raw_payload),
            })
    object_element = object_elements[0]
    if object_element["ordinal"] != 0 or object_element["header_start_bit"] != header_end:
        raise UnsupportedVariantError(
            "oamd", "object_element_order",
            "object element 不在当前固定位置窗口支持的首个 element 位置",
            details={
                "elements": _element_details(elements),
                "payload": bytes_descriptor(raw_payload),
                "repair_hint": "以 object element 的实际位置为基准重新定位对象窗口",
            })

    for element in elements:
        element_id = element["element_id"]
        if element_id in (OBJECT_ELEMENT_ID, TRIM_ELEMENT_ID):
            continue
        if element_id == EXTENDED_OBJECT_ELEMENT_ID:
            raise UnsupportedVariantError(
                "oamd", "extended_object_element",
                "OAMD 含可能改变坐标语义的 extended object element",
                details={
                    "element": _element_details([element])[0],
                    "elements": _element_details(elements),
                    "payload": bytes_descriptor(raw_payload),
                    "repair_hint": "解析 divergence/extended-precision position 后再应用轨迹",
                })
        raise UnsupportedVariantError(
            "oamd", f"unsupported_element_{element_id}",
            f"OAMD 含当前未覆盖的 element id {element_id}",
            details={
                "element": _element_details([element])[0],
                "elements": _element_details(elements),
                "payload": bytes_descriptor(raw_payload),
            })
    return object_element, elements


def _update_timing(bits, object_element, raw_payload):
    """从已定位的 object element 读取位置块开始偏移和 ramp 时长。"""
    reader = _BitReader(
        bits, object_element["data_start_bit"], object_element["body_end_bit"])
    try:
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
    except UnsupportedVariantError:
        raise
    except ValueError as exc:
        raise UnsupportedVariantError(
            "oamd", "object_element_syntax",
            "OAMD object element 的 timing 字段越界或不完整",
            details={
                "element": _element_details([object_element])[0],
                "payload": bytes_descriptor(raw_payload),
                "parser_error": str(exc),
            }) from exc

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


def frame_update(bits_one):
    """单帧 OAMD → 位置字段增量及其 sample offset/ramp duration。"""
    bits, raw_payload = _payload_bits(bits_one)
    if len(bits) < 14:
        raise UnsupportedVariantError(
            "oamd", "header_truncated",
            "OAMD payload 不足以容纳受支持的 header",
            details={"payload_bits": len(bits), "payload": bytes_descriptor(raw_payload)})
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
            "OAMD header 与当前位置字段布局不一致",
            details={
                "supported_header_prefix_hex": "1f88",
                "header_probe": header,
                "payload": bytes_descriptor(raw_payload),
                "repair_hint": "按新 header 的 program assignment 和 element 布局重新定位对象位置字段",
            })
    object_element, elements = _parse_elements(bits, header, 14, raw_payload)
    if object_element["body_end_bit"] < POSITION_WINDOW_END_BIT:
        raise UnsupportedVariantError(
            "oamd", "object_element_too_short",
            "OAMD object element 无法容纳当前固定位置窗口",
            details={
                "element": _element_details([object_element])[0],
                "required_position_end_bit": POSITION_WINDOW_END_BIT,
                "elements": _element_details(elements),
                "payload": bytes_descriptor(raw_payload),
            })
    block_offset, ramp_duration = _update_timing(bits, object_element, raw_payload)
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
                "OAMD 对象位置字段标记或位偏移发生变化",
                details={
                    "object_slot": obj,
                    "position_start_bit": start,
                    "q1_window_hex": f"{wq1:02x}",
                    "q2_window_hex": f"{wq2:02x}",
                    "q3_window_hex": f"{wq3:02x}",
                    "header_probe": header,
                    "elements": _element_details(elements),
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
