"""OAMD 位载荷 → 16 个对象槽的 q1/q2/q3 增量状态。

依据 ETSI TS 103 420 V1.2.1 clause 5。槽 0 为 bed/LFE，槽 1..15 为输出
ch1..15；bed/ISF/未激活对象没有位置字段，保持上一帧位置。element 目录按声明长度
驱动，未知 element 按边界跳过；个别编码器的 ``oa_element_size`` 比实际内容短时以
结构解析为准，差异记入 ``diagnostics``，不算变体错误。
"""
import math

import numpy as np

from variant_error import UnsupportedVariantError, bytes_descriptor

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
# Table 12 of ETSI TS 103 420：trim 电平表，4 bit 索引，单位 dB。
# 索引 15 的 -36 dB 是最小值（并非静音）。
TRIM_LUT = (6.0, 3.0, 1.5, 0.75, -0.75, -1.5, -3.0, -4.5,
            -6.0, -7.5, -9.0, -10.5, -12.0, -13.5, -15.0, -36.0)
# bal3d_y 的编码为 (code + 1) / 16，幅度落在 [1/16, 1]，取不到 0。
BALANCE_STEP = 16
# 5.5.3/5.5.5 divergence 码表（公开实现固定值），Q15 量纲。
DIVERGENCE_TABLE = (
    0, 0, 66, 117, 209, 331, 467, 659,
    829, 1042, 1309, 1643, 2059, 2302, 2573, 2872,
    3202, 3564, 3960, 4391, 4856, 5354, 5882, 6437,
    7014, 7605, 8204, 8803, 9394, 9970, 10524, 11051,
    11548, 12011, 12441, 12835, 13197, 13525, 13823, 14092,
    14335, 14553, 14748, 14924, 15081, 15221, 15347, 15459,
    15559, 15649, 15728, 15799, 15863, 15970, 16055, 16123,
    16176, 16219, 16267, 16301, 16332, 16355, 16368, 16384,
)
DIVERGENCE_QUICK_INDEX = (26, 29, 32, 63)
# 5.6.0.11 Table 11b：ISF 类型 → 对象数。
ISF_OBJECT_COUNTS = (4, 8, 10, 14, 15, 30)
# 5.6.1.1.4 Table 12：10 bit 标准 bed 掩码。array position 9（RC_L/RC_R）是 channel
# order 的第一项，字段按 array position 0→9 逐位传输，读成整数后 position 9 落在 LSB，
# 因此按 LSB→MSB 展开即得 channel order：(L,R), C, LFE, (Ls,Rs), … , LFE2。
BED_CHANNEL_LABELS = (
    "RC_L/RC_R", "RC_C", "RC_LFE", "RC_LS/RC_RS", "RC_LB/RC_RB",
    "RC_TFL/RC_TFR", "RC_TSL/RC_TSR", "RC_TBL/RC_TBR", "RC_LW/RC_RW", "RC_LFE2",
)
# 5.6.1.1.5 Table 13：17 bit 非标准 bed 掩码，同样按 channel order 排列（RC_L 最先，
# 对应 array position 16 = LSB）。
NONSTD_BED_LABELS = (
    "RC_L", "RC_R", "RC_C", "RC_LFE", "RC_LS", "RC_RS", "RC_LB", "RC_RB",
    "RC_TFL", "RC_TFR", "RC_TSL", "RC_TSR", "RC_TBL", "RC_TBR", "RC_LW", "RC_RW",
    "RC_LFE2",
)
MAX_SLOTS = 16


def standard_bed_channels(bed_mask):
    """5.6.1.1.4：标准 bed 掩码 → 按 channel order 排列的声道标签（LFE 留在原位）。"""
    channels = []
    for bit, labels in enumerate(BED_CHANNEL_LABELS):
        if (int(bed_mask) >> bit) & 1:
            channels.extend(labels.split("/"))
    return channels


def nonstandard_bed_channels(bed_mask):
    """5.6.1.1.5：非标准 bed 掩码 → 按 channel order 排列的声道标签。"""
    mask = int(bed_mask)
    return [label for bit, label in enumerate(NONSTD_BED_LABELS)
            if (mask >> bit) & 1]


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


def _variable_bits_max(reader, width, max_groups):
    """5.5.1 ``variable_bits_max(n, max_num_groups)``。"""
    value = reader.read(width)
    more = reader.read(1)
    num_group = 1
    if max_groups > num_group:
        if more:
            value = (value + 1) << width
        while more:
            value += reader.read(width)
            more = reader.read(1)
            if num_group >= max_groups:
                break
            if more:
                value = (value + 1) << width
                num_group += 1
    return value


def _element_details(elements):
    return [{
        "ordinal": element["ordinal"],
        "element_id": element["element_id"],
        "size_bytes": element["size_bytes"],
        "header_start_bit": element["header_start_bit"],
        "body_start_bit": element["body_start_bit"],
        "body_end_bit": element["body_end_bit"],
        "parsed_end_bit": element.get("parsed_end_bit"),
        "discard_unknown": element["discard_unknown"],
        "alternate_data_id": element["alternate_data_id"],
    } for element in elements]


def _syntax_error(variant, message, raw_payload, exc=None, **details):
    payload = {"payload": bytes_descriptor(raw_payload)}
    if exc is not None:
        payload["parser_error"] = str(exc)
    payload.update(details)
    return UnsupportedVariantError("oamd", variant, message, details=payload)


def _parse_program_assignment(reader, raw_payload):
    """5.5.3 ``program_assignment()``：bed / ISF / dynamic 对象数量。"""
    program = {
        "dynamic_object_only": bool(reader.read(1)),
        "lfe_present": False,
        "content_description": None,
        "bed_assignments": [],
        "num_bed_objects": 0,
        "b_bed_chan_distribute": False,
        "bed_mask": 0,
        "isf_idx": None,
        "num_isf_objects": 0,
        "num_dynamic_objects": None,
    }
    if program["dynamic_object_only"]:
        program["lfe_present"] = bool(reader.read(1))
        # 5.6.4.8：对象顺序为 bed → ISF → dynamic，LFE 属于 bed，排在最前。
        program["num_bed_objects"] = 1 if program["lfe_present"] else 0
        if program["lfe_present"]:
            program["bed_assignments"] = [{
                "instance": 0, "lfe_only": True, "channels": ["RC_LFE"],
            }]
            program["bed_mask"] = 0x8
        return program

    mask = reader.read(4)
    program["content_description"] = {
        "reserved": bool(mask & 0x8),
        "dynamic": bool(mask & 0x4),
        "isf": bool(mask & 0x2),
        "bed": bool(mask & 0x1),
    }
    if mask & 0x1:
        program["b_bed_chan_distribute"] = bool(reader.read(1))
        num_instances = (reader.read(3) + 2) if reader.read(1) else 1
        for instance in range(num_instances):
            if reader.read(1):                      # b_lfe_only
                program["bed_assignments"].append({
                    "instance": instance, "lfe_only": True, "channels": ["RC_LFE"],
                })
                continue
            if reader.read(1):                      # b_standard_chan_assign
                bed_mask = reader.read(10)
                standard = True
                channels = standard_bed_channels(bed_mask)
                wide_mask = standard_bed_mask(bed_mask)
            else:
                bed_mask = reader.read(17)
                standard = False
                channels = nonstandard_bed_channels(bed_mask)
                wide_mask = bed_mask
            program["bed_assignments"].append({
                "instance": instance, "lfe_only": False, "standard": standard,
                "mask": bed_mask, "channels": channels,
            })
            program["bed_mask"] |= wide_mask
        program["num_bed_objects"] = sum(
            len(assignment["channels"]) for assignment in program["bed_assignments"])
    if mask & 0x2:
        isf_idx = reader.read(3)
        program["isf_idx"] = isf_idx
        program["num_isf_objects"] = (ISF_OBJECT_COUNTS[isf_idx]
                                      if isf_idx < len(ISF_OBJECT_COUNTS) else 0)
    if mask & 0x4:
        count = reader.read(5)
        if count == 0x1F:
            count += reader.read(7)
        program["num_dynamic_objects"] = count + 1
    if mask & 0x8:
        # 5.6.4.1：reserved_data_size = reserved_data_size_bits + 1（字节）。
        reader.skip((reader.read(4) + 1) * 8)
    return program


# 17 bit 非标准 bed 标签在 wide mask 中的位顺序（与 NONSTD_BED_LABELS 相同）。
_STANDARD_BED_TO_WIDE = (0x3, 0x4, 0x8, 0x30, 0xC0,
                         0x300, 0xC00, 0x3000, 0xC000, 0x10000)


def standard_bed_mask(bed_mask):
    """5.6.1.1.4 的 10 bit 标准掩码 → 17 bit 非标准 bed 掩码。"""
    wide = 0
    for bit, expanded in enumerate(_STANDARD_BED_TO_WIDE):
        if (int(bed_mask) >> bit) & 1:
            wide |= expanded
    return wide


def _q15_ratio(numerator, denominator):
    """Q15 of ``numerator / denominator``, rounded half up and capped."""
    return min(32767, int(math.floor(32768.0 * numerator / denominator + 0.5)))


def _gain_q(gain_idx, gain_bits):
    """gain_idx/bits → 整数增益 Qg（单位增益 4096）。"""
    if gain_idx == 0:
        return 4096
    if gain_idx == 1:
        return 0
    if gain_idx == 2:
        bits = int(gain_bits)
        db = 15 - bits if bits < 15 else 14 - bits
        return int(math.floor(4096.0 * (10.0 ** (db / 20.0))))
    raise ValueError("gain_idx=3 需要同块前一对象的状态")


def _size_q(mode, values):
    """size_idx 与 5 bit 数值 → (sx, sy, sz) 整数 Q15。"""
    if mode == 1:
        value = _q15_ratio(int(values), 31)
        return (value, value, value)
    if mode == 2:
        return tuple(_q15_ratio(int(value), 31) for value in values)
    return (0, 0, 0)


_RENDER_DEFAULTS = {
    "qx": 16384, "qy": 16384, "qz": 0,
    "coarse": (31, 31, 0),
    "zone_idx": 0, "enable_elevation": True,
    "size_mode": 0, "size_q": (0, 0, 0), "snap": 0,
    "divergence_q": 0,
}


def _clamp_coarse(coarse):
    x, y, z = coarse
    return (min(max(int(x), 0), 62), min(max(int(y), 0), 62),
            min(max(int(z), -15), 15))


def _coarse_to_q15(coarse):
    x, y, z = coarse
    qx = _q15_ratio(int(x), 62)
    qy = _q15_ratio(int(y), 62)
    magnitude = _q15_ratio(abs(int(z)), 15)
    return qx, qy, (-magnitude if z < 0 else magnitude)


_POSITION_DELTA = (0, 1, 2, 3, -4, -3, -2, -1)
_REFINE_DELTA = (1, 2, -1, -2)


def _refine_xy(index):
    """Q15 code for one refined X/Y position index ``5*coarse + delta + 2``."""
    return min(32767, max(0, int(math.floor(
        (int(index) - 2) * 32768.0 / 310.0 + 0.5))))


def _refine_z(index):
    """Q15 code for one refined Z position index (signed)."""
    return min(32767, max(-32767, int(math.floor(
        (int(index) - 2) * 32768.0 / 75.0 + 0.5))))


def _copy_render_state(previous):
    state = {key: previous[key] for key in _RENDER_DEFAULTS}
    state["coarse"] = tuple(previous["coarse"])
    state["size_q"] = tuple(previous["size_q"])
    state["qx"], state["qy"], state["qz"] = previous["qx"], previous["qy"], previous["qz"]
    state["divergence_q"] = int(previous["divergence_q"])
    return state


def parse_object_info_block(reader, object_index, block_index, kind,
                            previous_block, raw_payload, retain_spatial=False,
                            object_states=None):
    """5.5.9 ``object_info_block``：展开一个对象的一个块。"""
    info = {
        "object": object_index,
        "block": block_index,
        "kind": kind,
        "in_bed_or_isf": kind != "dynamic",
        "not_active": bool(reader.read(1)),
    }
    inactive = info["not_active"]
    if inactive:
        info["basic_status"] = 0
        info["gain_q"] = 0
        info["priority_q"] = 0
        render = dict(_RENDER_DEFAULTS)
        render["coarse"] = (31, 31, 0)
        render["size_q"] = tuple(render["size_q"])
        info.update({key: render[key] for key in render if key != "coarse"})
        info["coarse"] = render["coarse"]
        info["render_status"] = 0
        info["qx"], info["qy"], info["qz"] = _coarse_to_q15(render["coarse"])
    else:
        basic_status = 1 if block_index == 0 else reader.read(2)
        info["basic_status"] = basic_status
        if basic_status == 0:
            info["gain_q"] = 0
            info["priority_q"] = 0
        elif basic_status == 1:
            gain_idx = reader.read(2)
            info["gain_idx"] = gain_idx
            gain_bits = reader.read(6) if gain_idx == 2 else None
            if gain_bits is not None:
                info["gain_bits"] = gain_bits
            if gain_idx == 3:
                if object_index == 0:
                    info["gain_q"] = 4096
                elif object_states[object_index - 1][block_index]["not_active"]:
                    info["gain_q"] = 0
                else:
                    info["gain_q"] = object_states[object_index - 1][block_index]["gain_q"]
            else:
                info["gain_q"] = _gain_q(gain_idx, gain_bits)
            default_priority = bool(reader.read(1))
            info["default_priority"] = default_priority
            if default_priority:
                info["priority_q"] = 32767
            else:
                bits = reader.read(5)
                info["priority_bits"] = bits
                info["priority_q"] = 1024 * bits
        elif basic_status == 2:
            info["gain_q"] = previous_block["gain_q"]
            info["priority_q"] = previous_block["priority_q"]
        else:
            flags = reader.read(2)
            info["basic_mask"] = flags
            info["gain_q"] = previous_block["gain_q"]
            info["priority_q"] = previous_block["priority_q"]
            if flags & 1:
                gain_idx = reader.read(2)
                info["gain_idx"] = gain_idx
                gain_bits = reader.read(6) if gain_idx == 2 else None
                if gain_bits is not None:
                    info["gain_bits"] = gain_bits
                if gain_idx == 3:
                    if object_index == 0:
                        info["gain_q"] = 4096
                    elif object_states[object_index - 1][block_index]["not_active"]:
                        info["gain_q"] = 0
                    else:
                        info["gain_q"] = object_states[object_index - 1][block_index]["gain_q"]
                else:
                    info["gain_q"] = _gain_q(gain_idx, gain_bits)
            if flags & 2:
                default_priority = bool(reader.read(1))
                info["default_priority"] = default_priority
                if default_priority:
                    info["priority_q"] = 32767
                else:
                    bits = reader.read(5)
                    info["priority_bits"] = bits
                    info["priority_q"] = 1024 * bits

        render_status = 0 if kind in ("bed", "isf") else (
            1 if block_index == 0 else reader.read(2))
        info["render_status"] = render_status
        if render_status in (1, 3):
            if render_status == 1:
                position_present = zone_present = size_present = screen_present = True
                flags = 0xF
            else:
                flags = reader.read(4)
                position_present = bool(flags & 0x1)
                zone_present = bool(flags & 0x2)
                size_present = bool(flags & 0x4)
                screen_present = bool(flags & 0x8)
            info["render_mask"] = flags
            state = _copy_render_state(previous_block) if render_status == 3 \
                else dict(_RENDER_DEFAULTS)
            state["coarse"] = tuple(state["coarse"])
            state["size_q"] = tuple(state["size_q"])
            if position_present:
                differential = (block_index > 0 and bool(reader.read(1)))
                if differential:
                    dx, dy, dz = (_POSITION_DELTA[reader.read(3)]
                                  for _axis in range(3))
                    coarse = (previous_block["coarse"][0] + dx,
                              previous_block["coarse"][1] + dy,
                              previous_block["coarse"][2] + dz)
                else:
                    x = reader.read(6)
                    y = reader.read(6)
                    z_sign = reader.read(1)
                    z = reader.read(4)
                    coarse = (x, y, z if z_sign else -z)
                state["coarse"] = _clamp_coarse(coarse)
                outside = bool(reader.read(1))
                infinite = False
                distance_index = None
                if outside:
                    infinite = bool(reader.read(1))
                    if not infinite:
                        distance_index = reader.read(4)
                if retain_spatial:
                    info["distance_specified"] = outside
                    info["at_infinity"] = infinite
                    info["distance_factor_idx"] = distance_index
            if zone_present:
                state["zone_idx"] = reader.read(3)
                state["enable_elevation"] = bool(reader.read(1))
            if size_present:
                size_idx = reader.read(2)
                state["size_mode"] = size_idx
                if size_idx == 1:
                    state["size_q"] = _size_q(1, reader.read(5))
                elif size_idx == 2:
                    state["size_q"] = _size_q(2, (reader.read(5), reader.read(5),
                                                   reader.read(5)))
                else:
                    state["size_q"] = (0, 0, 0)
            if screen_present:
                screen_ref = bool(reader.read(1))
                screen_bits = reader.read(3) if screen_ref else None
                depth_index = reader.read(2) if screen_ref else None
                if retain_spatial:
                    info["use_screen_ref"] = screen_ref
                    info["screen_factor_bits"] = screen_bits
                    info["depth_factor_idx"] = depth_index
            state["snap"] = 1 if reader.read(1) else 0
            info.update({key: state[key] for key in state if key != "coarse"})
            info["coarse"] = state["coarse"]
            info["qx"], info["qy"], info["qz"] = _coarse_to_q15(state["coarse"])
        elif render_status == 2:
            info.update({key: previous_block[key] for key in _RENDER_DEFAULTS
                         if key != "coarse"})
            info["coarse"] = tuple(previous_block["coarse"])
            info["size_q"] = tuple(previous_block["size_q"])
            info["qx"], info["qy"], info["qz"] = (
                previous_block["qx"], previous_block["qy"], previous_block["qz"])
            info["divergence_q"] = previous_block["divergence_q"]
        else:
            info.update({key: _RENDER_DEFAULTS[key] for key in _RENDER_DEFAULTS
                         if key != "coarse"})
            info["coarse"] = tuple(_RENDER_DEFAULTS["coarse"])
            info["size_q"] = tuple(_RENDER_DEFAULTS["size_q"])
            info["qx"], info["qy"], info["qz"] = _coarse_to_q15(info["coarse"])
    if reader.read(1):                              # b_additional_table_data_exists
        info["additional_table_bytes"] = reader.read(4) + 1
        reader.skip(info["additional_table_bytes"] * 8)
    return info


def _parse_object_element(reader, element, raw_payload, retain_spatial=False):
    """5.5.5 ``object_element()``：时间头 + 对象外/块内顺序的完整块状态。"""
    object_count = element["object_count"]
    try:
        sample_offset_code = reader.read(2)
        if sample_offset_code == 0:
            sample_offset = 0
        elif sample_offset_code == 1:
            sample_offset = SAMPLE_OFFSET_INDEX[reader.read(2)]
        elif sample_offset_code == 2:
            sample_offset = reader.read(5)
        else:
            raise UnsupportedVariantError(
                "oamd", "md_sample_offset_mode",
                "OAMD 使用了当前未覆盖的 MD sample-offset 模式",
                details={
                    "sample_offset_code": sample_offset_code,
                    "payload": bytes_descriptor(raw_payload),
                })
        block_count = reader.read(3) + 1
        blocks = []
        for _block in range(block_count):
            block_offset_factor = reader.read(6)
            ramp_code = reader.read(2)
            if ramp_code == 3:
                if reader.read(1):
                    ramp_duration = RAMP_DURATION_INDEX[reader.read(4)]
                else:
                    ramp_duration = reader.read(11)
            else:
                ramp_duration = RAMP_DURATIONS[ramp_code]
            blocks.append({
                "block_offset_samples": sample_offset + block_offset_factor * 32,
                "ramp_duration_samples": ramp_duration,
            })
        reserved_data_not_present = bool(reader.read(1))
        if not reserved_data_not_present:
            reader.read(5)
        kinds = element["object_kinds"]
        object_blocks = []
        for object_index in range(object_count):
            kind = kinds[object_index]
            states = []
            previous = None
            for block_index in range(block_count):
                info = parse_object_info_block(
                    reader, object_index, block_index, kind, previous,
                    raw_payload, retain_spatial, object_blocks)
                info["previous"] = previous
                states.append(info)
                previous = info
            object_blocks.append(states)
    except UnsupportedVariantError:
        raise
    except ValueError as exc:
        raise _syntax_error(
            "object_element_syntax",
            "OAMD object element 字段越界或不完整",
            raw_payload, exc,
            element=_element_details([element])[0],
            object_count=object_count,
        ) from exc
    return {
        "sample_offset": sample_offset,
        "blocks": blocks,
        "reserved_data_not_present": reserved_data_not_present,
        "objects": object_blocks,
    }


def _parse_extended_element(reader, element, object_element, raw_payload):
    """扩展对象元素：divergence 与高精度位置补充。"""
    blocks = object_element["blocks"]
    block_count = len(blocks)
    objects = object_element["objects"]
    object_count = len(objects)
    if object_element is None:
        raise UnsupportedVariantError(
            "oamd", "extended_without_object_element",
            "OAMD 扩展对象元素在默认主对象元素之前出现",
            details={"element": _element_details([element])[0],
                     "payload": bytes_descriptor(raw_payload)})
    try:
        divergence_present = bool(reader.read(1))
        if divergence_present:
            table = DIVERGENCE_TABLE
            quick = DIVERGENCE_QUICK_INDEX
            for object_index in range(object_count):
                for block_index in range(block_count):
                    state = objects[object_index][block_index]
                    if state["not_active"]:
                        state["divergence_q"] = 0
                        continue
                    if state["kind"] != "dynamic":
                        continue
                    present = bool(reader.read(1))
                    if not present:
                        state["divergence_q"] = 0
                        continue
                    mode = reader.read(2)
                    if mode == 0:
                        state["divergence_q"] = int(table[quick[reader.read(2)]])
                    elif mode == 1:
                        state["divergence_q"] = (
                            0 if block_index == 0
                            else objects[object_index][block_index - 1]["divergence_q"])
                    elif mode == 2:
                        state["divergence_q"] = int(table[reader.read(6)])
                    else:
                        reader.read(6)
                        state["divergence_q"] = 0
        refinement_present = bool(reader.read(1))
        if refinement_present:
            for object_index in range(object_count):
                for block_index in range(block_count):
                    state = objects[object_index][block_index]
                    if state["not_active"] or state["kind"] != "dynamic":
                        continue
                    present = bool(reader.read(1))
                    if not present:
                        continue
                    mask = reader.read(3)
                    if not mask:
                        continue
                    deltas = [0, 0, 0]
                    for axis in range(3):
                        if mask & (1 << axis):
                            deltas[axis] = _REFINE_DELTA[reader.read(2)]
                    qx, qy, qz = state["qx"], state["qy"], state["qz"]

                    def _sign(value):
                        return -1 if value < 0 else 1

                    def _coarse_xy(value):
                        return 31 + _sign(value - 16384) * (
                            (62 * abs(value - 16384) + 16384) >> 15)

                    def _coarse_z(value):
                        value = max(value, -32767)
                        return _sign(value) * ((15 * abs(value) + 16384) >> 15)

                    ix = _coarse_xy(qx)
                    iy = _coarse_xy(qy)
                    iz = _coarse_z(qz)
                    sign = _sign(iz)
                    qx = _refine_xy(5 * ix + 2 + deltas[0])
                    qy = _refine_xy(5 * iy + 2 + deltas[1])
                    qz = sign * _refine_z(sign * (5 * iz + deltas[2]) + 2)
                    state["qx"], state["qy"], state["qz"] = qx, qy, qz
    except ValueError as exc:
        raise _syntax_error(
            "extended_element_syntax",
            "OAMD 扩展对象元素字段越界或不完整",
            raw_payload, exc,
            element=_element_details([element])[0],
            object_count=object_count,
        ) from exc
    return {"divergence_present": divergence_present,
            "refinement_present": refinement_present}


def _parse_trim_element(reader, object_count, raw_payload):
    """``trim_element``：全局 trim/warp 配置，每个 payload 出现一次。

    该 element 描述整段节目的渲染校准意图，不随 block 变化，因此解析结果应当
    在 payload 之间保持，而不是逐帧重算。
    """
    trim = {
        "warp_mode": reader.read(2),
        "reserved": reader.read(2),
        "global_trim_mode": reader.read(2),
        "configs": [None] * 9,
        "disable_per_obj": None,
    }
    # global_trim_mode: 0 自动、1 禁用、2 按配置显式指定、3 保留。
    # 只有 2 才携带下面九个配置的显式数据。
    if trim["global_trim_mode"] == 2:
        for config in range(9):
            if reader.read(1):                       # b_default_trim
                trim["configs"][config] = {"default": True}
                continue
            if reader.read(1):                       # b_disable_trim
                trim["configs"][config] = {"disabled": True}
                continue
            present = reader.read(5)
            entry = {"present": present}
            if present & 0x01:
                entry["centre"] = TRIM_LUT[reader.read(4)]
            if present & 0x02:
                entry["surround"] = TRIM_LUT[reader.read(4)]
            if present & 0x04:
                entry["height"] = TRIM_LUT[reader.read(4)]
            if present & 0x08:
                sign = 1.0 if reader.read(1) else -1.0
                entry["bal3d_y_tb"] = (
                    (reader.read(4) + 1) / BALANCE_STEP * sign)
            if present & 0x10:
                sign = 1.0 if reader.read(1) else -1.0
                entry["bal3d_y_lis"] = (
                    (reader.read(4) + 1) / BALANCE_STEP * sign)
            trim["configs"][config] = entry
    if reader.read(1):                               # b_disable_trim_per_obj
        trim["disable_per_obj"] = [bool(reader.read(1))
                                   for _ in range(object_count)]
    return trim


def parse_trim(bits_one):
    """独立解析一个 OAMD payload 的 trim element。

    返回 ``None`` 表示该 payload 不含 trim element。异常统一转成
    ``UnsupportedVariantError``，与其它解析入口一致。
    """
    bits, raw_payload = _payload_bits(bits_one)
    try:
        version = bits[0] << 1 | bits[1]
    except IndexError:
        raise UnsupportedVariantError(
            "oamd", "header_truncated", "OAMD payload 不足以容纳 header",
            details={"payload_bits": len(bits),
                     "payload": bytes_descriptor(raw_payload)}) from None
    reader = _BitReader(bits, 2)
    try:
        if version == 3:
            version += reader.read(3)
        object_count_bits = reader.read(5)
        if object_count_bits == 0x1F:
            object_count_bits += reader.read(7)
        object_count = object_count_bits + 1
        program = _parse_program_assignment(reader, raw_payload)
    except UnsupportedVariantError:
        raise
    except ValueError as exc:
        raise _syntax_error(
            "header_truncated", "OAMD header 字段越界或不完整",
            raw_payload, exc, payload_bits=len(bits)) from exc
    alternate_present = bool(reader.read(1))
    try:
        element_count = reader.read(4)
    except ValueError as exc:
        raise _syntax_error(
            "element_count_truncated", "OAMD element 数量字段不完整",
            raw_payload, exc) from exc
    for ordinal in range(element_count):
        header_start = reader.position
        try:
            element_id = reader.read(4)
            size_bytes = _variable_bits_max(reader, 4, 4) + 1
        except ValueError as exc:
            raise _syntax_error(
                "element_header_truncated", "OAMD element header 不完整",
                raw_payload, exc,
                element_ordinal=ordinal, header_start_bit=header_start) from exc
        region_start = reader.position
        region_end = region_start + size_bytes * 8
        if region_end > reader.limit:
            raise UnsupportedVariantError(
                "oamd", "element_bounds",
                "OAMD element 声明长度超过 payload 边界",
                details={"element_ordinal": ordinal, "element_id": element_id,
                         "size_bytes": size_bytes,
                         "payload": bytes_descriptor(raw_payload)})
        if alternate_present:
            reader.read(4)
        reader.read(1)
        if element_id == TRIM_ELEMENT_ID:
            trim = _parse_trim_element(reader, object_count, raw_payload)
            trim["object_count"] = object_count
            return trim
        reader.position = region_end
    return None


def _parse_elements(reader, alternate_present, object_count, program,
                    raw_payload, retain_spatial=False):
    """5.5.4 ``oa_element_md()`` 目录：先选替代变体，再解析内容。"""
    try:
        element_count = reader.read(4)
        if element_count == 0xF:
            element_count += reader.read(5)
    except ValueError as exc:
        raise _syntax_error(
            "element_count_truncated", "OAMD element 数量字段不完整",
            raw_payload, exc) from exc
    if element_count == 0:
        raise UnsupportedVariantError(
            "oamd", "missing_object_element",
            "OAMD 没有声明任何 element",
            details={"payload": bytes_descriptor(raw_payload)})

    bed_isf_objects = program["num_bed_objects"] + program["num_isf_objects"]
    kinds = []
    for object_index in range(object_count):
        if object_index < program["num_bed_objects"]:
            kinds.append("bed")
        elif object_index < bed_isf_objects:
            kinds.append("isf")
        else:
            kinds.append("dynamic")
    elements = []
    object_element = None
    extended = None
    seen = set()
    for ordinal in range(element_count):
        header_start = reader.position
        try:
            element_id = reader.read(4)
            size_bytes = _variable_bits_max(reader, 4, 4) + 1
        except ValueError as exc:
            raise _syntax_error(
                "element_header_truncated", "OAMD element header 不完整",
                raw_payload, exc,
                element_ordinal=ordinal, header_start_bit=header_start) from exc
        region_start = reader.position
        region_end = region_start + size_bytes * 8
        if region_end > reader.limit:
            raise UnsupportedVariantError(
                "oamd", "element_bounds",
                "OAMD element 声明长度超过 payload 边界",
                details={
                    "element_ordinal": ordinal,
                    "element_id": element_id,
                    "size_bytes": size_bytes,
                    "body_start_bit": region_start,
                    "body_end_bit": region_end,
                    "payload_bits": reader.limit,
                    "payload": bytes_descriptor(raw_payload),
                })
        alternate_data_id = reader.read(4) if alternate_present else 0
        discard_unknown = bool(reader.read(1))
        element = {
            "ordinal": ordinal,
            "element_id": element_id,
            "size_bytes": size_bytes,
            "header_start_bit": header_start,
            "body_start_bit": region_start,
            "body_end_bit": region_end,
            "data_start_bit": reader.position,
            "alternate_data_id": alternate_data_id,
            "discard_unknown": discard_unknown,
            "object_count": object_count,
            "bed_isf_objects": bed_isf_objects,
        }
        selected = (element_id in (OBJECT_ELEMENT_ID, TRIM_ELEMENT_ID, 5)
                    and alternate_data_id == 0 and element_id not in seen)
        if not selected:
            element["parsed_end_bit"] = region_end
            reader.position = region_end
        elif element_id == OBJECT_ELEMENT_ID:
            element["object_kinds"] = kinds
            object_element = _parse_object_element(
                reader, element, raw_payload, retain_spatial)
            element["parsed_end_bit"] = reader.position
            element["object_element"] = object_element
            seen.add(element_id)
            reader.position = max(region_end, reader.position)
        elif element_id == 5:
            if object_element is None:
                raise UnsupportedVariantError(
                    "oamd", "extended_before_object_element",
                    "OAMD 扩展对象元素出现在默认主对象元素之前",
                    details={"element": _element_details(elements + [element])[0],
                             "payload": bytes_descriptor(raw_payload)})
            extended = _parse_extended_element(reader, element,
                                               object_element, raw_payload)
            element["parsed_end_bit"] = reader.position
            element["extended_element"] = extended
            seen.add(element_id)
            reader.position = max(region_end, reader.position)
        else:
            element["trim_element"] = _parse_trim_element(
                reader, object_count, raw_payload)
            element["parsed_end_bit"] = reader.position
            seen.add(element_id)
            reader.position = max(region_end, reader.position)
        elements.append(element)

    if object_element is None:
        raise UnsupportedVariantError(
            "oamd", "missing_object_element",
            "OAMD 缺少 object element",
            details={
                "elements": _element_details(elements),
                "payload": bytes_descriptor(raw_payload),
            })
    padding = reader.bits[reader.position:]
    if np.any(padding):
        raise UnsupportedVariantError(
            "oamd", "nonzero_padding",
            "OAMD payload 尾部 padding 含非零位",
            details={
                "elements": _element_details(elements),
                "padding_start_bit": reader.position,
                "padding_bits": "".join(str(int(bit)) for bit in padding[:64]),
                "payload": bytes_descriptor(raw_payload),
            })
    return object_element, elements


def _element_kinds(program, object_count):
    kinds = []
    bed_end = program["num_bed_objects"]
    isf_end = bed_end + program["num_isf_objects"]
    for index in range(object_count):
        kinds.append("bed" if index < bed_end
                     else ("isf" if index < isf_end else "dynamic"))
    return kinds


def program_bed_summary(bits_one):
    """只走 OAMD 头与 program_assignment()，返回 bed 声明与内容类型。

    ``channels`` 是按 bed_assignment() 展开的声道标签；``bed_only`` 表示这个 program
    只由 bed 组成（没有 ISF、没有动态对象）——此时 15 路 JOC 对象就是 bed 声道，
    位置由 bed 布局决定。跨层校验与完整帧解析之外的调用都只读这一段。
    """
    bits, raw_payload = _payload_bits(bits_one)
    try:
        version = bits[0] << 1 | bits[1]
    except IndexError:
        raise UnsupportedVariantError(
            "oamd", "header_truncated",
            "OAMD payload 不足以容纳 header",
            details={"payload_bits": len(bits),
                     "payload": bytes_descriptor(raw_payload)}) from None
    reader = _BitReader(bits, 2)
    try:
        if version == 3:
            version += reader.read(3)
        object_count_bits = reader.read(5)
        if object_count_bits == 0x1F:
            object_count_bits += reader.read(7)
        program = _parse_program_assignment(reader, raw_payload)
    except UnsupportedVariantError:
        raise
    except ValueError as exc:
        raise _syntax_error(
            "header_truncated", "OAMD header 字段越界或不完整",
            raw_payload, exc, payload_bits=len(bits)) from exc
    assignments = program["bed_assignments"]
    content = program.get("content_description") or {}
    return {
        "channels": [list(assignment["channels"]) for assignment in assignments],
        "assignments": assignments,
        "bed_only": bool(content.get("bed")) and not content.get("isf")
                    and not content.get("dynamic"),
        "num_bed_objects": program["num_bed_objects"],
        "num_isf_objects": program["num_isf_objects"],
        "num_dynamic_objects": program["num_dynamic_objects"],
        "isf_idx": program["isf_idx"],
        "dynamic_object_only": program["dynamic_object_only"],
        "lfe_present": program["lfe_present"],
        "b_bed_chan_distribute": program["b_bed_chan_distribute"],
        "bed_mask": program["bed_mask"],
    }


def program_bed_channels(bits_one):
    """只走 OAMD 头与 program_assignment()，返回声明为 bed 的声道标签列表。

    跨层一致性校验（规格 §11.2 校验 2）只需要这一段；完整帧解析见
    :func:`frame_update`，它还要继续解析全部对象 element。
    """
    return program_bed_summary(bits_one)["channels"]


def frame_state(bits_one, *, retain_spatial=False):
    """完整展开一帧 OAMD：每个对象每一块的属性、节目分区与扩展对象元素更新。"""
    bits, raw_payload = _payload_bits(bits_one)
    try:
        version = bits[0] << 1 | bits[1]
    except IndexError:
        raise UnsupportedVariantError(
            "oamd", "header_truncated",
            "OAMD payload 不足以容纳 header",
            details={"payload_bits": len(bits),
                     "payload": bytes_descriptor(raw_payload)}) from None
    reader = _BitReader(bits, 2)
    try:
        if version == 3:
            version += reader.read(3)
        object_count_bits = reader.read(5)
        if object_count_bits == 0x1F:
            object_count_bits += reader.read(7)
        object_count = object_count_bits + 1
        program = _parse_program_assignment(reader, raw_payload)
        alternate_present = bool(reader.read(1))
        object_element, elements = _parse_elements(
            reader, alternate_present, object_count, program, raw_payload,
            retain_spatial)
    except UnsupportedVariantError:
        raise
    except ValueError as exc:
        raise _syntax_error(
            "header_truncated", "OAMD header 字段越界或不完整",
            raw_payload, exc, payload_bits=len(bits)) from exc
    if version != 0:
        raise UnsupportedVariantError(
            "oamd", "oamd_version",
            "OAMD 使用了当前未覆盖的 syntax version",
            details={
                "version": version,
                "payload": bytes_descriptor(raw_payload),
            })
    blocks = object_element["blocks"]
    objects = object_element["objects"]
    # 对外契约：16 槽位置视图（block 0），保留旧 frame_update 的 values 形状。
    values = {(slot, field): None for slot in range(MAX_SLOTS)
              for field in ("q1", "q2", "q3")}
    for object_index, states in enumerate(objects):
        if object_index >= MAX_SLOTS:
            break
        state = states[0]
        if state["not_active"] or state.get("position") is None and \
                state["render_status"] == 0 and state["kind"] != "dynamic":
            # bed/ISF 与未激活对象没有位置字段：保持上一帧位置。
            continue
        values[(object_index, "q1")] = state["qx"]
        values[(object_index, "q2")] = state["qy"]
        qz = state["qz"]
        if not state["enable_elevation"]:
            # 描述符构建在 enable_elevation=0 时令 z=0。
            qz = 0
        values[(object_index, "q3")] = (
            qz if qz >= 0 or retain_spatial else 0)
    size_mismatch = [
        {
            "element_ordinal": element["ordinal"],
            "element_id": element["element_id"],
            "declared_size_bytes": element["size_bytes"],
            "declared_end_bit": element["body_end_bit"],
            "parsed_end_bit": element["parsed_end_bit"],
        }
        for element in elements
        if element["parsed_end_bit"] > element["body_end_bit"]
    ]
    trim = next((element["trim_element"] for element in elements
                 if "trim_element" in element), None)
    return {
        "values": values,
        "block_offset_samples": blocks[0]["block_offset_samples"],
        "ramp_duration_samples": blocks[0]["ramp_duration_samples"],
        "blocks": blocks,
        "sample_offset": object_element["sample_offset"],
        "object_count": object_count,
        "kinds": [states[0]["kind"] for states in objects],
        "objects": [dict(states[0]) for states in objects],
        "object_blocks": objects,
        "program": {
            "dynamic_object_only": program["dynamic_object_only"],
            "lfe_present": program["lfe_present"],
            "content_description": program["content_description"],
            "num_bed_objects": program["num_bed_objects"],
            "num_isf_objects": program["num_isf_objects"],
            "num_dynamic_objects": program["num_dynamic_objects"],
            "bed_channels": [list(assignment["channels"])
                             for assignment in program["bed_assignments"]],
            "bed_mask": program["bed_mask"],
            "b_bed_chan_distribute": program["b_bed_chan_distribute"],
            "isf_idx": program["isf_idx"],
        },
        "trim": trim,
        "elements": _element_details(elements),
        "size_mismatch": size_mismatch,
    }


def frame_update(bits_one, *, retain_spatial=False):
    """单帧 OAMD → 位置字段增量及其 sample offset/ramp duration。"""
    state = frame_state(bits_one, retain_spatial=retain_spatial)
    if state["object_count"] > MAX_SLOTS:
        raise UnsupportedVariantError(
            "oamd", "object_count",
            "OAMD 对象数超过当前 16 槽对象模型",
            details={
                "object_count": state["object_count"],
                "max_slots": MAX_SLOTS,
            })
    if len(state["blocks"]) != 1:
        # 旧接口（ADM/双耳/sofa 轨迹）只表达单块位置；多块由 frame_state 提供给
        # 扬声器渲染路径。这里保持与旧实现一致的显式拒绝，避免静默只取首块。
        raise UnsupportedVariantError(
            "oamd", "multiple_position_blocks",
            "OAMD 一帧含多个对象位置更新块，单次 frame_update 无法表达",
            details={
                "block_count": len(state["blocks"]),
                "blocks": state["blocks"],
                "repair_hint": "扬声器渲染路径改用 frame_state()",
            })
    return state


def frame_update_values(bits_one):
    """兼容接口：只返回 ``{(slot, field): q|None}``。"""
    return frame_update(bits_one)["values"]


class JocFieldState:
    """未更新/非法窗保持旧值；slot0 q1/q2 初始为中心 16384。"""

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
