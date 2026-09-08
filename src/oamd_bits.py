"""OAMD 位载荷 → 16 个对象槽的 q1/q2/q3 增量状态。

解析依据 ETSI TS 103 420 V1.2.1（Backwards-compatible object audio carriage
using Enhanced AC-3）clause 5：

* 5.5.2 ``object_audio_metadata_payload()``：版本、对象数、program assignment、
  element 目录；
* 5.5.3 ``program_assignment()``：bed / ISF / dynamic 三类对象及其数量；
* 5.5.4 ``oa_element_md()``：element id、字节长度、alternate data id；
* 5.5.5/5.5.6/5.5.7 object_element/md_update_info/block_update_info：
  ``start_sample = sample_offset + 32 * block_offset_factor``；
* 5.5.9/5.5.10/5.5.11 object_info_block/object_basic_info/object_render_info：
  逐对象的位置字段；bed 与 ISF 对象不携带 render info；
* 5.6.1.1.8~5.6.1.1.11 pos3D_X/Y/Z：横向/纵向 62 格、高度 15 格 + 符号位。

槽 0 是 bed/LFE；槽 1..15 对应输出 ch1..15 的对象元数据。bed/ISF 对象与
``b_object_not_active`` 对象没有位置字段（5.5.9），保持上一帧位置。

element 目录由声明长度驱动，因此 alternate_object_data_present、任意对象数、
多 element（trim/extended/未知 id 按声明边界跳过）都能解析。个别编码器写出的
``oa_element_size`` 比实际内容短（例如 Dolby 测试信号
``Audio_ID_..._special_6ch_..._ddp_joc.mp4`` 的 ID11 少 2 字节），此时以结构解析
出的实际位置为准，并把差异放进 ``diagnostics``，不当作变体错误。
"""
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
# 5.6.0.11 Table 11b：ISF 类型 → 对象数。
ISF_OBJECT_COUNTS = (4, 8, 10, 14, 15, 30)
# 5.6.1.1.4 Table 12：10 bit 标准 bed 掩码按位对应的声道（LSB = RC_L/RC_R）。
BED_CHANNEL_LABELS = (
    "RC_L/RC_R", "RC_C", "RC_LFE", "RC_LS/RC_RS", "RC_LB/RC_RB",
    "RC_TFL/RC_TFR", "RC_TSL/RC_TSR", "RC_TBL/RC_TBR", "RC_LW/RC_RW", "RC_LFE2",
)
# 5.6.1.1.5 Table 13：17 bit 非标准 bed 掩码按位对应的声道标签。
NONSTD_BED_LABELS = (
    "RC_LFE2", "RC_RW", "RC_LW", "RC_TBR", "RC_TBL", "RC_TSR", "RC_TSL",
    "RC_TFR", "RC_TFL", "RC_RB", "RC_LB", "RC_RS", "RC_LS", "RC_LFE", "RC_C",
    "RC_R", "RC_L",
)
MAX_SLOTS = 16


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
        "isf_idx": None,
        "num_isf_objects": 0,
        "num_dynamic_objects": None,
    }
    if program["dynamic_object_only"]:
        program["lfe_present"] = bool(reader.read(1))
        # 5.6.4.8：对象顺序为 bed → ISF → dynamic，LFE 属于 bed，排在最前。
        program["num_bed_objects"] = 1 if program["lfe_present"] else 0
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
                channels = []
                for bit in range(10):
                    if (bed_mask >> bit) & 1:
                        channels.extend(BED_CHANNEL_LABELS[bit].split("/"))
                program["bed_assignments"].append({
                    "instance": instance, "lfe_only": False, "standard": True,
                    "mask": bed_mask, "channels": channels,
                })
            else:
                bed_mask = reader.read(17)
                channels = [NONSTD_BED_LABELS[bit]
                            for bit in range(17) if (bed_mask >> bit) & 1]
                program["bed_assignments"].append({
                    "instance": instance, "lfe_only": False, "standard": False,
                    "mask": bed_mask, "channels": channels,
                })
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


def _parse_object_info_block(reader, object_index, in_bed_or_isf, raw_payload):
    """5.5.9 ``object_info_block(0)``：一个对象的属性更新。"""
    info = {
        "object": object_index,
        "in_bed_or_isf": bool(in_bed_or_isf),
        "not_active": bool(reader.read(1)),
    }
    # blk == 0 且对象激活时 basic info 恒为 full update（status 0b01）。
    basic_status = 0 if info["not_active"] else 1
    info["basic_status"] = basic_status
    if basic_status in (1, 3):
        if basic_status == 1:
            # 5.5.10：object_basic_info[] = {true, true}；
            # 5.6.4.12：array[1] = object_gain_idx，array[0] = b_default_object_priority。
            gain_present = priority_present = True
        else:
            flags = reader.read(2)
            gain_present = bool(flags & 1)
            priority_present = bool(flags & 2)
        if gain_present:
            gain_idx = reader.read(2)
            info["gain_idx"] = gain_idx
            if gain_idx == 2:
                info["gain_bits"] = reader.read(6)
        if priority_present:
            default_priority = bool(reader.read(1))
            info["default_priority"] = default_priority
            if not default_priority:
                info["priority_bits"] = reader.read(5)
    render_status = 0 if (info["not_active"] or info["in_bed_or_isf"]) else 1
    info["render_status"] = render_status
    if render_status in (1, 3):
        if render_status == 1:
            # 5.5.11：obj_render_info[] = {true, true, true, true}。
            position_present = zone_present = size_present = screen_present = True
        else:
            flags = reader.read(4)
            position_present = bool(flags & 0x1)
            zone_present = bool(flags & 0x2)
            size_present = bool(flags & 0x4)
            screen_present = bool(flags & 0x8)
        if position_present:
            # blk == 0 时 b_differential_position_specified 恒为 FALSE。
            x = reader.read(6)
            y = reader.read(6)
            z_sign = reader.read(1)
            z = reader.read(4)
            info["position"] = (x, y, z if z_sign else -z)
            if reader.read(1):                      # b_object_distance_specified
                if not reader.read(1):              # b_object_at_infinity
                    reader.read(4)                  # distance_factor_idx
        if zone_present:
            info["zone_constraints_idx"] = reader.read(3)
            info["enable_elevation"] = bool(reader.read(1))
        if size_present:
            size_idx = reader.read(2)
            info["object_size_idx"] = size_idx
            if size_idx == 1:
                reader.read(5)
            elif size_idx == 2:
                reader.read(15)
        if screen_present:
            if reader.read(1):                      # b_object_use_screen_ref
                reader.read(3)                      # screen_factor_bits
                reader.read(2)                      # depth_factor_idx
        info["snap"] = bool(reader.read(1))
    if reader.read(1):                              # b_additional_table_data_exists
        info["additional_table_bytes"] = reader.read(4) + 1
        reader.skip(info["additional_table_bytes"] * 8)
    return info


def _parse_object_element(reader, element, raw_payload):
    """5.5.5 ``object_element()``：时间信息 + 每个对象一条 object_info_block。"""
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
        objects = [
            _parse_object_info_block(
                reader, index, index < element["bed_isf_objects"], raw_payload)
            for index in range(object_count)
        ]
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
        "objects": objects,
    }


def _parse_elements(reader, alternate_present, object_count, program, raw_payload):
    """5.5.4 ``oa_element_md()`` 目录；对象 element 按结构解析，其余按声明长度跳过。"""
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
    elements = []
    object_element = None
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
        try:
            alternate_data_id = (reader.read(4)
                                 if alternate_present else None)
            discard_unknown = bool(reader.read(1))
        except ValueError as exc:
            raise _syntax_error(
                "element_control_bounds", "OAMD element 太短，无法容纳控制字段",
                raw_payload, exc,
                element_ordinal=ordinal, element_id=element_id,
                size_bytes=size_bytes) from exc
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
        if element_id == OBJECT_ELEMENT_ID:
            if object_element is not None:
                raise UnsupportedVariantError(
                    "oamd", "multiple_object_elements",
                    "OAMD 含多个 object element",
                    details={
                        "elements": _element_details(elements + [element]),
                        "payload": bytes_descriptor(raw_payload),
                    })
            object_element = _parse_object_element(
                reader, element, raw_payload)
            element["parsed_end_bit"] = reader.position
            element["object_element"] = object_element
            # 个别编码器声明的 oa_element_size 比实际内容短；以结构解析结果为准。
            reader.position = max(region_end, reader.position)
        else:
            # trim / extended / 未知 element：按声明长度整体跳过（5.5.4）。
            element["parsed_end_bit"] = region_end
            reader.position = region_end
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


def frame_update(bits_one):
    """单帧 OAMD → 位置字段增量及其 sample offset/ramp duration。"""
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
            reader, alternate_present, object_count, program, raw_payload)
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
    if object_count > MAX_SLOTS:
        raise UnsupportedVariantError(
            "oamd", "object_count",
            "OAMD 对象数超过当前 16 槽对象模型",
            details={
                "object_count": object_count,
                "max_slots": MAX_SLOTS,
                "payload": bytes_descriptor(raw_payload),
            })
    blocks = object_element["blocks"]
    if len(blocks) != 1:
        raise UnsupportedVariantError(
            "oamd", "multiple_position_blocks",
            "OAMD 一帧含多个对象位置更新块，单次 frame_update 无法表达",
            details={
                "block_count": len(blocks),
                "blocks": blocks,
                "payload": bytes_descriptor(raw_payload),
                "repair_hint": "按 ObjectInfoBlock 顺序逐块解析坐标，再生成分段 ADM ramp",
            })

    # 对外契约：始终给出 16 个槽；本帧没有覆盖到的槽保持上一帧位置。
    out = {(slot, field): None for slot in range(MAX_SLOTS)
           for field in ("q1", "q2", "q3")}
    for info in object_element["objects"]:
        slot = info["object"]
        position = info.get("position")
        if position is None:
            # bed/ISF 对象与未激活对象没有位置字段：保持上一帧位置。
            out[(slot, "q1")] = None
            out[(slot, "q2")] = None
            out[(slot, "q3")] = None
            continue
        x, y, z = position
        out[(slot, "q1")] = q_of(x, N_Q12) if 0 <= x <= N_Q12 else None
        out[(slot, "q2")] = q_of(y, N_Q12) if 0 <= y <= N_Q12 else None
        # 5.6.1.1.10/5.6.1.1.11：pos3D_Z 带符号；当前 16 槽模型只表达非负高度。
        out[(slot, "q3")] = q_of(z, N_Q3) if 0 <= z <= N_Q3 else (
            0 if z < 0 else None)

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
    return {
        "values": out,
        "block_offset_samples": blocks[0]["block_offset_samples"],
        "ramp_duration_samples": blocks[0]["ramp_duration_samples"],
        "object_count": object_count,
        "program": {
            "dynamic_object_only": program["dynamic_object_only"],
            "lfe_present": program["lfe_present"],
            "content_description": program["content_description"],
            "num_bed_objects": program["num_bed_objects"],
            "num_isf_objects": program["num_isf_objects"],
            "num_dynamic_objects": program["num_dynamic_objects"],
            "bed_channels": [list(assignment["channels"])
                             for assignment in program["bed_assignments"]],
        },
        "elements": _element_details(elements),
        "objects": object_element["objects"],
        "size_mismatch": size_mismatch,
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
