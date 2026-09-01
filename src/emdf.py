"""从常见 E-AC-3 同步帧直接提取连续 EMDF 容器。

扫描器检查八种全局位对齐，定位 ``0x5838`` 同步字，验证容器长度并解析各
payload config，因此不要求 EMDF 在原始 E-AC-3 文件中按字节对齐。

本模块有意只覆盖“完整 EMDF 容器在一个同步帧中连续出现”的常见情形。不解析
E-AC-3 mantissa，也不重组被音频数据隔开的多个 skip-field 碎片；遇到这种输入会
明确报错，让上层决定是否使用兼容桥。
"""
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import numpy as np

from variant_error import UnsupportedVariantError


SYNCWORD = 0x5838
REQUIRED_JOC_IDS = frozenset((11, 14))


class EmdfError(ValueError):
    """EMDF 或其 E-AC-3 传输结构不符合本实现支持的范围。"""


class BitReader:
    """MSB-first 位读取器；位置以源数据的绝对 bit offset 表示。"""

    def __init__(self, data, position=0, limit=None):
        self.data = memoryview(data)
        self.position = int(position)
        self.limit = len(self.data) * 8 if limit is None else int(limit)

    def read(self, count):
        count = int(count)
        if count < 0 or self.position + count > self.limit:
            raise EmdfError(f"位流越界 @bit{self.position}, need={count}, limit={self.limit}")
        value = 0
        while count:
            byte_pos = self.position >> 3
            removed_left = self.position & 7
            take = min(count, 8 - removed_left)
            shift = 8 - removed_left - take
            value = (value << take) | ((self.data[byte_pos] >> shift) & ((1 << take) - 1))
            self.position += take
            count -= take
        return value

    def skip(self, count):
        self.read(count)

    def read_bytes(self, count):
        return bytes(self.read(8) for _ in range(count))


def variable_bits(reader, width, max_groups=8):
    """读取 EMDF ``variable_bits(width)`` 变长整数。"""
    value = 0
    for _ in range(max_groups):
        value += reader.read(width)
        more = reader.read(1)
        if not more:
            return value
        value = (value + 1) << width
    raise EmdfError(f"variable_bits({width}) 延伸组过多")


@dataclass(frozen=True)
class EmdfContainer:
    start_bit: int
    raw: bytes
    payloads: dict
    sample_offsets: dict


def _parse_at(data, start_bit):
    """在已知 syncword 的 bit offset 解析一个 EMDF 容器。"""
    reader = BitReader(data, start_bit)
    if reader.read(16) != SYNCWORD:
        raise EmdfError(f"EMDF syncword 不匹配 @bit{start_bit}")
    length = reader.read(16)
    body_start = reader.position
    body_end = body_start + length * 8
    if body_end > reader.limit:
        raise EmdfError(f"EMDF 容器越界 @bit{start_bit}: length={length}")
    reader.limit = body_end

    version = reader.read(2)
    if version == 3:
        version += variable_bits(reader, 2)
    key_id = reader.read(3)
    if key_id == 7:
        key_id += variable_bits(reader, 3)
    # TS 103 420 JOC 使用 version=0/key_id=0；严格限制也能排除音频中的伪 marker。
    if version != 0 or key_id != 0:
        raise EmdfError(f"不支持的 EMDF version/key_id: {version}/{key_id}")

    payloads = {}
    sample_offsets = {}
    terminated = False
    while reader.position + 5 <= body_end:
        payload_id = reader.read(5)
        if payload_id == 0:
            terminated = True
            break
        if payload_id == 0x1F:
            payload_id += variable_bits(reader, 5)
        if payload_id in payloads:
            raise EmdfError(f"同一 EMDF 容器重复 payload id {payload_id}")

        has_sample_offset = bool(reader.read(1))
        sample_offset = (reader.read(12) >> 1) if has_sample_offset else 0
        if reader.read(1):
            variable_bits(reader, 11)       # duration
        if reader.read(1):
            variable_bits(reader, 2)        # group id
        if reader.read(1):
            reader.skip(8)                  # codec data

        if not reader.read(1):              # discard_unknown_payload
            frame_aligned = False
            if not has_sample_offset:
                frame_aligned = bool(reader.read(1))
                if frame_aligned:
                    reader.skip(2)
            if has_sample_offset or frame_aligned:
                reader.skip(7)

        payload_size = variable_bits(reader, 8)
        if reader.position + payload_size * 8 > body_end:
            raise EmdfError(
                f"payload id {payload_id} 越界: size={payload_size}, @bit{reader.position}")
        payloads[payload_id] = reader.read_bytes(payload_size)
        sample_offsets[payload_id] = sample_offset

    if not terminated:
        raise EmdfError("EMDF 容器缺少 payload id 0 终止符")
    total_bytes = 4 + length
    raw_reader = BitReader(data, start_bit, start_bit + total_bytes * 8)
    raw = raw_reader.read_bytes(total_bytes)
    return EmdfContainer(start_bit, raw, payloads, sample_offsets)


def _marker_offsets(data):
    """以 NumPy 批量检查八种位移，返回可能的 0x5838 bit offsets。"""
    source = np.frombuffer(data, dtype=np.uint8)
    if source.size < 4:
        return []
    offsets = []
    for shift in range(8):
        if shift == 0:
            aligned = source
        else:
            aligned = np.bitwise_or(
                np.left_shift(source[:-1].astype(np.uint16), shift) & 0xFF,
                np.right_shift(source[1:].astype(np.uint16), 8 - shift),
            ).astype(np.uint8)
        hits = np.flatnonzero((aligned[:-1] == 0x58) & (aligned[1:] == 0x38))
        offsets.extend(int(hit) * 8 + shift for hit in hits)
    return sorted(offsets)


def find_joc_emdf(frame):
    """返回同步帧中唯一、顶层连续且包含 ID11/ID14 的 JOC EMDF 容器。

    EMDF payload 是不透明字节串，其中可能自然出现另一个 ``0x5838``。若从这个
    内嵌 marker 开始的后续随机位恰好也能通过容器语法探测，它仍不是一个独立的
    transport 容器。因此，候选的起点一旦落在较早 JOC 容器的声明范围内，就只把
    它记作内嵌伪候选，不参与“多个容器”的判定。
    """
    matches = []
    offsets = _marker_offsets(frame)
    parsed_candidates = []
    parse_errors = []
    for start_bit in offsets:
        try:
            container = _parse_at(frame, start_bit)
        except EmdfError as exc:
            if len(parse_errors) < 8:
                parse_errors.append({"start_bit": start_bit, "error": str(exc)})
            continue
        parsed_candidates.append({
            "start_bit": start_bit,
            "payload_ids": list(container.payloads),
            "payload_lengths": {str(k): len(v) for k, v in container.payloads.items()},
        })
        if REQUIRED_JOC_IDS.issubset(container.payloads):
            matches.append(container)
    if not matches:
        raise UnsupportedVariantError(
            "emdf_transport", "no_contiguous_joc_container",
            "同步帧中未找到可连续解析且同时包含 ID11/ID14 的 EMDF 容器",
            details={
                "syncframe_bytes": len(frame),
                "marker_bit_offsets": offsets,
                "parsed_candidates": parsed_candidates,
                "candidate_parse_errors": parse_errors,
                "repair_hint": "检查 EMDF 是否跨多个 audio-block skip field 分片，或 payload config 是否变化",
            })
    top_level_matches = []
    nested_matches = []
    for container in sorted(matches, key=lambda item: item.start_bit):
        parent = next((candidate for candidate in top_level_matches
                       if candidate.start_bit < container.start_bit <
                       candidate.start_bit + len(candidate.raw) * 8), None)
        if parent is None:
            top_level_matches.append(container)
        else:
            nested_matches.append({
                "start_bit": container.start_bit,
                "end_bit": container.start_bit + len(container.raw) * 8,
                "parent_start_bit": parent.start_bit,
                "parent_end_bit": parent.start_bit + len(parent.raw) * 8,
            })
    if len(top_level_matches) != 1:
        starts = [item.start_bit for item in top_level_matches]
        raise UnsupportedVariantError(
            "emdf_transport", "multiple_joc_containers",
            "同步帧中存在多个可用 JOC EMDF，当前无法自动选择",
            details={
                "syncframe_bytes": len(frame),
                "joc_container_start_bits": starts,
                "nested_joc_candidates": nested_matches,
            })
    return top_level_matches[0]


def parse_container(data):
    """解析从 syncword 开始、已经重新按字节对齐保存的 EMDF 容器。"""
    container = _parse_at(data, 0)
    if len(container.raw) != len(data):
        raise EmdfError(f"EMDF 文件尾有额外数据: parsed={len(container.raw)}, file={len(data)}")
    return container


def iter_eac3_frames(data):
    """按 E-AC-3 ``frmsiz`` 遍历同步帧，拒绝静默重同步。"""
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data) or data[pos:pos + 2] != b"\x0b\x77":
            raise UnsupportedVariantError(
                "eac3_transport", "syncframe_header",
                "E-AC-3 同步帧头无效或出现了未处理的子流排列",
                details={
                    "byte_offset": pos,
                    "remaining_bytes": len(data) - pos,
                    "next_16_bytes_hex": data[pos:pos + 16].hex(),
                })
        size = ((((data[pos + 2] & 7) << 8) | data[pos + 3]) + 1) * 2
        if pos + size > len(data):
            raise UnsupportedVariantError(
                "eac3_transport", "truncated_syncframe",
                "E-AC-3 末帧长度超过输入剩余数据",
                details={
                    "byte_offset": pos,
                    "declared_frame_bytes": size,
                    "remaining_bytes": len(data) - pos,
                })
        yield data[pos:pos + size]
        pos += size


def extract_index(eac3_path, output_dir, max_frames=None):
    """将裸 E-AC-3 的连续 EMDF 保存为 ``frames.csv + emdf/``。"""
    output_dir = Path(output_dir)
    emdf_dir = output_dir / "emdf"
    emdf_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_eac3_frames(Path(eac3_path).read_bytes())
    rows = []
    for frame_number, frame in enumerate(frames):
        if max_frames is not None and frame_number >= max_frames:
            break
        try:
            container = find_joc_emdf(frame)
        except UnsupportedVariantError as exc:
            exc.add_context(frame=frame_number, details={"syncframe_bytes": len(frame)})
            raise
        except EmdfError as exc:
            raise UnsupportedVariantError(
                "emdf_transport", "container_syntax",
                "EMDF 容器语法无法解析",
                frame=frame_number,
                details={"syncframe_bytes": len(frame), "parser_error": str(exc)}) from exc
        digest = hashlib.sha256(container.raw).hexdigest()
        target = emdf_dir / f"{digest}.bin"
        if not target.is_file():
            target.write_bytes(container.raw)
        rows.append({
            "frame": frame_number,
            "emdf_hash": digest,
            "emdf_size": len(container.raw),
            "emdf_start_bit": container.start_bit,
            "payload_ids": ";".join(str(x) for x in container.payloads),
            "error": "",
        })
        if (frame_number + 1) % 1000 == 0:
            print(f"[metadata] {frame_number + 1} frames", flush=True)
    if not rows:
        raise EmdfError("E-AC-3 输入中没有可处理的同步帧")
    with (output_dir / "frames.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = ("frame", "emdf_hash", "emdf_size", "emdf_start_bit", "payload_ids", "error")
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_dir
