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
    # payload_id -> group_id（未声明 groupid 的载荷为 None）。规格 §8.2/Table 56：
    # 同一帧的 OAMD 与 JOC 载荷属于同一 group，凭此把一对载荷关联起来。
    group_ids: dict = None


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
    group_ids = {}
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
            group_ids[payload_id] = variable_bits(reader, 2)   # group id
        else:
            group_ids[payload_id] = None
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
    return EmdfContainer(start_bit, raw, payloads, sample_offsets, group_ids)


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


def find_joc_emdf(frame, usable=None, diagnostics=None, frame_number=None):
    """返回同步帧中唯一、顶层连续且包含 ID11/ID14 的 JOC EMDF 容器。

    EMDF payload 是不透明字节串，其中可能自然出现另一个 ``0x5838``。若从这个
    内嵌 marker 开始的后续随机位恰好也能通过容器语法探测，它仍不是一个独立的
    transport 容器。因此，候选的起点一旦落在较早 JOC 容器的声明范围内，就只把
    它记作内嵌伪候选，不参与“多个容器”的判定。

    一个同步帧里可能出现多个顶层容器（同一帧可能同时带两套 ID11/ID14）。
    ``usable`` 给出“这个容器能否作为本帧元数据”的分值（0 = 不是，越大越可信）：
    只有唯一最高分时用它，其余记入 ``diagnostics``；没有候选或最高分并列仍按变体
    报错，不猜。
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
        if usable is not None:
            scored = [(usable(item), item) for item in top_level_matches]
            best = max(score for score, _item in scored)
            winners = [item for score, item in scored if score == best]
            if best > 0 and len(winners) == 1:
                if diagnostics is not None:
                    diagnostics.append({
                        "stage": "emdf",
                        "variant": "extra_container_ignored",
                        "frame": frame_number,
                        "message": "同步帧内有多个含 ID11/ID14 的 EMDF 容器，只有一个能作为本帧元数据",
                        "details": {"joc_container_start_bits": starts,
                                    "used_start_bit": winners[0].start_bit,
                                    "pair_score": best},
                    })
                return winners[0]
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


# AC-3 同步帧长度表（16-bit word），索引为 fscod（0=48 kHz、1=44.1 kHz、2=32 kHz）
# 再取 frmsizecod；与 A/52 的比特率表等价。
_AC3_FRAME_WORDS = (
    (64, 64, 80, 80, 96, 96, 112, 112, 128, 128, 160, 160, 192, 192, 224, 224,
     256, 256, 320, 320, 384, 384, 448, 448, 512, 512, 640, 640, 768, 768,
     896, 896, 1024, 1024, 1152, 1152, 1280, 1280),
    (69, 70, 87, 88, 104, 105, 121, 122, 139, 140, 174, 175, 208, 209, 243, 244,
     278, 278, 348, 348, 417, 418, 487, 487, 557, 557, 696, 696, 835, 835,
     975, 975, 1114, 1114, 1253, 1253, 1393, 1393),
    (96, 96, 120, 120, 144, 144, 168, 168, 192, 192, 240, 240, 288, 288, 336, 336,
     384, 384, 480, 480, 576, 576, 672, 672, 768, 768, 960, 960, 1152, 1152,
     1344, 1344, 1536, 1536, 1728, 1728, 1920, 1920),
)


# E-AC-3 chanmap（TS 102 366 Annex E Table E2.5）：字段的最高位对应 bit 0。
CHANMAP_LOCATIONS = (
    "L", "C", "R", "Ls", "Rs", "Lc/Rc", "Lrs/Rrs", "Cs", "Ts", "Lsd/Rsd",
    "Lw/Rw", "Vhl/Vhr", "Vhc", "Lts/Rts", "LFE2", "LFE",
)
# 单个 bit 展开为两个相邻声道的成对位置；展开顺序为先左后右。
CHANMAP_PAIR_BITS = frozenset((5, 6, 9, 10, 11, 13))
# A/52 acmod 对应的全频段声道位置（不含 LFE，按解码顺序）。
ACMOD_LOCATIONS = {
    0: ("C",),                    # 1+1 双单声道
    1: ("C",),
    2: ("L", "R"),
    3: ("L", "C", "R"),
    4: ("L", "R", "Cs"),
    5: ("L", "C", "R", "Cs"),
    6: ("L", "R", "Ls", "Rs"),
    7: ("L", "C", "R", "Ls", "Rs"),
}


def expand_chanmap(chanmap):
    """按 Table E2.5 把 16 bit chanmap 展开为有序声道位置列表。"""
    locations = []
    for bit in range(16):
        if not (int(chanmap) >> (15 - bit)) & 1:
            continue
        locations.extend(CHANMAP_LOCATIONS[bit].split("/"))
    return locations


@dataclass(frozen=True)
class SyncFrame:
    """一个同步帧及其传输层（L2）字段。"""

    offset: int
    size: int
    codec: str          # "ac3" 或 "eac3"
    bsid: int
    strmtyp: int        # AC-3 没有子流类型，固定记为 0
    substreamid: int
    acmod: int
    lfeon: int
    chanmap: object     # 依赖子流的 16 bit chanmap；未声明为 None

    @property
    def key(self):
        """同一音频帧内唯一的子流标识。"""
        return (self.codec, self.strmtyp, self.substreamid)


def _read_bits(data, start, count):
    value = 0
    for index in range(start, start + count):
        value = (value << 1) | ((data[index >> 3] >> (7 - (index & 7))) & 1)
    return value


def sync_frame_at(data, offset):
    """解析 offset 处同步帧的传输层头。

    区分 AC-3 与 E-AC-3 的依据是 bsid（bit 40 起 5 bit）：bsid <= 8 为 AC-3，
    其帧长由 fscod/frmsizecod 的比特率表给出；bsid >= 16 为 E-AC-3，帧长由
    frmsiz 给出。依赖子流（strmtyp=1）在这里顺带解析出 chanmap。
    """
    if offset + 6 > len(data) or data[offset] != 0x0B or data[offset + 1] != 0x77:
        return None
    base = offset * 8
    bsid = _read_bits(data, base + 40, 5)
    if bsid <= 8:
        fscod = _read_bits(data, base + 32, 2)
        frmsizecod = _read_bits(data, base + 34, 6)
        if fscod > 2 or frmsizecod >= 38:
            return None
        size = _AC3_FRAME_WORDS[fscod][frmsizecod] * 2
        acmod = _read_bits(data, base + 48, 3)
        bit = 51
        if acmod & 0x1 and acmod != 0x1:
            bit += 2                      # cmixlev
        if acmod & 0x4:
            bit += 2                      # surmixlev
        if acmod == 0x2:
            bit += 2                      # dsurmod
        lfeon = _read_bits(data, base + bit, 1)
        return SyncFrame(offset, size, "ac3", bsid, 0, 0, acmod, lfeon, None)

    strmtyp = _read_bits(data, base + 16, 2)
    substreamid = _read_bits(data, base + 18, 3)
    frmsiz = _read_bits(data, base + 21, 11)
    # fscod==3 时的 fscod2 同样是 2 bit，因此 acmod 恒在 bit 36。
    acmod = _read_bits(data, base + 36, 3)
    lfeon = _read_bits(data, base + 39, 1)
    bit = 45                              # bsid 之后
    bit += 5                              # dialnorm
    compre = _read_bits(data, base + bit, 1)
    bit += 1
    if compre:
        bit += 8                          # compr
    if acmod == 0:
        bit += 5                          # dialnorm2
        compr2 = _read_bits(data, base + bit, 1)
        bit += 1
        if compr2:
            bit += 8
    chanmap = None
    if strmtyp == 1:
        if _read_bits(data, base + bit, 1):
            chanmap = _read_bits(data, base + bit + 1, 16)
    return SyncFrame(offset, (frmsiz + 1) * 2, "eac3", bsid, strmtyp, substreamid,
                     acmod, lfeon, chanmap)


def iter_syncframes(data):
    """按同步字长度遍历输入中的 AC-3 与 E-AC-3 同步帧。"""
    pos = 0
    while pos < len(data):
        frame = sync_frame_at(data, pos)
        if frame is None:
            raise UnsupportedVariantError(
                "eac3_transport", "syncframe_header",
                "同步帧头无效或出现了未处理的子流排列",
                details={
                    "byte_offset": pos,
                    "remaining_bytes": len(data) - pos,
                    "next_16_bytes_hex": data[pos:pos + 16].hex(),
                })
        if pos + frame.size > len(data):
            raise UnsupportedVariantError(
                "eac3_transport", "truncated_syncframe",
                "末帧长度超过输入剩余数据",
                details={
                    "byte_offset": pos,
                    "codec": frame.codec,
                    "declared_frame_bytes": frame.size,
                    "remaining_bytes": len(data) - pos,
                })
        yield frame
        pos += frame.size


def audio_frame_groups(frames):
    """把同步帧序列按音频帧分组。

    一个音频帧内每个 (编码, strmtyp, substreamid) 组合只出现一次，因此再次遇到
    已出现过的组合即表示下一个音频帧开始。独立子流缺席（只有依赖子流）或核心是
    AC-3 时同样成立。
    """
    group = []
    keys = set()
    for frame in frames:
        if frame.key in keys:
            yield group
            group = []
            keys = set()
        keys.add(frame.key)
        group.append(frame)
    if group:
        yield group


def iter_eac3_frames(data):
    """兼容入口：只产出同步帧字节。"""
    for frame in iter_syncframes(data):
        yield data[frame.offset:frame.offset + frame.size]


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
