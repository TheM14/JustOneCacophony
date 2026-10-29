"""Evolution sidecar 的 JOC/OAMD 纯 Python 解析、校验和可读输出。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from adm_atmos import q_to_adm_xyz
from bed_layout import (BED_CHANNEL_NAMES, EXTRA_PAIR_BY_CONFIG, bed_layout,
                        declared_bed_channels, extra_pair)
from evo_unpack import unpack_evolution
from emdf import (ACMOD_LOCATIONS, EmdfError, audio_frame_groups, expand_chanmap,
                  find_joc_emdf, iter_syncframes, parse_container)
from joc_decode import parse_joc, parse_joc_header
from oamd_bits import (JocFieldState, frame_update_values, program_bed_channels,
                       program_bed_summary)
from variant_error import UnsupportedVariantError, bytes_descriptor


def _joc_pair_score(container):
    """这个容器作为本帧 JOC 元数据的可信度（规格 §8.2 / Table 56）。

    §8.2：OAMD 与 JOC 载荷放在同一个 EMDF 容器里；Table 56：两者的 groupid
    必须相同。同一个同步帧里可能出现多个含 ID11/ID14 的顶层容器，其中不构成一对的那个
    （ID14 声明的对象数超出其字节所能容纳的范围、ID11 未声明 groupid）既不是一对
    合法载荷，也不可能是本帧的 JOC 参数。

    2 = 两个载荷 groupid 相同且 ID14 完整解析；1 = ID14 完整解析；0 = 其余。
    """
    payload = container.payloads.get(14)
    if payload is None or 11 not in container.payloads:
        return 0
    parses = True
    try:
        parse_joc(payload)
    except (IndexError, ValueError):
        parses = False
    group_ids = getattr(container, "group_ids", None) or {}
    first = group_ids.get(11)
    second = group_ids.get(14)
    associated = first is not None and second is not None and first == second
    if associated and parses:
        return 2
    return 1 if parses else 0


def _transport_record(group):
    """从一个音频帧的同步帧组提取规格 §11.2 需要的 L2 声明。"""
    core_locations = ()
    core_lfe = False
    chanmap = None
    for sync in group:
        if sync.codec == "ac3" or sync.strmtyp == 0:
            core_locations = ACMOD_LOCATIONS.get(sync.acmod, ())
            core_lfe = bool(sync.lfeon)
        elif sync.chanmap is not None:
            chanmap = sync.chanmap
    return {
        "core_locations": tuple(core_locations),
        "core_lfe": core_lfe,
        "chanmap": chanmap,
        "syncframes": [
            {
                "offset": sync.offset,
                "size": sync.size,
                "codec": sync.codec,
                "strmtyp": sync.strmtyp,
                "substreamid": sync.substreamid,
                "acmod": sync.acmod,
                "lfeon": sync.lfeon,
            }
            for sync in group
        ],
    }


def _bed_covers_core(bed_channels, core_positions):
    """声明的 bed 布局是否覆盖核心子流携带的全频段声道。

    OAMD 的 bed 掩码描述的是**内容**的 bed 布局，可以比 E-AC-3 核心实际编码的声道多
    （例如 9.1.6 bed 只编 5.1 核心）。因此只有"核心声道在 bed 布局里找不到位置"才是
    真正的不一致；多出来的 bed 声道是布局的一部分，不算冲突。
    """
    names = {BED_CHANNEL_NAMES.get(label, label)
             for instance in bed_channels for label in instance}
    return all(position in names for position in core_positions)


def check_cross_layer(parsed, *, frame, transport=None, bed_channels=None, bed_layout_info=None,
                      diagnostics=None):
    """规格 §11.2 的跨层一致性校验。

    校验 1/2（额外那对的身份）不一致时只记 diagnostics：通道身份不进 DSP
    （§10.1），床映射按 chanmap 给出的通道码做，音频与 k 一致时逐样本相同。
    校验 3（核心通道数）与 joc_ext_config_idx 会改变 DSP 输入，仍按变体报错。
    """
    config_idx = parsed["dmx_config_idx"]
    n_channels = parsed["n_channels"]
    expected_pair = EXTRA_PAIR_BY_CONFIG.get(config_idx)
    if parsed["ext_config_idx"] != 0:
        raise UnsupportedVariantError(
            "joc", "ext_config_idx_unsupported",
            "joc_ext_config_idx 取到当前未映射的值",
            frame=frame,
            details={
                "downmix_config": config_idx,
                "extension_config_idx": parsed["ext_config_idx"],
                "payload": bytes_descriptor(parsed.get("payload", b"")),
            })
    if diagnostics is not None and parsed.get("objects_clamped"):
        diagnostics.append({
            "stage": "joc",
            "variant": "num_objects_clamped",
            "frame": frame,
            "message": "对象数超过 15，已截断",
            "details": {"declared": parsed["num_objects_bits"] + 1,
                        "used": parsed["n_objects"]},
        })

    if transport is not None:
        chanmap = transport["chanmap"]
        core_locations = transport["core_locations"]
        details = {
            "config_idx": config_idx,
            "n_channels": n_channels,
            "syncframes": transport["syncframes"],
        }
        if n_channels == 7:
            if not core_locations or chanmap is None:
                raise UnsupportedVariantError(
                    "eac3", "core_extension_pair",
                    "7 通道配置需要独立子流与依赖子流 chanmap 同时在场",
                    frame=frame, details=details)
            pair = extra_pair(core_locations, chanmap)
            details["chanmap"] = chanmap
            details["chanmap_locations"] = expand_chanmap(chanmap)
            details["extra_pair"] = list(pair)
            if pair != expected_pair:
                if diagnostics is not None:
                    diagnostics.append({
                        "stage": "eac3",
                        "variant": "chanmap_cfg_mismatch",
                        "frame": frame,
                        "message": "依赖子流 chanmap 声明的额外对与 joc_dmx_config_idx 不一致，"
                                   "按 chanmap 的通道码映射，音频不受影响",
                        "details": {**details, "expected_pair": list(expected_pair or ())},
                    })
        elif chanmap is not None and extra_pair(core_locations, chanmap):
            raise UnsupportedVariantError(
                "eac3", "core_channel_count_mismatch",
                "5 通道配置的依赖子流声明了额外通道",
                frame=frame,
                details={**details, "chanmap_locations": expand_chanmap(chanmap)})
        # 校验 3：核心通道数（子流声明汇总，去掉 LFE）
        declared = set(core_locations)
        if chanmap is not None:
            declared |= set(expand_chanmap(chanmap))
        declared -= {"LFE", "LFE2"}
        details["declared_full_band_channels"] = len(declared)
        if len(declared) != n_channels:
            raise UnsupportedVariantError(
                "eac3", "core_channel_count_mismatch",
                "子流声明的全频段通道数与 joc_dmx_config_idx 不一致",
                frame=frame, details=details)

    if bed_channels is not None:
        core_positions = []
        if transport is not None:
            core_positions = list(transport["core_locations"])
            if transport["chanmap"] is not None:
                core_positions += expand_chanmap(transport["chanmap"])
        if not _bed_covers_core(bed_channels, core_positions):
            if diagnostics is not None:
                diagnostics.append({
                    "stage": "oamd",
                    "variant": "bed_mask_cfg_mismatch",
                    "frame": frame,
                    "message": "OAMD bed 布局没有覆盖核心子流的全频段声道，只影响 bed 槽位记录，"
                               "音频不受影响",
                    "details": {
                        "config_idx": config_idx,
                        "bed_channels": [list(instance) for instance in bed_channels],
                        "bed_layout": (bed_layout_info or {}).get("name"),
                        "core_positions": core_positions,
                    },
                })


class DirectPayloadIndex:
    """One-pass in-memory view of contiguous EMDF containers in an E-AC-3 file.

    The optional cache directory keeps the existing ``frames.csv + emdf/``
    contract, but normal processing reuses the containers already parsed during
    the scan instead of reading and parsing thousands of small files again.

    音频帧 = 同一 (编码, strmtyp, substreamid) 组合出现一次为止的同步帧组；
    JOC 通常只在依赖子流里，核心子流（AC-3 或 E-AC-3 独立子流）没有 EMDF。
    """

    def __init__(self, rows, subpayloads, sample_offsets, directory=None,
                 transports=None, diagnostics=None, config_idx=None, n_channels=None,
                 bed_assignments=None, bed_only=False):
        self.rows = rows
        self._subpayloads = subpayloads
        self._sample_offsets = sample_offsets
        self.directory = Path(directory) if directory is not None else None
        self.transports = transports if transports is not None else []
        self.diagnostics = diagnostics if diagnostics is not None else []
        self.config_idx = config_idx
        self.n_channels = n_channels
        # OAMD bed 声明（内容声床布局）：bed 对象没有位置字段，位置由布局决定。
        self.bed_assignments = bed_assignments if bed_assignments is not None else []
        self.bed_layout = bed_layout(self.bed_assignments)
        self.bed_only = bool(bed_only)

    @property
    def bed_channels(self):
        """bed 声道标签（JOC 对象槽顺序）；非 bed-only 内容返回空。

        bed 声道是固定通道，只有整个 program 都由 bed 组成时 15 路 JOC 对象才逐槽
        对应 bed 声道。含动态对象的内容仍按 OAMD 坐标渲染。
        """
        if not self.bed_only:
            return ()
        layout = self.bed_layout or {}
        return tuple(layout.get("channels") or ())

    def transport(self, row):
        return self.transports[int(row["frame"])] if self.transports else None

    def subpayloads(self, row):
        return self._subpayloads[int(row["frame"])]

    def subpayload_sample_offset(self, row, payload_id):
        offsets = self._sample_offsets[int(row["frame"])]
        return 0 if offsets is None else int(offsets.get(payload_id, 0))

    @classmethod
    def from_eac3(cls, eac3_path, max_frames=None, cache_dir=None):
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        emdf_dir = None
        if cache_dir is not None:
            emdf_dir = cache_dir / "emdf"
            emdf_dir.mkdir(parents=True, exist_ok=True)
        data = Path(eac3_path).read_bytes()
        rows = []
        payloads = []
        offsets = []
        transports = []
        diagnostics = []
        config_idx = None
        n_channels = None
        missing_joc_frames = 0
        bed_assignments = []
        bed_only = False
        bed_layout_info = None
        groups = audio_frame_groups(iter_syncframes(data))
        for frame_number, group in enumerate(groups):
            if max_frames is not None and frame_number >= max_frames:
                break
            transport = _transport_record(group)
            transports.append(transport)
            container = None
            for sync in group:
                chunk = data[sync.offset:sync.offset + sync.size]
                try:
                    container = find_joc_emdf(chunk, usable=_joc_pair_score,
                                              diagnostics=diagnostics,
                                              frame_number=frame_number)
                except UnsupportedVariantError as exc:
                    if exc.variant != "no_contiguous_joc_container":
                        exc.add_context(frame=frame_number,
                                        details={"syncframe_bytes": len(chunk)})
                        raise
                    continue
                except EmdfError as exc:
                    raise UnsupportedVariantError(
                        "emdf_transport", "container_syntax",
                        "EMDF 容器语法无法解析",
                        frame=frame_number,
                        details={"syncframe_bytes": len(chunk),
                                 "parser_error": str(exc)}) from exc
                break
            if container is None:
                missing_joc_frames += 1
                rows.append({
                    "frame": frame_number,
                    "emdf_hash": "",
                    "emdf_size": 0,
                    "emdf_start_bit": "",
                    "payload_ids": "",
                    "error": "no_joc",
                })
                payloads.append(None)
                offsets.append(None)
                continue

            if 14 in container.payloads:
                parsed = parse_joc_header(container.payloads[14])
                parsed["payload"] = container.payloads[14]
                frame_bed = (program_bed_summary(container.payloads[11])
                             if 11 in container.payloads else None)
                if frame_bed and not bed_assignments and frame_bed["channels"]:
                    bed_assignments = [list(instance) for instance in frame_bed["channels"]]
                    bed_only = bool(frame_bed["bed_only"])
                    bed_layout_info = bed_layout(bed_assignments)
                check_cross_layer(parsed, frame=frame_number, transport=transport,
                                  bed_channels=(frame_bed["channels"] if frame_bed else None),
                                  bed_layout_info=bed_layout_info,
                                  diagnostics=diagnostics)
                if config_idx is None:
                    config_idx = parsed["dmx_config_idx"]
                    n_channels = parsed["n_channels"]

            digest = hashlib.sha256(container.raw).hexdigest()
            if emdf_dir is not None:
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
            payloads.append(container.payloads)
            offsets.append(container.sample_offsets)
            if (frame_number + 1) % 1000 == 0:
                print(f"[metadata] {frame_number + 1} frames", flush=True)
        if not rows:
            raise EmdfError("E-AC-3 输入中没有可处理的同步帧")
        if config_idx is None:
            raise UnsupportedVariantError(
                "emdf_transport", "no_contiguous_joc_container",
                "输入中没有找到可连续解析且同时包含 ID11/ID14 的 EMDF 容器",
                details={"audio_frames": len(rows), "syncframe_bytes": len(data)})
        if missing_joc_frames:
            diagnostics.append({
                "stage": "eac3",
                "variant": "joc_frame_missing",
                "frame": None,
                "message": "部分音频帧没有 JOC 元数据，这些帧的对象通道静音",
                "details": {"frames": missing_joc_frames, "total": len(rows)},
            })
        if cache_dir is not None:
            with (cache_dir / "frames.csv").open("w", encoding="utf-8", newline="") as fp:
                fields = ("frame", "emdf_hash", "emdf_size", "emdf_start_bit", "payload_ids", "error")
                writer = csv.DictWriter(fp, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return cls(rows, payloads, offsets, cache_dir, transports, diagnostics,
                   config_idx, n_channels, bed_assignments, bed_only)

    def __len__(self):
        return len(self.rows)


class PayloadIndex:
    """旧 evolution sidecar 或直接 EMDF sidecar 的严格顺序视图。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        csv_path = self.directory / "frames.csv"
        self.payload_dir = self.directory / "payloads"
        self.emdf_dir = self.directory / "emdf"
        if not csv_path.is_file() or not (self.payload_dir.is_dir() or self.emdf_dir.is_dir()):
            raise FileNotFoundError(
                f"元数据目录需要 frames.csv 和 payloads/ 或 emdf/: {self.directory}")
        with csv_path.open(encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if not self.rows:
            raise ValueError("frames.csv 为空")
        self._sub_cache = {}
        self._sample_offset_cache = {}

    def payload(self, row):
        payload_hash = row.get("payload_hash", "")
        if not payload_hash:
            raise ValueError(f"frame {row.get('frame', '?')} 缺少 evolution payload")
        path = self.payload_dir / f"{payload_hash}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def subpayloads(self, row):
        """返回 ``{payload_id: bytes}``，屏蔽两种 sidecar 容器的差异。"""
        emdf_hash = row.get("emdf_hash", "")
        if emdf_hash:
            key = ("emdf", emdf_hash)
            if key not in self._sub_cache:
                path = self.emdf_dir / f"{emdf_hash}.bin"
                if not path.is_file():
                    raise FileNotFoundError(path)
                container = parse_container(path.read_bytes())
                self._sub_cache[key] = container.payloads
                self._sample_offset_cache[key] = container.sample_offsets
            return self._sub_cache[key]
        payload_hash = row.get("payload_hash", "")
        key = ("evolution", payload_hash)
        if key not in self._sub_cache:
            self._sub_cache[key], _ = unpack_evolution(self.payload(row), loose=True)
            self._sample_offset_cache[key] = {}
        return self._sub_cache[key]

    def subpayload_sample_offset(self, row, payload_id):
        """返回 EMDF payload 的外层 sample offset；旧 sidecar 没有该字段时为 0。"""
        self.subpayloads(row)
        emdf_hash = row.get("emdf_hash", "")
        key = (("emdf", emdf_hash) if emdf_hash else
               ("evolution", row.get("payload_hash", "")))
        return int(self._sample_offset_cache.get(key, {}).get(payload_id, 0))

    def __len__(self):
        return len(self.rows)


def _frame_record(frame, parsed, state):
    present = [i for i, obj in enumerate(parsed["objs"]) if obj["present"]]
    sparse = [i for i in present if parsed["objs"][i]["sparse"]]
    bands = Counter(parsed["objs"][i]["n_bands"] for i in present)
    positions = []
    for obj in range(1, 16):
        q1, q2, q3 = (state.q[(obj, key)] for key in ("q1", "q2", "q3"))
        x, y, z = q_to_adm_xyz(q1, q2, q3)
        positions.append([round(float(x), 7), round(float(y), 7), round(float(z), 7)])
    return {
        "frame": frame,
        "sequence": parsed["seq_count_bits"],
        "downmix_config": parsed["dmx_config_idx"],
        "extension_config": parsed["ext_config_idx"],
        "objects_declared": parsed["n_objects"],
        "objects_present": present,
        "sparse_objects": sparse,
        "parameter_bands": {str(k): v for k, v in sorted(bands.items())},
        "clipgain": parsed["clipgain"],
        "oamd_xyz": positions,
    }


def inspect(index, limit=None, print_frames=False):
    """逐帧完整解析 ID11/ID14，并返回可 JSON 序列化的汇总。"""
    rows = index.rows if limit is None else index.rows[:limit]
    oamd = JocFieldState()
    frame_records = []
    configs = Counter()
    active = Counter()
    band_totals = Counter()
    sparse_counts = Counter()
    clipgains = []
    for seq, row in enumerate(rows):
        frame_number = int(row.get("frame", seq))
        try:
            subs = index.subpayloads(row)
        except UnsupportedVariantError as exc:
            exc.add_context(frame=frame_number)
            raise
        except Exception as exc:
            raise UnsupportedVariantError(
                "metadata_container", "sidecar_or_emdf_parse",
                "元数据容器无法解析",
                frame=frame_number,
                details={"exception_type": type(exc).__name__, "parser_error": str(exc)}) from exc
        if subs is None:
            # 该音频帧没有任何 JOC EMDF（例如编码器未写首帧元数据）：对象静音。
            frame_records.append({"frame": frame_number, "no_joc": True})
            continue
        if 11 not in subs or 14 not in subs:
            raise UnsupportedVariantError(
                "emdf_payloads", "missing_required_payload",
                "EMDF 缺少 ID11/OAMD 或 ID14/JOC",
                frame=frame_number,
                details={
                    "payload_ids": list(subs),
                    "payload_lengths": {str(k): len(v) for k, v in subs.items()},
                })
        try:
            oamd.apply(frame_update_values(subs[11]))
        except UnsupportedVariantError as exc:
            exc.add_context(frame=frame_number, details={"payload_id": 11})
            raise
        except Exception as exc:
            raise UnsupportedVariantError(
                "oamd", "payload_syntax",
                "OAMD 字段解析失败",
                frame=frame_number,
                details={
                    "payload_id": 11,
                    "payload": bytes_descriptor(subs[11]),
                    "exception_type": type(exc).__name__,
                    "parser_error": str(exc),
                }) from exc
        try:
            parsed = parse_joc(subs[14])
        except UnsupportedVariantError as exc:
            exc.add_context(frame=frame_number, details={"payload_id": 14})
            raise
        except Exception as exc:
            payload = subs[14]
            header = {}
            if len(payload) >= 4:
                value = int.from_bytes(payload[:4], "big")
                header = {
                    "downmix_config": (value >> 29) & 7,
                    "objects_minus_one": (value >> 23) & 63,
                    "extension_config": (value >> 20) & 7,
                }
            raise UnsupportedVariantError(
                "joc", "payload_syntax",
                "JOC ID14 解析失败",
                frame=frame_number,
                details={
                    "payload_id": 14,
                    "header_probe": header,
                    "payload": bytes_descriptor(payload),
                    "exception_type": type(exc).__name__,
                    "parser_error": str(exc),
                    "repair_hint": "检查 JOC header、对象数、参数带、Huffman 或扩展字段",
                }) from exc
        if parsed["n_channels"] not in (5, 7):
            raise UnsupportedVariantError(
                "joc", "unsupported_configuration",
                "JOC 核心通道数超出当前渲染器范围",
                frame=frame_number,
                details={
                    "downmix_config": parsed["dmx_config_idx"],
                    "core_channels": parsed["n_channels"],
                    "objects": parsed["n_objects"],
                    "extension_config": parsed["ext_config_idx"],
                    "payload": bytes_descriptor(subs[14]),
                })
        check_cross_layer(
            parsed, frame=frame_number,
            transport=(index.transport(row) if hasattr(index, "transport") else None),
            bed_channels=program_bed_channels(subs[11]))
        rec = _frame_record(frame_number, parsed, oamd)
        configs[(parsed["dmx_config_idx"], parsed["ext_config_idx"],
                 parsed["n_channels"], parsed["n_objects"])] += 1
        active[len(rec["objects_present"])] += 1
        band_totals.update({int(k): v for k, v in rec["parameter_bands"].items()})
        sparse_counts[len(rec["sparse_objects"])] += 1
        clipgains.append(float(parsed["clipgain"]))
        if print_frames:
            print(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
        frame_records.append(rec)

    summary = {
        "frames": len(rows),
        "frame_samples": 1536,
        "sample_rate": 48000,
        "duration_sec": len(rows) * 1536 / 48000,
        "bed_layout": (getattr(index, "bed_layout", None) or {}).get("name"),
        "bed_channels": [
            BED_CHANNEL_NAMES.get(label, label)
            for label in getattr(index, "bed_channels", ())],
        "configurations": [
            {"downmix_config": k[0], "extension_config": k[1], "core_channels": k[2],
             "objects": k[3], "frames": count}
            for k, count in sorted(configs.items())
        ],
        "active_object_count_histogram": {str(k): v for k, v in sorted(active.items())},
        "parameter_band_totals": {str(k): v for k, v in sorted(band_totals.items())},
        "sparse_object_count_histogram": {str(k): v for k, v in sorted(sparse_counts.items())},
        "clipgain_min": min(clipgains),
        "clipgain_max": max(clipgains),
        "first_frame": frame_records[0],
        "last_frame": frame_records[-1],
    }
    emdf_rows = [row for row in rows if row.get("emdf_start_bit", "") != ""]
    if emdf_rows:
        starts = [int(row["emdf_start_bit"]) for row in emdf_rows]
        id_sets = Counter(row.get("payload_ids", "") for row in emdf_rows)
        summary["emdf_transport"] = {
            "continuous_containers": len(emdf_rows),
            "start_bit_min": min(starts),
            "start_bit_max": max(starts),
            "bit_alignment_histogram": {
                str(k): v for k, v in sorted(Counter(x & 7 for x in starts).items())
            },
            "payload_id_order_histogram": dict(sorted(id_sets.items())),
        }
    return summary


def write_summary(index, output, limit=None, print_frames=False):
    summary = inspect(index, limit=limit, print_frames=print_frames)
    output = Path(output)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
