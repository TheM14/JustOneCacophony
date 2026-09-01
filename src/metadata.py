"""Evolution sidecar 的 JOC/OAMD 纯 Python 解析、校验和可读输出。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from adm_atmos import q_to_adm_xyz
from evo_unpack import unpack_evolution
from emdf import EmdfError, find_joc_emdf, iter_eac3_frames, parse_container
from joc_decode import parse_joc
from oamd_bits import JocFieldState, frame_update_values
from variant_error import UnsupportedVariantError, bytes_descriptor


class DirectPayloadIndex:
    """One-pass in-memory view of contiguous EMDF containers in an E-AC-3 file.

    The optional cache directory keeps the existing ``frames.csv + emdf/``
    contract, but normal processing reuses the containers already parsed during
    the scan instead of reading and parsing thousands of small files again.
    """

    def __init__(self, rows, subpayloads, sample_offsets, directory=None):
        self.rows = rows
        self._subpayloads = subpayloads
        self._sample_offsets = sample_offsets
        self.directory = Path(directory) if directory is not None else None

    @classmethod
    def from_eac3(cls, eac3_path, max_frames=None, cache_dir=None):
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        emdf_dir = None
        if cache_dir is not None:
            emdf_dir = cache_dir / "emdf"
            emdf_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        payloads = []
        offsets = []
        frames = iter_eac3_frames(Path(eac3_path).read_bytes())
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
        if cache_dir is not None:
            with (cache_dir / "frames.csv").open("w", encoding="utf-8", newline="") as fp:
                fields = ("frame", "emdf_hash", "emdf_size", "emdf_start_bit", "payload_ids", "error")
                writer = csv.DictWriter(fp, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return cls(rows, payloads, offsets, cache_dir)

    def subpayloads(self, row):
        return self._subpayloads[int(row["frame"])]

    def subpayload_sample_offset(self, row, payload_id):
        return int(self._sample_offsets[int(row["frame"])].get(payload_id, 0))

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
        sparse = [i for i, obj in enumerate(parsed["objs"]) if obj["present"] and obj["sparse"]]
        if sparse:
            raise UnsupportedVariantError(
                "joc", "sparse_joc",
                "发现尚未验证的 Sparse JOC 帧",
                frame=frame_number,
                details={
                    "sparse_objects": sparse,
                    "downmix_config": parsed["dmx_config_idx"],
                    "extension_config": parsed["ext_config_idx"],
                    "objects": parsed["n_objects"],
                    "payload": bytes_descriptor(subs[14]),
                    "repair_hint": "需要 Sparse JOC 实际样本及对应输出建立回归后再启用",
                })
        if parsed["n_channels"] != 5 or parsed["n_objects"] > 15:
            raise UnsupportedVariantError(
                "joc", "unsupported_configuration",
                "JOC 核心通道数或对象数超出当前渲染器范围",
                frame=frame_number,
                details={
                    "downmix_config": parsed["dmx_config_idx"],
                    "core_channels": parsed["n_channels"],
                    "objects": parsed["n_objects"],
                    "extension_config": parsed["ext_config_idx"],
                    "payload": bytes_descriptor(subs[14]),
                })
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
