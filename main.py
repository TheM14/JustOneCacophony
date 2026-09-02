"""JustOneCacophony 的 E-AC-3 JOC 命令行入口。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np

import adm_assemble
from adm_validate import validate
from metadata import DirectPayloadIndex, PayloadIndex, write_summary
import oamd_tracks
from renderer import JocRenderer
from native_renderer import NativeBackendUnavailable, NativeJocRenderer
from speaker_backend import create_speaker_renderer
from speaker_layouts import (SPEAKER_LAYOUT_CHOICES, get_speaker_layout,
                             speaker_layout_display_name)
from speaker_wav import SpeakerPcmSpool, write_speaker_wav
from variant_error import UnsupportedVariantError, write_variant_report


RATE = 48000
FRAME_SAMPLES = 1536
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


def resolve_output(source, requested=None, speaker_layout=None):
    """解析成品路径；未指定时使用项目内的 ``output`` 目录。"""
    source = Path(source)
    if requested is not None:
        target = Path(requested)
    elif speaker_layout is not None:
        target = DEFAULT_OUTPUT_DIR / f"{source.stem}.{speaker_layout}.wav"
    else:
        target = DEFAULT_OUTPUT_DIR / (source.stem + ".adm.wav")
    return target.expanduser().resolve()


def executable(value, name):
    path = shutil.which(value) if value else None
    if path is None and value and Path(value).is_file():
        path = str(Path(value).resolve())
    if path is None:
        raise FileNotFoundError(f"找不到 {name}: {value!r}")
    return path


def run(command, label):
    print(f"[{label}]", flush=True)
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        tail = result.stderr[-4000:]
        raise RuntimeError(f"{label} 失败（exit {result.returncode}）\n{tail}")


def timed_call(timings, name, function, *args, **kwargs):
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[name] = time.perf_counter() - started


def extract_eac3(ffmpeg, source, target):
    if source.suffix.lower() in (".eac3", ".ec3"):
        return source
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", "0:a:0", "-vn", "-c:a", "copy", "-f", "eac3", str(target)],
        "FFmpeg 提取 E-AC-3")
    return target


def decode_core(ffmpeg, eac3, target, duration_sec=None):
    # 5.1(side) 的 f32le 顺序为 FL FR FC LFE SL SR；JOC 使用其中 0,1,2,4,5。
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(eac3),
               "-map", "0:a:0", "-vn"]
    if duration_sec is not None:
        command.extend(["-t", f"{duration_sec:.9f}"])
    command.extend(["-ac", "6", "-ar", str(RATE),
                    "-c:a", "pcm_f32le", "-f", "f32le", str(target)])
    run(command, "FFmpeg 解码核心 5.1 PCM")
    return target


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_speaker_output_format(requested_format, clip_action, peak, clipped_values,
                                 *, input_func=input, interactive=None):
    """Resolve int24 clipping interactively or through an explicit policy."""
    if requested_format != "int24" or clipped_values == 0:
        return requested_format
    print(
        f"[clip] int24 将发生削波：peak={peak:.9g}，超出 [-1,1] 的样本值={clipped_values}",
        file=sys.stderr, flush=True)
    action = clip_action
    if action == "ask":
        if interactive is None:
            interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
        if not interactive:
            raise RuntimeError(
                "检测到 int24 削波，但当前不是交互终端；请使用 "
                "--clip-action continue、--clip-action float32 或 --clip-action abort")
        while True:
            answer = input_func(
                "继续写 int24 并截断 [i] / 改为 float32 [f，默认] / 取消 [a]："
            ).strip().lower()
            if answer in ("", "f", "float", "float32"):
                action = "float32"
                break
            if answer in ("i", "int", "int24", "c", "continue"):
                action = "continue"
                break
            if answer in ("a", "abort", "q", "quit", "n", "no"):
                action = "abort"
                break
            print("请输入 i、f 或 a。", file=sys.stderr, flush=True)
    if action == "continue":
        print("[clip] 将继续写 int24，超范围值会截断到 [-1,1]。", flush=True)
        return "int24"
    if action == "float32":
        print("[clip] 已切换为 float32 WAV，不执行截断。", flush=True)
        return "float32"
    if action == "abort":
        raise RuntimeError("用户因 int24 削波取消输出")
    raise ValueError(f"未知 clip action: {action}")


def resolve_metadata(args, eac3, temp_dir):
    if args.metadata_dir:
        directory = Path(args.metadata_dir).resolve()
        return PayloadIndex(directory), "sidecar", directory
    if args.metadata_backend == "sidecar":
        raise ValueError("metadata-backend=sidecar 时必须提供 --metadata-dir")
    cache_dir = (args.metadata_cache.expanduser().resolve()
                 if args.metadata_cache else None)
    max_frames = (math.ceil(args.duration * RATE / FRAME_SAMPLES)
                  if args.duration is not None else None)
    index = DirectPayloadIndex.from_eac3(
        eac3, max_frames=max_frames, cache_dir=cache_dir)
    return index, "python-emdf-memory", cache_dir


def variant_call(output, source, function, *args, **kwargs):
    """执行一个阶段；遇到未知变体时在目标文件旁写结构化报告。"""
    try:
        return function(*args, **kwargs)
    except UnsupportedVariantError as exc:
        report_path = Path(str(output) + ".variant-error.json")
        write_variant_report(report_path, exc, input_path=source, output_path=output)
        print(f"[VARIANT] {exc}", file=sys.stderr, flush=True)
        print(f"[VARIANT] 维修报告: {report_path}", file=sys.stderr, flush=True)
        raise


def create_renderer(backend, gain, native_library=None, native_threads=None):
    """选择整帧 DSP 后端；auto 优先使用 lib 中当前平台的原生构建。"""
    if backend in ("auto", "native"):
        try:
            decoder = NativeJocRenderer(
                output_scale=gain, library_path=native_library, threads=native_threads)
            info = {
                "name": "native",
                "library": str(decoder.library_path),
                "build": decoder.build_info,
                "threads": decoder.threads,
            }
            print(f"[backend] native: {info['build']}  threads={info['threads']} "
                  f"({info['library']})", flush=True)
            return decoder, info
        except (NativeBackendUnavailable, OSError) as exc:
            print(f"[backend] native unavailable, falling back to Python: {exc}", flush=True)
    decoder = JocRenderer(output_scale=gain)
    info = {"name": "python", "library": None, "build": None, "threads": None}
    print("[backend] python/numpy", flush=True)
    return decoder, info


def render(index, bed_path, frame_count, raw_path, gain, progress_every,
           backend="auto", native_library=None, native_threads=None, frame_sink=None,
           speaker_renderer=None, speaker_sink=None, speaker_metadata_offset=1473):
    values = np.memmap(bed_path, dtype=np.float32, mode="r")
    frame_width = FRAME_SAMPLES * 6
    if values.size % frame_width:
        raise ValueError(f"FFmpeg PCM 长度不是 1536×6 的整数倍: {values.size}")
    bed = values.reshape(-1, FRAME_SAMPLES, 6)
    if len(bed) < frame_count:
        raise ValueError(f"PCM 只有 {len(bed)} 帧，元数据需要 {frame_count} 帧")
    output = (np.memmap(raw_path, dtype=np.float32, mode="w+",
                        shape=(frame_count, FRAME_SAMPLES, 16))
              if raw_path is not None else None)
    decoder, backend_info = create_renderer(backend, gain, native_library, native_threads)
    started = time.perf_counter()
    dsp_seconds = 0.0
    adm_stream_seconds = 0.0
    raw_write_seconds = 0.0
    speaker_render_seconds = 0.0
    speaker_write_seconds = 0.0
    try:
        for frame_number, row in enumerate(index.rows[:frame_count]):
            bed6 = np.asarray(bed[frame_number], dtype=np.float32)
            subs = index.subpayloads(row)
            stage = time.perf_counter()
            pcm16, _ = decoder.render_subpayloads(
                subs, bed6[:, [0, 1, 2, 4, 5]].T, bed6[:, 3])
            dsp_seconds += time.perf_counter() - stage
            if output is not None:
                stage = time.perf_counter()
                output[frame_number] = pcm16.T
                raw_write_seconds += time.perf_counter() - stage
            if frame_sink is not None:
                stage = time.perf_counter()
                frame_sink.write_frame(pcm16)
                adm_stream_seconds += time.perf_counter() - stage
            if speaker_renderer is not None:
                stage = time.perf_counter()
                speaker_pcm = speaker_renderer.render_frame(
                    pcm16.T, subs.get(11), speaker_metadata_offset)
                speaker_render_seconds += time.perf_counter() - stage
                stage = time.perf_counter()
                speaker_sink.write_frame(speaker_pcm)
                speaker_write_seconds += time.perf_counter() - stage
            done = frame_number + 1
            if done % progress_every == 0 or done == frame_count:
                elapsed = time.perf_counter() - started
                speed = done / max(elapsed, 1e-9)
                eta = (frame_count - done) / max(speed, 1e-9)
                print(f"[JOC:{backend_info['name']}] {done}/{frame_count}  "
                      f"{speed:.1f} frame/s  ETA {eta:.1f}s", flush=True)
        if output is not None:
            output.flush()
        elapsed = time.perf_counter() - started
    finally:
        close = getattr(decoder, "close", None)
        if close is not None:
            close()
        close = getattr(speaker_renderer, "close", None)
        if close is not None:
            close()
    breakdown = {
        "pipeline_wall_seconds": elapsed,
        "dsp_and_joc_parse_seconds": dsp_seconds,
        "adm_stream_write_seconds": adm_stream_seconds,
        "raw_float_write_seconds": raw_write_seconds,
        "speaker_render_seconds": speaker_render_seconds,
        "speaker_spool_write_seconds": speaker_write_seconds,
    }
    return dsp_seconds, backend_info, breakdown


def build_parser():
    parser = argparse.ArgumentParser(
        description="JustOneCacophony (JOC)：E-AC-3 JOC → 25ch ADM BWF 或扬声器 WAV")
    parser.add_argument("input", type=Path, help="输入 .m4a/.eac3/.ec3")
    parser.add_argument("-o", "--output", type=Path, help="输出文件；默认按模式和布局命名")
    parser.add_argument("--speaker-output", type=Path,
                        help="扬声器 WAV 路径；仅与 --speaker-layout 一起使用")
    parser.add_argument("--speaker-layout", choices=SPEAKER_LAYOUT_CHOICES,
                        help="直接扬声器渲染布局，例如 2.0、5.1、7.1.2")
    parser.add_argument("--speaker-format", choices=("float32", "int24"), default="float32",
                        help="扬声器 WAV 格式，默认 float32")
    parser.add_argument("--clip-action", choices=("ask", "continue", "float32", "abort"),
                        default="ask",
                        help="int24 削波处理：交互询问、继续截断、改 float32 或中止")
    parser.add_argument("--speaker-metadata-offset", type=int, default=1473,
                        help="扬声器渲染 metadata 相对帧偏移，默认 1473 samples")
    parser.add_argument("--gain-db", type=float, default=0.0,
                        help="成品增益 dB，默认 0（float32 系数 1.0）")
    parser.add_argument("--duration", type=float, help="只处理开头指定秒数")
    parser.add_argument("--object-delay-samples", type=int, default=1473,
                        help="可选的对象 PCM/OAMD 时间补偿，默认 1473 samples")
    parser.add_argument("--trajectory-mode", choices=("compact", "dense64"), default="compact",
                        help="对象轨迹表示；compact 用长线性插值压缩 AXML，dense64 保留逐 64-sample 块")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--backend", choices=("auto", "native", "python"), default="auto",
                        help="DSP 后端；auto 优先 C++，不可用时回退 Python")
    parser.add_argument("--native-library", type=Path,
                        help="显式指定原生库；默认从单层 lib 目录选择当前平台文件")
    parser.add_argument("--native-threads", type=int,
                        help="原生 DSP 总线程数；默认在 4 核以上使用 2，可用环境变量 EAC3JOC_NATIVE_THREADS 覆盖")
    metadata_source = parser.add_mutually_exclusive_group()
    metadata_source.add_argument("--metadata-dir", type=Path,
                                 help="含 frames.csv 和 emdf/ 或 payloads/ 的元数据 sidecar")
    metadata_source.add_argument("--metadata-cache", type=Path,
                                 help="把直接 EMDF 扫描或兼容桥结果持久保存到此目录")
    parser.add_argument("--metadata-backend", choices=("auto", "emdf", "sidecar"),
                        default="auto", help="直接扫描连续 EMDF，或读取现有 sidecar")
    parser.add_argument("--print-metadata", choices=("none", "summary", "frames"), default="none",
                        help="诊断元数据输出；默认 none，避免转换前重复完整解析")
    parser.add_argument("--metadata-json", type=Path, help="元数据汇总 JSON 路径")
    parser.add_argument("--metadata-only", action="store_true", help="解析/打印元数据后退出")
    parser.add_argument("--keep-raw", action="store_true", help="额外保留 16ch f32le 对象中间文件")
    parser.add_argument("--skip-sha256", action="store_true",
                        help="跳过最终文件 SHA-256 全量复扫以缩短大文件处理时间")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def main(argv=None):
    # Windows 控制台的活动代码页未必能表示日文文件名；保留信息并避免
    # UnicodeEncodeError 中断长任务。支持 UTF-8 的终端仍会原样显示。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    speaker_mode = args.speaker_layout is not None
    if args.speaker_output is not None and not speaker_mode:
        raise ValueError("--speaker-output 必须与 --speaker-layout 一起使用")
    if args.output is not None and args.speaker_output is not None:
        raise ValueError("-o/--output 与 --speaker-output 不能同时使用")
    if args.speaker_metadata_offset < 0:
        raise ValueError("speaker-metadata-offset 不能为负数")
    requested_output = (args.speaker_output if args.speaker_output is not None
                        else args.output)
    output = resolve_output(
        source, requested_output, args.speaker_layout if speaker_mode else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration 必须大于 0")
    if args.object_delay_samples < 0:
        raise ValueError("object-delay-samples 不能为负数")
    gain = np.float32(10.0 ** (args.gain_db / 20.0))
    if not np.isfinite(gain):
        raise ValueError("gain-db 超出 float32 范围")
    ffmpeg = executable(args.ffmpeg, "FFmpeg")

    total_started = time.perf_counter()
    timings = {}
    with tempfile.TemporaryDirectory(prefix="eac3joc-", dir=output.parent) as temporary:
        temp_dir = Path(temporary)
        eac3 = timed_call(timings, "extract_eac3", extract_eac3,
                          ffmpeg, source, temp_dir / "input.eac3")
        index, metadata_backend, metadata_cache_dir = timed_call(
            timings, "resolve_metadata", variant_call,
            output, source, resolve_metadata, args, eac3, temp_dir)
        timings["load_metadata_index"] = 0.0
        frame_count = len(index)
        if args.duration is not None:
            frame_count = min(frame_count, math.ceil(args.duration * RATE / FRAME_SAMPLES))
        duration_sec = frame_count * FRAME_SAMPLES / RATE
        need_metadata_summary = (
            args.metadata_only or args.metadata_json is not None or args.print_metadata != "none")
        if need_metadata_summary:
            metadata_json = (args.metadata_json or Path(str(output) + ".metadata.json")).resolve()
            summary = timed_call(
                timings, "metadata_summary", variant_call,
                output, source, write_summary, index, metadata_json, limit=frame_count,
                print_frames=args.print_metadata == "frames")
            if args.print_metadata == "summary":
                print("[metadata] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
            print(f"[metadata] backend={metadata_backend}  frames={frame_count}  -> {metadata_json}")
        else:
            metadata_json = None
            timings["metadata_summary"] = 0.0
            print(f"[metadata] backend={metadata_backend}  frames={frame_count}  summary=skipped")
        if args.metadata_only:
            return 0

        bed_path = timed_call(
            timings, "decode_core", decode_core,
            ffmpeg, eac3, temp_dir / "core51_f32le.raw", duration_sec)
        raw_path = (output.with_name(output.name + ".objects16.f32le")
                    if args.keep_raw else None)
        master = None
        speaker_backend_info = None
        speaker_wav_info = None
        speaker_clip_info = None
        speaker_actual_format = None
        if speaker_mode:
            layout = get_speaker_layout(args.speaker_layout)
            speaker_name = speaker_layout_display_name(layout)
            speaker_decoder, speaker_backend_info = create_speaker_renderer(
                layout, backend=args.backend, native_library=args.native_library)
            fallback = speaker_backend_info.get("fallback_reason")
            if fallback:
                print(f"[speaker] native unavailable, falling back to Python: {fallback}",
                      flush=True)
            print(f"[speaker] layout={speaker_name} backend={speaker_backend_info['name']} "
                  f"channels={layout.channel_count}", flush=True)
            spool = SpeakerPcmSpool(
                temp_dir / "speaker_interleaved_f32.raw",
                frame_count * FRAME_SAMPLES, layout.channel_count)
            try:
                render_seconds, renderer_backend, render_breakdown = timed_call(
                    timings, "render_and_stream", variant_call,
                    output, source, render, index, bed_path, frame_count, raw_path, gain,
                    max(1, args.progress_every), args.backend, args.native_library,
                    args.native_threads, None, speaker_decoder, spool,
                    args.speaker_metadata_offset)
                spool.finalize()
                speaker_actual_format = choose_speaker_output_format(
                    args.speaker_format, args.clip_action, spool.peak,
                    spool.clipped_values)
                speaker_wav_info = timed_call(
                    timings, "write_speaker_wav", write_speaker_wav,
                    output, spool.values, speaker_actual_format, rate=RATE)
                speaker_clip_info = {
                    "peak": spool.peak,
                    "over_unity_values": spool.clipped_values,
                    "requested_format": args.speaker_format,
                    "actual_format": speaker_actual_format,
                    "clip_action": args.clip_action,
                }
            finally:
                spool.close()
            timings["build_adm_tracks"] = 0.0
            timings["finalize_adm"] = 0.0
            timings["validate_adm"] = 0.0
            info = (f"speaker layout={speaker_name}, format={speaker_actual_format}, "
                    f"peak={speaker_clip_info['peak']:.9g}")
        else:
            master = adm_assemble.StreamingMaster(output, duration_sec, rate=RATE)
            try:
                render_seconds, renderer_backend, render_breakdown = timed_call(
                    timings, "render_and_stream", variant_call,
                    output, source, render, index, bed_path, frame_count, raw_path, gain,
                    max(1, args.progress_every), args.backend, args.native_library,
                    args.native_threads, master)
                tracks = timed_call(
                    timings, "build_adm_tracks", variant_call,
                    output, source, oamd_tracks.build_adm_tracks,
                    index, index.rows[:frame_count], rate=RATE, frame_samples=FRAME_SAMPLES,
                    object_delay_samples=args.object_delay_samples,
                    trajectory_mode=args.trajectory_mode)
                timed_call(timings, "finalize_adm", master.finalize, tracks)
            except Exception:
                master.abort()
                raise
            errors, info = timed_call(timings, "validate_adm", validate, str(output))
            if errors:
                raise RuntimeError("ADM 校验失败: " + "; ".join(errors))
        # Windows 不允许删除仍被 NumPy memmap 持有的临时 core/raw；显式回收闭包。
        import gc
        gc.collect()

    if args.skip_sha256:
        output_sha = None
        timings["sha256"] = 0.0
    else:
        output_sha = timed_call(timings, "sha256", sha256, output)
    total_seconds = time.perf_counter() - total_started
    report = {
        "input": str(source),
        "output": str(output),
        "mode": "speaker" if speaker_mode else "adm",
        "metadata": str(metadata_json) if metadata_json is not None else None,
        "metadata_backend": metadata_backend,
        "metadata_cache": str(metadata_cache_dir) if metadata_cache_dir is not None else None,
        "frames": frame_count,
        "duration_sec": duration_sec,
        "gain_db": args.gain_db,
        "gain_float32": float(gain),
        "object_delay_samples": None if speaker_mode else args.object_delay_samples,
        "trajectory_mode": None if speaker_mode else args.trajectory_mode,
        "render_seconds": render_seconds,
        "render_breakdown": render_breakdown,
        "renderer_backend": renderer_backend,
        "speaker_renderer_backend": speaker_backend_info,
        "speaker_layout": args.speaker_layout if speaker_mode else None,
        "speaker_metadata_offset": args.speaker_metadata_offset if speaker_mode else None,
        "speaker_clip": speaker_clip_info,
        "speaker_wav": speaker_wav_info,
        "streaming_adm": not speaker_mode,
        "kept_raw": str(raw_path) if raw_path is not None else None,
        "timings": timings,
        "total_seconds": total_seconds,
        "adm_validation": None if speaker_mode else info,
        "adm_metadata": getattr(master, "metadata_info", None) if master is not None else None,
        "sha256": output_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    report_path = Path(str(output) + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PASS] {output}")
    if report["sha256"] is None:
        print(f"[PASS] {info}; SHA-256 skipped")
    else:
        print(f"[PASS] {info}; SHA-256={report['sha256']}")
    if speaker_mode:
        print(f"[time] JOC-DSP={render_seconds:.2f}s ({renderer_backend['name']}) "
              f"speaker={render_breakdown['speaker_render_seconds']:.2f}s "
              f"pipeline={render_breakdown['pipeline_wall_seconds']:.2f}s "
              f"total={report['total_seconds']:.2f}s")
    else:
        print(f"[time] DSP={render_seconds:.2f}s ({renderer_backend['name']}) "
              f"render+ADM-stream={render_breakdown['pipeline_wall_seconds']:.2f}s "
              f"total={report['total_seconds']:.2f}s")
    print(f"[report] {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
