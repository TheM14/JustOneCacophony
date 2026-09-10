"""JustOneCacophony 的 E-AC-3 JOC 命令行入口。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
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
import adm_atmos
from adm_validate import validate
from metadata import DirectPayloadIndex, PayloadIndex, write_summary
import oamd_tracks
from renderer import JocRenderer
from native_renderer import NativeBackendUnavailable, NativeJocRenderer
from binaural_renderer import (
    DEFAULT_SOFA_HRTF,
    SofaBinauralRenderer,
    resolve_compiled_hrtf_cache,
    resolve_sofa_hrtf,
)
from rosella_binaural_renderer import (
    DEFAULT_PERSONALIZED_HEADPHONE,
    ROSSELLA_BLOCK_SAMPLES,
    ROSSELLA_LATENCY_SAMPLES,
    RosellaBinauralRenderer,
    resolve_personalized_headphone,
)
from sofa_hrtf_field import DEFAULT_HRTF_CACHE_DIR
from speaker_backend import create_speaker_renderer
from speaker_layouts import (SPEAKER_LAYOUT_CHOICES, get_speaker_layout,
                             speaker_layout_display_name)
from speaker_wav import BinauralPcmSpool, SpeakerPcmSpool, write_pcm_wav
from variant_error import UnsupportedVariantError, write_variant_report


RATE = 48000
FRAME_SAMPLES = 1536
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
EAC3_DRC_SCALE_MAX = 6.0
EAC3_TARGET_LEVEL_RANGE = (-31, 0)
EAC3_DECODER_OPTION_RE = re.compile(r"(?m)^\s*-([A-Za-z0-9_]+)\s+<")


def resolve_output(source, requested=None, speaker_layout=None, *, binaural=False):
    """解析成品路径；未指定时使用项目内的 ``output`` 目录。"""
    source = Path(source)
    if requested is not None:
        target = Path(requested)
    elif speaker_layout is not None:
        target = DEFAULT_OUTPUT_DIR / f"{source.stem}.{speaker_layout}.wav"
    elif binaural:
        target = DEFAULT_OUTPUT_DIR / f"{source.stem}.binaural.wav"
    else:
        target = DEFAULT_OUTPUT_DIR / (source.stem + ".adm.wav")
    return target.expanduser().resolve()


def _find_default_compiled_hrtf_cache():
    """在默认 cache 目录寻找唯一的 .jochrtf；无文件返回 None，多个则报错。"""
    directory = DEFAULT_HRTF_CACHE_DIR
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.jochrtf"))
    if not candidates:
        return None
    if len(candidates) > 1:
        listing = ", ".join(path.name for path in candidates[:8])
        raise ValueError(
            f"{directory} 下有多个 .jochrtf 缓存（{listing}…），无法自动选择；"
            "请用 --compiled-hrtf-cache PATH 或 --sofa-hrtf PATH 显式指定")
    return candidates[0]


def resolve_binaural_hrtf_input(args, *, required):
    """解析 binaural 的 HRTF 输入。

    无显式输入时按顺序回退：默认 HRTF/binaural.sofa → 默认 cache 目录下唯一的
    .jochrtf → 默认 HRTF/binaural.personalized_headphone → 报错。
    只校验路径，不做编译。
    """
    sofa = args.sofa_hrtf
    compiled = args.compiled_hrtf_cache
    private = args.personalized_headphone
    cache_policy = args.hrtf_cache_policy
    cache_dir = args.hrtf_cache_dir
    radius = args.hrtf_radius_m

    if compiled is not None and cache_policy is not None:
        raise ValueError("显式 .jochrtf 输入不能再指定 --hrtf-cache-policy")
    if compiled is not None and radius != 1.0:
        raise ValueError("显式 .jochrtf 输入不能再选择 SOFA radius shell")
    if private is not None and (cache_policy is not None or cache_dir is not None
                                or radius != 1.0):
        raise ValueError(
            "Rosella 模型输入不能使用 "
            "--hrtf-cache-policy/--hrtf-cache-dir/--hrtf-radius-m")

    if required and sofa is None and compiled is None and private is None:
        if DEFAULT_SOFA_HRTF.is_file():
            sofa = DEFAULT_SOFA_HRTF
        else:
            compiled = _find_default_compiled_hrtf_cache()
            if compiled is None and DEFAULT_PERSONALIZED_HEADPHONE.is_file():
                private = DEFAULT_PERSONALIZED_HEADPHONE

    if sofa is None and compiled is None and private is None:
        if cache_policy is not None or cache_dir is not None or radius != 1.0:
            raise ValueError("HRTF cache/radius 选项需要 --sofa-hrtf")
        if required:
            raise ValueError(
                "--binaural 未找到 HRTF 输入：默认 "
                f"{DEFAULT_SOFA_HRTF}、{DEFAULT_PERSONALIZED_HEADPHONE} 与 "
                f"{DEFAULT_HRTF_CACHE_DIR} 下的 .jochrtf 缓存都不存在；请用 "
                "--sofa-hrtf PATH、--compiled-hrtf-cache PATH 或 "
                "--personalized-headphone PATH 指定")
        return None

    if sofa is None and (cache_policy is not None or cache_dir is not None
                         or radius != 1.0):
        raise ValueError("HRTF cache/radius 选项需要 --sofa-hrtf")
    effective_policy = "memory" if cache_policy is None else cache_policy
    if cache_dir is not None and (sofa is None or effective_policy != "disk"):
        raise ValueError("--hrtf-cache-dir 仅与 SOFA 的 disk cache policy 一起使用")
    if sofa is not None:
        return {
            "kind": "sofa",
            "path": resolve_sofa_hrtf(sofa),
            "cache_policy": effective_policy,
            "cache_dir": (DEFAULT_HRTF_CACHE_DIR if cache_dir is None else
                          cache_dir.expanduser().resolve()),
        }
    if compiled is not None:
        return {
            "kind": "compiled_cache",
            "path": resolve_compiled_hrtf_cache(compiled),
            "cache_policy": None,
            "cache_dir": None,
        }
    if private is not None:
        return {
            "kind": "rosella",
            "path": resolve_personalized_headphone(private),
            "cache_policy": None,
            "cache_dir": None,
        }
    return None


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


def probe_eac3_decoder_options(ffmpeg):
    """读取 ``ffmpeg -h decoder=eac3`` 暴露的 AVOption 名。"""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-h", "decoder=eac3"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    options = frozenset(EAC3_DECODER_OPTION_RE.findall(result.stdout or ""))
    # decoder 名不存在时 ffmpeg 依然返回 0，因此以“解析不到任何选项”为失败。
    if not options:
        raise RuntimeError(
            "无法读取 FFmpeg 的 eac3 解码器选项（ffmpeg -h decoder=eac3）；"
            "需要带 E-AC-3 解码器的构建")
    return options


def ffmpeg_version(ffmpeg):
    """FFmpeg 版本字符串；探测失败返回空串，不影响渲染。"""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = (result.stdout or "").splitlines()
    line = lines[0].strip() if lines else ""
    prefix = "ffmpeg version "
    return line[len(prefix):].strip() if line.startswith(prefix) else line


def eac3_decode_options(drc_scale, target_level, available):
    """构造 ``-i`` 之前的 E-AC-3 解码选项，返回 ``(argv, report 片段)``。"""
    if "drc_scale" not in available:
        raise RuntimeError(
            "FFmpeg 的 eac3 解码器缺少 -drc_scale，无法关闭码流 DRC")
    # -drc_scale 始终显式下发：0（全动态范围）不是 ffmpeg 的默认值。
    argv = ["-drc_scale", format(float(drc_scale), ".10g")]
    if target_level:
        if "target_level" not in available:
            raise RuntimeError(
                "FFmpeg 的 eac3 解码器不支持 -target_level；请升级 FFmpeg "
                "或去掉 --eac3-target-level")
        argv += ["-target_level", str(int(target_level))]
    applied = {"drc_scale": float(drc_scale), "target_level": int(target_level)}
    return argv, applied


def extract_eac3(ffmpeg, source, target):
    if source.suffix.lower() in (".eac3", ".ec3"):
        return source
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map", "0:a:0", "-vn", "-c:a", "copy", "-f", "eac3", str(target)],
        "FFmpeg 提取 E-AC-3")
    return target


def decode_core(ffmpeg, eac3, target, duration_sec=None, *, options=()):
    # 5.1(side) 的 f32le 顺序为 FL FR FC LFE SL SR；JOC 使用其中 0,1,2,4,5。
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *options,
               "-i", str(eac3), "-map", "0:a:0", "-vn"]
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


def choose_pcm_output_format(requested_format, clip_action, peak, clipped_values,
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


# Backward-compatible public name used by existing tests and callers.
choose_speaker_output_format = choose_pcm_output_format


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
           speaker_renderer=None, speaker_sink=None, speaker_metadata_offset=1473,
           binaural_renderer=None, binaural_sink=None, binaural_metadata_offset=1473,
           raw_scale=1.0):
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
    binaural_render_seconds = 0.0
    binaural_write_seconds = 0.0
    elapsed = 0.0
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
                output[frame_number] = np.multiply(
                    pcm16.T, np.float32(raw_scale), dtype=np.float32)
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
            if binaural_renderer is not None:
                payload = subs.get(11)
                outer_offset = (
                    index.subpayload_sample_offset(row, 11)
                    if payload is not None and hasattr(index, "subpayload_sample_offset")
                    else 0
                )
                stage = time.perf_counter()
                binaural_pcm = binaural_renderer.render_frame(
                    pcm16.T, payload, binaural_metadata_offset,
                    outer_sample_offset=outer_offset)
                binaural_render_seconds += time.perf_counter() - stage
                if len(binaural_pcm):
                    stage = time.perf_counter()
                    binaural_sink.write_frame(binaural_pcm)
                    binaural_write_seconds += time.perf_counter() - stage
            done = frame_number + 1
            if done % progress_every == 0 or done == frame_count:
                elapsed = time.perf_counter() - started
                speed = done / max(elapsed, 1e-9)
                eta = (frame_count - done) / max(speed, 1e-9)
                print(f"[JOC:{backend_info['name']}] {done}/{frame_count}  "
                      f"{speed:.1f} frame/s  ETA {eta:.1f}s", flush=True)
        if binaural_renderer is not None:
            stage = time.perf_counter()
            binaural_tail = binaural_renderer.finish()
            binaural_render_seconds += time.perf_counter() - stage
            if len(binaural_tail):
                stage = time.perf_counter()
                binaural_sink.write_frame(binaural_tail)
                binaural_write_seconds += time.perf_counter() - stage
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
        close = getattr(binaural_renderer, "close", None)
        if close is not None:
            close()
    breakdown = {
        "pipeline_wall_seconds": elapsed,
        "dsp_and_joc_parse_seconds": dsp_seconds,
        "adm_stream_write_seconds": adm_stream_seconds,
        "raw_float_write_seconds": raw_write_seconds,
        "speaker_render_seconds": speaker_render_seconds,
        "speaker_spool_write_seconds": speaker_write_seconds,
        "binaural_render_seconds": binaural_render_seconds,
        "binaural_spool_write_seconds": binaural_write_seconds,
    }
    return dsp_seconds, backend_info, breakdown


def build_parser():
    parser = argparse.ArgumentParser(
        description=("JustOneCacophony (JOC)：E-AC-3 JOC → 25ch ADM BWF、"
                     "扬声器 WAV 或公开 SOFA 双耳 WAV"))
    parser.add_argument("input", type=Path, help="输入 .m4a/.eac3/.ec3")
    parser.add_argument("-o", "--output", type=Path, help="输出文件；默认按模式和布局命名")
    parser.add_argument("--speaker-output", type=Path,
                        help="扬声器 WAV 路径；仅与 --speaker-layout 一起使用")
    parser.add_argument("--binaural-output", type=Path,
                        help="双耳 WAV 路径；仅与 --binaural 一起使用")
    direct_mode = parser.add_mutually_exclusive_group()
    direct_mode.add_argument("--speaker-layout", choices=SPEAKER_LAYOUT_CHOICES,
                             help="直接扬声器渲染布局，例如 2.0、5.1、7.1.2")
    direct_mode.add_argument("--binaural", action="store_true",
                             help="直接 SOFA 双耳渲染；不生成临时 ADM BWF")
    parser.add_argument("--speaker-format", choices=("float32", "int24"), default="float32",
                        help="扬声器 WAV 格式，默认 float32")
    parser.add_argument("--binaural-format", choices=("float32", "int24"), default="float32",
                        help="双耳 WAV 格式，默认 float32")
    parser.add_argument("--clip-action", choices=("ask", "continue", "float32", "abort"),
                        default="ask",
                        help="int24 削波处理：交互询问、继续截断、改 float32 或中止")
    parser.add_argument("--speaker-metadata-offset", type=int, default=1473,
                        help="扬声器渲染 metadata 相对帧偏移，默认 1473 samples")
    parser.add_argument("--binaural-mode", choices=("off", "near", "mid", "far"),
                        default="mid",
                        help="双耳渲染模式，默认 mid（人为指定的渲染提示，非码流 "
                             "原始元数据）；直接双耳渲染与 ADM BWF 的 DBMD 提示共用。"
                             "off 仅用于 ADM BWF：关闭 DBMD 双耳提示（编码 0）")
    hrtf_input = parser.add_mutually_exclusive_group()
    hrtf_input.add_argument(
        "--sofa-hrtf", type=Path,
        help="SimpleFreeFieldHRIR SOFA；缺省时依次尝试 HRTF/binaural.sofa、"
             "output/hrtf-cache 下唯一的 .jochrtf、"
             "HRTF/binaural.personalized_headphone，均无则报错")
    hrtf_input.add_argument(
        "--compiled-hrtf-cache", type=Path,
        help="高级入口：显式读取 JOC .jochrtf compiled cache")
    hrtf_input.add_argument(
        "--personalized-headphone", type=Path, nargs="?",
        const=DEFAULT_PERSONALIZED_HEADPHONE,
        help="Rosella .personalized_headphone 模型；不带路径时默认 "
             "HRTF/binaural.personalized_headphone")
    parser.add_argument(
        "--hrtf-cache-policy", choices=("none", "memory", "disk"), default=None,
        help="SOFA 编译缓存；默认 memory，disk 写入可删除的 .jochrtf")
    parser.add_argument(
        "--hrtf-cache-dir", type=Path,
        help="disk cache 目录；默认 output/hrtf-cache")
    parser.add_argument(
        "--hrtf-radius-m", type=float, default=1.0,
        help="选择最近的 SOFA measurement-radius shell，默认 1.0 m")
    parser.add_argument("--binaural-tail-seconds", type=float, default=5.0,
                        help="双耳 room/filterbank flush 上限，默认 5 秒")
    parser.add_argument("--binaural-tail-threshold", type=float, default=1.0e-8,
                        help="双耳尾声裁切阈值，默认 1e-8；主体至少保留原时长")
    parser.add_argument("--binaural-chunk-frames", type=int, default=64,
                        help="双耳内部批处理 E-AC-3 帧数，默认 64")
    parser.add_argument("--gain-db", type=float, default=0.0,
                        help="成品增益 dB，默认 0；双耳路径以 float64 应用")
    parser.add_argument("--duration", type=float, help="只处理开头指定秒数")
    parser.add_argument("--object-delay-samples", type=int, default=1473,
                        help="对象 PCM/OAMD 时间补偿；ADM 与双耳默认 1473 samples")
    parser.add_argument("--trajectory-mode", choices=("compact", "dense64"), default="compact",
                        help="ADM 对象轨迹表示；直接双耳路径不序列化 AXML")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--eac3-drc-scale", type=float, default=0.0,
                        help="E-AC-3 解码器 -drc_scale：0=关闭码流 dynrng（全动态范围），"
                             "1=码流作者意图，>1 非对称；默认 0")
    parser.add_argument("--eac3-target-level", type=int, default=0,
                        help="E-AC-3 解码器 -target_level：按码流 dialnorm 归一化电平，"
                             "增益约 target_level - dialnorm dB；0=不施加，默认 0")
    parser.add_argument("--backend", choices=("auto", "native", "python"), default="auto",
                        help="JOC/扬声器 DSP 后端；SOFA 双耳 DSP 当前使用 Python")
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
    binaural_mode = bool(args.binaural)
    binaural_render_mode = args.binaural_mode
    if binaural_mode and binaural_render_mode == "off":
        raise ValueError(
            "--binaural-mode off 仅用于 ADM BWF 输出（关闭 DBMD 双耳提示）；"
            "直接双耳渲染请使用 near/mid/far")
    if args.speaker_output is not None and not speaker_mode:
        raise ValueError("--speaker-output 必须与 --speaker-layout 一起使用")
    if args.binaural_output is not None and not binaural_mode:
        raise ValueError("--binaural-output 必须与 --binaural 一起使用")
    specific_outputs = [value for value in (args.speaker_output, args.binaural_output)
                        if value is not None]
    if args.output is not None and specific_outputs:
        raise ValueError("-o/--output 与 --speaker-output/--binaural-output 不能同时使用")
    if len(specific_outputs) > 1:
        raise ValueError("--speaker-output 与 --binaural-output 不能同时使用")
    if args.speaker_metadata_offset < 0:
        raise ValueError("speaker-metadata-offset 不能为负数")
    hrtf_options_used = any((
        args.sofa_hrtf is not None,
        args.compiled_hrtf_cache is not None,
        args.personalized_headphone is not None,
        args.hrtf_cache_policy is not None,
        args.hrtf_cache_dir is not None,
        args.hrtf_radius_m != 1.0,
    ))
    if hrtf_options_used and not binaural_mode:
        raise ValueError("SOFA/HRTF 选项仅与 --binaural 一起使用")
    if (not math.isfinite(args.binaural_tail_seconds)
            or args.binaural_tail_seconds < 0):
        raise ValueError("binaural-tail-seconds 必须是非负有限值")
    if (not math.isfinite(args.binaural_tail_threshold)
            or args.binaural_tail_threshold < 0):
        raise ValueError("binaural-tail-threshold 必须是非负有限值")
    if args.binaural_chunk_frames <= 0:
        raise ValueError("binaural-chunk-frames 必须大于 0")
    if not math.isfinite(args.hrtf_radius_m) or args.hrtf_radius_m <= 0.0:
        raise ValueError("hrtf-radius-m 必须是正有限值")
    requested_output = (args.speaker_output if args.speaker_output is not None
                        else args.binaural_output if args.binaural_output is not None
                        else args.output)
    output = resolve_output(
        source, requested_output, args.speaker_layout if speaker_mode else None,
        binaural=binaural_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration 必须大于 0")
    if args.object_delay_samples < 0:
        raise ValueError("object-delay-samples 不能为负数")
    gain_float64 = 10.0 ** (args.gain_db / 20.0)
    gain = np.float32(gain_float64)
    if not math.isfinite(gain_float64) or not np.isfinite(gain):
        raise ValueError("gain-db 超出支持范围")
    if (not math.isfinite(args.eac3_drc_scale)
            or not 0.0 <= args.eac3_drc_scale <= EAC3_DRC_SCALE_MAX):
        raise ValueError(f"eac3-drc-scale 必须在 0..{EAC3_DRC_SCALE_MAX:g} 之间")
    if not (EAC3_TARGET_LEVEL_RANGE[0] <= args.eac3_target_level
            <= EAC3_TARGET_LEVEL_RANGE[1]):
        raise ValueError("eac3-target-level 必须在 -31..0 之间")
    binaural_hrtf_input = resolve_binaural_hrtf_input(
        args, required=binaural_mode and not args.metadata_only)
    ffmpeg = executable(args.ffmpeg, "FFmpeg")
    decode_options = ()
    decode_option_info = None
    if not args.metadata_only:
        available = probe_eac3_decoder_options(ffmpeg)
        decode_options, applied = eac3_decode_options(
            args.eac3_drc_scale, args.eac3_target_level, available)
        decode_option_info = {
            "version": ffmpeg_version(ffmpeg),
            "eac3_decode_options": applied,
        }
        print(f"[decode] ffmpeg {decode_option_info['version']}  "
              f"drc_scale={args.eac3_drc_scale:g}  "
              f"target_level={args.eac3_target_level}", flush=True)

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
            ffmpeg, eac3, temp_dir / "core51_f32le.raw", duration_sec,
            options=decode_options)
        raw_path = (output.with_name(output.name + ".objects16.f32le")
                    if args.keep_raw else None)
        master = None
        speaker_backend_info = None
        speaker_wav_info = None
        speaker_clip_info = None
        speaker_actual_format = None
        binaural_backend_info = None
        binaural_hrtf_report = None
        binaural_wav_info = None
        binaural_clip_info = None
        binaural_actual_format = None
        if speaker_mode:
            timings["create_binaural_renderer"] = 0.0
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
                speaker_actual_format = choose_pcm_output_format(
                    args.speaker_format, args.clip_action, spool.peak,
                    spool.clipped_values)
                speaker_wav_info = timed_call(
                    timings, "write_speaker_wav", write_pcm_wav,
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
        elif binaural_mode:
            hrtf_source = binaural_hrtf_input
            common_options = {
                "mode": binaural_render_mode,
                "object_delay_samples": args.object_delay_samples,
                "tail_seconds": args.binaural_tail_seconds,
                "output_gain": gain_float64,
                "chunk_frames": args.binaural_chunk_frames,
            }
            if hrtf_source["kind"] == "sofa":
                binaural_decoder = None
                if args.backend in ("auto", "native"):
                    try:
                        from sofa_native_backend import create_native_sofa_renderer
                        binaural_decoder = timed_call(
                            timings, "create_binaural_renderer",
                            create_native_sofa_renderer,
                            hrtf_source["path"],
                            cache_policy=hrtf_source["cache_policy"],
                            cache_dir=hrtf_source["cache_dir"],
                            shell_radius_m=args.hrtf_radius_m,
                            **common_options)
                    except (ImportError, OSError, RuntimeError, ValueError) as exc:
                        print(
                            f"[binaural] native SOFA backend unavailable "
                            f"({exc.__class__.__name__}: {exc}); "
                            f"falling back to Python", flush=True)
                        binaural_decoder = None
                if binaural_decoder is None:
                    binaural_decoder = timed_call(
                        timings, "create_binaural_renderer",
                        SofaBinauralRenderer.from_sofa,
                        hrtf_source["path"],
                        cache_policy=hrtf_source["cache_policy"],
                        cache_dir=hrtf_source["cache_dir"],
                        shell_radius_m=args.hrtf_radius_m,
                        **common_options)
            elif hrtf_source["kind"] == "rosella":
                binaural_decoder = timed_call(
                    timings, "create_binaural_renderer",
                    RosellaBinauralRenderer,
                    hrtf_source["path"],
                    mode=binaural_render_mode,
                    object_delay_samples=args.object_delay_samples,
                    tail_seconds=args.binaural_tail_seconds,
                    output_gain=gain_float64,
                    chunk_frames=args.binaural_chunk_frames,
                    backend=args.backend,
                    native_library=args.native_library)
            else:
                binaural_decoder = None
                if args.backend in ("auto", "native"):
                    try:
                        from sofa_native_backend import (
                            create_native_compiled_cache_renderer)
                        binaural_decoder = timed_call(
                            timings, "create_binaural_renderer",
                            create_native_compiled_cache_renderer,
                            hrtf_source["path"],
                            **common_options)
                    except (ImportError, OSError, RuntimeError, ValueError) as exc:
                        print(
                            f"[binaural] native SOFA backend unavailable "
                            f"({exc.__class__.__name__}: {exc}); "
                            f"falling back to Python", flush=True)
                        binaural_decoder = None
                if binaural_decoder is None:
                    binaural_decoder = timed_call(
                        timings, "create_binaural_renderer",
                        SofaBinauralRenderer.from_compiled_cache,
                        hrtf_source["path"],
                        **common_options)
            print(
                f"[binaural] mode={binaural_render_mode} "
                f"backend={binaural_decoder.dsp_backend} "
                f"precision=float64/complex128 "
                f"hrtf={hrtf_source['kind']}:{hrtf_source['path']}", flush=True)
            if hrtf_source["kind"] == "rosella":
                flush_samples = math.ceil(
                    (args.binaural_tail_seconds * RATE
                     + ROSSELLA_LATENCY_SAMPLES + ROSSELLA_BLOCK_SAMPLES)
                    / ROSSELLA_BLOCK_SAMPLES) * ROSSELLA_BLOCK_SAMPLES
                spool_capacity = frame_count * FRAME_SAMPLES + flush_samples
            else:
                spool_capacity = (
                    frame_count * FRAME_SAMPLES
                    + binaural_decoder.finish_capacity_samples)
            spool = BinauralPcmSpool(
                temp_dir / "binaural_interleaved_f64.raw",
                spool_capacity,
                tail_threshold=args.binaural_tail_threshold)
            try:
                render_seconds, renderer_backend, render_breakdown = timed_call(
                    timings, "render_and_stream", variant_call,
                    output, source, render, index, bed_path, frame_count, raw_path,
                    np.float32(1.0), max(1, args.progress_every),
                    backend=args.backend, native_library=args.native_library,
                    native_threads=args.native_threads,
                    binaural_renderer=binaural_decoder, binaural_sink=spool,
                    binaural_metadata_offset=args.object_delay_samples, raw_scale=gain)
                spool.finalize(minimum_samples=frame_count * FRAME_SAMPLES)
                binaural_actual_format = choose_pcm_output_format(
                    args.binaural_format, args.clip_action, spool.peak,
                    spool.clipped_values)
                binaural_wav_info = timed_call(
                    timings, "write_binaural_wav", write_pcm_wav,
                    output, spool.values, binaural_actual_format, rate=RATE)
                binaural_clip_info = {
                    "peak": spool.peak,
                    "over_unity_values": spool.clipped_values,
                    "requested_format": args.binaural_format,
                    "actual_format": binaural_actual_format,
                    "clip_action": args.clip_action,
                    "tail_threshold": args.binaural_tail_threshold,
                    "source_samples": frame_count * FRAME_SAMPLES,
                    "kept_samples": spool.sample_count,
                }
                binaural_backend_info = binaural_decoder.backend_info
                if hrtf_source["kind"] == "rosella":
                    binaural_hrtf_report = {
                        "input_kind": "rosella",
                        "input_path": str(binaural_decoder.model_path.resolve()),
                        "model_coefficient_sha256": (
                            binaural_decoder.model.coefficient_sha256),
                        "cache_policy": None,
                    }
                else:
                    binaural_hrtf_report = {
                        "input_kind": binaural_backend_info["hrtf_input_kind"],
                        "input_path": binaural_backend_info["hrtf_input_path"],
                        "source_sha256": (
                            binaural_backend_info["field"]["source_sha256"]),
                        "cache_policy": binaural_backend_info["cache_policy"],
                        "cache_key": (
                            binaural_backend_info["field"]["cache_key"]),
                        "format_version": (
                            binaural_backend_info["field"]["format_version"]),
                    }
            finally:
                spool.close()
            timings["build_adm_tracks"] = 0.0
            timings["finalize_adm"] = 0.0
            timings["validate_adm"] = 0.0
            info = (f"binaural mode={binaural_render_mode}, "
                    f"format={binaural_actual_format}, "
                    f"peak={binaural_clip_info['peak']:.9g}, "
                    f"samples={binaural_clip_info['kept_samples']}")
        else:
            timings["create_binaural_renderer"] = 0.0
            master = adm_assemble.StreamingMaster(
                output, duration_sec, rate=RATE,
                joc_binaural_mode=adm_atmos.JOC_BINAURAL_MODES[args.binaural_mode])
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
    mode_name = "speaker" if speaker_mode else "binaural" if binaural_mode else "adm"
    report = {
        "input": str(source),
        "output": str(output),
        "mode": mode_name,
        "metadata": str(metadata_json) if metadata_json is not None else None,
        "metadata_backend": metadata_backend,
        "metadata_cache": str(metadata_cache_dir) if metadata_cache_dir is not None else None,
        "frames": frame_count,
        "duration_sec": duration_sec,
        "gain_db": args.gain_db,
        "gain_float32": float(gain),
        "gain_float64": float(gain_float64),
        "object_delay_samples": (None if speaker_mode else args.object_delay_samples),
        "trajectory_mode": args.trajectory_mode if mode_name == "adm" else None,
        "binaural_mode_value": (
            adm_atmos.JOC_BINAURAL_MODES[args.binaural_mode]
            if mode_name == "adm" else None),
        "render_seconds": render_seconds,
        "render_breakdown": render_breakdown,
        "renderer_backend": renderer_backend,
        "speaker_renderer_backend": speaker_backend_info,
        "speaker_layout": args.speaker_layout if speaker_mode else None,
        "speaker_metadata_offset": args.speaker_metadata_offset if speaker_mode else None,
        "speaker_clip": speaker_clip_info,
        "speaker_wav": speaker_wav_info,
        "binaural_renderer_backend": binaural_backend_info,
        "binaural_mode": (
            args.binaural_mode if (binaural_mode or mode_name == "adm") else None),
        "binaural_hrtf": (
            binaural_hrtf_report if binaural_backend_info else None),
        "binaural_clip": binaural_clip_info,
        "binaural_wav": binaural_wav_info,
        "output_clip": speaker_clip_info if speaker_mode else binaural_clip_info,
        "output_wav": speaker_wav_info if speaker_mode else binaural_wav_info,
        "streaming_adm": mode_name == "adm",
        "kept_raw": str(raw_path) if raw_path is not None else None,
        "timings": timings,
        "total_seconds": total_seconds,
        "adm_validation": info if mode_name == "adm" else None,
        "adm_metadata": getattr(master, "metadata_info", None) if master is not None else None,
        "sha256": output_sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ffmpeg": decode_option_info,
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
    elif binaural_mode:
        print(f"[time] JOC-DSP={render_seconds:.2f}s ({renderer_backend['name']}) "
              f"binaural={render_breakdown['binaural_render_seconds']:.2f}s "
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
