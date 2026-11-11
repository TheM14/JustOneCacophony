"""Evolution id11/OAMD 帧序列 → ADM 15 对象关键帧。"""
import math

from adm_atmos import q_to_adm_xyz
from bed_layout import bed_object_positions
from oamd_bits import JocFieldState, frame_state, frame_update, parse_trim
from variant_error import UnsupportedVariantError

# 元数据更新时刻按渲染器处理块对齐，与 Dolby 的 processing_block_size 及
# speaker_renderer 的 block_size 落在同一网格。
METADATA_BLOCK_SAMPLES = 32


def align_metadata_sample(sample, block_samples=METADATA_BLOCK_SAMPLES):
    """把更新时刻量化到处理块边界。"""
    block = int(block_samples)
    if block <= 0:
        raise ValueError("block_samples must be positive")
    return block * ((int(sample) + block // 2 - 1) // block)


def _default_meta():
    """ADM 缺省渲染参数：点源、0 dB、无 divergence。"""
    return {"gain": 1.0, "size": (0.0, 0.0, 0.0), "divergence": 0.0,
            "snap": False}


def _apply_state(meta, state):
    """把一帧 OAMD 对象状态合并进跨帧保持的渲染参数。

    语义与 ``PythonSpeakerRenderer`` 的 metadata 路径一致：动态对象 block 0
    的 gain/size/snap 每帧都显式出现；divergence 只在扩展元素里出现，缺失时
    渲染器按 0 处理，因此这里也直接取 0，而不是保持旧值。未激活对象由渲染器
    静音，ADM 对象轨本身仍携带 PCM，必须写 gain=0。
    """
    if state.get("kind") != "dynamic":
        return
    if state.get("not_active"):
        meta["gain"] = 0.0
        return
    if state.get("render_status") == 1:
        size = (0.0, 0.0, 0.0)
        if state.get("size_mode") in (1, 2):
            size = tuple(min(max(float(value) / 32767.0, 0.0), 1.0)
                         for value in state.get("size_q", (0, 0, 0)))
        meta["gain"] = float(state.get("gain_q", 4096)) / 4096.0
        meta["size"] = size
        meta["snap"] = bool(state.get("snap", 0))
    meta["divergence"] = float(state.get("divergence_q", 0)) / 32767.0


def _lerp_xyz(start, target, amount):
    return tuple(a + (b - a) * amount for a, b in zip(start, target))


def _append_point(points, sample, xyz, interpolation_samples, meta):
    item = (int(sample), *map(float, xyz), int(interpolation_samples), dict(meta))
    if points and item[0] == points[-1][0]:
        points[-1] = item
    elif not points or item[0] > points[-1][0]:
        points.append(item)


def _expand_events_dense64(events, total_samples, rate, update_block_samples,
                           object_delay_samples, object_index):
    if not events:
        return [(0.0, 0.0, 0.0, 0.0, total_samples / float(rate), 0.0,
                 _default_meta())]

    block = int(update_block_samples)
    # 初始位置从成品 sample 0 起有效；合成延迟只作用于后续位置变化。
    current = events[0][1]
    current_meta = events[0][3]
    points = []
    _append_point(points, 0, current, 0, current_meta)

    for event_index, (coded_start, target, ramp_samples, meta) in enumerate(events[1:], 1):
        start = align_metadata_sample(coded_start + object_delay_samples, block)
        if start >= total_samples:
            break
        if start < points[-1][0]:
            raise UnsupportedVariantError(
                "oamd", "non_monotonic_position_updates",
                "对象位置更新时间倒退，无法生成连续 ADM 轨迹",
                details={"object": object_index, "sample": start,
                         "previous_sample": points[-1][0]})

        # 在运动起点保留上一位置，避免下游把长时间静止段直接连到首个中间点。
        if start > points[-1][0]:
            _append_point(points, start, current, 0, current_meta)

        ramp = max(0, int(ramp_samples))
        if ramp == 0:
            _append_point(points, start, target, 0, meta)
            current = target
            current_meta = meta
            continue

        end = start + math.ceil(ramp / block) * block
        if event_index + 1 < len(events):
            next_start = align_metadata_sample(
                events[event_index + 1][0] + object_delay_samples, block)
            if next_start < end:
                raise UnsupportedVariantError(
                    "oamd", "overlapping_position_ramps",
                    "同一对象的新位置更新在上一 ramp 完成前到达",
                    details={
                        "object": object_index,
                        "ramp_start_sample": start,
                        "ramp_end_sample": end,
                        "next_update_sample": next_start,
                        "repair_hint": "按 32-sample 状态机截断旧 ramp，再从当前插值位置启动新 ramp",
                    })

        # 从对齐后的更新点起按处理块推进，整条 ramp 覆盖 ramp_samples 个样本；
        # 几何式逼近望远镜化简为精确线性，每个中间点都落在真实 ramp 上。
        future = ramp
        elapsed = 0
        position = current
        while future > 0:
            amount = min(block / float(future), 1.0)
            position = _lerp_xyz(position, target, amount)
            elapsed += block
            sample = start + elapsed
            if sample >= total_samples:
                break
            _append_point(points, sample, position, block, meta)
            future -= block
        current = target
        current_meta = meta

    blocks = []
    for point_index, point in enumerate(points):
        sample, x, y, z, interpolation_samples, meta = point
        end = points[point_index + 1][0] if point_index + 1 < len(points) else total_samples
        duration_samples = max(0, end - sample)
        if duration_samples == 0:
            continue
        interpolation_samples = min(interpolation_samples, duration_samples)
        blocks.append((sample / float(rate), x, y, z,
                       duration_samples / float(rate),
                       interpolation_samples / float(rate), meta))
    return blocks



def _compact_events(events, total_samples, rate, update_block_samples,
                    object_delay_samples, object_index):
    """Represent each linear OAMD ramp with one ADM interpolation block.

    The update instant is quantized to the processing block boundary, the motion
    starts there immediately, and ``interpolationLength`` spans the full
    ``ramp_duration``. The interpreted motion therefore covers
    ``[align(start), align(start) + ramp_duration]``.
    """
    if not events:
        return [(0.0, 0.0, 0.0, 0.0, total_samples / float(rate), 0.0,
                 _default_meta())]

    points = []
    current = events[0][1]
    current_meta = events[0][3]
    _append_point(points, 0, current, 0, current_meta)

    block = int(update_block_samples)
    for event_index, (coded_start, target, ramp_samples, meta) in enumerate(events[1:], 1):
        event_start = align_metadata_sample(coded_start + object_delay_samples, block)
        if event_start >= total_samples:
            break
        ramp = max(0, int(ramp_samples))
        block_start = event_start
        if block_start >= total_samples:
            break
        ramp_end = block_start + ramp

        if block_start < points[-1][0]:
            raise UnsupportedVariantError(
                "oamd", "non_monotonic_compact_position_updates",
                "紧凑对象位置更新时间倒退",
                details={"object": object_index, "sample": block_start,
                         "previous_sample": points[-1][0]})

        if event_index + 1 < len(events):
            next_coded_start, _, next_ramp_samples = events[event_index + 1][:3]
            next_event_start = align_metadata_sample(
                next_coded_start + object_delay_samples, block)
            next_block_start = next_event_start
            if next_block_start < ramp_end:
                raise UnsupportedVariantError(
                    "oamd", "overlapping_compact_position_ramps",
                    "同一对象的新位置更新在上一紧凑 ramp 完成前到达",
                    details={
                        "object": object_index,
                        "ramp_start_sample": block_start,
                        "ramp_end_sample": ramp_end,
                        "next_update_sample": next_block_start,
                        "repair_hint": "对此变体使用 --trajectory-mode dense64 并检查 OAMD 调度",
                    })

        block_target = target
        block_interpolation = ramp
        available = total_samples - block_start
        if ramp > available:
            block_target = _lerp_xyz(current, target, available / float(ramp))
            block_interpolation = available
        _append_point(points, block_start, block_target, block_interpolation, meta)
        current = target
        current_meta = meta

    blocks = []
    for point_index, point in enumerate(points):
        sample, x, y, z, interpolation_samples, meta = point
        end = points[point_index + 1][0] if point_index + 1 < len(points) else total_samples
        duration_samples = max(0, end - sample)
        if duration_samples == 0:
            continue
        interpolation_samples = min(interpolation_samples, duration_samples)
        blocks.append((sample / float(rate), x, y, z,
                       duration_samples / float(rate),
                       interpolation_samples / float(rate), meta))
    return blocks


def _expand_events(events, total_samples, rate, update_block_samples,
                   object_delay_samples, object_index, trajectory_mode):
    if trajectory_mode == "compact":
        return _compact_events(events, total_samples, rate, update_block_samples,
                               object_delay_samples, object_index)
    if trajectory_mode == "dense64":
        return _expand_events_dense64(events, total_samples, rate, update_block_samples,
                                      object_delay_samples, object_index)
    raise ValueError(f"未知 trajectory_mode: {trajectory_mode}")

def read_atmos_meta(index, frames=None, max_scan=64):
    """从 OAMD payload 读取 trim/warp 配置。

    trim element 描述整段节目的校准意图，跨帧恒定，因此扫到第一帧含 trim 的
    payload 即可。返回 ``None`` 表示该节目没有 trim element。
    """
    rows = index.rows if frames is None else frames
    for row in rows[:max_scan]:
        subs = index.subpayloads(row)
        if not subs or 11 not in subs:
            continue
        meta = parse_trim(subs[11])
        if meta is not None:
            return meta
    return None


def build_adm_tracks(index, frames=None, rate=48000, frame_samples=1536,
                     update_block_samples=METADATA_BLOCK_SAMPLES,
                     object_delay_samples=1473, trajectory_mode="compact"):
    """从统一 metadata index 构造 15 条 ADM 轨迹。

    返回 ``[(name, [(rtime,x,y,z,duration,interpolation), ...]), ...]``。
    OAMD 的内外层 sample offset、block offset 和 ramp 均保留；更新时刻量化到
    ``update_block_samples`` 的块边界，ramp 覆盖完整的 ramp_duration。
    ``trajectory_mode="compact"`` 用一个长 ADM interpolation block 表示每条
    线性 ramp；``dense64`` 保留逐块展开作为兼容回退。
    ``object_delay_samples`` 将位置更新与对象 PCM 的 decoder 输出时刻对齐。
    slot1..15 与对象 PCM ch1..15 一一对应。
    """
    frames = index.rows if frames is None else frames
    state = JocFieldState()
    events = [[] for _ in range(15)]
    previous = [None] * 15
    meta_state = [_default_meta() for _ in range(15)]

    for seq, row in enumerate(frames):
        subs = index.subpayloads(row)
        timing = None
        full = None
        if subs is not None and 11 in subs:
            timing = frame_update(subs[11])
            state.apply(timing["values"])
            full = frame_state(subs[11])

        event_sample = seq * frame_samples
        ramp_samples = 0
        if timing is not None:
            outer_offset = (index.subpayload_sample_offset(row, 11)
                            if hasattr(index, "subpayload_sample_offset") else 0)
            event_sample += outer_offset + timing["block_offset_samples"]
            ramp_samples = timing["ramp_duration_samples"]

        q = state.q
        for obj in range(1, 16):
            xyz = q_to_adm_xyz(q[(obj, "q1")], q[(obj, "q2")], q[(obj, "q3")])
            if full is not None and obj < len(full["objects"]):
                _apply_state(meta_state[obj - 1], full["objects"][obj])
            meta = dict(meta_state[obj - 1])
            if previous[obj - 1] != (xyz, meta):
                events[obj - 1].append((event_sample, xyz, ramp_samples, meta))
                previous[obj - 1] = (xyz, meta)

    total_samples = len(frames) * frame_samples
    # bed 对象在 OAMD 里没有位置字段：位置由声明的 bed 布局决定，整片静止在该方位。
    bed_positions = bed_object_positions(getattr(index, "bed_channels", ()))
    tracks = []
    for obj in range(1, 16):
        fixed = bed_positions.get(obj)
        if fixed is not None:
            tracks.append((f"JOC_Object_{obj}",
                           [(0.0, fixed[0], fixed[1], fixed[2],
                             total_samples / float(rate), 0.0, _default_meta())]))
            continue
        tracks.append(
            (f"JOC_Object_{obj}",
             _expand_events(events[obj - 1], total_samples, rate,
                            update_block_samples, object_delay_samples, obj,
                            trajectory_mode)))
    return tracks
