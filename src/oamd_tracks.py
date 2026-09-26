"""Evolution id11/OAMD 帧序列 → ADM 15 对象关键帧。"""
import math

from adm_atmos import q_to_adm_xyz
from oamd_bits import JocFieldState, frame_update
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


def _lerp_xyz(start, target, amount):
    return tuple(a + (b - a) * amount for a, b in zip(start, target))


def _append_point(points, sample, xyz, interpolation_samples):
    item = (int(sample), *map(float, xyz), int(interpolation_samples))
    if points and item[0] == points[-1][0]:
        points[-1] = item
    elif not points or item[0] > points[-1][0]:
        points.append(item)


def _expand_events_dense64(events, total_samples, rate, update_block_samples,
                           object_delay_samples, object_index):
    if not events:
        return [(0.0, 0.0, 0.0, 0.0, total_samples / float(rate), 0.0)]

    block = int(update_block_samples)
    # 初始位置从成品 sample 0 起有效；合成延迟只作用于后续位置变化。
    current = events[0][1]
    points = []
    _append_point(points, 0, current, 0)

    for event_index, (coded_start, target, ramp_samples) in enumerate(events[1:], 1):
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
            _append_point(points, start, current, 0)

        ramp = max(0, int(ramp_samples))
        if ramp == 0:
            _append_point(points, start, target, 0)
            current = target
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
            _append_point(points, sample, position, block)
            future -= block
        current = target

    blocks = []
    for point_index, (sample, x, y, z, interpolation_samples) in enumerate(points):
        end = points[point_index + 1][0] if point_index + 1 < len(points) else total_samples
        duration_samples = max(0, end - sample)
        if duration_samples == 0:
            continue
        interpolation_samples = min(interpolation_samples, duration_samples)
        blocks.append((sample / float(rate), x, y, z,
                       duration_samples / float(rate),
                       interpolation_samples / float(rate)))
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
        return [(0.0, 0.0, 0.0, 0.0, total_samples / float(rate), 0.0)]

    points = []
    current = events[0][1]
    _append_point(points, 0, current, 0)

    block = int(update_block_samples)
    for event_index, (coded_start, target, ramp_samples) in enumerate(events[1:], 1):
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
            next_coded_start, _, next_ramp_samples = events[event_index + 1]
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
        _append_point(points, block_start, block_target, block_interpolation)
        current = target

    blocks = []
    for point_index, (sample, x, y, z, interpolation_samples) in enumerate(points):
        end = points[point_index + 1][0] if point_index + 1 < len(points) else total_samples
        duration_samples = max(0, end - sample)
        if duration_samples == 0:
            continue
        interpolation_samples = min(interpolation_samples, duration_samples)
        blocks.append((sample / float(rate), x, y, z,
                       duration_samples / float(rate),
                       interpolation_samples / float(rate)))
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

    for seq, row in enumerate(frames):
        subs = index.subpayloads(row)
        timing = None
        if 11 in subs:
            timing = frame_update(subs[11])
            state.apply(timing["values"])

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
            if previous[obj - 1] != xyz:
                events[obj - 1].append((event_sample, xyz, ramp_samples))
                previous[obj - 1] = xyz

    total_samples = len(frames) * frame_samples
    return [
        (f"JOC_Object_{obj}",
         _expand_events(events[obj - 1], total_samples, rate,
                        update_block_samples, object_delay_samples, obj,
                        trajectory_mode))
        for obj in range(1, 16)
    ]
