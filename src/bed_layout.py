"""JOC 核心通道的输入序、床声明与 ADM 落位。

这里只有"位置/身份"层：按通道码把 E-AC-3 解出的 PCM 摆进 JOC 输入序，并把额外
那一对落到 ADM 骨架的槽位。JOC 的 DSP 不读取这些信息（规格 §10.1），它只用
``(N, sigma)``。
"""
from emdf import expand_chanmap

# JOC 核心通道序列的前 5 位（规格 §2）：L R C Ls Rs。
JOC_CORE_POSITIONS = ("L", "R", "C", "Ls", "Rs")

# 额外那一对的身份由 k 决定（规格 §2）：k=1 是后环绕，k∈{2,4} 是前上方。
EXTRA_PAIR_BY_CONFIG = {1: ("Lrs", "Rrs"), 2: ("Vhl", "Vhr"), 4: ("Vhl", "Vhr")}

# 额外那一对在 10 槽 ADM 骨架里的槽位（规格 §10.2 的 RC_Lrs/RC_Rrs 与 RC_Lts/RC_Rts）。
# ADM 现契约是"除 LFE 槽外全部静音"（床通道不作为输出通道），因此这些槽位只被记录，
# 不承载音频；DSP 也不读它们。
BED_EXTRA_SLOTS_BY_CONFIG = {1: (6, 7), 2: (8, 9), 4: (8, 9)}

# 位置 → ffmpeg 声道名与其声道位。ffmpeg 的帧内声道顺序就是声道位的升序，
# 因此这张表同时给出"位置集合 → 帧内下标"的换算。
_FFMPEG_POSITIONS = {
    "L": ("FL", 0x1),
    "R": ("FR", 0x2),
    "C": ("FC", 0x4),
    "LFE": ("LFE", 0x8),
    "Lrs": ("BL", 0x10),
    "Rrs": ("BR", 0x20),
    "Lc": ("FLC", 0x40),
    "Rc": ("FRC", 0x80),
    "Cs": ("BC", 0x100),
    "Ls": ("SL", 0x200),
    "Rs": ("SR", 0x400),
    "Vhl": ("TFL", 0x1000),
    "Vhc": ("TFC", 0x2000),
    "Vhr": ("TFR", 0x4000),
    "Lts": ("TBL", 0x8000),
    "Rts": ("TBR", 0x20000),
}


def bed_order(locations):
    """把位置集合按 ffmpeg 的声道位顺序排好，返回 (位置, ffmpeg 名) 列表。

    未知位置返回 None，由调用方决定报哪种变体。
    """
    unique = []
    for position in locations:
        if position not in unique:
            unique.append(position)
    if any(position not in _FFMPEG_POSITIONS for position in unique):
        return None
    return [(position, _FFMPEG_POSITIONS[position][0])
            for position in sorted(unique, key=lambda p: _FFMPEG_POSITIONS[p][1])]


def extra_pair(core_locations, chanmap):
    """依赖子流的 chanmap 里超出核心声道集合的那一对位置。

    chanmap 会把核心已有的声道一并列出（例如 5.1 核心 + 前上方对会写
    ``Ls/Rs/Lrs/Rrs``），因此额外对是"chanmap 位置 − 核心位置"。
    """
    core = set(core_locations)
    return tuple(position for position in expand_chanmap(chanmap) if position not in core)


# OAMD §5.6.1.1.4 Table 12 的标准 bed 声道（LSB 在前，与 oamd_bits.BED_CHANNEL_LABELS 同序）。
# 一个 bed_assignment() 声明的声道就是按这个顺序展开的。
BED_LAYOUT_ORDER = (
    "RC_L/RC_R", "RC_C", "RC_LFE", "RC_LS/RC_RS", "RC_LB/RC_RB",
    "RC_TFL/RC_TFR", "RC_TSL/RC_TSR", "RC_TBL/RC_TBR", "RC_LW/RC_RW", "RC_LFE2",
)

# 标准 bed 声道 → 扬声器布局里的短名（与 speaker_layouts 的声道名同形，如 RC_LW → Lw）。
BED_CHANNEL_NAMES = {
    "RC_L": "L", "RC_R": "R", "RC_C": "C", "RC_LFE": "LFE",
    "RC_LS": "Ls", "RC_RS": "Rs", "RC_LB": "Lrs", "RC_RB": "Rrs",
    "RC_TFL": "Ltf", "RC_TFR": "Rtf", "RC_TSL": "Ltm", "RC_TSR": "Rtm",
    "RC_TBL": "Ltr", "RC_TBR": "Rtr", "RC_LW": "Lw", "RC_RW": "Rw",
}

# 每个 bed 声道的固定方位（ADM 笛卡尔坐标，x 向右、y 向前、z 向上），取自
# ETSI TS 103 420 V1.2.1 clause B.2.1.4 Table B.10：speakerLabel 各自对应唯一的
# position(X,Y,Z)，与目标扬声器布局无关。bed 对象在 OAMD 里没有位置字段，位置只由
# 声明的声道标签决定。
# Table B.10 把 Lw/Rw 的 Y 记作 0,677419，即 ADM 62 级网格上的 42/62 = 21/31。
BED_CHANNEL_ADM_POSITION = {
    "RC_L": (-1.0, 1.0, 0.0),
    "RC_R": (1.0, 1.0, 0.0),
    "RC_C": (0.0, 1.0, 0.0),
    "RC_LFE": (-1.0, 1.0, -1.0),
    "RC_LS": (-1.0, 0.0, 0.0),
    "RC_RS": (1.0, 0.0, 0.0),
    "RC_LB": (-1.0, -1.0, 0.0),
    "RC_RB": (1.0, -1.0, 0.0),
    "RC_TFL": (-1.0, 1.0, 1.0),
    "RC_TFR": (1.0, 1.0, 1.0),
    "RC_TSL": (-1.0, 0.0, 1.0),
    "RC_TSR": (1.0, 0.0, 1.0),
    "RC_TBL": (-1.0, -1.0, 1.0),
    "RC_TBR": (1.0, -1.0, 1.0),
    "RC_LW": (-1.0, 21.0 / 31.0, 0.0),
    "RC_RW": (1.0, 21.0 / 31.0, 0.0),
    "RC_LFE2": (1.0, 1.0, -1.0),
}

# LFE 不是空间声源：它的 PCM 走独立通路，不参与对象摆位。
BED_LFE_LABELS = frozenset(("RC_LFE", "RC_LFE2"))

# 标准 bed 布局名：由声明声道的集合唯一确定（9.1.6 这类常用叫法即由此得来）。
BED_LAYOUT_NAMES = {
    frozenset(("L", "R")): "2.0",
    frozenset(("L", "R", "C")): "3.0",
    frozenset(("L", "R", "C", "LFE")): "3.1",
    frozenset(("L", "R", "Ls", "Rs")): "4.0",
    frozenset(("L", "R", "C", "Ls", "Rs")): "5.0",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs")): "5.1",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs")): "7.1",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw")): "9.1",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Ltm", "Rtm")): "5.1.2",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf", "Ltr", "Rtr")):
        "5.1.4",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltm", "Rtm")): "7.1.2",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
               "Ltf", "Rtf", "Ltr", "Rtr")): "7.1.4",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
               "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr")): "7.1.6",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
               "Lw", "Rw", "Ltm", "Rtm")): "9.1.2",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs",
               "Lw", "Rw", "Ltf", "Rtf", "Ltr", "Rtr")): "9.1.4",
    frozenset(("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Lw", "Rw",
               "Ltf", "Rtf", "Ltm", "Rtm", "Ltr", "Rtr")): "9.1.6",
}

# ADM 7.1.2 床的槽位顺序：与 BED_EXTRA_SLOTS_BY_CONFIG 的 10 槽骨架一致。
ADM_BED_712 = ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf")


def adm_xyz_to_q(x, y, z):
    """ADM 笛卡尔坐标 → OAMD/渲染器使用的 Q15 方位三元组。"""
    q1 = int(round((float(x) + 1.0) * 0.5 * 32767.0))
    q2 = int(round((1.0 - float(y)) * 0.5 * 32767.0))
    q3 = int(round(float(z) * 32767.0))
    clamp = lambda value: max(0, min(32767, value))  # noqa: E731
    return clamp(q1), clamp(q2), clamp(q3)


def adm_xyz_to_render_coordinates(x, y, z):
    """ADM 笛卡尔坐标 → 渲染器方位 ``(u, v, w)``（u 向右、v 向后、w 向上，0..1）。"""
    return ((float(x) + 1.0) * 0.5, (1.0 - float(y)) * 0.5, float(z))


def declared_bed_channels(assignments):
    """把 ``program_assignment()`` 的 bed 声明展开成有序的标准声道标签。

    ``assignments`` 是 ``bed_assignments`` 列表：元素可以是带 ``channels`` 的字典，
    也可以直接是声道标签列表（``program_bed_channels()`` 的形式）。未识别标签保持
    原样，由调用方决定是否当作变体处理。
    """
    labels = []
    for assignment in assignments or ():
        if isinstance(assignment, dict):
            channels = assignment.get("channels", ())
        else:
            channels = assignment
        for label in channels:
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def bed_layout_name(labels):
    """按声明声道集合识别 bed 布局名；未知集合返回 None。"""
    names = [BED_CHANNEL_NAMES.get(label, label) for label in labels]
    return BED_LAYOUT_NAMES.get(frozenset(names))


def bed_layout(assignments):
    """声明的 bed 布局：``{"name", "labels", "channels"}``；没有 bed 声明时返回 None。

    ``channels`` 去掉 LFE/LFE2：JOC 的对象槽按这个顺序依次对应 15 路对象 PCM，
    LFE 走核心声道的独立通路。
    """
    labels = declared_bed_channels(assignments)
    if not labels:
        return None
    channels = [label for label in labels if label not in BED_LFE_LABELS]
    return {
        "name": bed_layout_name(labels),
        "labels": list(labels),
        "channels": channels,
    }


def bed_object_positions(labels):
    """JOC 对象槽 → 固定 ADM 位置：(1..15) 对应未含 LFE 的 bed 声道。

    positions 由声道标签唯一决定（Table B.10），与目标输出布局无关——同一个 bed 声道
    在扬声器布局和双耳渲染里都落在同一个源位置。返回的字典只包含能给出固定位置的
    标准声道；不是 bed 声道的对象不在其中，仍按 OAMD 坐标渲染。LFE 走独立通路，
    不出现在对象槽里。
    """
    positions = {}
    for index, label in enumerate(labels, 1):
        if index > 15:
            break
        if label in BED_LFE_LABELS:
            continue
        position = BED_CHANNEL_ADM_POSITION.get(label)
        if position is not None:
            positions[index] = position
    return positions


def joc_input_indices(locations, pair):
    """返回 JOC 输入序 (L, R, C, Ls, Rs, 额外对…) 在床 PCM 里的下标。"""
    if pair is None:
        pair = ()
    order = bed_order(locations)
    if order is None:
        return None
    index_of = {position: index for index, (position, _name) in enumerate(order)}
    required = list(JOC_CORE_POSITIONS) + list(pair)
    if any(position not in index_of for position in required):
        return None
    if len(required) != len(set(required)):
        return None
    return [index_of[position] for position in required]
