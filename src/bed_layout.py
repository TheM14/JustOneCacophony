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
