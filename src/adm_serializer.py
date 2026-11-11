"""把 7.1.2 bed、15 个对象及其位置轨迹序列化为 ADM axml。

序列化结果采用固定元素顺序、属性顺序和十六进制 ADM 标识符，便于稳定输出和校验。
"""
BED_NAMES = ["RoomCentricLeft", "RoomCentricRight", "RoomCentricCenter", "RoomCentricLFE",
             "RoomCentricLeftSideSurround", "RoomCentricRightSideSurround",
             "RoomCentricLeftRearSurround", "RoomCentricRightRearSurround",
             "RoomCentricLeftTopSurround", "RoomCentricRightTopSurround"]
BED_LABELS = ["RC_L", "RC_R", "RC_C", "RC_LFE", "RC_Lss", "RC_Rss",
              "RC_Lrs", "RC_Rrs", "RC_Lts", "RC_Rts"]
BED_POS = [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 1.0, -1.0),
           (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
           (-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
N_OBJ = 15

def ts(seconds):
    s = int(seconds)
    frac = int(round((seconds - s) * 100000))
    if frac >= 100000:
        s += 1; frac = 0
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{frac:05d}"

def esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def _fmt_float(value):
    return f"{float(value):.6f}"


def _trim_records(atmos_meta):
    """把 OAMD trim element 展开成 DolbyTrimModeSetup 的 records。

    ``configs`` 每项有三种形态：
      * ``{"default": True}``  —— 交给渲染器推导，不写任何值子元素；
      * ``{"disabled": True}`` —— 该配置不施加 trim，写全 0；
      * 含 centre/surround/height/bal3d_y_tb/bal3d_y_lis 的显式配置。
    缺省字段一律补 0。
    """
    lines = ["<DolbyTrimModeSetup>"]
    for index, record in enumerate(atmos_meta.get("configs") or ()):
        if record is None:
            continue
        lines.append("<records>")
        lines.append(f"<type>{index}</type>")
        if isinstance(record, dict) and not record.get("default"):
            if record.get("disabled"):
                values = {"centerTrim": 0.0, "surroundTrim": 0.0,
                          "heightTrim": 0.0, "balanceOverhead": 0.0,
                          "balanceListener": 0.0}
            else:
                values = {
                    "centerTrim": record.get("centre", 0.0),
                    "surroundTrim": record.get("surround", 0.0),
                    "heightTrim": record.get("height", 0.0),
                    "balanceOverhead": record.get("bal3d_y_tb", 0.0),
                    "balanceListener": record.get("bal3d_y_lis", 0.0),
                }
            for name, value in values.items():
                lines.append(f"<{name}>{_fmt_float(value)}</{name}>")
        lines.append("</records>")
    lines.append("</DolbyTrimModeSetup>")
    return lines


def _warp_xml_value(atmos_meta):
    """OAMD 的 warp_mode → ADM ``DolbyWarpMode`` 取值。

    warp_mode 选择下混/渲染模式，四档依次为 Direct render、Direct render
    with room balance、Dolby Pro Logic IIx、Standard (Lo/Ro)，ADM 沿用同一
    编号，因此直接照搬码流值。
    """
    return max(0, min(3, int(atmos_meta.get("warp_mode", 0))))


def _atmos_master_block(atmos_meta):
    """Dolby 扩展容器：内容信息 + trim 记录 + warp。"""
    lines = ["<DolbyAtmosMasterFile>"]
    warp = _warp_xml_value(atmos_meta)
    lines.append("<DolbyMasterContentInfo>")
    lines.append(f"<render_warp_mode>{warp}</render_warp_mode>")
    lines.append("</DolbyMasterContentInfo>")
    lines.extend(_trim_records(atmos_meta))
    lines.append(f"<DolbyWarpMode>{warp}</DolbyWarpMode>")
    lines.append("</DolbyAtmosMasterFile>")
    return lines


def _object_meta_lines(meta):
    """把 OAMD 的逐对象渲染参数写成 ADM 的 audioBlockFormat 子元素。

    只写 ADM 能表达的字段：size → width/height/depth（相对单位 0..1），
    gain → gain（线性），divergence → objectDivergence（笛卡尔坐标系用
    positionRange）。缺省值（size 0、gain 1.0、divergence 0）省略。
    """
    if not meta:
        return []
    lines = []
    size = meta.get("size") or (0.0, 0.0, 0.0)
    # OAMD 的 size 三轴顺序为 width, depth, height。
    width, depth, height = size
    if width > 0.0 or depth > 0.0 or height > 0.0:
        lines.append(f'<width>{width:.6f}</width>')
        lines.append(f'<height>{height:.6f}</height>')
        lines.append(f'<depth>{depth:.6f}</depth>')
    gain = meta.get("gain")
    if gain is not None and abs(float(gain) - 1.0) > 1e-9:
        lines.append(f'<gain gainUnit="linear">{float(gain):.6f}</gain>')
    divergence = meta.get("divergence")
    if divergence is not None and float(divergence) > 0.0:
        lines.append(f'<objectDivergence positionRange="{float(divergence):.6f}"/>')
    return lines


def build_axml(obj_tracks, duration_sec, atmos_meta=None):
    """obj_tracks: [(name, [(rtime, x, y, z, dur, interpolation, meta), ...]) ×15]

    ``meta`` 可省略；给出时写 width/height/depth、gain、objectDivergence。
    ``atmos_meta`` 为 ``oamd_bits.parse_trim`` 的结果；给出时把 trim 与 warp
    写进 ADM。
    """
    w = []
    a = w.append
    a('<?xml version="1.0" encoding="utf-8"?>')
    a('<ebuCoreMain xsi:schemaLocation="urn:ebu:metadata-schema:ebuCore_2016 ebucore.xsd" '
      'lang="en" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
      'xmlns="urn:ebu:metadata-schema:ebuCore_2016">')
    a('<coreMetadata><format><audioFormatExtended>')
    a(f'<audioProgramme audioProgrammeID="APR_1001" audioProgrammeName="EAC3JOC_Export" '
      f'start="{ts(0)}" end="{ts(duration_sec)}">')
    a('<audioContentIDRef>ACO_1001</audioContentIDRef>')
    a('<audioContentIDRef>ACO_1002</audioContentIDRef>')
    a('</audioProgramme>')
    a('<audioContent audioContentID="ACO_1001" audioContentName="EAC3JOC_Master_Content">')
    a('<audioObjectIDRef>AO_1001</audioObjectIDRef>')
    a('<dialogue mixedContentKind="0">2</dialogue>')
    a('</audioContent>')
    a('<audioContent audioContentID="ACO_1002" audioContentName="Objects">')
    for i in range(N_OBJ):
        a(f'<audioObjectIDRef>AO_{0x100b + i:04x}</audioObjectIDRef>')
    a('<dialogue mixedContentKind="0">2</dialogue>')
    a('</audioContent>')
    a(f'<audioObject audioObjectID="AO_1001" audioObjectName="Bed" '
      f'start="{ts(0)}" duration="{ts(duration_sec)}">')
    a('<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>')
    for i in range(10):
        a(f'<audioTrackUIDRef>ATU_{i + 1:08x}</audioTrackUIDRef>')
    a('</audioObject>')
    for i in range(N_OBJ):
        a(f'<audioObject audioObjectID="AO_{0x100b + i:04x}" audioObjectName="Audio Object {i+1}" '
          f'start="{ts(0)}" duration="{ts(duration_sec)}">')
        a(f'<audioPackFormatIDRef>AP_0003{0x1001 + i:04x}</audioPackFormatIDRef>')
        a(f'<audioTrackUIDRef>ATU_{11 + i:08x}</audioTrackUIDRef>')
        a('</audioObject>')
    a('<audioPackFormat audioPackFormatID="AP_00011001" audioPackFormatName="EAC3JOCBedPack" '
      'typeDefinition="DirectSpeakers" typeLabel="0001">')
    for i in range(10):
        a(f'<audioChannelFormatIDRef>AC_0001{0x1001 + i:04x}</audioChannelFormatIDRef>')
    a('</audioPackFormat>')
    for i in range(N_OBJ):
        a(f'<audioPackFormat audioPackFormatID="AP_0003{0x1001 + i:04x}" '
          f'audioPackFormatName="JOC_Object_{i+1}" typeDefinition="Objects" typeLabel="0003">')
        a(f'<audioChannelFormatIDRef>AC_0003{0x1001 + i:04x}</audioChannelFormatIDRef>')
        a('</audioPackFormat>')
    for i in range(10):
        a(f'<audioChannelFormat audioChannelFormatID="AC_0001{0x1001 + i:04x}" '
          f'audioChannelFormatName="{BED_NAMES[i]}" typeDefinition="DirectSpeakers" typeLabel="0001">')
        a(f'<audioBlockFormat audioBlockFormatID="AB_0001{0x1001 + i:04x}_00000001">')
        a('<cartesian>1</cartesian>')
        x, y, z = BED_POS[i]
        a(f'<position coordinate="X">{x:.10f}</position>')
        a(f'<position coordinate="Y">{y:.10f}</position>')
        if z != 0:
            a(f'<position coordinate="Z">{z:.10f}</position>')
        a(f'<speakerLabel>{BED_LABELS[i]}</speakerLabel>')
        a('</audioBlockFormat>')
        a('</audioChannelFormat>')
    for i, (oname, kfs) in enumerate(obj_tracks):
        a(f'<audioChannelFormat audioChannelFormatID="AC_0003{0x1001 + i:04x}" '
          f'audioChannelFormatName="{oname}" typeDefinition="Objects" typeLabel="0003">')
        for k, keyframe in enumerate(kfs):
            t, x, y, z, dur = keyframe[:5]
            interpolation = keyframe[5] if len(keyframe) > 5 else 0.0
            meta = keyframe[6] if len(keyframe) > 6 else None
            a(f'<audioBlockFormat audioBlockFormatID="AB_0003{0x1001 + i:04x}_{k + 1:08x}" '
              f'rtime="{ts(t)}" duration="{ts(dur)}">')
            a('<cartesian>1</cartesian>')
            a(f'<position coordinate="X">{x:.10f}</position>')
            a(f'<position coordinate="Y">{y:.10f}</position>')
            if z != 0:
                a(f'<position coordinate="Z">{z:.10f}</position>')
            for line in _object_meta_lines(meta):
                a(line)
            a(f'<jumpPosition interpolationLength="{interpolation:.5f}">1</jumpPosition>')
            a('</audioBlockFormat>')
        a('</audioChannelFormat>')
    for i in range(10):
        a(f'<audioTrackUID UID="ATU_{i + 1:08x}" bitDepth="24" sampleRate="48000">')
        a(f'<audioTrackFormatIDRef>AT_0001{0x1001 + i:04x}_01</audioTrackFormatIDRef>')
        a('<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>')
        a('</audioTrackUID>')
    for i in range(N_OBJ):
        a(f'<audioTrackUID UID="ATU_{11 + i:08x}" bitDepth="24" sampleRate="48000">')
        a(f'<audioTrackFormatIDRef>AT_0003{0x1001 + i:04x}_01</audioTrackFormatIDRef>')
        a(f'<audioPackFormatIDRef>AP_0003{0x1001 + i:04x}</audioPackFormatIDRef>')
        a('</audioTrackUID>')
    for i in range(10):
        a(f'<audioTrackFormat audioTrackFormatID="AT_0001{0x1001 + i:04x}_01" '
          f'audioTrackFormatName="PCM_{BED_NAMES[i]}" formatDefinition="PCM" formatLabel="0001">')
        a(f'<audioStreamFormatIDRef>AS_0001{0x1001 + i:04x}</audioStreamFormatIDRef>')
        a('</audioTrackFormat>')
    for i in range(N_OBJ):
        a(f'<audioTrackFormat audioTrackFormatID="AT_0003{0x1001 + i:04x}_01" '
          f'audioTrackFormatName="PCM_JOC_Object_{i+1}" formatDefinition="PCM" formatLabel="0001">')
        a(f'<audioStreamFormatIDRef>AS_0003{0x1001 + i:04x}</audioStreamFormatIDRef>')
        a('</audioTrackFormat>')
    for i in range(10):
        a(f'<audioStreamFormat audioStreamFormatID="AS_0001{0x1001 + i:04x}" '
          f'audioStreamFormatName="PCM_{BED_NAMES[i]}" formatDefinition="PCM" formatLabel="0001">')
        a(f'<audioChannelFormatIDRef>AC_0001{0x1001 + i:04x}</audioChannelFormatIDRef>')
        a('<audioPackFormatIDRef>AP_00011001</audioPackFormatIDRef>')
        a(f'<audioTrackFormatIDRef>AT_0001{0x1001 + i:04x}_01</audioTrackFormatIDRef>')
        a('</audioStreamFormat>')
    for i in range(N_OBJ):
        a(f'<audioStreamFormat audioStreamFormatID="AS_0003{0x1001 + i:04x}" '
          f'audioStreamFormatName="PCM_JOC_Object_{i+1}" formatDefinition="PCM" formatLabel="0001">')
        a(f'<audioChannelFormatIDRef>AC_0003{0x1001 + i:04x}</audioChannelFormatIDRef>')
        a(f'<audioPackFormatIDRef>AP_0003{0x1001 + i:04x}</audioPackFormatIDRef>')
        a(f'<audioTrackFormatIDRef>AT_0003{0x1001 + i:04x}_01</audioTrackFormatIDRef>')
        a('</audioStreamFormat>')
    if atmos_meta:
        w.extend(_atmos_master_block(atmos_meta))
    a('</audioFormatExtended></format></coreMetadata>')
    a('</ebuCoreMain>')
    return ''.join(w).encode('utf-8')
