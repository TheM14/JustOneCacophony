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

def build_axml(obj_tracks, duration_sec):
    """obj_tracks: [(name, [(rtime, x, y, z, dur), ...]) ×15]"""
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
            a(f'<audioBlockFormat audioBlockFormatID="AB_0003{0x1001 + i:04x}_{k + 1:08x}" '
              f'rtime="{ts(t)}" duration="{ts(dur)}">')
            a('<cartesian>1</cartesian>')
            a(f'<position coordinate="X">{x:.10f}</position>')
            a(f'<position coordinate="Y">{y:.10f}</position>')
            if z != 0:
                a(f'<position coordinate="Z">{z:.10f}</position>')
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
    a('</audioFormatExtended></format></coreMetadata>')
    a('</ebuCoreMain>')
    return ''.join(w).encode('utf-8')
