"""生成 25 声道 RF64 ADM BWF 及其 axml、chna、dbmd 元数据。

输出由 10 声道 7.1.2 bed 和 15 路对象组成；RF64 尺寸字段在写入完成后回填。
"""
import operator
import struct
import numpy as np
import xml.etree.ElementTree as ET

NS = "urn:ebu:metadata-schema:ebuCore_2016"
XSI = "http://www.w3.org/2001/XMLSchema-instance"

BED_NAMES = ["RoomCentricLeft", "RoomCentricRight", "RoomCentricCenter",
             "RoomCentricLFE", "RoomCentricLeftSideSurround",
             "RoomCentricRightSideSurround", "RoomCentricLeftRearSurround",
             "RoomCentricRightRearSurround", "RoomCentricLeftTopSurround",
             "RoomCentricRightTopSurround"]
BED_LABELS = ["RC_L", "RC_R", "RC_C", "RC_LFE", "RC_Lss", "RC_Rss",
              "RC_Lrs", "RC_Rrs", "RC_Lts", "RC_Rts"]
BED_POS = [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
           (-1.0, 1.0, -1.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
           (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

N_OBJ = 15
JOC_BINAURAL_MODES = {
    "off": 0,
    "near": 1,
    "far": 2,
    "mid": 3,
    "unspecified": 4,
}
JOC_BINAURAL_MODE_DEFAULT = "unspecified"

def q_to_adm_xyz(q1, q2, q3):
    posX = min(1.0, round(q1 * 62 / 32767.0) / 62.0)
    posY = min(1.0, round(q2 * 62 / 32767.0) / 62.0)
    posZ = round(q3 * 15 / 32767.0) / 15.0
    posZ = max(-1.0, min(1.0, posZ))
    return posX * 2 - 1, 1 - posY * 2, posZ

def ts(seconds):
    s = int(seconds)
    frac = int(round((seconds - s) * 100000))
    if frac >= 100000:
        s += 1; frac = 0
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{frac:05d}"

def sub(parent, tag, attrib=None, text=None):
    e = ET.SubElement(parent, tag)
    if attrib:
        for k, v in attrib.items():
            e.set(k, v)
    if text is not None:
        e.text = text
    return e

def add_refs(parent, tag, ids):
    for i in ids:
        sub(parent, tag, text=i)

def obj_block(cf, bid, t, x, y, z, dur, interpolation=0.0):
    b = sub(cf, "audioBlockFormat", {
        "audioBlockFormatID": bid, "rtime": ts(t), "duration": ts(dur)})
    sub(b, "cartesian", text="1")
    for c, v in (("X", x), ("Y", y), ("Z", z)):
        if c == "Z" and v == 0:
            continue
        p = sub(b, "position", {"coordinate": c})
        p.text = f"{v:.10f}"
    sub(b, "jumpPosition", {"interpolationLength": f"{interpolation:.5f}"}, text="1")

def build_axml(obj_tracks, duration_sec):
    adm = ET.Element("ebuCoreMain", {
        "xmlns": NS, "xmlns:xsi": XSI,
        "xsi:schemaLocation": f"{NS} ebucore.xsd", "lang": "en"})
    core = sub(adm, "coreMetadata")
    fmt = sub(core, "format")
    af = sub(fmt, "audioFormatExtended")

    prog = sub(af, "audioProgramme", {
        "audioProgrammeID": "APR_1001", "audioProgrammeName": "EAC3JOC_Export",
        "start": ts(0), "end": ts(duration_sec)})
    add_refs(prog, "audioContentIDRef", ("ACO_1001", "ACO_1002"))
    bc = sub(af, "audioContent", {"audioContentID": "ACO_1001",
                                  "audioContentName": "EAC3JOC_Master_Content"})
    add_refs(bc, "audioObjectIDRef", ["AO_1001"])
    sub(bc, "dialogue", {"mixedContentKind": "0"})
    oc = sub(af, "audioContent", {"audioContentID": "ACO_1002",
                                  "audioContentName": "Objects"})
    add_refs(oc, "audioObjectIDRef", ["AO_%04x" % (0x100b + i) for i in range(N_OBJ)])
    sub(oc, "dialogue", {"mixedContentKind": "0"})

    bed_o = sub(af, "audioObject", {"audioObjectID": "AO_1001", "audioObjectName": "Bed",
                                    "start": ts(0), "duration": ts(duration_sec)})
    sub(bed_o, "audioPackFormatIDRef", text="AP_00011001")
    add_refs(bed_o, "audioTrackUIDRef", ["ATU_%08x" % (i + 1) for i in range(10)])
    for i in range(N_OBJ):
        o = sub(af, "audioObject", {"audioObjectID": "AO_%04x" % (0x100b + i),
                                    "audioObjectName": f"Audio Object {i+1}",
                                    "start": ts(0), "duration": ts(duration_sec)})
        sub(o, "audioPackFormatIDRef", text="AP_0003%04x" % (0x1001 + i))
        add_refs(o, "audioTrackUIDRef", ["ATU_%08x" % (i + 11)])

    bp = sub(af, "audioPackFormat", {"audioPackFormatID": "AP_00011001",
                                     "audioPackFormatName": "EAC3JOCBedPack",
                                     "typeDefinition": "DirectSpeakers", "typeLabel": "0001"})
    add_refs(bp, "audioChannelFormatIDRef", ["AC_0001%04x" % (0x1001 + i) for i in range(10)])
    for i in range(N_OBJ):
        pk = sub(af, "audioPackFormat", {"audioPackFormatID": "AP_0003%04x" % (0x1001 + i),
                                         "audioPackFormatName": f"JOC_Object_{i+1}",
                                         "typeDefinition": "Objects", "typeLabel": "0003"})
        add_refs(pk, "audioChannelFormatIDRef", ["AC_0003%04x" % (0x1001 + i)])

    for i in range(10):
        cf = sub(af, "audioChannelFormat", {"audioChannelFormatID": "AC_0001%04x" % (0x1001 + i),
                                            "audioChannelFormatName": BED_NAMES[i],
                                            "typeDefinition": "DirectSpeakers", "typeLabel": "0001"})
        b = sub(cf, "audioBlockFormat", {"audioBlockFormatID": "AB_0001%04x_00000001" % (0x1001 + i)})
        sub(b, "cartesian", text="1")
        x, y, z = BED_POS[i]
        for c, v in (("X", x), ("Y", y), ("Z", z)):
            if c == "Z" and v == 0:
                continue
            p = sub(b, "position", {"coordinate": c})
            p.text = f"{v:.10f}"
        sub(b, "speakerLabel", text=BED_LABELS[i])

    for i, (oname, kfs) in enumerate(obj_tracks):
        cf = sub(af, "audioChannelFormat", {"audioChannelFormatID": "AC_0003%04x" % (0x1001 + i),
                                            "audioChannelFormatName": oname,
                                            "typeDefinition": "Objects", "typeLabel": "0003"})
        for k, keyframe in enumerate(kfs):
            t, x, y, z, dur = keyframe[:5]
            interpolation = keyframe[5] if len(keyframe) > 5 else 0.0
            obj_block(cf, "AB_0003%04x_%08x" % (0x1001 + i, k + 1),
                      t, x, y, z, dur, interpolation)

    for i in range(10):
        t = sub(af, "audioTrackUID", {"UID": "ATU_%08x" % (i + 1),
                                      "bitDepth": "24", "sampleRate": "48000"})
        sub(t, "audioTrackFormatIDRef", text="AT_0001%04x_01" % (0x1001 + i))
        sub(t, "audioPackFormatIDRef", text="AP_00011001")
    for i in range(N_OBJ):
        t = sub(af, "audioTrackUID", {"UID": "ATU_%08x" % (i + 11),
                                      "bitDepth": "24", "sampleRate": "48000"})
        sub(t, "audioTrackFormatIDRef", text="AT_0003%04x_01" % (0x1001 + i))
        sub(t, "audioPackFormatIDRef", text="AP_0003%04x" % (0x1001 + i))

    for i in range(10):
        tf = sub(af, "audioTrackFormat", {"audioTrackFormatID": "AT_0001%04x_01" % (0x1001 + i),
                                          "audioTrackFormatName": "PCM_" + BED_NAMES[i],
                                          "formatDefinition": "PCM", "formatLabel": "0001"})
        sub(tf, "audioStreamFormatIDRef", text="AS_0001%04x" % (0x1001 + i))
    for i in range(N_OBJ):
        tf = sub(af, "audioTrackFormat", {"audioTrackFormatID": "AT_0003%04x_01" % (0x1001 + i),
                                          "audioTrackFormatName": "PCM_JOC_Object_%d" % (i + 1),
                                          "formatDefinition": "PCM", "formatLabel": "0001"})
        sub(tf, "audioStreamFormatIDRef", text="AS_0003%04x" % (0x1001 + i))

    for i in range(10):
        sf = sub(af, "audioStreamFormat", {"audioStreamFormatID": "AS_0001%04x" % (0x1001 + i),
                                           "audioStreamFormatName": "PCM_" + BED_NAMES[i],
                                           "formatDefinition": "PCM", "formatLabel": "0001"})
        sub(sf, "audioChannelFormatIDRef", text="AC_0001%04x" % (0x1001 + i))
        sub(sf, "audioPackFormatIDRef", text="AP_00011001")
        sub(sf, "audioTrackFormatIDRef", text="AT_0001%04x_01" % (0x1001 + i))
    for i in range(N_OBJ):
        sf = sub(af, "audioStreamFormat", {"audioStreamFormatID": "AS_0003%04x" % (0x1001 + i),
                                           "audioStreamFormatName": "PCM_JOC_Object_%d" % (i + 1),
                                           "formatDefinition": "PCM", "formatLabel": "0001"})
        sub(sf, "audioChannelFormatIDRef", text="AC_0003%04x" % (0x1001 + i))
        sub(sf, "audioPackFormatIDRef", text="AP_0003%04x" % (0x1001 + i))
        sub(sf, "audioTrackFormatIDRef", text="AT_0003%04x_01" % (0x1001 + i))

    return ET.tostring(adm, encoding="utf-8", xml_declaration=True)

def build_chna():
    out = bytearray()
    out += struct.pack("<HH", 25, 25)
    for i in range(10):
        out += struct.pack("<H", i + 1)
        out += ("ATU_%08x" % (i + 1)).encode()
        out += ("AT_0001%04x_01" % (0x1001 + i)).encode()
        out += b"AP_00011001" + b"\x00"
    for i in range(N_OBJ):
        out += struct.pack("<H", i + 11)
        out += ("ATU_%08x" % (i + 11)).encode()
        out += ("AT_0003%04x_01" % (0x1001 + i)).encode()
        out += ("AP_0003%04x" % (0x1001 + i)).encode() + b"\x00"
    return bytes(out)

def _checksum(seg):
    s = len(seg)
    for b in seg:
        s += b
    return (~s + 1) & 0xFF

def build_dbmd(object_count=25, joc_binaural_mode=4):
    """仅覆盖 segment 10 中 JOC object slots 10..24 的 mode 低 3 bit。"""
    mode = operator.index(joc_binaural_mode)
    if mode not in JOC_BINAURAL_MODES.values():
        raise ValueError(f"invalid JOC binaural render mode: {mode}")
    out = bytearray(struct.pack("<I", 0x01000006))
    dd = bytearray(96)
    dd[1] = 0x47
    dd[5] = 0x60
    dd[8] = 0x24; dd[9] = 0x24
    out.append(7); out += struct.pack("<H", 96); out += bytes(dd)
    out.append(_checksum(dd))
    at = bytearray(248)
    c0 = b"Created with EAC3JOC"; c1 = b"EAC3JOC Python Renderer"
    at[0:len(c0)] = c0
    at[32:32 + len(c1)] = c1
    at[96], at[97], at[98] = 2, 1, 0
    at[103] = 0x03
    at[106] = 0x01
    at[111] = 0x22; at[112] = 0xFF
    out.append(9); out += struct.pack("<H", 248); out += bytes(at)
    out.append(_checksum(at))
    ob = bytearray(5 + 262 + object_count)
    ob[0:4] = struct.pack("<I", 0xF8726FBD)
    ob[4] = object_count
    for i in range(5 + 262, len(ob)):
        ob[i] = 0x84
    # sync (4), count (2), reserved (1), nine 15-byte config trims,
    # then one trim-bypass byte per track before the headphone modes.
    # Preserve the existing template's bed fields and trailing bytes.
    object_modes = 4 + 2 + 1 + 9 * 15 + object_count
    for i in range(10, min(object_count, 10 + N_OBJ)):
        ob[object_modes + i] = (ob[object_modes + i] & 0xF8) | mode
    out.append(10); out += struct.pack("<H", len(ob)); out += bytes(ob)
    out.append(_checksum(ob))
    out += b"\x00\x00"
    return bytes(out)

class Sink25:
    def __init__(self, path, channels, rate):
        self.ch = channels; self.rate = rate; self.frames = 0
        self.fp = open(path, "wb+")
        self.fp.write(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE")
        self._chunk(b"ds64", b"\x00" * 64)
        self._chunk(b"fmt ", self._fmt())
        self._chunk(b"data", b"")
    def _chunk(self, cid, body):
        self.fp.write(cid + struct.pack("<I", len(body)) + body)
        if len(body) & 1:
            self.fp.write(b"\x00")
    def _fmt(self):
        return struct.pack("<HHIIHH", 1, self.ch, self.rate,
                           self.rate * self.ch * 3, self.ch * 3, 24)
    def write_block(self, arr):
        arr = arr.reshape(-1, self.ch)
        i24 = (np.clip(arr, -1.0, 1.0) * 8388607.0).astype(np.int32)
        self.fp.write(i24.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())
        self.frames += arr.shape[0]
    def finalize(self, axml_bytes, chna_bytes, dbmd_bytes):
        data_len = self.frames * self.ch * 3
        self._chunk(b"axml", axml_bytes)
        self._chunk(b"chna", chna_bytes)
        self._chunk(b"dbmd", dbmd_bytes)
        self.fp.seek(0, 2); total = self.fp.tell()
        self.fp.seek(0); head = self.fp.read()
        m = head.find(b"data")
        if m >= 0:
            self.fp.seek(m + 4); self.fp.write(struct.pack("<I", data_len))
        m = head.find(b"ds64")
        if m >= 0:
            self.fp.seek(m + 8)
            self.fp.write(struct.pack("<QQQI", total - 8, data_len, self.frames, 0))
        self.fp.flush()
        self.fp.close()

def build_master(out_path, bed_mm, obj_mm, kf_tracks, duration_sec, rate=48000,
                 block=480000, joc_binaural_mode=4):
    n = min(bed_mm.shape[0], obj_mm.shape[0])
    try:
        from . import adm_serializer
    except ImportError:
        import adm_serializer
    serial_axml = adm_serializer.build_axml
    axml = serial_axml(kf_tracks, duration_sec)
    chna = build_chna()
    dbmd = build_dbmd(25, joc_binaural_mode=joc_binaural_mode)
    sink = Sink25(out_path, 25, rate)
    for st in range(0, n, block):
        en = min(n, st + block)
        blk = np.hstack((np.asarray(bed_mm[st:en], dtype=np.float32),
                         np.asarray(obj_mm[st:en, 1:16], dtype=np.float32)))
        sink.write_block(blk)
    sink.finalize(axml, chna, dbmd)
    print(f"master25 -> {out_path} ({duration_sec:.2f}s, 25ch, axml={len(axml)}B, "
          f"chna={len(chna)}B, dbmd={len(dbmd)}B)")
