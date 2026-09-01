"""校验 ADM BWF 的 RF64、通道、axml、chna、dbmd 和对象引用结构。

用法：``python src/adm_validate.py <file.wav> [more.wav ...]``，全部通过时退出码为 0。
"""
import struct, sys, re, os

def fail(msgs, m): msgs.append(m)

def walk_chunks(path):
    chunks, ds64 = [], {}
    with open(path, "rb") as f:
        riff = f.read(4); f.read(4); wave = f.read(4)
        if riff not in (b"RIFF", b"RF64"):
            return None, None, f"File does not have a 'RIFF' or 'RF64' chunk"
        if wave != b"WAVE":
            return None, None, "File does not have a required 'WAVE' chunk"
        while True:
            off = f.tell()
            cid = f.read(4)
            if len(cid) < 4: break
            sz = struct.unpack("<I", f.read(4))[0]
            if cid == b"ds64":
                body = f.read(sz + (sz & 1))
                riff64, data64, sample64, _ = struct.unpack("<QQQI", body[:28])
                ds64 = dict(riff64=riff64, data64=data64, sample64=sample64)
                chunks.append(("ds64", off, sz)); continue
            chunks.append((cid.decode("latin1"), off, sz))
            eff = ds64.get("data64", sz) if (sz == 0xFFFFFFFF and cid == b"data") else sz
            f.seek(off + 8 + eff + (eff & 1))
    return chunks, ds64, None

def read_body(path, chunks, cid):
    for c, off, sz in chunks:
        if c == cid:
            with open(path, "rb") as f:
                f.seek(off + 8)
                return f.read(sz)
    return None

def parse_chna(body):
    n_track, n_uid = struct.unpack("<HH", body[:4])
    rows, p = [], 4
    while p + 40 <= len(body):
        trk = struct.unpack("<H", body[p:p+2])[0]
        uid = body[p+2:p+14].rstrip(b"\x00").decode()
        tf  = body[p+14:p+28].rstrip(b"\x00").decode()
        pk  = body[p+28:p+40].rstrip(b"\x00").decode()
        rows.append((trk, uid, tf, pk)); p += 40
    return n_track, n_uid, rows

def decode_channel_input(ao_id):
    """将 ``AO_xxxx`` 的十六进制标识符解码为低 12 位通道输入号。"""
    m = re.fullmatch(r"AO_([0-9a-fA-F]+)", ao_id)
    if not m:
        return None
    v = int(m.group(1), 16)
    if v > 0x1FFF:            # 超过 12 位通道域
        return None
    return v & 0x0FFF

def ts_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)

def validate(path, axml_override=None, chna_override=None):
    msgs = []
    chunks, ds64, err = walk_chunks(path)
    if err:
        return [err]
    have = {c for c, _, _ in chunks}
    for need in ("fmt ", "data", "axml", "chna", "dbmd"):
        if need not in have:
            fail(msgs, f"File does not have a required '{need.strip()}' chunk")
    if msgs:
        return msgs
    fmt = read_body(path, chunks, "fmt ")
    f_tag, f_ch, f_rate, _, _, f_bits = struct.unpack("<HHIIHH", fmt[:16])
    chna_body = chna_override if chna_override is not None else read_body(path, chunks, "chna")
    n_track, n_uid, rows = parse_chna(chna_body)
    if f_ch != n_track:
        fail(msgs, f"Mismatched number of audio channels and chna entries "
                   f"(fmt={f_ch} chna={n_track})")
    ax_raw = axml_override if axml_override is not None else read_body(path, chunks, "axml")
    ax = ax_raw.decode("utf-8")

    # --- audioObjectID 十六进制通道输入号解码 ---
    objs = re.findall(r'audioObjectID="(AO_[0-9a-zA-Z]+)"', ax)
    bed_ch, obj_ch = [], []
    for ao in objs:
        cid = decode_channel_input(ao)
        if cid is None:
            fail(msgs, f"Invalid ADM BWF XML format: cannot decode channel "
                       f"input ID from AudioObjectID '{ao}'")
            continue
        if ao == "AO_1001":
            bed_ch.append(cid)
        else:
            if cid <= 10:
                fail(msgs, f"Source channel index should be greater than 10 "
                           f"for objects ('{ao}' -> {cid})")
            obj_ch.append(cid)

    # UID 十六进制 → 必须与 chna 表一致
    uid_map = {uid: trk for trk, uid, tf, pk in rows}
    for uid in re.findall(r'UID="(ATU_[0-9a-zA-Z]+)"', ax):
        if uid not in uid_map:
            fail(msgs, f"'{uid}' is not referenced in 'chna' chunk UID table")
            continue
        m = re.fullmatch(r"ATU_([0-9a-fA-F]+)", uid)
        if m:
            v = int(m.group(1), 16)
            if v > 128:
                fail(msgs, f"Channel index out of range (UID {uid} -> {v})")

    # 轨数一致性：axml audioTrackUID 数 == fmt 声道数
    n_tu = len(re.findall(r"<audioTrackUID ", ax))
    if n_tu != f_ch:
        fail(msgs, f"Number of channels declared in ADM ({n_tu}) does not "
                   f"match 'fmt ' chunk ({f_ch})")

    # sampleRate / bitDepth 一致
    for sr in set(re.findall(r'sampleRate="(\d+)"', ax)):
        if int(sr) != f_rate:
            fail(msgs, f"Mismatched track sample rate between ADM and WAV ({sr} vs {f_rate})")
    for bd in set(re.findall(r'bitDepth="(\d+)"', ax)):
        if int(bd) != f_bits:
            fail(msgs, f"Mismatched track bit depth between ADM and WAV ({bd} vs {f_bits})")

    # audioProgramme 唯一性 / audioContent ≥1
    if ax.count("<audioProgramme ") != 1:
        fail(msgs, "ADM has more than one audioProgramme object -- there must be only one"
             if ax.count("<audioProgramme ") > 1
             else "ADM does not have a required audioProgramme object")
    if "<audioContent " not in ax:
        fail(msgs, "audioProgramme object does not have a required audioContent object")

    # 每个 channelFormat ≥1 blockFormat + 对象块链连续性
    cfs = re.findall(r'<audioChannelFormat [^>]*typeLabel="0003".*?</audioChannelFormat>', ax, re.S)
    object_block_formats = 0
    for seg in cfs:
        cf_id = re.search(r'audioChannelFormatID="([^"]+)"', seg).group(1)
        block_xml = re.findall(r'<audioBlockFormat [^>]*rtime="[^"]+".*?</audioBlockFormat>',
                               seg, re.S)
        object_block_formats += len(block_xml)
        blocks = []
        for block_index, block in enumerate(block_xml, 1):
            timing = re.search(r'rtime="([^"]+)" duration="([^"]+)"', block)
            if timing is None:
                continue
            rtime, duration = timing.groups()
            blocks.append((rtime, duration))
            jump = re.search(
                r'<jumpPosition interpolationLength="([^"]+)">1</jumpPosition>', block)
            if jump is not None and float(jump.group(1)) > ts_sec(duration) + 1e-8:
                fail(msgs, f"Interpolation length exceeds duration in block format "
                           f"{block_index} of {cf_id}: {jump.group(1)} > {duration}")
        if not blocks:
            fail(msgs, f"AudioChannelFormat {cf_id} is missing audioBlockFormat sub-element")
            continue
        for i in range(len(blocks) - 1):
            end_i = ts_sec(blocks[i][0]) + ts_sec(blocks[i][1])
            nxt = ts_sec(blocks[i + 1][0])
            if abs(end_i - nxt) > 2e-5:
                fail(msgs, f"Time gap between block format {i+1} and {i+2} of {cf_id}: "
                           f"{end_i:.5f} vs {nxt:.5f}")
    return msgs, dict(fmt_ch=f_ch, fmt_rate=f_rate, fmt_bits=f_bits,
                      chna=n_track, objects=len(obj_ch), bed=len(bed_ch),
                      trackUIDs=n_tu, axml_bytes=len(ax_raw),
                      audioBlockFormats=ax.count("<audioBlockFormat "),
                      objectBlockFormats=object_block_formats)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--axml-file", default=None, help="用该文件内容替换 wav 内 axml（对照实验）")
    ap.add_argument("--chna-file", default=None, help="用该文件内容替换 wav 内 chna（对照实验）")
    a = ap.parse_args()
    ax_o = open(a.axml_file, "rb").read() if a.axml_file else None
    ch_o = open(a.chna_file, "rb").read() if a.chna_file else None
    rc = 0
    for p in a.files:
        r = validate(p, axml_override=ax_o, chna_override=ch_o)
        name = os.path.basename(p)
        if isinstance(r, list):
            msgs, info = r, {}
        else:
            msgs, info = r
        if msgs:
            rc = 1
            print(f"[FAIL] {name}")
            for m in msgs:
                print("   -", m)
        else:
            print(f"[PASS] {name}  {info}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
