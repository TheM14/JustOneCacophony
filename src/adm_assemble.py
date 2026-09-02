"""把 LFE、15 路对象 PCM 和对象轨迹组装为 ADM BWF。

固定输出契约：
  EAC3JOC 重放输出 = 16ch（ch0 = LFE + ch1-15 = 15 对象）；
  最终 ADM BWF = 7.1.2 bed（L R C Ls Rs Lb Rb + LFE + Ltf Rtf = 10ch）
  —— 除 LFE 外全部静音；
  15 对象 = ch1-15 直接填充对象轨；轨迹 = OAMD（q1/q2/q3 → xyz）。
"""
import os

import numpy as np

import adm_atmos


def assemble_from_raw(raw16_path, out_path, scale=1.0, kf_tracks=None,
                      duration_sec=None, rate=48000, joc_binaural_mode=4):
    """16ch f32 交织 raw → 25ch ADM BWF（空 7.1.2 bed + LFE + 15 对象）。

    raw16: (n, 16) 交织（ch0 = LFE，ch1-15 = 对象）。
    scale: 1.0 = 默认 0 dB，不附加输出缩放。该参数与 joc_clipgain 无关；
           主命令行已在渲染阶段应用用户增益，因此这里传 1.0。
    kf_tracks: 可选轨迹关键帧（OAMD 输出，格式 [(obj_id, [(t, x, y, z), ...]), ...]）；
               缺省 = 静止参考位置（adm_atmos 默认）。
    """
    raw = np.memmap(raw16_path, dtype=np.float32, mode="r")
    n = len(raw) // 16
    raw = raw[:n * 16].reshape(-1, 16)
    if duration_sec is None:
        duration_sec = n / rate
    # 惰性视图：adm_atmos 按块读取，避免全片 25ch 在内存中展开。
    class BedView:
        shape = (n, 10)

        def __getitem__(self, key):
            src = np.asarray(raw[key], dtype=np.float32)
            one = src.ndim == 1
            if one:
                src = src[None, :]
            out = np.zeros((len(src), 10), dtype=np.float32)
            out[:, 3] = np.multiply(src[:, 0], np.float32(scale), dtype=np.float32)
            return out[0] if one else out

    class ObjView:
        shape = (n, 16)

        def __getitem__(self, key):
            return np.multiply(np.asarray(raw[key], dtype=np.float32),
                               np.float32(scale), dtype=np.float32)
    if kf_tracks is None:
        kf_tracks = []
        for oi in range(15):
            # 静止参考位置（q1=q2=q3=0 → 原点；实际坐标按 OAMD 输出填入）
            kf_tracks.append(("JOC_Object_%d" % (oi + 1),
                              [(0.0, 0.0, 0.0, 0.0, max(duration_sec, 1e-6))]))
    adm_atmos.build_master(out_path, BedView(), ObjView(), kf_tracks,
                           duration_sec, rate=rate,
                           joc_binaural_mode=joc_binaural_mode)
    # 及时释放 Windows 文件句柄，允许 TemporaryDirectory 删除中间 raw。
    raw._mmap.close()
    return out_path


class StreamingMaster:
    """Incrementally write renderer frames into the final 25-channel ADM BWF.

    This removes the default 16-channel float32 intermediate file. The mapping
    remains identical to :func:`assemble_from_raw`: bed channel 3 receives LFE,
    bed channels 0..2/4..9 are silent, and output objects 1..15 map to ADM
    channels 10..24.
    """

    def __init__(self, out_path, duration_sec, rate=48000, block_samples=131072,
                 joc_binaural_mode=4):
        if block_samples < 1536:
            raise ValueError("block_samples must be at least one E-AC-3 frame")
        self.out_path = os.fspath(out_path)
        self.duration_sec = float(duration_sec)
        self.rate = int(rate)
        self.joc_binaural_mode = joc_binaural_mode
        self._sink = adm_atmos.Sink25(self.out_path, 25, self.rate)
        self._buffer = np.empty((int(block_samples), 25), dtype=np.float32)
        self._used = 0
        self._finalized = False

    def _flush(self):
        if self._used:
            self._sink.write_block(self._buffer[:self._used])
            self._used = 0

    def write_frame(self, pcm16):
        pcm = np.asarray(pcm16, dtype=np.float32)
        if pcm.shape != (16, 1536):
            raise ValueError(f"renderer frame must be (16,1536), got {pcm.shape}")
        source = 0
        while source < 1536:
            available = len(self._buffer) - self._used
            count = min(available, 1536 - source)
            target = self._buffer[self._used:self._used + count]
            target.fill(0.0)
            target[:, 3] = pcm[0, source:source + count]
            target[:, 10:25] = pcm[1:16, source:source + count].T
            self._used += count
            source += count
            if self._used == len(self._buffer):
                self._flush()

    def finalize(self, kf_tracks):
        if self._finalized:
            raise RuntimeError("StreamingMaster already finalized")
        self._flush()
        try:
            from . import adm_serializer
        except ImportError:
            import adm_serializer
        axml = adm_serializer.build_axml(kf_tracks, self.duration_sec)
        chna = adm_atmos.build_chna()
        dbmd = adm_atmos.build_dbmd(
            25, joc_binaural_mode=self.joc_binaural_mode)
        trajectory_blocks = sum(len(track[1]) for track in kf_tracks)
        self.metadata_info = {
            "axml_bytes": len(axml),
            "trajectory_blocks": trajectory_blocks,
            "chna_bytes": len(chna),
            "dbmd_bytes": len(dbmd),
        }
        self._sink.finalize(axml, chna, dbmd)
        self._finalized = True
        print(f"master25 -> {self.out_path} ({self.duration_sec:.2f}s, 25ch, "
              f"axml={len(axml)}B, chna={len(chna)}B, dbmd={len(dbmd)}B)")
        return self.out_path

    def abort(self):
        if self._finalized:
            return
        sink = getattr(self, "_sink", None)
        fp = getattr(sink, "fp", None)
        if fp is not None and not fp.closed:
            fp.close()

    def __del__(self):
        try:
            self.abort()
        except Exception:
            pass
