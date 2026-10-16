"""把 E-AC-3 核心 PCM 和 JOC 元数据重建为 LFE 加 15 路对象 PCM。

管线（每帧）：
  核心 PCM（N=5 或 7 路全频段通道，按 k 查表）→ QMF 分析 x → 填充器 z = Σ m·x（每对象）
  → 对象时域组装（前步、64 点 FFT、旋转、合成窗）→ ×16 + clamp
  → 对象波形 × joc_clipgain
  + LFE 专用 1217-sample 环形延迟
  → 16ch（ch0 = LFE, ch1-15 = 15 对象）→ ×output_scale

output_scale:
  默认 1.0（0 dB）。用户选择的 dB 在命令行转换为 float32 系数；该系数
  不进入 JOC 数学本体，也不是 joc_clipgain。
"""
import numpy as np

from joc_decode import SIGMA_DMX_CONFIGS, parse_joc, diff_decode, dequantize
from joc_qmf import (N, QMF5_WINDOW, qmf_analysis_frame,
                     surround_post_frame, interp_matrix)
from evo_unpack import unpack_evolution

CORE_CHANNELS = [0, 1, 2, 4, 5]  # L R C Ls Rs（EAC3 5.1 核心顺序）


class JocRenderer:
    """E-AC-3 JOC 帧渲染器。状态跨帧保持（FIFO/插值 prev/合成状态）。"""

    def __init__(self, output_scale=1.0):
        # 成品增益明确按 float32 运算；默认值为 1.0。
        self.output_scale = np.float32(output_scale)
        self._prev = {}           # 每对象插值 prev（m 的帧末 sb 值）
        self._synthesis_state = np.zeros((15, 640), dtype=np.float64)
        # 分析 QMF：最多七路，各保留 9×64 FIFO。L/R/C 与 N=7 时的第 6、7 路
        # （额外那一对）在这里做 10 时隙延迟；Ls/Rs 的同一延迟在环绕通路内完成。
        self._analysis_fifo = np.zeros((7, 9, 64), dtype=np.float64)
        self._analysis_delay = np.zeros((5, 10, 64), dtype=np.float32)
        # 哨兵：复位后（或整条流的第一帧）直接用新标量，不做帧内过渡。
        self._analysis_phase = np.float32(-0.5)
        self._surround_qmf_delay = np.zeros((2, 10, 64), dtype=np.complex128)
        self._surround_dc_hist = np.zeros((2, 20), dtype=np.complex128)
        # ch0/LFE 使用循环历史，读指针恒落后写指针 1217 个采样。
        # phase=1/16 与输出端 *16 抵消，净效果是 LFE 延迟。
        self._lfe_delay = np.zeros(1217, dtype=np.float64)
        self._last_x = None       # 仅供逐槽诊断读取，不参与状态推进
        self._rot = None          # 每带旋转表

    def held_analysis_scalar(self):
        """当前分析端标量；还没有过 JOC 帧时用结构常数 1/16。"""
        value = self._analysis_phase
        if value <= 0.0 and value != 0.0:      # 哨兵
            return np.float32(0.0625)
        return value

    @staticmethod
    def decode_payload(payload_bytes):
        """id14 载荷 → (parse 结果, mix_q, mix_dq)。"""
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        out = parse_joc(subs[14])
        mix_q = diff_decode(out)
        mix_dq = dequantize(out, mix_q)
        return out, mix_q, mix_dq

    @staticmethod
    def decode_subpayloads(subs):
        """已解析 EMDF 子载荷 → (parse 结果, mix_q, mix_dq)。"""
        if 14 not in subs:
            raise ValueError("EMDF 缺少 ID14/JOC")
        out = parse_joc(subs[14])
        mix_q = diff_decode(out)
        mix_dq = dequantize(out, mix_q)
        return out, mix_q, mix_dq

    def qmf_x(self, bed_pcm, phase_new=0.0625, apply_dc_filter=True, rotate=True):
        """核心 N ch PCM → 对象矩阵使用的复数 QMF ``x``。

        phase_new 是分析端标量：结构常数 1/16 × 本帧 joc_clipgain。先以 float32
        对当前帧应用它；相比上一帧变化时仅前 256 个样本从旧值线性过渡。缩放后的
        L/R/C 与 N=7 时的第 6、7 路延迟 10 槽，Ls/Rs 不延迟，再进入分析 QMF；
        出口的 ×16 抵消 1/16，净效果是 ×joc_clipgain。``apply_dc_filter`` 与 ``rotate`` 是规格 §4 的同一个
        σ(k) 开关，控制 Ls/Rs 的 band-0 21-tap DC 补偿与 -j 旋转。
        """
        pcm = np.asarray(bed_pcm, dtype=np.float32)
        n_channels = pcm.shape[0] if pcm.ndim == 2 else 0
        if n_channels not in (5, 7) or pcm.shape != (n_channels, 1536):
            raise ValueError(f"核心帧应为 (5,1536) 或 (7,1536)，实际 {pcm.shape}")
        new_phase = np.float32(phase_new)
        old_phase = self._analysis_phase
        if old_phase <= 0.0 and old_phase != 0.0:      # 哨兵：首帧/复位后不做过过渡
            old_phase = new_phase
        gains = np.full(1536, new_phase, dtype=np.float32)
        if old_phase != new_phase:
            step = np.float32((new_phase - old_phase) / np.float32(256.0))
            gains[:256] = old_phase + np.arange(256, dtype=np.float32) * step
        scaled = np.multiply(pcm, gains[None, :], dtype=np.float32).reshape(
            n_channels, 24, 64)

        # 走外部延迟线的通道：前级 L/R/C，以及 N=7 时额外那一对（索引 5、6）。
        plain = [0, 1, 2] + ([5, 6] if n_channels == 7 else [])
        blocks = np.empty_like(scaled)
        for lane, ch in enumerate(plain):
            delayed = np.concatenate((self._analysis_delay[lane], scaled[ch]), axis=0)
            blocks[ch] = delayed[:24]
            self._analysis_delay[lane] = delayed[24:]
        blocks[3:5] = scaled[3:5]

        fifo = self._analysis_fifo[:n_channels]
        x, self._analysis_fifo[:n_channels] = qmf_analysis_frame(fifo, blocks)
        self._analysis_phase = new_phase
        x[3:5], self._surround_qmf_delay, self._surround_dc_hist = surround_post_frame(
            x[3:5], self._surround_qmf_delay, self._surround_dc_hist,
            apply_dc_filter=apply_dc_filter, rotate=rotate)
        return x

    def object_z(self, out, mix_dq, x):
        """计算 ``z[obj] = Σ_ch m[ch,sb,ts]·x[ch,sb,ts]``。"""
        z_all = {}
        new_prev = {}
        for obj, o in enumerate(out["objs"]):
            if not o["present"]:
                continue
            dq = mix_dq[obj]
            prev = self._prev.get(obj)
            if prev is None:
                prev = np.zeros((out["n_channels"], N), dtype=np.float64)
            m = interp_matrix(o, dq, prev)          # [ch][sb][ts]（内部按 o["n_bands"] 映射）
            z = np.sum(x * m, axis=0)               # [sb][ts]（复）
            z_all[obj] = z
            new_prev[obj] = m[:, :, -1]
        self._prev.update(new_prev)
        return z_all

    def _rot_table_86840(self):
        """生成 ``θ=πk/128`` 的 ``0.5·(sin θ, cos θ)`` 旋转表。"""
        if self._rot is None:
            k = np.arange(64)
            theta = np.pi * k / 128.0
            self._rot = np.empty(128, dtype=np.float64)
            self._rot[0::2] = 0.5 * np.sin(theta)   # sin 分量
            self._rot[1::2] = 0.5 * np.cos(theta)   # cos 分量
        return self._rot

    @staticmethod
    def _front_step(src):
        """重排 128 个交织复数分量：
        outA[2k] = src[4k]、outA[2k+1] = −src[4k+1]（区 [0:64]）；
        outB[126−2k] = src[4k+2]、outB[127−2k] = src[4k+3]（区 [64:128] 倒序）。"""
        src = np.asarray(src, dtype=np.float64)
        zone = np.empty_like(src)
        k = np.arange(32)
        zone[..., 2 * k] = src[..., 4 * k]
        zone[..., 2 * k + 1] = -src[..., 4 * k + 1]
        zone[..., 126 - 2 * k] = src[..., 4 * k + 2]
        zone[..., 127 - 2 * k] = src[..., 4 * k + 3]
        return zone

    @staticmethod
    def _h880_vec(a2):
        """对 128 个 re/im 交织值执行标准 64 点复数 FFT。"""
        a2 = np.asarray(a2, dtype=np.float64)
        zc = a2[..., 0::2] + 1j * a2[..., 1::2]
        f = np.fft.fft(zc, axis=-1)
        a1 = np.empty_like(a2)
        a1[..., 0::2] = f.real
        a1[..., 1::2] = f.imag
        return a1

    @staticmethod
    def _h0b0_vec(a2, a3):
        """NumPy 向量化的 ``out = 2·复乘(a2, a3)``，re/im 交织。"""
        a2 = np.asarray(a2, dtype=np.float64)
        a3 = np.asarray(a3, dtype=np.float64)
        shape = a2.shape[:-1] + (16, 4)
        v11 = a2[..., 1::2].reshape(shape)
        v12 = a2[..., 0::2].reshape(shape)
        v13 = a3[..., 0::2].reshape(a3.shape[:-1] + (16, 4))
        v14 = a3[..., 1::2].reshape(a3.shape[:-1] + (16, 4))
        v15 = (v11 * v14 - v12 * v13) * 2.0
        v16 = (v12 * v14 + v11 * v13) * 2.0
        out = np.empty_like(a2)
        out[..., 0::2] = v16.reshape(a2.shape[:-1] + (64,))
        out[..., 1::2] = v15.reshape(a2.shape[:-1] + (64,))
        return out

    @staticmethod
    def _qmf5_vec(state, win_flat, rot):
        """推进合成窗状态并返回 64 个时域样本。"""
        state = np.asarray(state, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)
        single = state.ndim == 1
        if single:
            state = state[None, :]
        if rot.ndim == 1:
            rot = np.broadcast_to(rot, (len(state), len(rot)))
        rot2 = rot.reshape(len(state), 16, 8)
        v22 = rot2[..., 0:8:2]
        v19 = rot2[..., 1:8:2]
        S = state[:, :576].reshape(len(state), 16, 9, 4)
        w10 = win_flat[:640].reshape(10, 64)
        idx = np.arange(16)[:, None] * 4 + np.arange(4)
        w0 = w10[0, idx][None, ...]
        w_all = w10[1:10, idx].transpose(1, 0, 2)[None, ...]
        out64 = (2.0 * (w0 * v22 + S[:, :, 0])).reshape(len(state), 64)
        # 输出使用窗行 0，状态第 0 行从窗行 1 开始推进。
        S[:, :, 0] = w_all[:, :, 0] * v19 + S[:, :, 1]
        for k in range(7):
            alt = v22 if k % 2 == 0 else v19
            S[:, :, k + 1] = w_all[:, :, k + 1] * alt + S[:, :, k + 2]
        S[:, :, 8] = w_all[:, :, 8] * v19
        return out64[0] if single else out64

    def dll_synth_objects(self, z_all):
        """批量执行对象逆 QMF，返回对象 PCM 字典。

        每个对象保持独立合成状态，64 点 FFT 和窗核在对象维批量计算。
        """
        object_ids = sorted(z_all)
        if not object_ids:
            return {}
        z = np.stack([z_all[obj] for obj in object_ids], axis=0)
        state = self._synthesis_state[object_ids].copy()
        rot868 = self._rot_table_86840()
        pcm = np.zeros((len(object_ids), 1536), dtype=np.float64)
        for tsg in range(6):
            ring = np.zeros((len(object_ids), 512), dtype=np.float64)
            chunk = z[:, :, tsg * 4:tsg * 4 + 4].transpose(0, 2, 1)
            slots = ring.reshape(len(object_ids), 4, 128)
            slots[..., 0::2] = chunk.real
            slots[..., 1::2] = chunk.imag
            for it in range(4):
                zone = self._front_step(ring[:, 128 * it:128 * it + 128])
                zone = self._h0b0_vec(self._h880_vec(zone), rot868)
                ring[:, 64 * it:64 * it + 64] = self._qmf5_vec(state, QMF5_WINDOW, zone)
            pcm[:, tsg * 256:(tsg + 1) * 256] = np.clip(16.0 * ring[:, :256], -1.0, 1.0)
        self._synthesis_state[object_ids] = state
        return {obj: pcm[i] for i, obj in enumerate(object_ids)}

    def dll_synth_object(self, obj, z):
        """单对象兼容入口；整帧渲染使用对象维批量实现。"""
        return self.dll_synth_objects({obj: z})[obj]

    @staticmethod
    def _win_flat():
        """返回逆 QMF 使用的 640 项有效合成窗表。"""
        return QMF5_WINDOW

    def decode_lfe(self, lfe_pcm):
        """将核心 ch3/LFE 经过跨帧 1217-sample 延迟后输出到 ch0。"""
        current = np.asarray(lfe_pcm, dtype=np.float64)
        if current.shape != (1536,):
            raise ValueError(f"LFE 帧应为 (1536,)，实际 {current.shape}")
        delayed = np.concatenate([self._lfe_delay, current])
        out = np.clip(delayed[:1536], -1.0, 1.0)
        self._lfe_delay = delayed[-1217:].copy()
        return out

    def render_missing_frame(self, bed_pcm, lfe_pcm=None, config_idx=3):
        """没有 JOC 元数据的音频帧：对象通道静音，只保留 LFE 旁通。

        QMF 分析与对象合成状态照常推进，否则后续帧的时间轴会整体少一格。
        """
        sigma = config_idx in SIGMA_DMX_CONFIGS
        # 本帧没有 JOC 元数据，分析端标量保持上一帧的值（槽位不被改写）。
        x = self.qmf_x(bed_pcm, phase_new=self.held_analysis_scalar(),
                       apply_dc_filter=sigma, rotate=sigma)
        self._last_x = x.copy()
        z_all = {obj: np.zeros((x.shape[1], x.shape[2]), dtype=np.complex128)
                 for obj in range(15)}
        self.dll_synth_objects(z_all)
        pcm16 = np.zeros((16, 1536), dtype=np.float64)
        if lfe_pcm is not None:
            pcm16[0] = self.decode_lfe(lfe_pcm)
        pcm16_f32 = np.asarray(pcm16, dtype=np.float32)
        return np.multiply(pcm16_f32, self.output_scale, dtype=np.float32), z_all

    def render_frame(self, payload_bytes, bed_pcm, lfe_pcm=None):
        """payload_bytes = evolution 载荷（含 id14 JOC）；bed_pcm = N×1536 核心 PCM。
        lfe_pcm = 1536 核心 LFE；提供时生成 ch0，省略时 ch0 静音
        （保留给只验证对象/z 链的诊断脚本）。
        返回 (pcm16, z_all)：pcm16 = 16×1536（ch0 LFE + 15 对象）f32，
        已用 FP32 系数乘 output_scale。"""
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        return self.render_subpayloads(subs, bed_pcm, lfe_pcm)

    def render_subpayloads(self, subs, bed_pcm, lfe_pcm=None):
        """以已拆出的 EMDF payload 字典渲染一帧，避免绑定 transport 容器。"""
        out, mix_q, mix_dq = self.decode_subpayloads(subs)
        # σ(k) 同时门控 -j 旋转与 band-0 FIR（规格 §4）。
        sigma = out["dmx_config_idx"] in SIGMA_DMX_CONFIGS
        x = self.qmf_x(bed_pcm, phase_new=0.0625 * out["clipgain"],
                       apply_dc_filter=sigma, rotate=sigma)
        self._last_x = x.copy()
        z_all = self.object_z(out, mix_dq, x)
        # joc_clipgain 已在分析端进入 qmf_x，与用户 output_scale 分离。
        objs_pcm = self.dll_synth_objects(z_all)
        pcm16 = np.zeros((16, 1536), dtype=np.float64)
        pcm16[0] = self.decode_lfe(lfe_pcm) if lfe_pcm is not None else 0.0
        for obj, p in objs_pcm.items():
            pcm16[obj + 1] = p
        pcm16_f32 = np.asarray(pcm16, dtype=np.float32)
        return np.multiply(pcm16_f32, self.output_scale, dtype=np.float32), z_all


