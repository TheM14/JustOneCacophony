"""把 E-AC-3 核心 PCM 和 JOC 元数据重建为 LFE 加 15 路对象 PCM。

管线（每帧）：
  核心 PCM（N=5 或 7 路全频段通道，按 k 查表）→ QMF 分析 x → 对象矩阵 z = Σ m·x
  → 对象时域组装（前步、64 点 FFT、旋转、合成窗）→ ×16 + clamp
  → LFE 专用 1217-sample 环形延迟（逐点乘 clipgain 斜坡）
  → 16ch（ch0 = LFE, ch1-15 = 15 对象）→ ×output_scale
  无元数据帧改为 8 路核心 PCM identity 直通（槽 0..7，每路延迟 1217），槽 8..15 为零。

跨帧状态：seq 状态机（state/expected/无效帧计数）、每对象的 prev 矩阵与存在标记、
QMF 分析与合成状态、LFE 延迟环、直通渲染的 8 路核心 PCM 历史。

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
FRAME_SAMPLES = 1536
MAX_OBJECTS = 15
MAX_CORE_CHANNELS = 7
CORE_OUT_CHANNELS = 8
LFE_SLOT = 3
LFE_DELAY = 1217
SEQ_MODULUS = 1024
# 没有 JOC 元数据的帧：状态机短路分支使用这个状态码。
STATUS_NO_METADATA = 8
# 渲染层给出的“核心声道数变化”状态码。
STATUS_CORE_CHANNEL_CHANGE = 9


def next_seq(value):
    """seq 的推进规则：1023 之后回到 1。"""
    return 1 if value == SEQ_MODULUS - 1 else value + 1


class JocRenderer:
    """E-AC-3 JOC 帧渲染器。状态跨帧保持（FIFO/插值 prev/合成状态/状态机）。"""

    def __init__(self, output_scale=1.0):
        # 成品增益明确按 float32 运算；默认值为 1.0。
        self.output_scale = np.float32(output_scale)
        # 每对象插值 prev：上一帧发布的末时槽矩阵与“prev 存在”标记；槽位固定 15 个。
        self._prev_mtx = np.zeros((MAX_OBJECTS, MAX_CORE_CHANNELS, N), dtype=np.float64)
        self._prev_present = np.zeros(MAX_OBJECTS, dtype=bool)
        self._synthesis_state = np.zeros((MAX_OBJECTS, 640), dtype=np.float64)
        # 分析 QMF：最多七路，各保留 9×64 FIFO。L/R/C 与 N=7 时的第 6、7 路
        # （额外那一对）在这里做 10 时隙延迟；Ls/Rs 的同一延迟在环绕通路内完成。
        self._analysis_fifo = np.zeros((MAX_CORE_CHANNELS, 9, 64), dtype=np.float64)
        self._analysis_delay = np.zeros((5, 10, 64), dtype=np.float32)
        # 哨兵：复位后（或整条流的第一帧）直接用新标量，不做帧内过渡。
        self._analysis_phase = np.float32(-0.5)
        self._surround_qmf_delay = np.zeros((2, 10, 64), dtype=np.complex128)
        self._surround_dc_hist = np.zeros((2, 20), dtype=np.complex128)
        # ch0/LFE 使用循环历史，读指针恒落后写指针 1217 个采样。
        # 逐点增益斜坡在写入延迟环之前乘上，读出后不再乘任何增益。
        self._lfe_delay = np.zeros(LFE_DELAY, dtype=np.float64)
        self._lfe_clipgain = None         # LFE 增益斜坡的起点；None = 还没有历史
        # 直通渲染的 8 路核心 PCM 延迟历史：保存上一帧输入。
        self._core_hist = np.zeros((CORE_OUT_CHANNELS, FRAME_SAMPLES), dtype=np.float64)
        # seq 状态机与逐帧分派所需的跨帧状态。
        self._state = 0
        self._expected = 0
        self._invalid_count = 0
        self._last_frame_valid = False
        self._n_channels_prev = None
        self._config_idx = 3
        self._last_x = None               # 仅供逐槽诊断读取，不参与状态推进
        self._rot = None                  # 每带旋转表

    def held_analysis_scalar(self):
        """当前分析端标量；还没有过 JOC 帧时用结构常数 1/16。"""
        value = self._analysis_phase
        if value <= 0.0 and value != 0.0:      # 哨兵
            return np.float32(0.0625)
        return value

    def held_clipgain(self):
        """当前生效的完整 clipgain；没有历史时为 1.0。"""
        return float(self.held_analysis_scalar()) * 16.0

    @staticmethod
    def decode_payload(payload_bytes):
        """id14 载荷 → (parse 结果, mix_q, mix_dq)。"""
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        return JocRenderer.decode_subpayloads(subs)

    @staticmethod
    def decode_subpayloads(subs):
        """已解析 EMDF 子载荷 → (parse 结果, mix_q, mix_dq)。

        整帧状态不是 0 时只返回解析结果：这一帧按无效帧处理，不做矩阵还原。
        """
        if 14 not in subs:
            raise ValueError("EMDF 缺少 ID14/JOC")
        out = parse_joc(subs[14])
        if out["status"] != 0:
            return out, {}, {}
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
        if n_channels not in (5, 7) or pcm.shape != (n_channels, FRAME_SAMPLES):
            raise ValueError(f"核心帧应为 (5,1536) 或 (7,1536)，实际 {pcm.shape}")
        new_phase = np.float32(phase_new)
        old_phase = self._analysis_phase
        if old_phase <= 0.0 and old_phase != 0.0:      # 哨兵：首帧/复位后不做过过渡
            old_phase = new_phase
        gains = np.full(FRAME_SAMPLES, new_phase, dtype=np.float32)
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

    def _clear_unused_channels(self, n_channels):
        """核心声道数变化后，只清除新布局中未使用声道的历史状态。"""
        if n_channels >= MAX_CORE_CHANNELS:
            return
        self._analysis_fifo[n_channels:] = 0.0
        self._prev_mtx[:, n_channels:, :] = 0.0
        if n_channels == 5:
            # N=5 不用额外那一对，10 时隙延迟线 3、4 号槽不再推进。
            self._analysis_delay[3:] = 0.0

    def _maybe_clear_lfe_ring(self, status):
        """声明声道数与实际核心声道数失配（本帧 status=9）时清空 LFE 环。"""
        if status == STATUS_CORE_CHANNEL_CHANGE:
            self._lfe_delay.fill(0.0)

    def _advance_state(self, status, seq):
        """按 seq 推进状态机，返回进入本帧前的 ``state``。

        status：0 正常；2..7 解析层无效（7 = ID14 载荷在 joc_data 内提前结束）；
        8 无元数据；9 核心声道数变化。
        """
        pre_state = self._state
        valid = status in (0, STATUS_CORE_CHANNEL_CHANGE)
        if status == STATUS_NO_METADATA and pre_state != 0:
            # 无元数据帧不推进期望、不比较 seq。
            self._state = 3
            return pre_state
        self._expected = next_seq(self._expected)
        if valid and seq == 0 and pre_state != 0:
            # seq 回到 0：节目重新开始，硬切并清空对象历史。
            self._state = 4
            return pre_state

        state = self._state
        if state == 0:
            if valid:
                state = 1
                self._expected = seq
        elif state == 1:
            if valid:
                if seq != self._expected:
                    state = 4
            else:
                state = 2
                self._invalid_count = 1
        elif state == 2:
            if valid:
                state = 1
                self._expected = seq
            elif self._invalid_count == 5:
                state = 3
            else:
                self._invalid_count += 1
        elif state == 3:
            if valid:
                state = 1
                self._expected = seq
            else:
                self._invalid_count += 1
        elif state == 4:
            if valid:
                state = 1
                self._expected = seq
            else:
                state = 2
                self._invalid_count = 1
        self._state = state
        if status == STATUS_CORE_CHANNEL_CHANGE:
            self._state = 4
        self._last_frame_valid = valid
        return pre_state

    def object_z(self, out, mix_dq, x):
        """计算恢复正常渲染时的 ``z[obj] = Σ_ch m[ch,sb,ts]·x[ch,sb,ts]``。

        对象槽固定 15 个：本帧声明范围内 present=1 的对象按插值规则生成矩阵；
        present 由 1 变 0（或落在声明范围之外）的对象按“present=1 且矩阵全零”
        继续发布，直到它重新 present=1；从未出现过的槽输出零。
        """
        n_channels = x.shape[0]
        objs = out["objs"]
        z_all = {}
        for obj in range(MAX_OBJECTS):
            present = obj < len(objs) and objs[obj]["present"]
            if present:
                prev = self._prev_mtx[obj][:n_channels] if self._prev_present[obj] else None
                mtx = interp_matrix(objs[obj], mix_dq[obj], prev)
                z_all[obj] = np.sum(x * mtx, axis=0)
                self._prev_mtx[obj][:n_channels] = mtx[:, :, -1]
                self._prev_present[obj] = True
            elif self._prev_present[obj]:
                z_all[obj] = np.zeros((N, 24), dtype=np.complex128)
                self._prev_mtx[obj][:n_channels] = 0.0
                self._prev_present[obj] = True
            else:
                z_all[obj] = np.zeros((N, 24), dtype=np.complex128)
        return z_all

    def _hold_objects(self, x):
        """保持渲染：24 个时隙全部使用上一帧发布的矩阵，不更新 prev。"""
        n_channels = x.shape[0]
        z_all = {}
        for obj in range(MAX_OBJECTS):
            z_all[obj] = np.sum(x * self._prev_mtx[obj][:n_channels][:, :, None], axis=0)
        return z_all

    @staticmethod
    def _zero_objects():
        return {obj: np.zeros((N, 24), dtype=np.complex128)
                for obj in range(MAX_OBJECTS)}

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
        pcm = np.zeros((len(object_ids), FRAME_SAMPLES), dtype=np.float64)
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

    def decode_lfe(self, lfe_pcm, clipgain=None):
        """核心 ch3/LFE 乘逐点增益斜坡，再经跨帧 1217-sample 延迟输出到 ch0。

        clipgain 为本帧完整增益 G_new；None 表示本帧没有新值（沿用当前值）。
        斜坡：n<256 时 G_old + n·(G_new−G_old)/256，n≥256 时 G_new；
        首帧（没有历史）时 G_old = G_new。
        """
        current = np.asarray(lfe_pcm, dtype=np.float64)
        if current.shape != (FRAME_SAMPLES,):
            raise ValueError(f"LFE 帧应为 (1536,)，实际 {current.shape}")
        new_gain = self.held_clipgain() if clipgain is None else float(clipgain)
        old_gain = new_gain if self._lfe_clipgain is None else self._lfe_clipgain
        gains = np.full(FRAME_SAMPLES, new_gain, dtype=np.float64)
        if old_gain != new_gain:
            gains[:256] = old_gain + np.arange(256, dtype=np.float64) * (
                (new_gain - old_gain) / 256.0)
        self._lfe_clipgain = new_gain
        delayed = np.concatenate([self._lfe_delay, current * gains])
        out = np.clip(delayed[:FRAME_SAMPLES], -1.0, 1.0)
        self._lfe_delay = delayed[-LFE_DELAY:].copy()
        return out

    @staticmethod
    def _validate_bed(bed_pcm):
        pcm = np.asarray(bed_pcm, dtype=np.float32)
        n_channels = pcm.shape[0] if pcm.ndim == 2 else 0
        if n_channels not in (5, 7) or pcm.shape != (n_channels, FRAME_SAMPLES):
            raise ValueError(f"核心帧应为 (5,1536) 或 (7,1536)，实际 {pcm.shape}")
        return pcm

    @staticmethod
    def _core8(core, lfe):
        """JOC 输入序 5/7 路（不含 LFE）→ 8 平面核心序 L,R,C,LFE,Ls,Rs,(Lrs,Rrs)。"""
        out = np.zeros((CORE_OUT_CHANNELS, FRAME_SAMPLES), dtype=np.float64)
        out[0:3] = core[0:3]
        out[LFE_SLOT] = lfe
        rest = core.shape[0] - 3
        if rest > 0:
            out[4:4 + rest] = core[3:]
        return out

    @staticmethod
    def _delayed_core(hist, current):
        """直通渲染：8 路核心 PCM 各延迟 1217 个采样。"""
        out = np.empty((CORE_OUT_CHANNELS, FRAME_SAMPLES), dtype=np.float64)
        out[:, :LFE_DELAY] = hist[:, FRAME_SAMPLES - LFE_DELAY:]
        out[:, LFE_DELAY:] = current[:, :FRAME_SAMPLES - LFE_DELAY]
        return out

    def _render_frame(self, *, status, seq, out, mix_dq, bed_pcm, lfe_pcm,
                      config_idx=None):
        """每帧执行一次状态机并按结果分派渲染。"""
        pcm = self._validate_bed(bed_pcm)
        n_channels = pcm.shape[0]
        if self._n_channels_prev is not None and n_channels != self._n_channels_prev:
            self._clear_unused_channels(n_channels)
        self._advance_state(status, seq)
        state = self._state
        self._maybe_clear_lfe_ring(status)

        # 兼容性说明（status 8 无元数据帧的状态机改动）：
        # 变更前：status 8 无条件走 passthrough，流中途也一样。passthrough 用
        #   延迟后的 8 路 core 平面替换对象渲染，对象合成在下一帧从静音重新开始，
        #   于是每个无元数据帧都会在帧界产生一次台阶，元数据恢复时再产生一次；
        #   对不够标准的编码器（JOC EMDF 容器写得不好）这听起来就是噼啪声。
        # 变更后：status 8 和其他无效状态一样交给状态机——state 0（还没渲染过
        #   有效帧，即普通 E-AC-3 或 JOC 首帧之前）仍然 passthrough（核心 PCM
        #   直通本来就是为这种流准备的）；state 3（流中途）走 hold，沿用上一帧
        #   的对象矩阵继续渲染，床与对象都保持连续。
        # 说明：这是为兼容性做的取舍，不代表规范解释一定如此——"无元数据"也可以
        #   被论证为就是 passthrough 的情形；若要回到参考行为，恢复这里的无条件
        #   passthrough 分支即可。
        if status == STATUS_NO_METADATA and state == 0:
            mode = "passthrough"
        elif state == 4:
            mode = "hard_cut"
        elif state in (2, 3):
            mode = "hold"
        elif state == 1 and status == 0:
            mode = "normal"
        else:
            mode = "passthrough"

        if out is not None and out.get("n_channels") is not None:
            config = out["dmx_config_idx"]
            self._config_idx = config
        elif config_idx is not None:
            config = config_idx
            self._config_idx = config
        else:
            config = self._config_idx
        sigma = config in SIGMA_DMX_CONFIGS

        valid_metadata = (out is not None and out.get("status") == 0
                          and out.get("n_channels") is not None)
        if mode in ("normal", "hard_cut") and valid_metadata:
            clipgain = float(out["clipgain"])
            phase_new = 0.0625 * clipgain
        else:
            clipgain = self.held_clipgain()
            phase_new = self.held_analysis_scalar()

        lfe = (np.zeros(FRAME_SAMPLES, dtype=np.float64)
               if lfe_pcm is None else np.asarray(lfe_pcm, dtype=np.float64))
        if lfe.shape != (FRAME_SAMPLES,):
            raise ValueError(f"LFE 帧应为 (1536,)，实际 {lfe.shape}")
        core8 = self._core8(pcm, lfe)

        x = self.qmf_x(pcm, phase_new=phase_new, apply_dc_filter=sigma, rotate=sigma)
        self._last_x = x.copy()
        pcm16 = np.zeros((MAX_OBJECTS + 1, FRAME_SAMPLES), dtype=np.float64)

        if mode == "normal":
            z_all = self.object_z(out, mix_dq, x)
            objs_pcm = self.dll_synth_objects(z_all)
            pcm16[0] = self.decode_lfe(lfe, clipgain)
            for obj, pcm_obj in objs_pcm.items():
                pcm16[obj + 1] = pcm_obj
        elif mode == "hold":
            z_all = self._hold_objects(x)
            objs_pcm = self.dll_synth_objects(z_all)
            pcm16[0] = self.decode_lfe(lfe, clipgain)
            for obj, pcm_obj in objs_pcm.items():
                pcm16[obj + 1] = pcm_obj
        elif mode == "hard_cut":
            # 全部对象矩阵清零，保留“prev 存在”标记，下一有效帧从 0 淡入。
            z_all = self._zero_objects()
            self.dll_synth_objects(z_all)
            self.decode_lfe(lfe, clipgain)          # 环照常写入并推进写游标
            self._prev_present[:] = True
            self._prev_mtx[:] = 0.0
        else:                                        # passthrough
            z_all = self._zero_objects()
            self.dll_synth_objects(z_all)
            self.decode_lfe(lfe, clipgain)
            pcm16[:CORE_OUT_CHANNELS] = self._delayed_core(self._core_hist, core8)

        self._core_hist[:] = core8
        self._n_channels_prev = n_channels
        pcm16_f32 = np.asarray(pcm16, dtype=np.float32)
        return np.multiply(pcm16_f32, self.output_scale, dtype=np.float32), z_all

    def render_missing_frame(self, bed_pcm, lfe_pcm=None, config_idx=3):
        """没有 JOC 元数据的音频帧：8 路核心 PCM identity 直通（每路延迟 1217 采样）。"""
        return self._render_frame(status=STATUS_NO_METADATA, seq=None,
                                  out=None, mix_dq=None,
                                  bed_pcm=bed_pcm, lfe_pcm=lfe_pcm,
                                  config_idx=config_idx)

    def render_frame(self, payload_bytes, bed_pcm, lfe_pcm=None):
        """payload_bytes = evolution 载荷（含 id14 JOC）；bed_pcm = N×1536 核心 PCM。

        返回 (pcm16, z_all)：pcm16 = 16×1536（ch0 LFE + 15 对象）f32，
        已用 FP32 系数乘 output_scale。"""
        subs, _ = unpack_evolution(payload_bytes, loose=True)
        return self.render_subpayloads(subs, bed_pcm, lfe_pcm)

    def render_subpayloads(self, subs, bed_pcm, lfe_pcm=None):
        """以已拆出的 EMDF payload 字典渲染一帧，避免绑定 transport 容器。"""
        out, mix_q, mix_dq = self.decode_subpayloads(subs)
        pcm = self._validate_bed(bed_pcm)
        status = out["status"]
        if (status == 0 and out["n_channels"] is not None
                and self._n_channels_prev is not None
                and out["n_channels"] != self._n_channels_prev):
            status = STATUS_CORE_CHANNEL_CHANGE
        config_idx = (out["dmx_config_idx"] if out["n_channels"] is not None else None)
        return self._render_frame(status=status, seq=out.get("seq_count_bits"),
                                  out=out, mix_dq=mix_dq,
                                  bed_pcm=pcm, lfe_pcm=lfe_pcm,
                                  config_idx=config_idx)
