# JustOneCacophony — E-AC-3 JOC 解码与渲染数学

[English](math.en.md) · [返回 README](../README.md)

本文只说明 JustOneCacophony 研究路径中使用的信号模型和公式：JOC 参数如何与核心 PCM 结合并重建对象信号，以及 OAMD 坐标如何转换为扬声器增益。

这些公式描述项目当前研究的 JOC 矩阵参数（dense 与 sparse 两条差分语法）与普通点对象路径，不代表对所有 E-AC-3 JOC 变体的完整定义。

## 1. 总体路径与记号

对象重建路径：

```text
E-AC-3 核心 5.1 PCM
  + ID14 JOC 矩阵参数
  → analysis QMF
  → 参数带展开与时间插值
  → 对象矩阵
  → inverse QMF
  → LFE + 15 路对象 PCM
```

扬声器渲染路径：

```text
LFE + 15 路对象 PCM
  + ID11 OAMD 坐标与更新时间
  → 目标布局 region
  → 等功率声像
  → 位置补偿
  → 逐样本增益斜坡
  → 扬声器 PCM
```

主要记号：

| 符号 | 含义 |
|---|---|
| $c=0\ldots4$ | 核心声道 L、R、C、Ls、Rs |
| $o=0\ldots14$ | 15 个 JOC 对象 |
| $b=0\ldots63$ | 复 QMF 子带 |
| $t=0\ldots23$ | 每帧 24 个 64-sample 时槽 |
| $p(b)$ | QMF 子带 $b$ 对应的 JOC 参数带 |
| $X_{c,b,t}$ | 核心声道的 analysis-QMF 值 |
| $M_{o,c,b,t}$ | 对象矩阵系数 |
| $Z_{o,b,t}$ | 对象的 inverse-QMF 输入 |
| $y_o[n]$ | 对象时域 PCM |

一帧的采样数为

$$
N_f=1536=24\times64.
$$

## 2. JOC 矩阵参数

每个对象、每个数据点的量化矩阵 `joc_mix_mtx_q` 都定义在 $N_q$ 个量化级上。标志位 `b_joc_sparse` 选择两条差分语法之一：dense 为每个核心声道各送一路 MTX 差分，sparse 每参数带只送一个 active 声道与一路系数差分。

### 2.1 Dense 差分还原

令 `quant_idx` 为 $q_i\in\{0,1\}$，量化级数为

$$
N_q=
\begin{cases}
96, & q_i=0,\\
192, & q_i=1.
\end{cases}
$$

中心偏移为

$$
O_q=\frac{N_q}{2}.
$$

对对象 $o$、数据点 $d$、核心声道 $c$ 和参数带 $p$，编码差分 $\Delta_{o,d,c,p}$ 还原为

$$
Q_{o,d,c,0}=
\left(O_q+\Delta_{o,d,c,0}\right)\bmod N_q,
$$

$$
Q_{o,d,c,p}=
\left(Q_{o,d,c,p-1}+\Delta_{o,d,c,p}\right)\bmod N_q,
\qquad p>0.
$$

### 2.2 Sparse 差分还原

令 $I_{o,d,p}$ 为 `joc_channel_idx` 符号（IDX）， $V_{o,d,p}$ 为 `joc_vec` 符号（VEC）， $N_c\in\lbrace5,7\rbrace$ 为核心声道数。每参数带只有一个 active 声道

$$
A_{o,d,p}=
\begin{cases}
I_{o,d,0}, & p=0,\\
\left(A_{o,d,p-1}+I_{o,d,p}\right)\bmod N_c, & p>0,
\end{cases}
$$

其中 $I_{o,d,0}$ 是 3 bit 绝对声道号，其余 IDX 符号是相对上一个 **active 声道**的增量。系数是一个跨参数带连续的单累加器

$$
\kappa_{o,d,-1}=O^{(s)}_q,\qquad
\kappa_{o,d,p}=
\left(\kappa_{o,d,p-1}+V_{o,d,p}\right)\bmod N_q,
$$

sparse 起点比 dense 的中心偏移高两个量化级：

$$
O^{(s)}_q=
\begin{cases}
50, & q_i=0,\\
100, & q_i=1.
\end{cases}
$$

active 声道切换时累加器**不**重置。完整矩阵为

$$
Q_{o,d,c,p}=
\begin{cases}
\kappa_{o,d,p}, & c=A_{o,d,p},\\
\dfrac{N_q}{2}, & c\neq A_{o,d,p}.
\end{cases}
$$

非 active 项取 $N_q/2$，即去量化后恰为 0。

### 2.3 去量化

矩阵系数的去量化值为

$$
D_{o,d,c,p}=
\left(Q_{o,d,c,p}-\frac{N_q}{2}\right)
\frac{820}{4096(1+q_i)}.
$$

因此 coarse 模式的有效分母为 4096，fine 模式为 8192。

### 2.4 JOC clipgain

若 clipgain 字段由整数 $x$ 和尾数 $y$ 组成，则

$$
G_{\mathrm{clip}}=
1+\frac{y}{32}2^{x-4}.
$$

它并入分析端标量 $S=\frac{1}{16}G_{\mathrm{clip}}$，乘在核心 bed 的每个采样上（QMF 分析之前）；
对象 inverse QMF 之后的 $\times16$ 抵消 $1/16$，净效果是乘 $G_{\mathrm{clip}}$，因此钳位 $[-1,1]$
处在增益**之后**。标量按帧保留两个槽，本帧前 256 个采样从上一帧值线性过渡到本帧值、之后为常数，
复位后的第一帧直接取新值。LFE 走旁通通路，不参与该标量。

## 3. 参数带展开与时间插值

### 3.1 参数带到 QMF 子带

JOC 矩阵按参数带编码，而 QMF 使用 64 个子带。令 $p(b)$ 表示子带 $b$ 所属的参数带，则每个参数带系数展开为

$$
D_{o,d,c,b}=D_{o,d,c,p(b)}.
$$

常见的 12-band 映射为

$$
\begin{aligned}
\mathcal B_0 &= \{0\}, &
\mathcal B_1 &= \{1\}, &
\mathcal B_2 &= \{2\}, &
\mathcal B_3 &= \{3\},\\
\mathcal B_4 &= \{4,5\}, &
\mathcal B_5 &= \{6,7\}, &
\mathcal B_6 &= \{8,9,10\}, &
\mathcal B_7 &= \{11,12,13\},\\
\mathcal B_8 &= \{14,15,16,17\}, &
\mathcal B_9 &= \{18,\ldots,22\},\\
\mathcal B_{10} &= \{23,\ldots,34\}, &
\mathcal B_{11} &= \{35,\ldots,63\}.
\end{aligned}
$$

其中 $p(b)=k$ 当且仅当 $b\in\mathcal B_k$。其他参数带数使用各自的子带边界。

### 3.2 单数据点插值

令上一帧末值为 $P_{o,c,b}$，当前目标值为 $D_{o,c,p(b)}$。对时槽 $t=0\ldots23$：

$$
\alpha_t=\frac{t+1}{24},
$$

$$
M_{o,c,b,t}=
(1-\alpha_t)P_{o,c,b}
+\alpha_tD_{o,c,p(b)}.
$$

因此第一时槽已经推进 ramp 的 $1/24$，最后一时槽等于当前目标：

$$
M_{o,c,b,23}=D_{o,c,p(b)}.
$$

该值随后成为下一帧的 previous 状态。

### 3.3 多数据点

当一帧含两个数据点时，`offset_ts` 给出分段边界。每一段在上一目标和下一目标之间使用相同的线性关系；阶跃模式则在指定时槽直接切换目标。

## 4. 核心 PCM 的 analysis QMF

矩阵输入使用 $N$ 个核心声道（$N\in\{5,7\}$，由 $joc\_dmx\_config\_idx$ 查表得到，
$k=0\ldots4$ 对应 $N=5,7,7,5,7$）；LFE 走独立路径。核心 PCM 先缩放为

$$
\widetilde x_c[n]=\frac{x_c[n]}{16}.
$$

令 $\mathcal A_b$ 表示带 polyphase 历史状态的 64-band analysis-QMF 算子，则

$$
X_{c,b,t}=
\mathcal A_b\left(
\widetilde x_c[64t],\ldots,\widetilde x_c[64t+63];
\mathbf s^{\mathrm A}_{c,t}
\right).
$$

该过程依次包含 analysis window/polyphase、调制、64 点 FFT 和子带重排。历史状态跨时槽和帧连续推进。

## 5. 核心声道的 QMF 域处理

L、R、C 在进入对象矩阵前延迟 10 个 QMF 时槽：

$$
\widehat X_{c,b,t}=X_{c,b,t-10},
\qquad c\in\{L,R,C\}.
$$

Ls、Rs 的 10 时槽延迟在环绕通路内完成，延迟线照常推进；$b>0$ 的 $-j$ 旋转与 band-0 的
21-tap 复 FIR 由**同一个开关** $\sigma(k)=1\iff k\in\{3,4\}$ 门控：

$$
\widehat X_{c,b,t}=
\begin{cases}
-j\,X_{c,b,t-10}, & \sigma(k)=1,\ c\in\{Ls,Rs\},\ b>0,\\[1mm]
\sum_{m=0}^{20}h_mX_{c,0,t-m}, & \sigma(k)=1,\ c\in\{Ls,Rs\},\ b=0,\\[1mm]
X_{c,b,t-10}, & \text{其他情况}.
\end{cases}
$$

$N=7$ 时第 6、7 个核心声道（Lb/Rb 或 Tfl/Tfr）与前级相同，只做 10 时槽延迟：
通道身份只影响元数据与渲染，不进入 DSP。这些延迟和滤波历史属于解码状态，
不能按帧独立清零。

## 6. 对象矩阵

对每个对象 $o$、子带 $b$ 和时槽 $t$，对象频域值为 $N$ 个核心声道的线性组合：

$$
Z_{o,b,t}=
\sum_{c=0}^{N-1}
M_{o,c,b,t}\widehat X_{c,b,t}.
$$

analysis 输入的 $1/16$ 缩放会在 inverse QMF 输出端由 $\times16$ 抵消，因此矩阵本身不需要额外经验增益。

## 7. 对象 inverse QMF

### 7.1 子带重排

将 64 个复子带写成 128 个交织实数 `src`。对 $k=0\ldots31$：

$$
\begin{aligned}
\mathrm{zone}[2k] &= \mathrm{src}[4k],\\
\mathrm{zone}[2k+1] &= -\mathrm{src}[4k+1],\\
\mathrm{zone}[126-2k] &= \mathrm{src}[4k+2],\\
\mathrm{zone}[127-2k] &= \mathrm{src}[4k+3].
\end{aligned}
$$

把 `zone` 重新视为 64 个复数后执行未归一化 64 点 FFT：

$$
F_k=
\sum_{n=0}^{63}
\mathrm{zone}_n
\exp\left(-j\frac{2\pi kn}{64}\right).
$$

### 7.2 调制与合成

定义旋转系数

$$
r_k=
\frac12\left(
\sin\frac{\pi k}{128}
+j\cos\frac{\pi k}{128}
\right),
$$

并计算

$$
R_k=2F_kr_k.
$$

令 $\mathcal S$ 表示带 640 项 synthesis window 和跨时槽状态的 polyphase 合成算子：

$$
\mathbf y_{o,t}=
\mathcal S\left(
\mathbf R_{o,t},W,\mathbf s^{\mathrm S}_{o,t}
\right).
$$

对象输出为

$$
y_o[64t+r]=
\mathrm{clip}\left(
16\,\mathbf y_{o,t}[r],-1,1
\right)G_{\mathrm{clip}},
$$

其中 $r=0\ldots63$。synthesis 状态必须按时槽连续推进。

## 8. LFE 路径

LFE 不经过对象矩阵或 inverse QMF，而是使用 1217-sample 延迟。输入与输出端的比例因子抵消后：

$$
y_{\mathrm{LFE}}[n]=
\mathrm{clip}\left(
x_{\mathrm{LFE,core}}[n-1217],-1,1
\right).
$$

## 9. OAMD 坐标

横向和纵向网格使用 $N=62$，高度网格使用 $N=15$。量化函数为

$$
q_N(k)=
\min\left(
32767,
\left\lfloor\frac{32768k}{N}+\frac12\right\rfloor
\right).
$$

OAR 坐标为

$$
u=\frac{q_1}{32768},
\qquad
v=\frac{q_2}{32768},
\qquad
w=\frac{q_3}{32768}.
$$

其最大运行值为 $32767/32768$，不是精确的 1。

转换为 ADM 网格时：

$$
k_1=\mathrm{round}\left(\frac{62q_1}{32767}\right),
\quad
k_2=\mathrm{round}\left(\frac{62q_2}{32767}\right),
\quad
k_3=\mathrm{round}\left(\frac{15q_3}{32767}\right),
$$

$$
X=2\frac{k_1}{62}-1,
\qquad
Y=1-2\frac{k_2}{62},
\qquad
Z=\frac{k_3}{15}.
$$

连续坐标关系为

$$
u=\frac{X+1}{2},
\qquad
v=\frac{1-Y}{2},
\qquad
w=Z.
$$

## 10. 等功率扬声器声像

### 10.1 一维插值

相邻扬声器坐标为 $a_0<a_1$，对象位置为 $a$。归一化位置为

$$
\tau=\frac{a-a_0}{a_1-a_0}.
$$

区间内的增益为

$$
g_0(\tau)=\cos\left(\frac\pi2\tau\right),
\qquad
g_1(\tau)=\sin\left(\frac\pi2\tau\right),
$$

并满足

$$
g_0^2(\tau)+g_1^2(\tau)=1.
$$

区间外的对象位置夹到最近端点。

### 10.2 二维 region

每一行先沿 $u$ 得到横向增益向量 $\mathbf h_r(u)$。若对象位于相邻两行 $r_0,r_1$ 之间：

$$
\eta=\frac{v-v_{r_0}}{v_{r_1}-v_{r_0}},
$$

$$
a_0=\cos\left(\frac\pi2\eta\right),
\qquad
a_1=\sin\left(\frac\pi2\eta\right).
$$

二维点增益为

$$
\mathbf G_{\mathrm{2D}}(u,v)=
\mathbf h(u)\odot\mathbf v(v).
$$

对于只有一对水平环绕、没有独立 side/rear 两对的布局（5.1 系列，以及 4.0 这类
只有一对后置声道的布局），仅在 OAMD `warp_mode=1` 时，纵向坐标使用

$$
v_{\mathrm{floor}}=
\mathrm{clamp}(2v,0,1).
$$

`warp_mode=0` 或其他布局使用 $v_{\mathrm{floor}}=v$。高度层不做该缩放。
未携带 warp 元数据时采用 0；命令行不提供覆盖选项。warp 2/3 使用 §12.5 的中间布局与折叠矩阵。
有 front/back balance 时，先求这里的分支坐标，再应用 §12.3 的 balance。

### 10.3 高度层

三维布局分别计算地面层增益 $\mathbf G_f$ 和高度层增益 $\mathbf G_h$：

$$
\mathbf G_{\mathrm{point}}(u,v,w)=
\cos\left(\frac\pi2w\right)\mathbf G_f
+
\sin\left(\frac\pi2w\right)\mathbf G_h.
$$

当地面层与高度层扬声器集合不重叠、且各层内部使用等功率插值时：

$$
\left\|\mathbf G_{\mathrm{point}}\right\|_2=1.
$$

## 11. 布局位置补偿

本节是 trim 为 automatic 时的逐对象增益。disabled 取 1；explicit 改用 §12.2，
不能把 automatic 增益与 explicit trim 再乘一遍。

令 $N_h$ 为相关高度扬声器数，$N_f$ 为相关附加水平扬声器数：

$$
H=\min\left(\frac{N_h}{4},1\right),
\qquad
F=\min\left(\frac{N_f}{4},1\right).
$$

最大位置补偿为

$$
A_{\max}=
-\max\left(4.5-1.5H-3F,0\right)
\quad\text{dB}.
$$

前后与高度位置权重为

$$
p_v=\mathrm{clamp}\left(\frac v{0.6},0,1\right),
$$

$$
p_w=\mathrm{clamp}\left(\frac{w-0.2}{0.8},0,1\right),
$$

$$
p=\mathrm{clamp}(p_v+p_w,0,1).
$$

线性补偿增益为

$$
G_{\mathrm{pos}}=10^{A_{\max}p/20}.
$$

对象的目标增益向量为

$$
\mathbf G_{\mathrm{target}}=
G_{\mathrm{object}}
G_{\mathrm{pos}}
\mathbf G_{\mathrm{point}}.
$$

## 12. OAMD trim 与前后平衡

### 12.1 配置选择

OAMD trim element 把节目的校准数据分成九个配置，由目标布局的附加水平扬声器数
$N_f$ 与高度扬声器数 $N_h$ 选定：

$$
\kappa_f(N)=
\begin{cases}
0, & N=0,\\
1, & N\in\{2,3\},\\
2, & \text{其他},
\end{cases}
\qquad
\kappa_h(N)=
\begin{cases}
0, & N=0,\\
1, & N=2,\\
2, & \text{其他}.
\end{cases}
$$

$$
c_{\mathrm{trim}}=\kappa_f(N_f)+3\,\kappa_h(N_h).
$$

九个配置依次为“no / some / many 附加水平 × no / some / many 高度”。

trim 与前后平衡只在七种布局下施加：2.0、3.1、4.0、5.1、5.1.2、5.1.4、7.1。
更宽的布局（7.1.2、7.1.4、9.1.4、9.1.6）取 $g_t=1$、$b=0$。

warp、trim 和 balance 均从 OAMD 元数据读取。各参数的取值与作用为：

| 参数 | 取值或单位 | 行为 |
|---|---|---|
| `warp_mode` | 0 / 1 / 2 / 3 | 直接渲染 / 地面 Y 缩放 / PLIIx 折叠 / Standard（Lo/Ro）折叠 |
| `global_trim_mode` | 0 / 1 / 2 | automatic / disabled / 按目标布局配置选择；3 为保留值，明确报错 |
| $T_s$、$T_h$ | dB | 显式 surround、height trim 参数，正值按 0 处理 |
| $T_c$ | dB | 显式 centre trim 参数，保留符号 |
| $b_{\mathrm{lis}}$、$b_{\mathrm{tb}}$ | $[-1,1]$，步长 $1/16$ | 听音平面、顶部的前后 balance |
| 逐对象 trim bypass | 布尔值 | 关闭该对象的 balance 和 trim 增益，保留 warp |

全局模式为 2 时，所选配置的 default 或 disabled 标志分别采用 automatic 或 disabled；
缺失配置采用 automatic，其他配置使用其显式参数。automatic 使用 §11 的位置补偿，
disabled 使用单位增益，explicit 使用 §12.2，三种模式互斥。

初始没有 trim element 时采用 `warp_mode=0`、automatic trim。后续 payload 不含该
element 时保持上一组控制；出现新 element 时，在对应元数据事件边界更新。命令行不提供
warp 覆盖选项。

### 12.2 逐对象电平

centre、surround、height 是位置相关 trim 函数的参数，不是混音后各输出声道的固定增益。
令 $y_t$ 为原始 Y 经 balance 后的坐标，$z=\max(w,0)$：

$$
a_s=\mathrm{clamp}(2y_t,0,1)(1-z),\qquad a_h=z,
$$

$$
a_c=1-\mathrm{clamp}\bigl(10\max(|z|-0.05,\ y_t-0.05,\ |u-0.5|-0.05),0,1\bigr).
$$

explicit 状态下，surround/height 参数上限为 0 dB，centre 保留有符号电平：

$$
D=a_s\min(T_s,0)+a_h\min(T_h,0)+a_cT_c,\qquad g_t=10^{D/20}.
$$

同一个 $g_t$ 乘该对象的整条声像增益向量，然后才累加不同对象。automatic 使用 §11；
disabled 或逐对象 trim bypass 取 $g_t=1$。LFE 仍直接通过。
增益求值、声像插值和对象累加均使用 float64。

### 12.3 前后平衡

每个配置还携带两个前后平衡量：$b_{\mathrm{lis}}$ 作用于听音平面，
$b_{\mathrm{tb}}$ 作用于顶部。二者按 $1/16$ 网格量化，只在显式配置下非零。

记 $n=\mathrm{round}(16\,|b|)$、$s=n/18$，单层映射为

$$
B_b(y)=
\begin{cases}
y, & y\le 0.05\ \text{或}\ y\ge 0.95,\\[2pt]
(1+s)y-0.05s, & 0.05<y<0.5,\ b>0,\\[2pt]
(1-s)(y-0.5)+0.5+0.45s, & 0.5\le y<0.95,\ b>0,\\[2pt]
(1-s)y+0.05s, & 0.05<y<0.5,\ b<0,\\[2pt]
(1+s)(y-0.5)+0.5-0.45s, & 0.5\le y<0.95,\ b<0.
\end{cases}
$$

它把 $[0.05,0.95]$ 单调映到自身：两端不动，$y=0.5$ 处两段取值相同，斜率分别为
$1+s$ 与 $1-s$。首尾各 $5\%$ 的墙区以及 $b=0$ 的情形都原样通过，因此映射不会
越界，也无需额外钳位。

高度 $w$ 把两层混合起来：

$$
y'=(1-w)\,B_{b_{\mathrm{lis}}}(y)+w\,B_{b_{\mathrm{tb}}}(y).
$$

前后轴 $y$ 的取向见 §9，$0$ 为前、$1$ 为后；$x$ 与 $z$ 不变。

### 12.4 施加位置

记 §12.3 的高度混合 balance 映射为 $B_z(y)$，§10.2 的 warp 映射为 $W(y)$。
warp 0/1 直接渲染点对象时，分支顺序为

$$
y_f=B_z(W(y)),\qquad y_h=B_z(y),\qquad y_t=B_z(y).
$$

地面和高度声像分别使用 $y_f,y_h$，trim 函数使用 $y_t$。最后组合

$$
g_{o,c}=G_{\mathrm{object}}\,g_t(u,y_t,z)\,P_c(u,y_f,y_h,z).
$$

因此可概括为 warp → balance → trim 与声像增益合成，但不能让所有阶段共用一个
已经 warp 的 Y。balance 仅在 explicit 且未逐对象 bypass 时启用；warp 不受 trim
状态或 trim bypass 控制。源坐标保持原值，ADM 与 binaural 不预处理此变换。

地面坐标依次经过 warp、balance，再用于地面声像；顶部坐标和 trim 求值坐标只经过
balance。trim 标量乘到该对象的整条声像增益向量后，再累加到输出声道。LFE 直接通过。

### 12.5 warp 2/3 的环绕折叠

目标为 5.1、5.1.2 或 5.1.4 时，warp 2/3 先把对象渲染到对应的 7.1、7.1.2 或
7.1.4 中间布局，再将后环绕折叠到目标的环绕对。4.0 同样先增加一对后环绕，
中间布局不增加 centre 或 LFE。其他现有布局直接渲染。

中间布局只改变扬声器几何与声像分配。Trim 配置、两层 Balance 和 Auto Trim 的
最大位置补偿仍按最终目标布局选择。例如，5.1.2 使用配置 4，最大位置补偿为
$-2.25$ dB，即使其中间布局是 7.1.2，也继续使用这些参数。中间布局已有独立
side/rear 两对，地面 Y 不再乘 2；Balance 和逐对象 Trim 在中间渲染时求值。

设中间布局输出为 $L_s,R_s,L_{rs},R_{rs}$，最终目标的环绕输出为 $L'_s,R'_s$：

$$
L'_s=L_s+aL_{rs}+bR_{rs},\qquad
R'_s=R_s+bL_{rs}+aR_{rs}.
$$

| 模式 | $a$ | $b$ |
|---|---|---|
| 2：PLIIx | $\sqrt{3}/2$ | $1/2$ |
| 3：Standard（Lo/Ro） | $1$ | $0$ |

两种矩阵都按线性幅度相加，不做折叠后的能量归一化。共同的前方、高度和 LFE
声道按对应位置直接通过。折叠后的对象增益向量进入 §13 的目标增益更新与斜坡，
然后累加对象音频。

完整顺序为：按目标布局选择校准参数 → 在中间布局应用 Balance、Trim 并计算声像
增益 → 折叠水平环绕 → 更新增益斜坡与混音。逐对象 bypass 关闭 Balance 和 Trim，
但仍保留 warp 2/3 的布局选择与折叠。上述模式属于通用扬声器渲染控制；E-AC-3 JOC
公开定义的 warp 值仍为 0/1，源对象轨迹、ADM 和 binaural 不预处理折叠结果。

## 13. OAMD 时间对齐与增益斜坡

OAMD 更新的编码位置为

$$
s_{\mathrm{coded}}=
s_{\mathrm{frame}}
+s_{\mathrm{outer}}
+s_{\mathrm{OAMD}}
+32f_{\mathrm{block}}.
$$

decoder 输出 PCM timeline 上的理论更新位置为

$$
s_{\mathrm{theoretical}}=
s_{\mathrm{coded}}+d_{\mathrm{decoder}},
\qquad d_{\mathrm{decoder}}=1473.
$$

扬声器 renderer 保留现有的处理块长度 $B=32$，更新点对齐为

$$
\widehat s=
B\left\lfloor
\frac{s_{\mathrm{theoretical}}+B/2-1}{B}
\right\rfloor.
$$

因此，对 frame-aligned 更新有 `align32(1473)=1472`。1473 是 metadata interface 的理论 decoder delay；1472 是当前 32-sample control block 中的有效边界。inverse-QMF 使用的 640 项 window/state 不属于这条 metadata timing 公式。

给定 ramp duration $D$，block 数为

$$
K=
\left\lfloor
\frac{D+B/2-1}{B}
\right\rfloor.
$$

若当前增益为 $g_0$、目标为 $g_1$，则每 block 的增量为

$$
\Delta g=\frac{g_1-g_0}{K}.
$$

第 $j$ 个 block 内的样本 $r=0\ldots B-1$ 使用

$$
g_{j,r}=g_j+\frac rB\Delta g,
\qquad
g_{j+1}=g_j+\Delta g.
$$

如果中途没有新的 metadata 更新，该过程等价于总长度 $KB$ 的逐样本线性斜坡。

## 14. 最终扬声器混音

对目标输出声道 $c$：

$$
y_c[n]=
\delta_{c,\mathrm{LFE}}x_{\mathrm{LFE}}[n]
+
\sum_{o=1}^{15}x_o[n]g_{o,c}[n].
$$

其中

$$
\delta_{c,\mathrm{LFE}}=
\begin{cases}
1, & c\text{ 为目标布局的 LFE},\\
0, & \text{其他声道}.
\end{cases}
$$

没有 LFE 输出的布局不把输入 LFE 混入其他声道。对象完成累加后，再按目标格式要求排列输出声道。

若输出 PCM24，量化关系为

$$
y_{24}[n]=
\mathrm{trunc}\left(
8388607\,\mathrm{clip}(y[n],-1,1)
\right).
$$

## 15. 公式适用范围

- JOC 矩阵部分同时描述 dense MTX 与 sparse IDX/VEC 两条差分语法。
- 扬声器声像部分描述普通点对象；extent、spread、divergence 等模式需要额外模型。
- 多个 OAMD position block 必须按其时间顺序调度。
- limiter 属于独立后处理，不包含在上述混音公式中。
- trim 与前后平衡只作用于扬声器渲染；双耳渲染不读取它们。
