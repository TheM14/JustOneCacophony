# JustOneCacophony — E-AC-3 JOC 解码与渲染数学

[English](math.en.md) · [返回 README](../README.md)

本文只说明 JustOneCacophony 研究路径中使用的信号模型和公式：JOC 参数如何与核心 PCM 结合并重建对象信号，以及 OAMD 坐标如何转换为扬声器增益。

这些公式描述项目当前研究的 dense JOC 与普通点对象路径，不代表对所有 E-AC-3 JOC 变体的完整定义。

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

## 2. Dense JOC 矩阵参数

### 2.1 差分还原

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
Q_{o,d,c,0}
=
\left(O_q+\Delta_{o,d,c,0}\right)\bmod N_q,
$$

$$
Q_{o,d,c,p}
=
\left(Q_{o,d,c,p-1}+\Delta_{o,d,c,p}\right)\bmod N_q,
\qquad p>0.
$$

### 2.2 去量化

矩阵系数的去量化值为

$$
D_{o,d,c,p}
=
\left(Q_{o,d,c,p}-\frac{N_q}{2}\right)
\frac{820}{4096(1+q_i)}.
$$

因此 coarse 模式的有效分母为 4096，fine 模式为 8192。

### 2.3 JOC clipgain

若 clipgain 字段由整数 $x$ 和尾数 $y$ 组成，则

$$
G_{\mathrm{clip}}
=
1+\frac{y}{32}2^{x-4}.
$$

它在对象 inverse QMF 之后作用于对象 PCM，不作用于 LFE。

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
M_{o,c,b,t}
=
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

矩阵输入使用核心声道 L、R、C、Ls、Rs；LFE 走独立路径。核心 PCM 先缩放为

$$
\widetilde x_c[n]=\frac{x_c[n]}{16}.
$$

令 $\mathcal A_b$ 表示带 polyphase 历史状态的 64-band analysis-QMF 算子，则

$$
X_{c,b,t}
=
\mathcal A_b\!\left(
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

Ls、Rs 同样延迟 10 个时槽，并在 $b>0$ 时作 $-j$ 旋转：

$$
\widehat X_{c,b,t}=-jX_{c,b,t-10},
\qquad c\in\{Ls,Rs\},\ b>0.
$$

环绕声道的 band 0 还经过 21-tap 复 FIR：

$$
\widehat X_{c,0,t}
=
\sum_{k=0}^{20}h_kX_{c,0,t-k}.
$$

这些延迟和滤波历史属于解码状态，不能按帧独立清零。

## 6. 对象矩阵

对每个对象 $o$、子带 $b$ 和时槽 $t$，对象频域值为五个核心声道的线性组合：

$$
Z_{o,b,t}
=
\sum_{c=0}^{4}
M_{o,c,b,t}\widehat X_{c,b,t}.
$$

analysis 输入的 $1/16$ 缩放会在 inverse QMF 输出端由 $\times16$ 抵消，因此矩阵本身不需要额外经验增益。

## 7. 对象 inverse QMF

### 7.1 子带重排

将 64 个复子带写成 128 个交织实数 `src`。对 $k=0\ldots31$：

$$
\begin{aligned}
\operatorname{zone}[2k] &= \operatorname{src}[4k],\\
\operatorname{zone}[2k+1] &= -\operatorname{src}[4k+1],\\
\operatorname{zone}[126-2k] &= \operatorname{src}[4k+2],\\
\operatorname{zone}[127-2k] &= \operatorname{src}[4k+3].
\end{aligned}
$$

把 `zone` 重新视为 64 个复数后执行未归一化 64 点 FFT：

$$
F_k
=
\sum_{n=0}^{63}
\operatorname{zone}_n
\exp\!\left(-j\frac{2\pi kn}{64}\right).
$$

### 7.2 调制与合成

定义旋转系数

$$
r_k
=
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
\mathbf y_{o,t}
=
\mathcal S\!\left(
\mathbf R_{o,t},W,\mathbf s^{\mathrm S}_{o,t}
\right).
$$

对象输出为

$$
y_o[64t+r]
=
\operatorname{clip}\!\left(
16\,\mathbf y_{o,t}[r],-1,1
\right)G_{\mathrm{clip}},
$$

其中 $r=0\ldots63$。synthesis 状态必须按时槽连续推进。

## 8. LFE 路径

LFE 不经过对象矩阵或 inverse QMF，而是使用 1217-sample 延迟。输入与输出端的比例因子抵消后：

$$
y_{\mathrm{LFE}}[n]
=
\operatorname{clip}\!\left(
x_{\mathrm{LFE,core}}[n-1217],-1,1
\right).
$$

## 9. OAMD 坐标

横向和纵向网格使用 $N=62$，高度网格使用 $N=15$。量化函数为

$$
q_N(k)
=
\min\!\left(
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
k_1=\operatorname{round}\!\left(\frac{62q_1}{32767}\right),
\quad
k_2=\operatorname{round}\!\left(\frac{62q_2}{32767}\right),
\quad
k_3=\operatorname{round}\!\left(\frac{15q_3}{32767}\right),
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
\mathbf G_{\mathrm{2D}}(u,v)
=
\mathbf h(u)\odot\mathbf v(v).
$$

对于只有一对水平环绕、没有独立 side/rear 两对的 5.1 系列布局，纵向坐标使用

$$
v_{\mathrm{floor}}
=
\operatorname{clamp}(2v,0,1).
$$

其他布局使用 $v_{\mathrm{floor}}=v$。

### 10.3 高度层

三维布局分别计算地面层增益 $\mathbf G_f$ 和高度层增益 $\mathbf G_h$：

$$
\mathbf G_{\mathrm{point}}(u,v,w)
=
\cos\left(\frac\pi2w\right)\mathbf G_f
+
\sin\left(\frac\pi2w\right)\mathbf G_h.
$$

当地面层与高度层扬声器集合不重叠、且各层内部使用等功率插值时：

$$
\left\|\mathbf G_{\mathrm{point}}\right\|_2=1.
$$

## 11. 布局位置补偿

令 $N_h$ 为相关高度扬声器数，$N_f$ 为相关附加水平扬声器数：

$$
H=\min\left(\frac{N_h}{4},1\right),
\qquad
F=\min\left(\frac{N_f}{4},1\right).
$$

最大位置补偿为

$$
A_{\max}
=
-\max\left(4.5-1.5H-3F,0\right)
\quad\text{dB}.
$$

前后与高度位置权重为

$$
p_v=\operatorname{clamp}\left(\frac v{0.6},0,1\right),
$$

$$
p_w=\operatorname{clamp}\left(\frac{w-0.2}{0.8},0,1\right),
$$

$$
p=\operatorname{clamp}(p_v+p_w,0,1).
$$

线性补偿增益为

$$
G_{\mathrm{pos}}=10^{A_{\max}p/20}.
$$

对象的目标增益向量为

$$
\mathbf G_{\mathrm{target}}
=
G_{\mathrm{object}}
G_{\mathrm{pos}}
\mathbf G_{\mathrm{point}}.
$$

## 12. OAMD 时间对齐与增益斜坡

OAMD 更新的编码位置为

$$
s_{\mathrm{coded}}
=
s_{\mathrm{frame}}
+s_{\mathrm{outer}}
+s_{\mathrm{OAMD}}
+32f_{\mathrm{block}}.
$$

对处理块长度 $B=32$，更新点对齐为

$$
\widehat s
=
B\left\lfloor
\frac{s_{\mathrm{coded}}+B/2-1}{B}
\right\rfloor.
$$

给定 ramp duration $D$，block 数为

$$
K
=
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

## 13. 最终扬声器混音

对目标输出声道 $c$：

$$
y_c[n]
=
\delta_{c,\mathrm{LFE}}x_{\mathrm{LFE}}[n]
+
\sum_{o=1}^{15}x_o[n]g_{o,c}[n].
$$

其中

$$
\delta_{c,\mathrm{LFE}}
=
\begin{cases}
1, & c\text{ 为目标布局的 LFE},\\
0, & \text{其他声道}.
\end{cases}
$$

没有 LFE 输出的布局不把输入 LFE 混入其他声道。对象完成累加后，再按目标格式要求排列输出声道。

若输出 PCM24，量化关系为

$$
y_{24}[n]
=
\operatorname{trunc}\left(
8388607\,\operatorname{clip}(y[n],-1,1)
\right).
$$

## 14. 公式适用范围

- JOC 矩阵部分描述 dense JOC；Sparse JOC 使用不同的稀疏系数/索引路径。
- 扬声器声像部分描述普通点对象；extent、spread、divergence 等模式需要额外模型。
- 多个 OAMD position block 必须按其时间顺序调度。
- limiter 属于独立后处理，不包含在上述混音公式中。
