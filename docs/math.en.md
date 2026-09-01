# JustOneCacophony — E-AC-3 JOC decoding and rendering mathematics

[中文](math.md) · [Back to README](../README.en.md)

This document covers only the signal model and formulas used in the JustOneCacophony research path: how JOC parameters combine with core PCM to reconstruct object signals, and how OAMD coordinates become speaker gains.

The formulas describe the dense-JOC and ordinary point-object paths studied by the project. They are not a complete definition of every E-AC-3 JOC variant.

## 1. Overall path and notation

Object reconstruction:

```text
E-AC-3 core 5.1 PCM
  + ID14 JOC matrix parameters
  → analysis QMF
  → parameter-band expansion and time interpolation
  → object matrix
  → inverse QMF
  → LFE + 15 object PCM channels
```

Speaker rendering:

```text
LFE + 15 object PCM channels
  + ID11 OAMD coordinates and update timing
  → target-layout region
  → equal-power panning
  → position compensation
  → sample-wise gain ramp
  → speaker PCM
```

Main notation:

| Symbol | Meaning |
|---|---|
| $c=0\ldots4$ | core channels L, R, C, Ls, Rs |
| $o=0\ldots14$ | 15 JOC objects |
| $b=0\ldots63$ | complex QMF subbands |
| $t=0\ldots23$ | 24 64-sample slots per frame |
| $p(b)$ | JOC parameter band corresponding to QMF subband $b$ |
| $X_{c,b,t}$ | analysis-QMF value for a core channel |
| $M_{o,c,b,t}$ | object-matrix coefficient |
| $Z_{o,b,t}$ | inverse-QMF input for an object |
| $y_o[n]$ | time-domain object PCM |

The number of samples in one frame is

$$
N_f=1536=24\times64.
$$

## 2. Dense-JOC matrix parameters

### 2.1 Differential reconstruction

Let `quant_idx` be $q_i\in\{0,1\}$. The number of quantization levels is

$$
N_q=
\begin{cases}
96, & q_i=0,\\
192, & q_i=1.
\end{cases}
$$

The center offset is

$$
O_q=\frac{N_q}{2}.
$$

For object $o$, data point $d$, core channel $c$, and parameter band $p$, the coded difference $\Delta_{o,d,c,p}$ reconstructs to

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

### 2.2 Dequantization

The dequantized matrix coefficient is

$$
D_{o,d,c,p}
=
\left(Q_{o,d,c,p}-\frac{N_q}{2}\right)
\frac{820}{4096(1+q_i)}.
$$

The effective denominator is therefore 4096 in coarse mode and 8192 in fine mode.

### 2.3 JOC clipgain

If the clipgain field consists of integer $x$ and mantissa $y$, then

$$
G_{\mathrm{clip}}
=
1+\frac{y}{32}2^{x-4}.
$$

It is applied to object PCM after inverse QMF and does not apply to LFE.

## 3. Parameter-band expansion and time interpolation

### 3.1 Parameter bands to QMF subbands

The JOC matrix is coded in parameter bands, while the QMF contains 64 subbands. Let $p(b)$ identify the parameter band containing subband $b$. A parameter-band coefficient expands as

$$
D_{o,d,c,b}=D_{o,d,c,p(b)}.
$$

The common 12-band mapping is

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

Thus $p(b)=k$ if and only if $b\in\mathcal B_k$. Other parameter-band counts use their corresponding subband boundaries.

### 3.2 One-data-point interpolation

Let $P_{o,c,b}$ be the previous frame-end value and $D_{o,c,p(b)}$ the current target. For slot $t=0\ldots23$:

$$
\alpha_t=\frac{t+1}{24},
$$

$$
M_{o,c,b,t}
=
(1-\alpha_t)P_{o,c,b}
+\alpha_tD_{o,c,p(b)}.
$$

The first slot has therefore advanced by $1/24$ of the ramp, while the last slot equals the current target:

$$
M_{o,c,b,23}=D_{o,c,p(b)}.
$$

This value then becomes the previous state for the next frame.

### 3.3 Multiple data points

When a frame contains two data points, `offset_ts` gives the segment boundary. Each segment uses the same linear relation between the previous and next targets; step mode switches targets at the designated slot.

## 4. Analysis QMF for core PCM

The matrix input uses core channels L, R, C, Ls, and Rs; LFE follows a separate path. Core PCM is first scaled as

$$
\widetilde x_c[n]=\frac{x_c[n]}{16}.
$$

Let $\mathcal A_b$ denote the 64-band analysis-QMF operator with polyphase history state. Then

$$
X_{c,b,t}
=
\mathcal A_b\!\left(
\widetilde x_c[64t],\ldots,\widetilde x_c[64t+63];
\mathbf s^{\mathrm A}_{c,t}
\right).
$$

This consists of the analysis window/polyphase stage, modulation, a 64-point FFT, and subband reordering. History state advances continuously across slots and frames.

## 5. QMF-domain processing of core channels

L, R, and C are delayed by ten QMF slots before entering the object matrix:

$$
\widehat X_{c,b,t}=X_{c,b,t-10},
\qquad c\in\{L,R,C\}.
$$

Ls and Rs use the same ten-slot delay and a $-j$ rotation for $b>0$:

$$
\widehat X_{c,b,t}=-jX_{c,b,t-10},
\qquad c\in\{Ls,Rs\},\ b>0.
$$

Band 0 of each surround channel additionally passes through a 21-tap complex FIR:

$$
\widehat X_{c,0,t}
=
\sum_{k=0}^{20}h_kX_{c,0,t-k}.
$$

These delays and filter histories are decoder state and cannot be reset independently for every frame.

## 6. Object matrix

For each object $o$, subband $b$, and slot $t$, the object's frequency-domain value is a linear combination of the five core channels:

$$
Z_{o,b,t}
=
\sum_{c=0}^{4}
M_{o,c,b,t}\widehat X_{c,b,t}.
$$

The $1/16$ analysis-input scale is canceled by the $\times16$ factor after inverse QMF, so the matrix itself needs no additional empirical gain.

## 7. Object inverse QMF

### 7.1 Subband reorder

Write the 64 complex subbands as 128 interleaved real values in `src`. For $k=0\ldots31$:

$$
\begin{aligned}
\operatorname{zone}[2k] &= \operatorname{src}[4k],\\
\operatorname{zone}[2k+1] &= -\operatorname{src}[4k+1],\\
\operatorname{zone}[126-2k] &= \operatorname{src}[4k+2],\\
\operatorname{zone}[127-2k] &= \operatorname{src}[4k+3].
\end{aligned}
$$

Treat `zone` as 64 complex values and apply an unnormalized 64-point FFT:

$$
F_k
=
\sum_{n=0}^{63}
\operatorname{zone}_n
\exp\!\left(-j\frac{2\pi kn}{64}\right).
$$

### 7.2 Modulation and synthesis

Define the rotation coefficient

$$
r_k
=
\frac12\left(
\sin\frac{\pi k}{128}
+j\cos\frac{\pi k}{128}
\right),
$$

and compute

$$
R_k=2F_kr_k.
$$

Let $\mathcal S$ denote polyphase synthesis with a 640-value synthesis window and cross-slot state:

$$
\mathbf y_{o,t}
=
\mathcal S\!\left(
\mathbf R_{o,t},W,\mathbf s^{\mathrm S}_{o,t}
\right).
$$

Object output is

$$
y_o[64t+r]
=
\operatorname{clip}\!\left(
16\,\mathbf y_{o,t}[r],-1,1
\right)G_{\mathrm{clip}},
$$

where $r=0\ldots63$. Synthesis state must advance continuously by slot.

## 8. LFE path

LFE bypasses the object matrix and inverse QMF and uses a 1217-sample delay. After the input and output scale factors cancel:

$$
y_{\mathrm{LFE}}[n]
=
\operatorname{clip}\!\left(
x_{\mathrm{LFE,core}}[n-1217],-1,1
\right).
$$

## 9. OAMD coordinates

The lateral and longitudinal grids use $N=62$; the height grid uses $N=15$. The quantizer is

$$
q_N(k)
=
\min\!\left(
32767,
\left\lfloor\frac{32768k}{N}+\frac12\right\rfloor
\right).
$$

OAR coordinates are

$$
u=\frac{q_1}{32768},
\qquad
v=\frac{q_2}{32768},
\qquad
w=\frac{q_3}{32768}.
$$

Their maximum runtime value is $32767/32768$, not exactly 1.

For conversion to the ADM grid:

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

The continuous-coordinate relation is

$$
u=\frac{X+1}{2},
\qquad
v=\frac{1-Y}{2},
\qquad
w=Z.
$$

## 10. Equal-power speaker panning

### 10.1 One-dimensional interpolation

Let adjacent speaker coordinates be $a_0<a_1$ and object position be $a$. The normalized position is

$$
\tau=\frac{a-a_0}{a_1-a_0}.
$$

Gains inside the interval are

$$
g_0(\tau)=\cos\left(\frac\pi2\tau\right),
\qquad
g_1(\tau)=\sin\left(\frac\pi2\tau\right),
$$

and satisfy

$$
g_0^2(\tau)+g_1^2(\tau)=1.
$$

Object positions outside the interval are clamped to the nearest endpoint.

### 10.2 Two-dimensional regions

Each row first produces a horizontal gain vector $\mathbf h_r(u)$. If the object lies between adjacent rows $r_0,r_1$:

$$
\eta=\frac{v-v_{r_0}}{v_{r_1}-v_{r_0}},
$$

$$
a_0=\cos\left(\frac\pi2\eta\right),
\qquad
a_1=\sin\left(\frac\pi2\eta\right).
$$

The two-dimensional point gain is

$$
\mathbf G_{\mathrm{2D}}(u,v)
=
\mathbf h(u)\odot\mathbf v(v).
$$

For 5.1-family layouts with one horizontal surround pair rather than separate side and rear pairs, the longitudinal coordinate is

$$
v_{\mathrm{floor}}
=
\operatorname{clamp}(2v,0,1).
$$

Other layouts use $v_{\mathrm{floor}}=v$.

### 10.3 Height layer

Three-dimensional layouts compute floor gain $\mathbf G_f$ and height gain $\mathbf G_h$ separately:

$$
\mathbf G_{\mathrm{point}}(u,v,w)
=
\cos\left(\frac\pi2w\right)\mathbf G_f
+
\sin\left(\frac\pi2w\right)\mathbf G_h.
$$

When the floor and height speaker sets do not overlap and each layer uses equal-power interpolation:

$$
\left\|\mathbf G_{\mathrm{point}}\right\|_2=1.
$$

## 11. Layout-dependent position compensation

Let $N_h$ be the number of relevant height speakers and $N_f$ the number of relevant additional horizontal speakers:

$$
H=\min\left(\frac{N_h}{4},1\right),
\qquad
F=\min\left(\frac{N_f}{4},1\right).
$$

Maximum position compensation is

$$
A_{\max}
=
-\max\left(4.5-1.5H-3F,0\right)
\quad\text{dB}.
$$

Longitudinal and height weights are

$$
p_v=\operatorname{clamp}\left(\frac v{0.6},0,1\right),
$$

$$
p_w=\operatorname{clamp}\left(\frac{w-0.2}{0.8},0,1\right),
$$

$$
p=\operatorname{clamp}(p_v+p_w,0,1).
$$

The linear compensation gain is

$$
G_{\mathrm{pos}}=10^{A_{\max}p/20}.
$$

The object's target-gain vector is

$$
\mathbf G_{\mathrm{target}}
=
G_{\mathrm{object}}
G_{\mathrm{pos}}
\mathbf G_{\mathrm{point}}.
$$

## 12. OAMD time alignment and gain ramps

The coded position of an OAMD update is

$$
s_{\mathrm{coded}}
=
s_{\mathrm{frame}}
+s_{\mathrm{outer}}
+s_{\mathrm{OAMD}}
+32f_{\mathrm{block}}.
$$

For processing-block length $B=32$, the aligned update point is

$$
\widehat s
=
B\left\lfloor
\frac{s_{\mathrm{coded}}+B/2-1}{B}
\right\rfloor.
$$

For ramp duration $D$, the number of blocks is

$$
K
=
\left\lfloor
\frac{D+B/2-1}{B}
\right\rfloor.
$$

If current gain is $g_0$ and target gain is $g_1$, the increment per block is

$$
\Delta g=\frac{g_1-g_0}{K}.
$$

Sample $r=0\ldots B-1$ of block $j$ uses

$$
g_{j,r}=g_j+\frac rB\Delta g,
\qquad
g_{j+1}=g_j+\Delta g.
$$

If no new metadata update intervenes, this is equivalent to a sample-wise linear ramp of total length $KB$.

## 13. Final speaker mix

For target output channel $c$:

$$
y_c[n]
=
\delta_{c,\mathrm{LFE}}x_{\mathrm{LFE}}[n]
+
\sum_{o=1}^{15}x_o[n]g_{o,c}[n].
$$

Here

$$
\delta_{c,\mathrm{LFE}}
=
\begin{cases}
1, & c\text{ is the target layout's LFE channel},\\
0, & \text{otherwise}.
\end{cases}
$$

A layout without LFE output does not mix input LFE into other channels. After object accumulation, output channels are ordered as required by the target format.

For PCM24 output, quantization is

$$
y_{24}[n]
=
\operatorname{trunc}\left(
8388607\,\operatorname{clip}(y[n],-1,1)
\right).
$$

## 14. Scope of the formulas

- The JOC matrix section describes dense JOC; Sparse JOC uses a different sparse coefficient/index path.
- The speaker-panning section describes ordinary point objects; extent, spread, divergence, and similar modes require additional models.
- Multiple OAMD position blocks must be scheduled in time order.
- A limiter is separate post-processing and is not included in the mixing equations above.
