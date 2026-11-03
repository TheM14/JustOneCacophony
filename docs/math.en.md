# JustOneCacophony — E-AC-3 JOC decoding and rendering mathematics

[中文](math.md) · [Back to README](../README.en.md)

This document covers only the signal model and formulas used in the JustOneCacophony research path: how JOC parameters combine with core PCM to reconstruct object signals, and how OAMD coordinates become speaker gains.

The formulas describe the JOC matrix parameters (both the dense and the sparse differential syntax) and the ordinary point-object paths studied by the project. They are not a complete definition of every E-AC-3 JOC variant.

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

## 2. JOC matrix parameters

For every object and data point, the quantized matrix `joc_mix_mtx_q` is defined on $N_q$ quantization levels. The `b_joc_sparse` flag selects one of two differential syntaxes: dense sends one MTX difference per core channel, while sparse sends one active channel plus one coefficient difference per parameter band.

### 2.1 Dense differential reconstruction

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
Q_{o,d,c,0}=
\left(O_q+\Delta_{o,d,c,0}\right)\bmod N_q,
$$

$$
Q_{o,d,c,p}=
\left(Q_{o,d,c,p-1}+\Delta_{o,d,c,p}\right)\bmod N_q,
\qquad p>0.
$$

### 2.2 Sparse differential reconstruction

Let $I_{o,d,p}$ be the `joc_channel_idx` symbol (IDX), $V_{o,d,p}$ the `joc_vec` symbol (VEC), and $N_c\in\lbrace5,7\rbrace$ the number of core channels. Each parameter band has exactly one active channel:

$$
A_{o,d,p}=
\begin{cases}
I_{o,d,0}, & p=0,\\
\left(A_{o,d,p-1}+I_{o,d,p}\right)\bmod N_c, & p>0,
\end{cases}
$$

where $I_{o,d,0}$ is a 3-bit absolute channel index and every later IDX symbol is an increment relative to the previous **active channel**. The coefficient is a single accumulator running across parameter bands:

$$
\kappa_{o,d,-1}=O^{(s)}_q,\qquad
\kappa_{o,d,p}=
\left(\kappa_{o,d,p-1}+V_{o,d,p}\right)\bmod N_q,
$$

with a sparse starting point two quantization levels above the dense center offset:

$$
O^{(s)}_q=
\begin{cases}
50, & q_i=0,\\
100, & q_i=1.
\end{cases}
$$

The accumulator is **not** reset when the active channel changes. The complete matrix is

$$
Q_{o,d,c,p}=
\begin{cases}
\kappa_{o,d,p}, & c=A_{o,d,p},\\
\dfrac{N_q}{2}, & c\neq A_{o,d,p}.
\end{cases}
$$

Non-active entries take $N_q/2$, which dequantizes to exactly 0.

### 2.3 Dequantization

The dequantized matrix coefficient is

$$
D_{o,d,c,p}=
\left(Q_{o,d,c,p}-\frac{N_q}{2}\right)
\frac{820}{4096(1+q_i)}.
$$

The effective denominator is therefore 4096 in coarse mode and 8192 in fine mode.

### 2.4 JOC clipgain

If the clipgain field consists of integer $x$ and mantissa $y$, then

$$
G_{\mathrm{clip}}=
1+\frac{y}{32}2^{x-4}.
$$

It is folded into the analysis-side scalar $S=\frac{1}{16}G_{\mathrm{clip}}$, which multiplies
every core-bed sample before the QMF analysis; the $\times16$ after the object inverse QMF cancels the
$1/16$, so the net effect is a multiply by $G_{\mathrm{clip}}$ and the $[-1,1]$ clamp sits **after** the
gain. The scalar keeps two per-frame slots: the first 256 samples of a frame ramp linearly from the
previous frame's value to this frame's value and the rest stays constant, and the first frame after a
reset takes the new value directly. LFE bypasses this scalar.

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
M_{o,c,b,t}=
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

The matrix input uses $N$ core channels ($N\in\{5,7\}$, read from $joc\_dmx\_config\_idx$: $k=0\ldots4$ maps to $N=5,7,7,5,7$); LFE follows a separate path. Core PCM is first scaled as

$$
\widetilde x_c[n]=\frac{x_c[n]}{16}.
$$

Let $\mathcal A_b$ denote the 64-band analysis-QMF operator with polyphase history state. Then

$$
X_{c,b,t}=
\mathcal A_b\left(
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

Ls and Rs take the same delay inside the surround path, whose delay line advances
regardless; the $-j$ rotation for $b>0$ and the band-0 21-tap complex FIR are gated by
the **same switch** $\sigma(k)=1\iff k\in\{3,4\}$:

$$
\widehat X_{c,b,t}=
\begin{cases}
-j\,X_{c,b,t-10}, & \sigma(k)=1,\ c\in\{Ls,Rs\},\ b>0,\\[1mm]
\sum_{m=0}^{20}h_mX_{c,0,t-m}, & \sigma(k)=1,\ c\in\{Ls,Rs\},\ b=0,\\[1mm]
X_{c,b,t-10}, & \text{otherwise}.
\end{cases}
$$

For $N=7$ the sixth and seventh core channels (Lb/Rb or Tfl/Tfr) behave like the front
channels and only take the ten-slot delay: channel identity affects metadata and rendering,
not the DSP. These delays and filter histories are decoder state and cannot be reset
independently for every frame.

## 6. Object matrix

For each object $o$, subband $b$, and slot $t$, the object's frequency-domain value is a linear combination of the $N$ core channels:

$$
Z_{o,b,t}=
\sum_{c=0}^{N-1}
M_{o,c,b,t}\widehat X_{c,b,t}.
$$

The $1/16$ analysis-input scale is canceled by the $\times16$ factor after inverse QMF, so the matrix itself needs no additional empirical gain.

## 7. Object inverse QMF

### 7.1 Subband reorder

Write the 64 complex subbands as 128 interleaved real values in `src`. For $k=0\ldots31$:

$$
\begin{aligned}
\mathrm{zone}[2k] &= \mathrm{src}[4k],\\
\mathrm{zone}[2k+1] &= -\mathrm{src}[4k+1],\\
\mathrm{zone}[126-2k] &= \mathrm{src}[4k+2],\\
\mathrm{zone}[127-2k] &= \mathrm{src}[4k+3].
\end{aligned}
$$

Treat `zone` as 64 complex values and apply an unnormalized 64-point FFT:

$$
F_k=
\sum_{n=0}^{63}
\mathrm{zone}_n
\exp\left(-j\frac{2\pi kn}{64}\right).
$$

### 7.2 Modulation and synthesis

Define the rotation coefficient

$$
r_k=
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
\mathbf y_{o,t}=
\mathcal S\left(
\mathbf R_{o,t},W,\mathbf s^{\mathrm S}_{o,t}
\right).
$$

Object output is

$$
y_o[64t+r]=
\mathrm{clip}\left(
16\,\mathbf y_{o,t}[r],-1,1
\right)G_{\mathrm{clip}},
$$

where $r=0\ldots63$. Synthesis state must advance continuously by slot.

## 8. LFE path

LFE bypasses the object matrix and inverse QMF and uses a 1217-sample delay. After the input and output scale factors cancel:

$$
y_{\mathrm{LFE}}[n]=
\mathrm{clip}\left(
x_{\mathrm{LFE,core}}[n-1217],-1,1
\right).
$$

## 9. OAMD coordinates

The lateral and longitudinal grids use $N=62$; the height grid uses $N=15$. The quantizer is

$$
q_N(k)=
\min\left(
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
\mathbf G_{\mathrm{2D}}(u,v)=
\mathbf h(u)\odot\mathbf v(v).
$$

For layouts with one horizontal surround pair rather than separate side and rear pairs (the 5.1
family, and 4.0, which carries one rear pair only), OAMD `warp_mode=1` uses

$$
v_{\mathrm{floor}}=
\mathrm{clamp}(2v,0,1).
$$

Warp 0 and other layouts use $v_{\mathrm{floor}}=v$. The height layer is not scaled.
An absent warp indication uses 0; no CLI override is provided. Modes 2/3 use
the intermediate layout and folding matrix in §12.5. Balance follows this scaling.

### 10.3 Height layer

Three-dimensional layouts compute floor gain $\mathbf G_f$ and height gain $\mathbf G_h$ separately:

$$
\mathbf G_{\mathrm{point}}(u,v,w)=
\cos\left(\frac\pi2w\right)\mathbf G_f
+
\sin\left(\frac\pi2w\right)\mathbf G_h.
$$

When the floor and height speaker sets do not overlap and each layer uses equal-power interpolation:

$$
\left\|\mathbf G_{\mathrm{point}}\right\|_2=1.
$$

## 11. Layout-dependent position compensation

This is the per-object gain for automatic trim. Disabled trim uses unity;
explicit trim uses §12.2 instead, without multiplying automatic compensation again.

Let $N_h$ be the number of relevant height speakers and $N_f$ the number of relevant additional horizontal speakers:

$$
H=\min\left(\frac{N_h}{4},1\right),
\qquad
F=\min\left(\frac{N_f}{4},1\right).
$$

Maximum position compensation is

$$
A_{\max}=
-\max\left(4.5-1.5H-3F,0\right)
\quad\text{dB}.
$$

Longitudinal and height weights are

$$
p_v=\mathrm{clamp}\left(\frac v{0.6},0,1\right),
$$

$$
p_w=\mathrm{clamp}\left(\frac{w-0.2}{0.8},0,1\right),
$$

$$
p=\mathrm{clamp}(p_v+p_w,0,1).
$$

The linear compensation gain is

$$
G_{\mathrm{pos}}=10^{A_{\max}p/20}.
$$

The object's target-gain vector is

$$
\mathbf G_{\mathrm{target}}=
G_{\mathrm{object}}
G_{\mathrm{pos}}
\mathbf G_{\mathrm{point}}.
$$

## 12. OAMD trim and front-back balance

### 12.1 Configuration selection

The OAMD trim element splits a programme's calibration data into nine
configurations, chosen by the target layout's additional horizontal speaker
count $N_f$ and height speaker count $N_h$:

$$
\kappa_f(N)=
\begin{cases}
0, & N=0,\\
1, & N\in\{2,3\},\\
2, & \text{otherwise},
\end{cases}
\qquad
\kappa_h(N)=
\begin{cases}
0, & N=0,\\
1, & N=2,\\
2, & \text{otherwise}.
\end{cases}
$$

$$
c_{\mathrm{trim}}=\kappa_f(N_f)+3\,\kappa_h(N_h).
$$

The nine configurations enumerate "no / some / many additional horizontal
speakers by no / some / many height speakers".

Trim and front-back balance apply to seven layouts only: 2.0, 3.1, 4.0, 5.1,
5.1.2, 5.1.4 and 7.1.  Wider layouts (7.1.2, 7.1.4, 9.1.4, 9.1.6) are left
untrimmed, with $g_t=1$ and $b=0$.

Warp, trim and balance controls are read from OAMD metadata:

| Parameter | Value or unit | Behavior |
|---|---|---|
| `warp_mode` | 0 / 1 / 2 / 3 | Direct rendering / floor Y scaling / PLIIx folding / Standard (Lo/Ro) folding |
| `global_trim_mode` | 0 / 1 / 2 | Automatic / disabled / per-layout configuration; reserved value 3 is rejected |
| $T_s$, $T_h$ | dB | Explicit surround and height trim, with positive values treated as zero |
| $T_c$ | dB | Signed explicit centre trim |
| $b_{\mathrm{lis}}$, $b_{\mathrm{tb}}$ | $[-1,1]$, step $1/16$ | Listener-plane and overhead front-back balance |
| Per-object trim bypass | Boolean | Disables the object's balance and trim gain while retaining warp |

With global mode 2, the selected configuration's default or disabled flag
chooses automatic or disabled trim. A missing configuration uses automatic;
otherwise its explicit parameters apply. Automatic uses §11, disabled uses
unity, and explicit uses §12.2. These modes are mutually exclusive.

Without an initial trim element, warp is 0 and trim is automatic. A later
payload without that element retains the previous controls; a new element
updates them at the corresponding metadata event boundary. The CLI has no
warp override.

### 12.2 Per-object levels

Centre, surround and height levels parameterize a position-dependent object
gain; they are not fixed gains applied to the output channels after mixing.
For original Y after balance $y_t$ and $z=\max(w,0)$:

$$
a_s=\mathrm{clamp}(2y_t,0,1)(1-z),\quad a_h=z,
$$

$$
a_c=1-\mathrm{clamp}\bigl(10\max(|z|-0.05,y_t-0.05,|u-0.5|-0.05),0,1\bigr).
$$

Explicit trim uses

$$
D=a_s\min(T_s,0)+a_h\min(T_h,0)+a_cT_c,\qquad g_t=10^{D/20}.
$$

The scalar multiplies the object's whole panning vector before accumulation.
Automatic trim uses §11; disabled trim or per-object bypass uses unity. LFE
passes through. Gain evaluation, panning interpolation and object accumulation
use float64.

### 12.3 Front-back balance

Each configuration also carries two front-back balance values:
$b_{\mathrm{lis}}$ for the listener plane and $b_{\mathrm{tb}}$ for the overhead
layer. Both are quantised to a $1/16$ grid and are non-zero only in an explicit
configuration.

Writing $n=\mathrm{round}(16\,|b|)$ and $s=n/18$, the single-layer map is

$$
B_b(y)=
\begin{cases}
y, & y\le 0.05\ \text{or}\ y\ge 0.95,\\[2pt]
(1+s)y-0.05s, & 0.05<y<0.5,\ b>0,\\[2pt]
(1-s)(y-0.5)+0.5+0.45s, & 0.5\le y<0.95,\ b>0,\\[2pt]
(1-s)y+0.05s, & 0.05<y<0.5,\ b<0,\\[2pt]
(1+s)(y-0.5)+0.5-0.45s, & 0.5\le y<0.95,\ b<0.
\end{cases}
$$

It maps $[0.05,0.95]$ monotonically onto itself: the two walls stay fixed, the
two halves meet at $y=0.5$, and their slopes are $1+s$ and $1-s$. The 5 percent
wall bands and the case $b=0$ pass through unchanged, so the map never leaves
the interval and needs no extra clamp.

The height $w$ blends the two layers:

$$
y'=(1-w)\,B_{b_{\mathrm{lis}}}(y)+w\,B_{b_{\mathrm{tb}}}(y).
$$

The front-back axis $y$ is oriented as in §9, with $0$ at the front wall and
$1$ at the back wall; $x$ and $z$ are unchanged.

### 12.4 Where it is applied

Let $B_z(y)$ denote the balance map of §12.3 and $W(y)$ the warp map of §10.2.
Direct point rendering in warp modes 0/1 keeps three coordinates:

$$
y_f=B_z(W(y)),\qquad y_h=B_z(y),\qquad y_t=B_z(y).
$$

Floor and height panning use $y_f,y_h$; trim evaluation uses $y_t$. The final gain is

$$
g_{o,c}=G_{\mathrm{object}}\,g_t(u,y_t,z)\,P_c(u,y_f,y_h,z).
$$

Warp precedes balance, followed by trim/pan gain composition, but not every
stage shares a warped Y. Balance requires explicit trim without object bypass;
warp remains independent of trim state and bypass. Source positions are kept
unchanged for ADM and binaural.

Floor coordinates pass through warp and then balance before floor panning.
Height and trim coordinates pass through balance alone. The trim scalar
multiplies the object's entire panning vector before accumulation into output
channels. LFE passes through.

### 12.5 Surround folding in warp modes 2/3

For a 5.1, 5.1.2 or 5.1.4 target, warp modes 2/3 first render objects to the
corresponding 7.1, 7.1.2 or 7.1.4 intermediate layout, then fold the rear
surrounds into the target surround pair. The 4.0 layout also adds a rear pair
without adding centre or LFE. Other existing layouts render directly.

The intermediate layout changes speaker geometry and panning only. The trim
configuration, both balance layers and the maximum automatic position
compensation remain those of the final target. For example, 5.1.2 uses
configuration 4 and a maximum compensation of $-2.25$ dB even when its
intermediate layout is 7.1.2. With separate side and rear pairs in the
intermediate layout, floor Y is not doubled. Balance and per-object trim
are evaluated during intermediate rendering.

For intermediate outputs $L_s,R_s,L_{rs},R_{rs}$ and target surrounds
$L'_s,R'_s$:

$$
L'_s=L_s+aL_{rs}+bR_{rs},\qquad
R'_s=R_s+bL_{rs}+aR_{rs}.
$$

| Mode | $a$ | $b$ |
|---|---|---|
| 2: PLIIx | $\sqrt{3}/2$ | $1/2$ |
| 3: Standard (Lo/Ro) | $1$ | $0$ |

Both matrices add linear amplitudes without normalizing energy after the
fold. Shared front, height and LFE channels pass through to their matching
positions. The folded object gain vector enters the target-gain update and
ramp in §13 before object audio is accumulated.

The complete order is: select calibration for the target layout → apply
balance and trim while panning in the intermediate layout → fold horizontal
surrounds → update gain ramps and mix. Per-object bypass disables balance
and trim while retaining layout selection and folding. These modes are
generic speaker-rendering controls; the publicly defined E-AC-3 JOC warp
values remain 0/1. Source trajectories, ADM and binaural do not bake in the
folded result.

## 13. OAMD time alignment and gain ramps

The coded position of an OAMD update is

$$
s_{\mathrm{coded}}=
s_{\mathrm{frame}}
+s_{\mathrm{outer}}
+s_{\mathrm{OAMD}}
+32f_{\mathrm{block}}.
$$

The theoretical update position on the decoder-output PCM timeline is

$$
s_{\mathrm{theoretical}}=
s_{\mathrm{coded}}+d_{\mathrm{decoder}},
\qquad d_{\mathrm{decoder}}=1473.
$$

The speaker renderer retains the existing processing-block length $B=32$, so the aligned update point is

$$
\widehat s=
B\left\lfloor
\frac{s_{\mathrm{theoretical}}+B/2-1}{B}
\right\rfloor.
$$

Thus, for frame-aligned updates, `align32(1473)=1472`. The 1473 value is the theoretical decoder delay at the metadata interface; 1472 is its effective boundary in the current 32-sample control block. The 640-value inverse-QMF window/state is not part of this metadata-timing formula.

For ramp duration $D$, the number of blocks is

$$
K=
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

## 14. Final speaker mix

For target output channel $c$:

$$
y_c[n]=
\delta_{c,\mathrm{LFE}}x_{\mathrm{LFE}}[n]
+
\sum_{o=1}^{15}x_o[n]g_{o,c}[n].
$$

Here

$$
\delta_{c,\mathrm{LFE}}=
\begin{cases}
1, & c\text{ is the target layout's LFE channel},\\
0, & \text{otherwise}.
\end{cases}
$$

A layout without LFE output does not mix input LFE into other channels. After object accumulation, output channels are ordered as required by the target format.

For PCM24 output, quantization is

$$
y_{24}[n]=
\mathrm{trunc}\left(
8388607\,\mathrm{clip}(y[n],-1,1)
\right).
$$

## 15. Scope of the formulas

- The JOC matrix section covers both the dense MTX and the sparse IDX/VEC differential syntax.
- The speaker-panning section describes ordinary point objects; extent, spread, divergence, and similar modes require additional models.
- Multiple OAMD position blocks must be scheduled in time order.
- A limiter is separate post-processing and is not included in the mixing equations above.
- Trim and front-back balance apply to speaker rendering only;
  binaural rendering reads neither of them.
