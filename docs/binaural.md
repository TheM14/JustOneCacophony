# 双耳渲染

[English](binaural.en.md) · [返回 README](../README.md)

JOC 的双耳输出有两条输入路线：

| 输入 | 渲染 | 距离与环境 |
|---|---|---|
| `.personalized_headphone`（默认） | Rosella DSP、滤波器组与模型参数 | 模型自身的 Near/Mid/Far 与房间处理，行为不变 |
| `.jochrtf`（保留项） | 77 带方向场、64-QMF/hybrid 分析与合成 | 默认只输出直达声；`--jochrtf-room on` 开启项目房间 |

`.jochrtf` 由独立的离线编译器从 SOFA 测量数据拟合得到；本仓库负责加载和渲染
编译结果，编译器产出的缓存用 `--compiled-hrtf-cache` 加载。

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf --jochrtf-room on
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf --jochrtf-room on
```

不指定输入时使用项目目录下的 `HRTF/binaural.personalized_headphone`；`.jochrtf` 只能
显式选择，不自动生成，也不回退。双耳 DSP 全程 float64/complex128，参数每 128 样本更新。

## 点声源参考尺度

`.jochrtf` 路线沿用 Near 约 1 m、Mid 约 2.193 m、Far 约 6.402 m 的点声源参考尺度，默认
Mid。对于 ADM 右/前/上坐标 `p=(x,y,z)`：

```text
q_SOFA    = (y, -x, z) / ||p||
d_nominal = reference_distance_m * ||p||
```

参考距离只给对象坐标规定标称米制尺度，不改变方向场的响应，也不额外增加 1/r 衰减或
传播延迟。声源与听者重合时沿用该对象最后一个有效方向，没有历史时取正前方。

## `.jochrtf` 房间

默认关闭。`--jochrtf-room on` 启用项目自有房间，算法与 Rosella 之前的公开路线一致，
直接在代码里计算，不读取任何资产：

- 六个一阶镜像反射（`public_room.first_order_image_sources`，18×18×14 m 原房间几何）；
- 四路正交 Hadamard 反馈、两级全通扩散、低通阻尼的共享 FDN 晚期（RT60 0.85 s）；
- 早反射按测量半径/路径距离给增益，晚期发送按档位与归一化半径给增益。

它不是测量 BRIR，也不是外部标准房间。关闭时跳过房间处理，只输出方向场的双耳直达声；
房间参数在构造时确定，没有预设、没有外部文件。

## 对象接入

输入为每帧 `[1536,16]` PCM：0 是 LFE，1–15 是已经重建的对象／bed 槽。直接使用 ID11
时间线，不重新解码，也不先渲染到扬声器布局。保留 1473 样本对象时间补偿和 32 样本元数据
对齐，参数每 128 样本更新。

LFE 继续使用独立低通处理，不发送至房间。多位置块、ISF、显式无穷远以及需要屏幕校准的
锚定明确报错。`finish()` 仅排空一次，`--binaural-tail-seconds` 规定滤波器组与房间尾声的
排空预算。

## 轻量格式

`.jochrtf` v1 是无 pickle 的 NumPy ZIP，保存 JSON 元数据、77 个频带中心、
`complex128[36,2,77]` 方向系数、`float64[36,2]` 延迟系数及 `float64[2,2]` 边界。加载时
校验格式版本、形状、标量范围，按成员重算载荷哈希，并重新计算 cache key 以确认系数与
配置一致；不匹配的文件直接拒绝。

格式、cache key 与哈希校验由本仓库维护；拟合编译器在独立目录里，两者共用同一个格式
模块，不复制定义。压缩表示保留自身的角向与频带近似误差，不等同于逐点测量的精度。

滤波器组来源见[data说明](../data/README.md)。方向插值与 MCA 的参考文献见
[研究实现依据](https://doi.org/10.1109/TASLP.2023.3313908)；算法独立实现，外部测量数据
仍适用其自身许可。
