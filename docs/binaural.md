# 双耳渲染

[English](binaural.en.md) · [返回 README](../README.md)

## 三条独立路线

| 输入 | 直接声 | 距离与环境 |
|---|---|---|
| SOFA | 完整复数HRIR局部插值、自适应长度FIR、128样本分区卷积 | 外部参考配置的Near/Mid/Far |
| `.jochrtf` | 保留原77带／五阶方向场及延迟表示 | 与SOFA相同的三档参考合同 |
| `.personalized_headphone` | 原DSP及滤波器组 | 原Near/Mid/Far及环境处理 |

SOFA不会调用编译器、生成压缩缓存或自动回退。原压缩格式、校验器和离线编译能力保留。旧 `sofa_binaural_backend.py` 作为已有底层实现仍在，CLI工厂使用新的 `ReferenceSofaBackend`。

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --reference-config C:\config\reference.json
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-interpolation lf_mca_zero
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --binaural-mode near
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --binaural-mode far
python src\compile_hrtf.py C:\HRTF\subject.sofa C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
```

默认SOFA为项目目录下的 `HRTF/binaural.sofa`，路径不依赖当前工作目录；其他位置用 `--sofa-hrtf PATH` 显式指定。单球壳自动选择，多球壳须用 `--hrtf-radius-m` 指定最近的测量球壳。个性化默认路径不变。三条双耳路线都支持 `--binaural-mode near|mid|far`，CLI默认mid；off仅适用于ADM DBMD提示。

## 外部参考配置

`--reference-config` > `BINAURAL_REFERENCE_CONFIG` > 本地 `local/binaural_reference.json`。缺失、SHA-256不符或不支持的映射明确报错。环境数据与具体身份说明不随源码提供。配置示例：

```json
{"schema":"binaural-reference-v1","profile_id":"reference-v1",
 "model_path":"C:/path/to/reference-data.json","model_sha256":"<64 hex digits>",
 "profile_index":3,"room_impulse_slots":4096,"level_calibration_directions":128,
 "compiled_receiver_positions_m":[[0,0.09,0],[0,-0.09,0]]}
```

`model_path` 指向受现有参数加载器支持的外部数据，支持相对配置文件的路径；`profile_index` 固定公共电平校准锚点，选择渲染档位使用 `--binaural-mode`，不改校准锚点。压缩格式没有耳位字段，因此压缩输入必须显式配置耳位，且报告其来源。完整SOFA使用文件自身耳位。不同SOFA共用环境尺度，测量半径不定义场景距离或房间大小。

参考映射执行所选档位的量化、轴缩放、边界与近中心正则规则。有限盒外位置沿边界射线延伸，保留ETSI距离因子，不能再次裁回坐标盒。Near共同直接声系数为1、房间发送为0；Mid/Far为：

```text
r_eff = h0 * reference_distance_m
a = 1 / sqrt(1 + c * r_eff^2)
b = r_eff * a
```

房间发送乘对象信号后进入独立环境状态。完整SOFA直接声与环境分支分别处理，再对齐实现延迟并求和；没有额外的1/r或绝对飞行时间。输出增益只应用一次，不隐式限幅。

## 完整HRIR数学与时间基准

严格导入器继续校验 `SimpleFreeFieldHRIR` 0.4/1.0/1.1、FIR、两个可由位置识别左右的耳、零Emitter偏移、自由场、单位和有限数值。重采样时Data.Delay的采样点数同比例变化。当前距离模型要求固定、对称并以听者中心为原点的耳位；不支持的几何明确拒绝。

SOFA为前/左/上，ADM为右/前/上，变换为 `(y,-x,z)`。单位方向三维凸包构建局部三角网格，查询使用射线与平面三角形的非负、和为1的重心权重；不做经纬平面插值或外推。精确节点返回完整复数响应。跨越大测量缺口的面仍是低支持区域，网格包围原点不能证明覆盖精度。

FFT最少2048点，按完整HRIR与Data.Delay自动扩展为满足时间预算的2次幂，自动长度上限16384点；不裁短输入。每耳在200–1500 Hz作线性相位回归，对齐后插值完整剩余复数响应，再恢复加权延迟。Data.Delay恰好施加一次，不能从耳位再加一次ITD。默认 `lf_logmag_phase` 对数幅度与对齐复数插值相位结合；可选 `lf_mca_zero` 使用41个连续Gammatone功率近似带、PCHIP和±12 dB修正限幅。这是独立MCA近似，并非第三方项目逐位复现。

距离补充采用归一化刚性球传递函数 `H_sphere(r)/H_sphere(r_measurement)` 的复数比值。测量半径处严格为1；不叠加第二套耳间延迟。球半径由对称耳位推导。中心使用参考正则半径与零ITD的球面面积加权能量平均响应平滑混合。该模型是人头径向变化的近似，不能把角度插值实验当作任意距离准确性证明。

直接声电平按固定等面积方向及100–16000 Hz对数频率权重，与校准锚点的参考直接声匹配总双耳能量；每份SOFA只得到一个全局系数，三档共用该系数，不逐档、逐方向、逐耳或逐频率归一化。三档共用相同的HRIR插值、距离补充、卷积和时间补偿流程，只切换参考参数与其直接声/发送规则。

保留选定FFT长度的全部FIR采样，共同移位FFT长度的一半以存储前振铃。渲染时使用全方向双耳低频延迟平均值的中位数、取整为单一录音时间原点，给环境分支补齐该原点与实现延迟后统一补偿；报告此偏移。查询得到的原始复数HRIR不改时间基准。这个全局时间参考变换不根据目标距离增加传播时间。

每对象独立卷积历史，更新时旧/新输出线性交叉淡化128样本，保持输入历史及独立overlap；环境发送平滑更新。环境计算按1536样本批处理，不改变128样本参数时间线，排空时裁掉额外计算填充。采样率固定48 kHz，内部float64/complex128。

## 对象接入与支持范围

输入为每帧 `[1536,16]` PCM：0是LFE，1–15是已经重建的对象/bed槽。直接使用ID11时间线，不重新解码，不先转扬声器布局。保留1473样本内容时间补偿和32样本元数据对齐；新后端每128样本更新参数。

SOFA专用解析开关保留负高度、边界外因子及锚定字段；旧解析器默认行为与个性化时间线不变。有限盒外更新进入时间插值；当前多位置块、ISF、显式无穷远和需要屏幕校准的锚定明确返回结构化不支持错误。`finish()`只排空一次，并保留配置的尾声预算。

## 轻量格式

`.jochrtf` v1仍为无pickle的NumPy ZIP，保存JSON元数据、77个频带中心、`complex128[36,2,77]`方向系数、`float64[36,2]`延迟系数和`float64[2,2]`边界。原格式、cache key、哈希校验、原子写入及源数据许可关系不变。它保留方向压缩的固有误差；与完整SOFA的距离语义一致不代表频响精度相同。滤波器组公开来源见 [data说明](../data/README.md)。

数学依据：[Duda与Martens 1998](https://www.ece.ucdavis.edu/cipic/wp-content/uploads/sites/12/2015/04/cipic_JASA_Nov_1998.pdf)，[MCA原论文](https://doi.org/10.1109/TASLP.2023.3313908)，[ETSI TS 103 420](https://www.etsi.org/deliver/etsi_ts/103400_103499/103420/01.02.01_60/ts_103420v010201p.pdf)。算法独立实现，不复制第三方HRTF源码；数据许可仍随外部数据适用。
