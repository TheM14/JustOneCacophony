# 双耳渲染

[English](binaural.en.md) · [返回 README](../README.md)

## 三条独立路线

| 输入 | 渲染 | 距离与环境 |
|---|---|---|
| SOFA | 完整复数HRIR方向插值、自适应长度FIR、128样本分区卷积 | 默认纯直达声；`--sofa-room`可选项目房间；不做径向人头合成 |
| `.jochrtf` | 原77带／五阶方向场及延迟表示 | 同样默认纯直达声，可选相同房间 |
| `.personalized_headphone` | 原DSP、滤波器组及模型参数 | 原Near/Mid/Far及房间处理保持不变 |

SOFA不会调用编译器、生成压缩缓存或自动回退。`.jochrtf`格式、校验器和离线编译能力保留。SOFA与压缩输入通过`DirectSofaBackend`接入对象时间线。

```powershell
python main.py input.m4a --binaural
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-interpolation lf_mca_zero
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --source-reference-distance-m 2.193
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-room
python main.py input.m4a --binaural --sofa-hrtf C:\HRTF\subject.sofa --sofa-room --sofa-room-preset legacy
python src\compile_hrtf.py C:\HRTF\subject.sofa C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --compiled-hrtf-cache C:\HRTF\subject.jochrtf
python main.py input.m4a --binaural --personalized-headphone --binaural-mode far --rosella-filterbank 2
```

默认SOFA为项目目录下的`HRTF/binaural.sofa`，不依赖当前工作目录；其他位置用`--sofa-hrtf PATH`指定。单球壳自动选择，多球壳须用`--hrtf-radius-m`选择一个测量球壳，当前不做径向插值。默认个性化路径仍为`HRTF/binaural.personalized_headphone`。

SOFA与压缩输入沿用Near/Mid/Far的点声源参考尺度，默认Mid。个性化的Near/Mid/Far和ADM输出的DBMD提示保持原行为。

## 点声源参考尺度

Near/Mid/Far的参考距离分别为1.00000465／2.19327927／6.40177584 m，可用`--source-reference-distance-m`覆盖所选档位。对于ADM右／前／上坐标`p=(x,y,z)`：

```text
q_SOFA = (y, -x, z) / ||p||
d_nominal = reference_distance_m * ||p||
H_render(ear, f) = H_selected_shell(ear, f, q_SOFA)
```

参考距离给对象坐标规定标称米制尺度，不会把测量半径为1.2 m的HRIR合成为2.193 m真值，也不会更改SOFA测量元数据。默认直达声模式不额外应用1/r衰减、球头距离比值、人工中心响应、房间发送、direct/room能量耦合或目标距离传播延迟；节目增益和SOFA响应刻度保持不变。开启房间后采用原项目距离耦合，用户指定输出增益仍只应用一次。

声源与听者重合时方向没有定义，沿用该对象最后一个有效方向；没有历史时取正前方。该规则不合成零ITD响应。有限盒外坐标保留元数据距离因子，不裁回单位盒。

SOFA输入不需要个性化模型或本机配置；关闭房间时也不读取房间资产。报告记录输入路径、SHA-256、测量半径、插值方法与房间开关。

## 可选项目房间

`--sofa-room`默认关闭，复用原`.jochrtf`路线的`public_room.py`和`reference_distance.py`算法，提供原始与紧凑两套参数：六个一阶镜像反射，以及四路正交Hadamard反馈、两级全通扩散、低通阻尼的共享FDN。它是项目自有听感预设，目标是提供接近理想听音环境、以直达声为主的房间感，不是测量BRIR或强制标准房间。它不依赖Rosella房间表。

启用后Near/Mid默认使用`compact`紧凑房间，Far默认使用`legacy`原大房间，以保留Far单位坐标盒的空间范围。`--sofa-room-preset compact|legacy`可显式选择；`legacy`指原项目大房间。紧凑房间尺寸引用[Zahorik 2009表1模型9](https://pmc.ncbi.nlm.nih.gov/articles/PMC2730711/)，保留原镜像／FDN算法，将晚期延迟、扩散时间与RT60参数按`V/S`比例缩短。参数及来源记录在资产元数据中。这仅借用文献几何，不是该论文BRIR算法的复现。

完整HRIR路线对六个到达方向分别查询HRIR，以完整频率网格上的相位斜坡实现分数采样延迟。反射路径相对直达路径的额外时间为`(L_path-L_direct)/c`，保留原HRIR的`Data.Delay`，不再量化到77带。直接声、反射与晚期发送采用原项目的增益公式。房间几何只覆盖其内部声源；开启房间时不静默裁剪盒外位置。

晚期FDN在48 kHz下用单位脉冲生成双声道float64 FIR：紧凑版为`data/sofa-room-compact.npz`，37,383点、RT60参数约0.3894秒；原版为`data/sofa-room.npz`，81,600点、RT60参数0.85秒。两者截断后的探测尾能量比均约`2.2e-13`。运行时通过PCM域FFT卷积实现各自同一线性时不变系统。早反射继续使用所选SOFA；晚期输出保持原项目的双声道FDN定义。

晚期分支补齐卷积实现移位和所选SOFA的共同录音延迟估计后，与直接声及早反射相加。房间关闭时不创建这些状态；开启时按1536样本计算批次处理，参数仍每128样本更新，排空时裁去额外批处理填充。`.jochrtf`的早反射保持其77带表示误差。

可用`python src\generate_sofa_room.py`重生成两套资产，或以`--preset compact|legacy`只生成一套。原算法来源、参数、系数哈希及尾部精度预算写入NPZ元数据。

## 完整HRIR数学与时间基准

导入器校验`SimpleFreeFieldHRIR` 0.4/1.0/1.1、FIR、两个可由位置识别左右的耳、零Emitter偏移、自由场、单位和有限数值。采样率统一为48 kHz；重采样时`Data.Delay`的采样点数同比例变化。耳位须在测量之间固定；方向插值不再要求用对称耳位拟合球头。

当前没有实现`SimpleFreeFieldHRIR`以外的BRIR或其他SOFA Convention。不支持的文件明确拒绝。后端已按响应类型预留BRIR提示及忽略`--sofa-room`的规则；当前验证的是该策略接口，不代表已实现BRIR读取。

测量单位方向的三维凸包构成局部三角网格，查询使用射线与面三角形的非负、和为1的重心权重。不使用经纬平面插值，不外推未覆盖半球。精确节点返回测量复数响应。

低频对齐延迟在200–1500 Hz拟合。默认`lf_logmag_phase`对对数幅度插值，原相位通过时间对齐的复数残差及插值延迟恢复；备选`lf_mca_zero`使用听觉带幅度校正。两种方法都保留所选FFT长度的完整FIR，不替换为最小相位。

`Data.Delay`在频域只叠加一次。公共移位为FFT长度的一半，用于保存插值前振铃；输出只补偿卷积／滤波器组的实现延迟。文件提供的共同录音延迟、左右耳相对延迟和频率相关相位关系均保留，不再减去拟合录音原点，也不根据目标距离增加飞行时间。真实序列的DC／Nyquist频点按实数端点处理。

每对象拥有独立卷积历史。更新时旧／新输出在128样本内线性交叉淡化，两个滤波器均使用完整输入历史。滤波器更新同样延后公共FIR实现移位，避免移除实现延迟后将元数据更新提前。完整HRIR路线直接在PCM域卷积；QMF/hybrid仅用于显式压缩输入。内部精度为float64/complex128，不做隐式限幅。

## 对象接入

输入为每帧`[1536,16]` PCM：0是LFE，1–15是已经重建的对象／bed槽。直接使用ID11时间线，不重新解码，也不先渲染到扬声器布局。保留1473样本对象时间补偿和32样本元数据对齐，参数每128样本更新。

LFE继续使用独立低通处理，不发送至可选房间。SOFA专用解析保留负高度、边界外因子及锚定字段；个性化解析与时间线不变。当前多位置块、ISF、显式无穷远以及需要屏幕校准的锚定明确报错。`finish()`仅排空一次，`--binaural-tail-seconds`规定FIR／滤波器组及可选房间尾声的排空预算。

## 轻量格式

`.jochrtf` v1仍为无pickle的NumPy ZIP，保存JSON元数据、77个频带中心、`complex128[36,2,77]`方向系数、`float64[36,2]`延迟系数及`float64[2,2]`边界。原格式、cache key、哈希校验、原子写入及源数据许可关系不变。压缩输入保留自身的方向与频率表示误差，不等同于完整HRIR精度。

滤波器组来源见[data说明](../data/README.md)。方向插值和MCA的参考文献见[研究实现依据](https://doi.org/10.1109/TASLP.2023.3313908)；算法独立实现，外部测量数据仍适用其自身许可。
