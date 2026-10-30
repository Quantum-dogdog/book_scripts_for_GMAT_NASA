# step6 → GMAT 输入示例（EMEJ：地球→火星→地球→木星）

数据源：`data/emej_gmat.csv`（**DE421 `.bsp` / jplephem.spk**，11 个事件）。

## 0. 三条硬约定

1. **坐标系**：所有日心 `x/y/z`、`vx/vy/vz`、`dv` 都在 **SunMJ2000Eq** 里，即 `Origin=Sun, Axes=MJ2000Eq`。在 GMAT 里可以建成任意不冲突的名字，例如 `HelioMJ2000Eq`。**不要**用 `Origin=Sun, Axes=BodyInertial`：太阳 IAU 极与地球 J2000 极差约 26°，1.8 km/s 的 DSM 会被转出几百 m/s 的误差。
2. **时间**：CSV 内部 MJD 是 **TDB**。`epoch_tdb` 配 `Sat.DateFormat = TDBGregorian`；`epoch_utc` 配 `UTCGregorian`。两者相差 69.184 s（TT−UTC），不要混用。
3. **B 平面**：`BT_target_body_km / BR_target_body_km` 才是 GMAT 的 `BdotT / BdotR` 目标；坐标系为 `Origin=<Body>, Axes=BodyInertial`（NASA 官方 Mars B-plane 教程同款写法）。`BT_body_km / BR_body_km` 是桥接 TCM 实际打到的值，用来对账。

## 1. 事件总表

| # | leg | event | body | TDB | 距上一事件 (d) | 动作 |
|---:|---:|---|---|---|---:|---|
| 0 | 0 | departure | Earth | 16 Oct 2024 11:50:02.554 |  | TCM 已在 14 Oct 2024 08:55:41 施加，**不要重复** |
| 1 | 0 | dsm | - | 28 Oct 2024 03:37:34.859 | 11.658013 | DSM 1.841919 km/s（pre-burn） |
| 2 | 0 | flyby_entry | Mars | 27 Feb 2025 00:44:38.003 | 121.879898 | — |
| 3 | 0 | flyby_periapsis | Mars | 27 Feb 2025 15:44:00.659 | 0.624568 | — |
| 4 | 1 | flyby_exit | Mars | 28 Feb 2025 06:43:22.881 | 0.624563 | TCM 0.008125 km/s（post-burn） |
| 5 | 1 | dsm | - | 19 Nov 2025 02:45:31.497 | 263.834822 | DSM 0.456147 km/s（pre-burn） |
| 6 | 1 | flyby_entry | Earth | 16 Dec 2026 09:00:24.655 | 392.260337 | — |
| 7 | 1 | flyby_periapsis | Earth | 17 Dec 2026 03:07:24.576 | 0.754860 | — |
| 8 | 2 | flyby_exit | Earth | 17 Dec 2026 21:14:25.535 | 0.754872 | TCM 0.018359 km/s（post-burn） |
| 9 | 2 | dsm | - | 19 Oct 2029 17:43:56.642 | 1036.853832 | DSM 0.102351 km/s（pre-burn） |
| 10 | 2 | arrival | Jupiter | 05 Apr 2030 12:34:40.084 | 167.785225 | — |

## 2. 逐事件输入数据

### [0] departure — 出发（日心，SOI 外）

```text
TDB : 16 Oct 2024 11:50:02.554
UTC : 16 Oct 2024 11:48:53.370
r   : 136528794.021156 54647312.883641 23852805.761765  km      (SunMJ2000Eq)
v   :  -14.725483993   28.283830383   13.158415790  km/s
dv  :    0.005864273    0.011212854    0.005660117  km/s  |dv| = 0.013861988
burn: post_burn_at_earlier_epoch; do_not_reapply @ 14 Oct 2024 08:55:41.162
rel : -453050.372748  603228.455458  426314.396742  km      (EarthMJ2000Eq)
relv:   -2.475422843    3.289853917    2.325137169  km/s
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Jupiter, Saturn}
```

### [1] dsm — 深空机动 DSM

```text
TDB : 28 Oct 2024 03:37:34.859
UTC : 28 Oct 2024 03:36:25.675
r   : 119017667.332747 81829964.516922 36530898.760627  km      (SunMJ2000Eq)
v   :  -19.926273778   25.506408664   11.929557587  km/s
dv  :   -1.023709806    1.184980310    0.969797087  km/s  |dv| = 1.841919241
burn: pre_burn; apply dv at this epoch @ 28 Oct 2024 03:37:34.859
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Jupiter, Saturn}
```

### [2] flyby_entry — 飞越进入（SOI 入口）

```text
TDB : 27 Feb 2025 00:44:38.003
UTC : 27 Feb 2025 00:43:28.819
r   : -173637062.413558 157712274.388170 77038850.594806  km      (SunMJ2000Eq)
v   :  -23.636061434   -6.699529271   -2.762274227  km/s
rel :  -19587.469629 -541886.863850  195803.453025  km      (MarsMJ2000Eq)
relv:    0.299337597   10.005308790   -3.660350838  km/s
B   : BdotT=-3376.589  BdotR=2328.593  km   (MarsFlyby, BodyInertial)
      achieved BT=-3376.066  BR=2328.629  km; rp target=3742.058 km, actual=3741.075 km
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Jupiter, Saturn}
```

### [3] flyby_periapsis — 飞越近点

```text
TDB : 27 Feb 2025 15:44:00.659
UTC : 27 Feb 2025 15:42:51.475
r   : -174911216.647494 157350050.065843 76889502.634615  km      (SunMJ2000Eq)
v   :  -23.481037203   -5.602772838   -1.947289633  km/s
rel :   -3078.133794    -312.571055   -2104.849150  km      (MarsMJ2000Eq)
relv:    1.214262404   11.105051819   -3.424843960  km/s
fm  : CentralBody=Mars; PointMasses={}
```

### [4] flyby_exit — 飞越离开（下一段起点）

```text
TDB : 28 Feb 2025 06:43:22.881
UTC : 28 Feb 2025 06:42:13.697
r   : -176101918.083227 157028733.032777 76790734.312804  km      (SunMJ2000Eq)
v   :  -22.005393284   -6.006142897   -1.855193934  km/s
dv  :   -0.004401185    0.006462101    0.002210352  km/s  |dv| = 0.008124952
burn: post_burn; state includes dv; do_not_reapply @ 28 Feb 2025 06:43:22.881
rel :   99817.994660  550499.784172 -141121.775674  km      (MarsMJ2000Eq)
relv:    1.908545363   10.184434263   -2.567325674  km/s
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Mars, Jupiter, Saturn}
```

### [5] dsm — 深空机动 DSM

```text
TDB : 19 Nov 2025 02:45:31.497
UTC : 19 Nov 2025 02:44:22.313
r   : -344488542.703800 -104624910.697885 -33782296.628352  km      (SunMJ2000Eq)
v   :    4.985875039  -11.587437298   -5.365088567  km/s
dv  :   -0.141406299    0.331544989    0.279556981  km/s  |dv| = 0.456146826
burn: pre_burn; apply dv at this epoch @ 19 Nov 2025 02:45:31.497
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Mars, Jupiter, Saturn}
```

### [6] flyby_entry — 飞越进入（SOI 入口）

```text
TDB : 16 Dec 2026 09:00:24.655
UTC : 16 Dec 2026 08:59:15.471
r   : 15374987.999210 133541653.168317 57812686.232930  km      (SunMJ2000Eq)
v   :  -31.088640088   15.170419304    7.704563263  km/s
rel :   75284.563643 -816126.368853 -428286.096757  km      (EarthMJ2000Eq)
relv:   -0.974276783   12.443379216    6.523335800  km/s
B   : BdotT=11372.488  BdotR=-26.894  km   (EarthFlyby, BodyInertial)
      achieved BT=11374.490  BR=-22.712  km; rp target=9532.415 km, actual=9533.768 km
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Mars, Jupiter, Saturn}
```

### [7] flyby_periapsis — 飞越近点

```text
TDB : 17 Dec 2026 03:07:24.576
UTC : 17 Dec 2026 03:06:15.392
r   : 13343659.975008 134525929.954759 58313643.400829  km      (SunMJ2000Eq)
v   :  -34.214255097   16.723987864    8.665492297  km/s
rel :    9242.615913    2195.431140     788.079231  km      (EarthMJ2000Eq)
relv:   -4.063087685   14.361979744    7.642405853  km/s
fm  : CentralBody=Earth; PointMasses={}
```

### [8] flyby_exit — 飞越离开（下一段起点）

```text
TDB : 17 Dec 2026 21:14:25.535
UTC : 17 Dec 2026 21:13:16.351
r   : 10999906.114812 135412949.546612 58777958.964599  km      (SunMJ2000Eq)
v   :  -35.930821346   13.325330004    6.990748021  km/s
dv  :   -0.001940305    0.016322341    0.008177661  km/s  |dv| = 0.018359132
burn: post_burn; state includes dv; do_not_reapply @ 17 Dec 2026 21:14:25.535
rel : -366970.039700  747072.250171  403535.677464  km      (EarthMJ2000Eq)
relv:   -5.748256749   11.328556623    6.125859213  km/s
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Earth, Mars, Saturn}
```

### [9] dsm — 深空机动 DSM

```text
TDB : 19 Oct 2029 17:43:56.642
UTC : 19 Oct 2029 17:42:47.458
r   : -608914251.700454 -519550482.194362 -207073789.397267  km      (SunMJ2000Eq)
v   :    4.663304077   -4.249454637   -2.007554006  km/s
dv  :    0.059837200   -0.081151022   -0.017600217  km/s  |dv| = 0.102351095
burn: pre_burn; apply dv at this epoch @ 19 Oct 2029 17:43:56.642
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Earth, Mars, Saturn}
```

### [10] arrival — 到达

```text
TDB : 05 Apr 2030 12:34:40.084
UTC : 05 Apr 2030 12:33:30.900
r   : -525857921.897927 -568868429.174940 -231030173.388646  km      (SunMJ2000Eq)
v   :    6.709640393   -2.423089692   -1.257278722  km/s
rel :      -0.473778      -0.327066      -0.007242  km      (JupiterMJ2000Eq)
relv:   -3.224443138    5.077127206   -0.144459921  km/s
miss: 0.575752 km
fm  : CentralBody=Sun; PointMasses={Mercury, Venus, Earth, Mars, Saturn}
```


## 3. 最小输入清单（照这个填就行）

| 类别 | 数量 | 从哪里拿 | 填到 GMAT 哪里 |
|---|---:|---|---|
| 初值状态 | 3 | `departure` + 两个 `flyby_exit` 行（`epoch_tdb + x/y/z + vx/vy/vz`） | `Sat.Epoch` + `Sat.X...VZ`，`CoordinateSystem = HelioMJ2000Eq` |
| 出发 TCM | 1 | `departure` 行 `burn_epoch_tdb + dvx/dvy/dvz` | **已在 14 Oct 2024 施加，仅记账，不要再 Maneuver** |
| DSM | 3 | 三个 `dsm` 行（时刻 + dv） | `ImpulsiveBurn` + `Maneuver`，在对应时刻之后执行 |
| 飞越 TCM | 2 | 两个 `flyby_exit` 行（时刻 + dv，post-burn） | 新段落初值已含 dv，不要重复 |
| B 平面目标 | 2 | `flyby_entry` 行 `BT_target_body_km / BR_target_body_km` | `Achieve DC1(Sat.<Body>Flyby.BdotT/BdotR = ...)` |
| 力模型 | 5 段 | 3 段日心 + 2 段天体中心（`flyby_periapsis` 行） | `ForceModel.CentralBody` + `PointMasses` |
| 到达 | 1 | `arrival` 行 `epoch_tdb + 状态` | 传播终点 + `Report` 对 `miss_km` |

## 4. 最小可跑骨架（leg0：出发 → 火星 B 平面）

下面这段只搭 leg0，语法按你仓库里 `Simulate Juno Probe.script` 的写法。把 `HelioMJ2000Eq` 建成 `Origin=Sun, Axes=MJ2000Eq`，不要建成 `BodyInertial`。

```text
%---------- 坐标系 ----------
Create CoordinateSystem HelioMJ2000Eq;
GMAT HelioMJ2000Eq.Origin = Sun;
GMAT HelioMJ2000Eq.Axes   = MJ2000Eq;

Create CoordinateSystem MarsFlyby;
GMAT MarsFlyby.Origin = Mars;
GMAT MarsFlyby.Axes   = BodyInertial;

%---------- 力模型：leg0 排除 Earth/Mars ----------
Create ForceModel FM_leg0;
GMAT FM_leg0.CentralBody            = Sun;
GMAT FM_leg0.PrimaryBodies          = {Sun};
GMAT FM_leg0.PointMasses            = {Mercury, Venus, Jupiter, Saturn};
GMAT FM_leg0.Drag                   = None;
GMAT FM_leg0.SRP                    = Off;
GMAT FM_leg0.RelativisticCorrection = Off;

Create Propagator Prop_leg0;
GMAT Prop_leg0.FM                        = FM_leg0;
GMAT Prop_leg0.Type                      = PrinceDormand78;
GMAT Prop_leg0.InitialStepSize           = 600;
GMAT Prop_leg0.Accuracy                  = 1e-12;
GMAT Prop_leg0.MinStep                   = 1e-6;
GMAT Prop_leg0.MaxStep                   = 86400;
GMAT Prop_leg0.StopIfAccuracyIsViolated  = true;

%---------- DSM0：28 Oct 2024 03:37:34.859 TDB，pre-burn ----------
Create ImpulsiveBurn DSM0;
GMAT DSM0.CoordinateSystem = HelioMJ2000Eq;
GMAT DSM0.Element1         = -1.023709806;
GMAT DSM0.Element2         =  1.184980310;
GMAT DSM0.Element3         =  0.969797087;

%---------- 初值：16 Oct 2024 11:50:02.554 TDB ----------
Create Spacecraft Sat;
GMAT Sat.DateFormat       = TDBGregorian;
GMAT Sat.Epoch            = '16 Oct 2024 11:50:02.554';
GMAT Sat.CoordinateSystem = HelioMJ2000Eq;
GMAT Sat.StateType        = Cartesian;
GMAT Sat.X                =  136528794.021156;
GMAT Sat.Y                =   54647312.883641;
GMAT Sat.Z                =   23852805.761765;
GMAT Sat.VX               =     -14.725483993;
GMAT Sat.VY               =      28.283830383;
GMAT Sat.VZ               =      13.158415790;
GMAT Sat.DryMass          = 1000;

Create DifferentialCorrector DC1;
GMAT DC1.MaximumIterations = 50;
GMAT DC1.DerivativeMethod  = ForwardDifference;

BeginMissionSequence;

% 目标：BT/BR 打到设计值（3 个 DSM 分量对 2 个目标；你的 Juno 脚本也是 3 对 2）
Target 'Target Mars B-plane' DC1 {SolveMode = Solve, ExitMode = SaveAndContinue, ShowProgressWindow = true};
   Vary 'Vary DSM0.x' DC1(DSM0.Element1 = -1.023709806, {Perturbation = 0.0001});
   Vary 'Vary DSM0.y' DC1(DSM0.Element2 =  1.184980310, {Perturbation = 0.0001});
   Vary 'Vary DSM0.z' DC1(DSM0.Element3 =  0.969797087, {Perturbation = 0.0001});
   Propagate 'Prop to DSM0'     Prop_leg0(Sat) {Sat.ElapsedDays = 11.658013};
   Maneuver  'Apply DSM0'       DSM0(Sat);
   Propagate 'Prop to Mars entry' Prop_leg0(Sat) {Sat.ElapsedDays = 121.879898};
   Achieve 'Achieve BdotT' DC1(Sat.MarsFlyby.BdotT = -3376.589, {Tolerance = 0.1});
   Achieve 'Achieve BdotR' DC1(Sat.MarsFlyby.BdotR =  2328.593, {Tolerance = 0.1});
EndTarget;
```

跑完 targeter 后，`Sat.MarsFlyby.BdotT/BdotR` 应接近 `-3376.589 / 2328.593 km`；CSV 里桥接自己的 TCM 打到的是 `-3376.066 / 2328.629 km`，差 0.52 km，属于正常。

## 5. 后续两段怎么接

**火星飞越段（27 Feb 2025 00:44:38.003 → 28 Feb 2025 06:43:22.881 TDB，1.249131 d）**

- 用 `flyby_entry` 行的 `rel_* / relv_*`（MarsMJ2000Eq）作为初值；
- `CentralBody = Mars`，`PrimaryBodies = {Mars}`，`PointMasses = {}`；
- 传播 0.624568 d 到近点（rp = 3742.058 km），再传播 0.624563 d 到出口；
- 出口状态已经是加过 TCM 的（`post_burn`，dv `(-0.004401185, 0.006462101, 0.002210352) km/s`），**不要重复加**；换回日心系直接接 leg1。

**leg1：火星 → 地球（B 平面目标 `11372.488 / -26.894 km`）**

- 起点用 `flyby_exit` 行的日心状态（**已含** 8.12 m/s TCM，不要再加）；
- 力模型 `PointMasses = {Mercury, Venus, Mars, Jupiter, Saturn}`，排除 Earth；
- 传播 263.834822 d 到 DSM1（19 Nov 2025 02:45:31.497 TDB），施加 `(-0.141406299, 0.331544989, 0.279556981) km/s`；
- 再传播 392.260337 d 到地球入口，`Achieve` `EarthFlyby.BdotT/BdotR = 11372.488 / -26.894`。
- 这次用 `.bsp`，leg1 收住了：实际 `11374.490 / -22.712 km`，最大偏差 4.18 km；GMAT 只打 BT/BR 两目标会更轻松。

**leg2：地球 → 木星**

- 起点用第二个 `flyby_exit` 行（已含 18.36 m/s 末段 TCM）；
- 力模型 `PointMasses = {Mercury, Venus, Earth, Mars, Saturn}`，排除 Jupiter；
- 传播 1036.853832 d 到 DSM2（19 Oct 2029 17:43:56.642 TDB），施加 `(0.059837200, -0.081151022, -0.017600217) km/s`；
- 再传播 167.785225 d 到 05 Apr 2030 12:34:40.084 TDB，位置交会残差 0.576 km。

## 6. 四个容易踩的坑

1. **别把 `SunXyz`/`BodyInertial` 当日心 J2000 赤道用**。GMAT 的 `BodyInertial` 用的是天体 IAU 自转轴；太阳极与地球 J2000 极差约 26°。本 CSV 的 `dv` 是 `MJ2000Eq` 轴，必须建 `Origin=Sun, Axes=MJ2000Eq`。
2. **TDB 和 UTC 差 69.184 s**。用 `epoch_tdb` 就配 `TDBGregorian`，用 `epoch_utc` 就配 `UTCGregorian`，不要一个格式填另一列。
3. **burn 语义看 `burn_semantics`**：
   - `pre_burn`（DSM 行）：行状态是点火前，`Maneuver` 在这一时刻执行；
   - `post_burn`（`flyby_exit` 行）：行状态已含 dv，**不要再 Maneuver**；
   - `post_burn_at_earlier_epoch`（`departure` 行）：TCM 在更早的 14 Oct 2024 已施加，**不要重复**。
4. **B 平面的 `BT_target_*` 才是目标**，`BT_*` 是桥接实际打到的值。GMAT 的 `BdotT/BdotR` 符号约定在不同版本可能整体反号；先跑一个飞越对一眼，若 BT/BR 同时反号，就整体取负。
