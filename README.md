# 行星探测轨道设计与借力弹道链（教材配套脚本）

脚本按 `stepN_职责` 命名，文件名即执行序；唯一例外是 **step7 在 step6 之前跑**  
（step7 产出 `nodes.json`，step6 消费它）。`step8v4.py` 自动适配 pykep 2.6 / 3.x；  
`step10.py` 接在链尾做 GMAT 外层打靶。`*.script` 是配套的高保真 GMAT 脚本。

| 脚本                            | 章节          | 干啥                                                                                                                               | 依赖                    |
| ----------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `step1_发射窗口与上升段闭合.py`         | §1.1~1.2    | 文昌发射窗口 + 逃逸双曲线 + 上升段闭合链 + GMAT 四套钟状态                                                                                             | numpy                 |
| `step2_B平面设计与差分修正.py`         | §1.2.3/§1.3 | 圆锥拼接标称 B 平面 + 日心 N 体差分修正                                                                                                         | numpy                 |
| `step3_日心段中途修正.py`            | §1.4        | 灵敏度 K、修正脉冲、蒙特卡洛色散、碰撞概率                                                                                                           | 自动加载 step2            |
| `step4_火星捕获段设计.py`            | §1.5        | 闭式捕获预算 + 间接法燃料最优对称弧 + 三种推力模型 PSO                                                                                                 | scipy 必需              |
| `step5_捕获段误差分析.py`            | §1.5.4      | 恒角速度标称捕获 + 测量/执行/姿态三类误差蒙特卡洛                                                                                                      | scipy                 |
| `step6_引力辅助修正与EMEJ桥接.py`      | §3.4        | 书链（火借力→Flora 微分修正）+ 二体序列桥接到精确星历，出 GMAT CSV                                                                                       | scipy + 星历            |
| `step7_EMEJ借力优化导出.py`         | —           | pykep MGA-1DSM（地火地木）SADE 优化 + 导出 nodes.json                                                                                      | pykep + pygmo         |
| `step8v4.py`                  | —        | 纯 MGA 借力优化：飞越近心点零脉冲（rp 为输出量）、圈数 k 整数基因、直飞与谐振同场竞争；pykep 2.6/3.x 自适应、文昌几何后检与 chain-close 发射窗口分层扫描、`--polish` 精修、`--eph de421` 精确星历 | pykep + pygmo + scipy |
| `step9_复现验证_skippingstone.py` | —           | 逐段复算 Skipping_stone 的 E-V-E-E-J：Lambert 多圈 + 前向瞄准打靶 + 病态程度 + 自由搜索                                                                | pykep + scipy         |
| `step10.py`                   | —           | GMAT 外层打靶：把内层微分校正器当黑箱 y=f(x)，只调最后一根腿的 Achieve 值，有限差分 + 阻尼牛顿/Broyden 收敛，只 print 改哪几行                                              | 纯标准库 + GMAT R2022a    |

## GMAT 脚本

| 脚本                     | 干啥                                                                        | 力模型                                       |
| ---------------------- | ------------------------------------------------------------------------- | ----------------------------------------- |
| `step1forward.script`  | 200 km 停泊轨道 TMI 注入后的正向检验：地心 → 日心 → 火心三段传播，停在 `Mars.Periapsis`（2020-07 窗口） | Earth+Luna+Sun / Sun+五大点质量 / Mars 4×4 重力场 |
| `step1backward.script` | 同一状态反向：`Maneuver BackProp` 打掉 TMI 脉冲，再用 `Propagate BackProp` 倒推回点火前的停泊轨道  | Earth 单体（pureearth）                       |
| `step6v4.script`       | EVEEJ 全序列高保真打靶：DE421/点质量力模型 + 分段 B 平面微分校正，由 step6 的节点 CSV 喂进来             | 日心全点质量 + 各行星近心段                           |

## 用法

```bash
cd ss                            # data/ 与星历都按本目录解析

# 第 1 章链
python step1_发射窗口与上升段闭合.py 2020-07-25 2021-02-11
python step2_B平面设计与差分修正.py --dep "2020-07-21 17:59:19" --arr 2021-01-30
        # dep 建议填 step1 报告给出的实际出 SOI 时刻
python step3_日心段中途修正.py                  # 加 --disp 只出色散，--coll 只出碰撞概率
python step4_火星捕获段设计.py                   # 加 --indirect 走间接法，--pso 走三种推力模型
python step5_捕获段误差分析.py                   # 默认 400 样本，--nmc 100 快速看趋势

# EMEJ 借力链（先导出 nodes.json，再桥接）
python step7_EMEJ借力优化导出.py --xfile data/emej_x_recon.json \
        --out emej_nodes_run.json --selfcheck --no-plot
        # 复放冠军染色体；重优化就换掉 --xfile、给 --seed/--gen/--pop
python step6_引力辅助修正与EMEJ桥接.py --nodes data/emej_nodes_repro.json \
        --csv emej_gmat_run.csv
python step6_引力辅助修正与EMEJ桥接.py --book   # 书链：表3.27/3.28 微分修正

# Skipping_stone 复现
python step9_复现验证_skippingstone.py          # 加 --coarse 只跑 A 段，几秒

# step8v4：带文昌发射窗口与上升段闭合
python step8v4.py --case eveej2029 --rp-min earth=6678,venus=6352 \
        --pop 50 --gen 50 --seed 5 --dsm-max 200 --islands 7 --workers 7 \
        --polish --wenchang --chain-close --cc-span 5.0 --cc-step 0.1 \
        --wc-max-coast 30 --out 1.json --csv 1.csv
        # --eph de421 换精确星历；--epoch-shift 手动覆盖 jpl_lp 的 -0.5 d
        # --chain-close 要读 step1，step1.py 和中文全名都认，找不到直接报错

# step10：外层打靶，先看结构再求解
python step10.py --self-test                   # 纯 Python 自检，不调 GMAT
python step10.py --script step6v4.script --epoch 34731.721007756 --inspect
python step10.py --script step6v4.script --epoch 34731.721007756 \
        --start -3174.805517 12486.774499 --tol 100 --max-iter 6 --fd-step 50 \
        --scan 34648 34668 0.25
        # --epoch 必须显式给，单位沿用 script（TDBModJulian），别取 script 里最大那个
        # --start 的数值顺序按 --inspect 打印的 control 顺序；--scan 步长要够细
```



## 环境

```bash
export PYTHONIOENCODING=utf-8                       # Windows 控制台是 GBK
DE421_PATH="E:/ss/de421.bsp" python step6_引力辅助修正与EMEJ桥接.py --book
        # 桥接/星历档由 DE421_PATH 指定；换一份 SPK 会改动桥接残差（木星那一段最敏感）
```

step8v4 用兼容层认 pykep 版本（3.x 走 `udpla.jpl_lp`，2.x 走 `planet.jpl_lp`），两版都  
不用改调用；jpl_lp 节点历元相对标准 J2000 恒差 −0.5 d，`--eph de421` 强制 0，  
`--epoch-shift` 手动覆盖。GMAT 的 `TDBModJulian` 基准是 JD 2430000.0（1941-01-05 12:00），  
与标准 MJD 差 29999.5 d。  
step10 找 GmatConsole 的顺序是 `--gmat-console` > `GMAT_CONSOLE` > `--gmat-root`/`GMAT_ROOT`

> PATH > 各平台默认安装路径；Windows 输出走 cp936，上真机先跑 `--self-test` 和 `--inspect`。

pykep 3.0.1 解释器退出时 C++ 析构顺序会堆崩溃（结果已打完、退出码不干净），  
step7/step8/step9 在出口统一 `os._exit()` 规避；pykep 走 conda-forge 时装的是 openblas.dll，  
环境里别再混第二个 BLAS，否则 `np.linalg.solve` 原生崩溃（抓不到异常）。

## data/

| 角色       | 文件                                                                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 脚本的输入    | `emej_x_recon.json`（step7 复放）、`emej_nodes_repro.json`（step6 桥接）、`nodes_example.json`（`--dump-nodes` 模板）                                               |
| 直接喂 GMAT | `emej_gmat.csv`（EMEJ 11 事件）、`eveej_puremga_1989_gmat.csv`（VEEGA）、`bridge_gmat.csv`（地火 Flora）；逐事件填法见 `GMAT_EMEJ_input.md`                              |
| 复现锚点     | `eveej_puremga_1989_nodes.json`（含 `meta.x`，`--replay` 吃它）、`eveej_puremga_1989.csv`                                                                    |
| 结论留档     | `eveej_puremga_报告.md`；`eveej_gmat.csv` 与 `eveej_realrp31_gmat.csv`/`eveej_realrp31_nodes.json`/`eveej_realrp31_GMAT_burns.md`（VEEGA 两条线的 GMAT 输入与烧写单） |

喂 GMAT 的三条硬约定（CSV 头里也写着）：

- 日心 `x/y/z`、`vx/vy/vz`、`dv` 都是 **SunMJ2000Eq**（`Origin=Sun, Axes=MJ2000Eq`），  
  不是 `BodyInertial`；太阳 IAU 极与地球 J2000 极差约 26°。
- `epoch_tdb` 配 `Sat.DateFormat = TDBGregorian`，`epoch_utc` 配 `UTCGregorian`，两者差 69.184 s。
- `BT_body_km/BR_body_km` 对应 `BdotT/BdotR`（`Origin=<Body>, Axes=BodyInertial`）；  
  `BT_target_*` 是设计目标，`BT_*` 是桥接实际打到的值。

