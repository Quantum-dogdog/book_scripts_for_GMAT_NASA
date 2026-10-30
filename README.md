# 火星探测轨道设计与借力弹道链（教材配套脚本）

九个脚本按 `stepN_职责` 命名，文件名即执行序；唯一例外是 **step7 在 step6 之前跑**
（step7 产出 `nodes.json`，step6 消费它）。

| 脚本 | 章节 | 干啥 | 耗时 | 依赖 |
|---|---|---|---|---|
| `step1_发射窗口与上升段闭合.py` | §1.1~1.2 | 文昌发射窗口 + 逃逸双曲线 + 上升段闭合链 + GMAT 四套钟状态 | 10 s | numpy |
| `step2_B平面设计与差分修正.py` | §1.2.3/§1.3 | 圆锥拼接标称 B 平面 + 日心 N 体差分修正 | 5 s | numpy |
| `step3_日心段中途修正.py` | §1.4 | 灵敏度 K、修正脉冲、蒙特卡洛色散、碰撞概率 | 5 s | 自动加载 step2 |
| `step4_火星捕获段设计.py` | §1.5 | 闭式捕获预算 + 间接法燃料最优对称弧 + 三种推力模型 PSO | 15 s | scipy 必需 |
| `step5_捕获段误差分析.py` | §1.5.4 | 恒角速度标称捕获 + 测量/执行/姿态三类误差蒙特卡洛 | ~9 min | scipy |
| `step6_引力辅助修正与EMEJ桥接.py` | §3.4 | 书链（火借力→Flora 微分修正）+ 二体序列桥接到精确星历，出 GMAT CSV | 1~3 min | scipy + 星历 |
| `step7_EMEJ借力优化导出.py` | — | pykep MGA-1DSM（地火地木）SADE 优化 + 导出 nodes.json | 5 s ~ 分钟级 | pykep + pygmo |
| `step8_纯MGA借力优化.py` | §3.4 | 纯 MGA：飞越近心点零脉冲（rp 为输出量）；圈数 k 是整数基因，直飞与谐振同场竞争 | 分钟级 | pykep + pygmo + scipy |
| `step9_复现验证_skippingstone.py` | — | 逐段复算 Skipping_stone 的 E-V-E-E-J：Lambert 多圈 + 前向瞄准打靶 + 病态程度 + 自由搜索 | 1~2 min | 复用 step8 |

## 用法

```bash
cd ss                            # data/ 与星历都按本目录解析

# 第 1 章链
python3 step1_发射窗口与上升段闭合.py 2020-07-25 2021-02-11
python3 step2_B平面设计与差分修正.py --dep "2020-07-21 17:59:19" --arr 2021-01-30
        # dep 建议填 step1 报告给出的实际出 SOI 时刻
python3 step3_日心段中途修正.py                  # 加 --disp 只出色散，--coll 只出碰撞概率
python3 step4_火星捕获段设计.py                   # 加 --indirect 走间接法，--pso 走三种推力模型
python3 step5_捕获段误差分析.py                   # 默认 400 样本，--nmc 100 快速看趋势

# EMEJ 借力链（先导出 nodes.json，再桥接）
python3 step7_EMEJ借力优化导出.py --xfile data/emej_x_recon.json \
        --out emej_nodes_run.json --selfcheck --no-plot
        # 复放冠军染色体；重优化就换掉 --xfile、给 --seed/--gen/--pop
python3 step6_引力辅助修正与EMEJ桥接.py --nodes data/emej_nodes_repro.json \
        --csv emej_gmat_run.csv
python3 step6_引力辅助修正与EMEJ桥接.py --book   # 书链：表3.27/3.28 微分修正

# 纯 MGA 与 Skipping_stone 复现
python3 step8_纯MGA借力优化.py --case eveej --rp-min earth=6678,venus=6352 \
        --polish --wenchang --out eveej_puremga_1989_nodes_run.json \
        --csv eveej_puremga_1989_run.csv
python3 step8_纯MGA借力优化.py --case eveej --scan \
        --scan-start 1995-01-01 --scan-end 2005-01-01 --scan-step 20
        # 逐片窗口多进程；--t0 是 nargs=2，起止日期用空格分隔
python3 step8_纯MGA借力优化.py --case eveej --x-seed data/eveej_puremga_1989_nodes.json \
        --replay --out replay_nodes.json          # 逐位复放，回归对拍用
python3 step9_复现验证_skippingstone.py          # 加 --coarse 只跑 A 段，几秒
```

`--out`/`--csv` 的例子都写当前目录（Windows 下没有 /tmp，别照抄 Linux 习惯），
跑出来的 *_run.* / replay_* 是临时产物，看完删掉即可；data/ 里留的是复现基准。

## 输出约定

脚本只打**算出来的结果**：不打自检过程，也不打教材里的对照值——数自己拿去对。
stdout 的行数和版式算固定接口，新信息一律放显式开关后面。三个开关值得知道：

- step7 `--selfcheck`：导出表是 pykep 之外手写的，下标错位不会让轨迹画歪，这层网把
  导出的 DSM 与 pykep 内部值逐段对一遍，不一致直接 FAIL 非零退出。
- step8 跑完打一张**分支族谱**：各岛末种群按圈数 k 归族，每族取 J 最小再精算。冠军只代表
  GA 选了哪一支，族谱表才回答"谐振那一支有没有被搜到"。`--no-families` 关表。
- step8 默认一岛一进程（`--islands auto`），`--serial` 退回单进程；同一 `--seed` 下
  并行与串行逐位一致。蒙特卡洛（step1~step5）种子写死，两次 stdout 可以直接 diff。

## 环境

```bash
export PYTHONIOENCODING=utf-8                       # Windows 控制台是 GBK
DE421_PATH="E:/ss/de405.bsp" python3 step6_引力辅助修正与EMEJ桥接.py --book
        # 桥接/星历档由 DE421_PATH 指定；换一份 SPK 会改动桥接残差（木星那一段最敏感）
```

pykep 3.0.1 解释器退出时 C++ 析构顺序会堆崩溃（结果已打完、退出码不干净），
step7/step8/step9 在出口统一 `os._exit()` 规避；pykep 走 conda-forge 时装的是 openblas.dll，
环境里别再混第二个 BLAS，否则 `np.linalg.solve` 原生崩溃（抓不到异常）。

## data/

| 角色 | 文件 |
|---|---|
| 脚本的输入 | `emej_x_recon.json`（step7 复放）、`emej_nodes_repro.json`（step6 桥接）、`nodes_example.json`（`--dump-nodes` 模板） |
| 直接喂 GMAT | `emej_gmat.csv`（EMEJ 11 事件）、`eveej_puremga_1989_gmat.csv`（VEEGA）、`bridge_gmat.csv`（地火 Flora）；逐事件填法见 `GMAT_EMEJ_input.md` |
| 复现锚点 | `eveej_puremga_1989_nodes.json`（含 `meta.x`，`--replay` 吃它）、`eveej_puremga_1989.csv` |
| 结论留档 | `eveej_puremga_报告.md`；`eveej_gmat.csv` 与 `eveej_realrp31_gmat.csv`/`eveej_realrp31_nodes.json`/`eveej_realrp31_GMAT_burns.md`（VEEGA 两条线的 GMAT 输入与烧写单） |

喂 GMAT 的三条硬约定（CSV 头里也写着）：

- 日心 `x/y/z`、`vx/vy/vz`、`dv` 都是 **SunMJ2000Eq**（`Origin=Sun, Axes=MJ2000Eq`），
  不是 `BodyInertial`；太阳 IAU 极与地球 J2000 极差约 26°。
- `epoch_tdb` 配 `Sat.DateFormat = TDBGregorian`，`epoch_utc` 配 `UTCGregorian`，两者差 69.184 s。
- `BT_body_km/BR_body_km` 对应 `BdotT/BdotR`（`Origin=<Body>, Axes=BodyInertial`）；
  `BT_target_*` 是设计目标，`BT_*` 是桥接实际打到的值。

## 版本

旧版用 git 取，不在目录里堆副本：`git log --oneline` → `git show <hash>:step4_火星捕获段设计.py`。
