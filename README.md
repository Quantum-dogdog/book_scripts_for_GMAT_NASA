# dihuo · 地火任务设计链（教材配套脚本合集）

同一条书的配套脚本链：火星探测轨道设计与控制（第 1 章）→ 引力辅助与借力弹道
（§3.4）→ 地-火-地-木（EMEJ）MGA 任务链 → 纯 MGA（零脉冲借力）与 Skipping_stone
复现验证。九个主脚本按 `stepN_职责` 命名，
文件名即执行序（唯一例外：**step7 在 step6 之前跑**——step7 产出 nodes.json，
step6 消费它）。

| 脚本 | 章节 | 干啥 | 耗时 | 依赖 |
|---|---|---|---|---|
| `step1_发射窗口与上升段闭合.py` | §1.1~1.2 | 文昌当日发射窗口 + 逃逸双曲线 + 定时闭合解 + 上升段闭合链，导出 GMAT 四钟状态（A1/A1c/A2/B） | 10 s | numpy |
| `step2_B平面设计与差分修正.py` | §1.2.3/§1.3 | 圆锥拼接标称 B 平面 + 日心 N 体精确模型差分修正（GMAT 对拍后终版：`--dep` 接受时刻、`--book` 书表对账、`--selftest` 自检） | 5 s | numpy（scipy 可选） |
| `step3_日心段中途修正.py` | §1.4 | 灵敏度/修正脉冲/蒙特卡洛色散/碰撞概率，自带回归 | 5 s | 同上；会自动加载 step2 交叉校验 |
| `step4_火星捕获段设计.py` | §1.5 | 闭式捕获预算 + 间接法对称弧 + PSO + primer 协态校验 | 15 s | numpy + **scipy 必需** |
| `step5_捕获段误差分析.py` | §1.5.4 | 表1.15 标称 + 表1.16 三类误差蒙特卡洛 | ~9 min | 同上 |
| `step6_引力辅助修正与EMEJ桥接.py` | §3.4 | 书链（火借力→Flora 微分修正）+ 中保真桥接器（nodes→精确星历，输出 GMAT 就绪 CSV/TDB-UTC/B 平面） | 1~3 min | numpy + scipy；DE421（优先 `de421.bsp`，否则 PyPI `de421` + `jplephem`） |
| `step7_EMEJ借力优化导出.py` | — | pykep MGA-1DSM（地火地木）SADE 优化 + 导出 nodes.json | 5 s（复放）/ 分钟级（重优化） | **pykep + pygmo** |
| `step8_纯MGA借力优化.py` | §3.4 | **纯 MGA**：无 DSM、飞越近心点零脉冲（圆锥拼接，rp 为输出量）；SADE 全局搜索 + SLSQP 精修 + 窗口扫描 + `--resonant` 谐振返回段 | 分钟级（扫窗 ~6 min） | **pykep + pygmo** + scipy |
| `step9_复现验证_skippingstone.py` | — | 复现/交叉验证 `Skipping_stone` 打印的 E-V-E-E-J 轨迹：逐段零脉冲打靶 + 病态程度 + DE421 复核 | 1~2 min | pykep + pygmo + step6/step8 + DE421 |

## 用法

```bash
cd dihuo                       # 星历与文件名都按本目录解析，先 cd 进来

# 发射段→火心段交棒链（step1 出 SOI 时刻带进 step2）
python3 step1_发射窗口与上升段闭合.py 2020-07-25 2021-02-11
        # 报告里 A1 钟给出实际出 SOI 时刻（如 2020-07-24 17:27:52）
python3 step2_B平面设计与差分修正.py --dep "2020-07-21 17:59:19" --arr 2021-01-30
        # dep/arr 可带 HH:MM:SS；建议直接填 step1 报告的出 SOI 时刻
python3 step2_B平面设计与差分修正.py --book       # 书表 1.3/1.5 对账
python3 step2_B平面设计与差分修正.py --selftest   # 10 项 B 平面不变量自检

# 第 1 章链（B 平面 → 中途修正 → 捕获 → 误差）
python3 step2_B平面设计与差分修正.py
python3 step3_日心段中途修正.py --check      # 快速回归（双引擎对拍等 3 项）
python3 step3_日心段中途修正.py --reg       # 完整回归（表1.8 项预期 FAIL，见 AUDIT）
python3 step4_火星捕获段设计.py --indirect --primer
python3 step5_捕获段误差分析.py              # 默认 nmc=400，可 --nmc 100 快验

# EMEJ 借力链（先优化导出，再桥接）
python3 step7_EMEJ借力优化导出.py --xfile audit/emej_x_recon.json \
        --out /tmp/nodes.json --no-plot      # 复放冠军染色体（5 s，逐位复现）
python3 step6_引力辅助修正与EMEJ桥接.py --nodes audit/emej_nodes_repro.json \
        --csv data/emej_gmat.csv             # GMAT-ready 全链（11 事件）
python3 step6_引力辅助修正与EMEJ桥接.py --nodes audit/emej_nodes_repro.json \
        --csv /tmp/repro.csv --legacy-csv    # 旧混合坐标系，仅审计对拍
python3 step6_引力辅助修正与EMEJ桥接.py --book              # 书链（§3.4 Flora）

# step7 重新优化（不固定种子，每次不同）
python3 step7_EMEJ借力优化导出.py --seed 42 --gen 30 --islands 8 --pop 15 \
        --out data/emej_nodes_new.json --no-plot

# 纯 MGA（零脉冲借力）：1984-1994 每 90 天切一窗扫，再把低能种子做 defect-only 精修
python3 step8_纯MGA借力优化.py --case eveej --rp-min earth=6678,venus=6352 \
        --w-fb 2000 --scan --scan-start 1984-01-01 --scan-end 1994-01-01 \
        --scan-step 90 --scan-width 150 --out data/eveej_scan.json \
        --csv data/eveej_puremga_scan.csv
python3 step8_纯MGA借力优化.py --case eveej --rp-min earth=6678,venus=6352 \
        --polish --wenchang --out data/eveej_puremga_1989_nodes.json \
        --csv data/eveej_puremga_1989.csv
        # 注意 --t0 是 nargs=2，起止日期要用空格分隔，不要写 --t0 2029-11-01,2029-12-01

# 谐振返回段（skipping-stone 那种 1:1 地-地借力）：该段不走 Lambert，
# 由内层打靶解 (δ,ψ,τ)，TOF 变成输出量；段号从 0 计数且不能是第 0 段
python3 step8_纯MGA借力优化.py --names earth,venus,earth,earth \
        --t0 10896.5 10971.5 --tof 155,170 320,340 700,760 --revs 1,1,3 \
        --resonant 2:2 --rp-min earth=6678,venus=6352 \
        --w-dep 1 --w-arr 0 --w-fb 2000 --pop 16 --gen 40
        # 换谐振阶：--resonant 2:3（3 圈，TOF≈1096 d）
        # 换行星/换腿：--names earth,venus,venus,earth --resonant 1:3
        # 多段谐振：  --resonant 1:2,3:2
python3 step8_纯MGA借力优化.py --names earth,venus,earth,earth,jupiter \
        --resonant 2:2 --t0 10896.5 10971.5 \
        --tof 155,170 320,340 700,760 820,940 --revs 1,1,3,1 \
        --rp-min earth=6678,venus=6352 --w-fb 2000

# 复现验证：逐段检查 Skipping_stone 打印的轨迹能不能用零脉冲纯 MGA 打出来
python3 step9_复现验证_skippingstone.py            # 全量（含 DE421 复核）
python3 step9_复现验证_skippingstone.py --coarse    # 只跑粗搜，快
python3 step9_复现验证_skippingstone.py --no-de421  # 没有 de421.bsp 时
```

audit/ 里还放了两个独立复核脚本：`audit/verify_tau.py`（τ 独立复核）、
`audit/recon_x.py`（染色体反演）。纯 MGA 与谐振验证的完整结论写在
`data/eveej_puremga_报告.md`（§8 交叉验证、§9 `--resonant`），
step9 原始输出在 `data/skippingstone_复现验证.txt`。

## DE421 星历获取（step6 桥接用）

优先读本目录（或 `/tmp`、`/data`）的 `de421.bsp`（16.8 MB，不随 git 仓库分发）：

```bash
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp -O de421.bsp
# 或从 https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ 下载 DE421
# 也可用环境变量 DE421_PATH=/path/to/de421.bsp 指定
```

没有 `.bsp` 时，step6 自动回退到 PyPI 的 Chebyshev 数据包：

```bash
pip install de421 jplephem
```

此时内部走 `jplephem.ephem.Ephemeris` 旧接口（地球由 `earthmoon - earth_share*moon`
合成），数值与 `.bsp` 同源，只是不再需要 16.8 MB 的 BSP 文件。两者都没有时才回退到
Standish 开普勒根数，并在 CSV 头的 `ephemeris:` 行明确标出。

## GMAT 输入（step6 输出）

`data/emej_gmat.csv`（EMEJ，11 事件）和 `data/bridge_gmat.csv`（地火 Flora，5 事件）
是默认导出格式；逐事件填法见 `data/GMAT_EMEJ_input.md`。三条硬约定：

- 日心 `x/y/z`、`vx/vy/vz`、`dv` 都是 **SunMJ2000Eq**（`Origin=Sun, Axes=MJ2000Eq`），
  不是 `BodyInertial`；太阳 IAU 极与地球 J2000 极差约 26°。
- `epoch_tdb` 配 `Sat.DateFormat = TDBGregorian`；`epoch_utc` 配 `UTCGregorian`，
  两者差 69.184 s。
- `BT_body_km/BR_body_km` 对应 `BdotT/BdotR`，坐标系为
  `Origin=<Body>, Axes=BodyInertial`；`BT_target_body_km/BR_target_body_km` 是设计目标，
  `BT_*` 是桥接实际打到的值。

CSV 头里还写了每段的 `fm_central_body` / `fm_point_masses`、burn 语义（pre/post）和
GMAT 设置片段；`--legacy-csv` 只用于和旧审计 CSV 对趋势，坐标系仍是旧版的黄道/赤道混合。

## 目录

```
dihuo/
├── step1 ~ step9（.py）   主脚本：step2 = GMAT 对拍后终版（dihuo2），
│                          旧书版在 versions/；其余 = 各源仓库终版
│                          step8/step9 = 纯 MGA（零脉冲）+ skipping-stone 复现验证
├── data/                  emej_nodes.json / emej.csv / emej_gmat.csv /
│                          bridge_gmat.csv / GMAT_EMEJ_input.md / nodes_example.json
│                          eveej_puremga_报告.md / skippingstone_复现验证.txt /
│                          eveej_puremga_scan.csv / eveej_puremga_*.csv|.json /
│                          evee_resonant_*_nodes.json、evve_resonant_*（谐振产物）
├── de421.bsp              DE421 星历（可选；缺省时用 PyPI de421 包回退）
├── audit/                 EMEJ 审计复现产物 + 独立复核脚本 + CHANGELOG
├── originals/             emej 上传原件字节级副本
├── versions/              step2/3/4 旧版导出（免 git 取用）
├── AUDIT.md               两链合并审计报告
└── README.md              本文件
```

## 版本历史

dihuo 是收编仓库：step2-5 来自 mars-probe-design（git 5 commits），
step6/step7 及 audit 来自 emej-audit（git 带 tag v0-v5 + emej-chain，另有
emej-history.bundle 备份）。dihuo 自身 git 亦有完整收编记录。取旧版：

```bash
git log --oneline                  # 列版本
git show <hash>:step4_火星捕获段设计.py   # 看某版本
git show <hash>:step4_火星捕获段设计.py > 旧版.py
```

更深的历史（6.py 六版演进、emej 原件 md5 凭证）见 `audit/emej_CHANGELOG.md`
与 emej-audit 仓库 README 的 git 标签对照表。

改名联动说明：step3 L304 的引擎加载路径已同步指向 step2 新文件名（唯一一处
功能性改动），其余为纯改名；step6/step7 未改动。

step8（`--resonant` 谐振返回模式）与 step9 的增量、以及本次合并掉重名文件
（`...(1)` / `... (2)` 后缀）的取舍依据，见根目录 `VERSION.md`。
