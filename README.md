# dihuo · 地火任务设计链（教材配套脚本合集）

同一条书的配套脚本链：火星探测轨道设计与控制（第 1 章）→ 引力辅助与借力弹道
（§3.4）→ 地-火-地-木（EMEJ）MGA 任务链。七个主脚本按 `stepN_职责` 命名，
文件名即执行序（唯一例外：**step7 在 step6 之前跑**——step7 产出 nodes.json，
step6 消费它）。

| 脚本 | 章节 | 干啥 | 耗时 | 依赖 |
|---|---|---|---|---|
| `step1_发射窗口与上升段闭合.py` | §1.1~1.2 | 文昌当日发射窗口 + 逃逸双曲线 + 定时闭合解 + 上升段闭合链，导出 GMAT 四钟状态（A1/A1c/A2/B） | 10 s | numpy |
| `step2_B平面设计与差分修正.py` | §1.2.3/§1.3 | 圆锥拼接标称 B 平面 + 日心 N 体精确模型差分修正（GMAT 对拍后终版：`--dep` 接受时刻、`--book` 书表对账、`--selftest` 自检） | 5 s | numpy（scipy 可选） |
| `step3_日心段中途修正.py` | §1.4 | 灵敏度/修正脉冲/蒙特卡洛色散/碰撞概率，自带回归 | 5 s | 同上；会自动加载 step2 交叉校验 |
| `step4_火星捕获段设计.py` | §1.5 | 闭式捕获预算 + 间接法对称弧 + PSO + primer 协态校验 | 15 s | numpy + **scipy 必需** |
| `step5_捕获段误差分析.py` | §1.5.4 | 表1.15 标称 + 表1.16 三类误差蒙特卡洛 | ~9 min | 同上 |
| `step6_引力辅助修正与EMEJ桥接.py` | §3.4 | 书链（火借力→Flora 微分修正）+ 中保真桥接器（nodes→精确星历 CSV，GMAT 可读） | 1~3 min | numpy + scipy；DE421（本目录自带 `de421.bsp`） |
| `step7_EMEJ借力优化导出.py` | — | pykep MGA-1DSM（地火地木）SADE 优化 + 导出 nodes.json | 5 s（复放）/ 分钟级（重优化） | **pykep + pygmo** |

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
        --csv /tmp/repro.csv                 # 桥接全链（11 事件）
diff /tmp/repro.csv audit/emej_repro.csv     # 应 byte-identical（审计已验）
python3 step6_引力辅助修正与EMEJ桥接.py --book              # 书链（§3.4 Flora）

# step7 重新优化（不固定种子，每次不同）
python3 step7_EMEJ借力优化导出.py --seed 42 --gen 30 --islands 8 --pop 15 \
        --out data/emej_nodes_new.json --no-plot
```

audit/ 里还放了两个独立复核脚本：`audit/verify_tau.py`（τ 独立复核）、
`audit/recon_x.py`（染色体反演）。

## DE421 星历获取（step6 桥接用）

`de421.bsp`（16.8 MB）不随 git 仓库分发，放本目录即自动生效：

```bash
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp -O de421.bsp
# 或从 https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ 下载 DE421
```

## 目录

```
dihuo/
├── step1 ~ step7（.py）   主脚本：step2 = GMAT 对拍后终版（dihuo2），
│                          旧书版在 versions/；其余 = 各源仓库终版
├── data/                  emej_nodes.json / emej.csv / nodes_example.json
├── de421.bsp              DE421 星历（step6 桥接用，放本目录自动生效）
├── audit/                 EMEJ 审计复现产物 + 独立复核脚本 + CHANGELOG
├── originals/             emej 上传原件字节级副本
├── versions/              step2/3/4 旧版导出（免 git 取用）
├── AUDIT.md               两链合并审计报告
└── README.md              本文件
```

## 版本历史

dihuo 是收编仓库：step2~5 来自 mars-probe-design（git 5 commits），
step6/step7 及 audit 来自 emej-audit（git 带 tag v0~v5 + emej-chain，另有
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
