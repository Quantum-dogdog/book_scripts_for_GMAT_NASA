# emej 链版本变更记录

来源：10/27 上传的 11 个文件（6 个 6.py 版本 + emej_export.py + 3 个数据件）。
上传顺序即演进顺序（行数单调递增 748→874→949→1140→1265→1327）。
git 标签与原件 md5 见 README.md。

## 总叙事

一条「教材复现」如何长成「MGA 任务设计工具」：

- v0–v2：复现教材 §3.4（地球→火星借力→Flora）。三个版本都在与"微分修正不收敛"
  搏斗——升星历（DE421）、加全局搜索（PSO）、扩到 6 元混合修正，书表 3.27/3.28
  始终无法同时满足。**审计发现其中一个隐藏推手是 τ 公式错误（见 AUDIT F2）：
  书链的 τ 度量存在硬下限，τ→0 目标永远差 ~27 s，DC 残差停滞在 33.8 km。**
- v3：架构转向（bridge pivot）。放弃在书算例上死磕，改为把 pykep 脉冲 MGA 序列
  桥接进精确力模型、导出 GMAT 可用 CSV。PSO/6 元 DC 全部删除。
- v4：输入通用化（nodes JSON，飞越数不限）。
- v5：支持 MGA-1DSM 的深空机动；出发处理改为与 pykep 一致的"天体中心+vinf"
  （旧的反向积分缺陷在代码注释中有记录：段误差放大到 1e7 km）。
- emej_export.py + emej_nodes.json + emej.csv：地-火-地-木（EMEJ）完整链交付，
  CSV 含 2 飞越 + 3 DSM + 交会残差 0.576 km。

## 逐版变更

### v0-book-repro（748 行）
- Lambert 普适变量解（Stumpff 保护 + brentq），Standish 近似根数星历。
- §3.4.1 拼接锥两腿设计 + fsolve 种子（直击书 B 平面）。
- §3.4.2 精确多体模型（日+地+火）3 元微分修正；B 平面、τ（式 3.108）。
- 诚实结论：3 分量 vd 不足以同时修 B 平面与命中 Flora（书中先全局寻优的原因）。

### v1-de421-pso（874 行，+126）
- 接入 jplephem/DE421：SSB 减太阳位矢得日心、绕 x 轴 −ε 转黄道；不可用回退 Standish。
- 删除内置 Lambert（外部不再需要）。
- 种子改 4 元 fsolve（vd,tM 对 [BT,BR,τ,|vinf|]）。
- 出发状态改"地球 SOI 出口沿 vinf 偏移"（避免落在地心引力奇异点）。
- 新增 PSO 5D 全局搜索（Kennedy–Eberhart 标准）与 --pso 开关。

### v2-dc6var（949 行，+75）
- 新增 6 元混合 DC（dc_6var_hybrid，least_squares/TRF）：
  [vd(3),tM,tf,phi] 同时约束 B 平面 + Flora 交会。
- 报告增加 §3.4.2-扩展小节；结论：|vi| 约束冲突，需要精确代价的 PSO。

### v3-bridge-pivot（1140 行，+191）【架构转向】
- 删除：PSO、6 元混合 DC、书链种子的大部分复杂度。
- 新增"中型保真桥接器"：
  - `_B_Eph` 行星历表缓存（1 天网格 + Hermite 插值，仅供摄动体）；
  - `_b_lambert` Lambert 回归（桥接需要）；
  - `_b_bplane` 状态法 B 平面 + τ；
  - `build_bridge_sequence`：patched-conic MGA → 逐段 TCM 打靶 → 飞越双曲线重建；
  - `export_bridge_csv`：GMAT SunMJ2000Eq（J2000 地球赤道）坐标导出。

### v4-nodes-json（1265 行，+125）
- 输入通用化：nodes 字典（departure/flybys/arrival），飞越个数不限。
- `build_bridge_nodes`（书链种子产出 nodes）+ `seq_from_nodes`（nodes → 桥接序列）。
- `--dump-nodes` 模板导出（nodes_example.json 即其产物）。
- 两级打靶：gpos（位置粗修）→ gfun（B 平面+τ 精修）。
- 飞越几何由进/离 vinf + rp 重建（`_b_flyby_geom`：转角定 e、近星点反解）。

### v5-dsm（1327 行，+62）
- DSM（MGA-1DSM）支持：nodes 可选 "dsms" 数组与各段一一对应；
  `_b_prop_leg` 段内脉冲；CSV 增 dsm 行（给 PRE-burn 状态 + dv 列）。
- 出发重构：与 pykep 完全一致，t0 从"天体中心 + vinf"起算
  （代码注释记录旧反向积分缺陷：103~10^3 天段误差放大到 1e7 km）。
- 飞越近星点速度改用设计 vinf 反算（保 |v∞out|=|v∞in|=设计值）。
- `_b_hyper_exit_days`：出发 SOI 出口历元/状态（正确公式：acosh + e·sinhH−H）。

### emej-chain（数据 + 前端）
- `emej_export.py`：pykep MGA-1DSM（地火地木）SADE 优化 + 导出 nodes.json；
  内置 `_check_against_udp`（对 pykep 自身 _compute_dvs 校验 DSM）。
- `emej_nodes.json`：优化结果（2 飞越 + 3 DSM + meta）。
- `emej.csv`：桥接输出（须 ≥ v5 生成——含 dsm 行）。
  **复现凭证：从 emej_nodes.json 重跑 6.py 得到的 CSV 与交付件逐字节一致。**

## 丢失项

- `*_x.json`（冠军染色体）未上传——SADE 不定 seed 时结果不可复现；
  建议补交并存入 data/（AUDIT F7）。
