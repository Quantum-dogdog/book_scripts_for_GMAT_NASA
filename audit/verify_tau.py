# -*- coding: utf-8 -*-
"""审计独立复核：
(1) τ 公式检验——CSV flyby_entry 状态二体积分到近星点，真值 vs CSV tau 列
(2) B 平面 T/R 基：黄道 N vs 赤道 N（GMAT SunMJ2000Eq 语义）下的 BT/BR 差异
(3) 时间尺度：UTC-as-TDB 的 69.184 s → 行星位移量级
(4) 出发/到达状态 vs nodes 一致性抽查
"""
import csv, math, sys
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, "/home/z/my-project/emej-audit")
import os
os.environ.setdefault("DE421_PATH", "/home/z/my-project/moon2earth/de421.bsp")
import importlib.util
spec = importlib.util.spec_from_file_location("m6", "/home/z/my-project/emej-audit/6.py")
# 只借用其常数与 _b_state：避免触发 import 时的书链种子 fsolve——手工复制所需量
GM = {"sun": 1.32712440018e11, "earth": 398600.435436, "mars": 42828.37,
      "jupiter": 1.26686534e8}
EPS = math.radians(23.4392911)
RX = np.array([[1, 0, 0], [0, math.cos(EPS), math.sin(EPS)], [0, -math.sin(EPS), math.cos(EPS)]])
from jplephem.spk import SPK
K = SPK.open(os.environ["DE421_PATH"])
PAIRS = {"earth": [(0, 3), (3, 399)], "mars": [(0, 4), (4, 499)], "jupiter": [(0, 5)]}
DAY = 86400.0

def bstate(name, t_mjd):
    jd = t_mjd + 2400000.5
    r = np.zeros(3); v = np.zeros(3)
    for c, t in PAIRS[name]:
        p, vv = K[(c, t)].compute_and_differentiate(jd)
        r += np.asarray(p, float); v += np.asarray(vv, float)
    ps, vs = K[(0, 10)].compute_and_differentiate(jd)
    r -= np.asarray(ps, float); v -= np.asarray(vs, float)
    return RX @ r, (RX @ v) / DAY      # 黄道系

def read_csv(path):
    rows = []
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    rd = csv.DictReader(lines)
    for r in rd:
        rows.append(r)
    return rows

rows = read_csv("/home/z/my-project/emej-audit/data/emej.csv")

def state_of(r):   # CSV 为赤道系 -> 转黄道
    y = np.array([float(r["x_km"]), float(r["y_km"]), float(r["z_km"]),
                  float(r["vx_km_s"]), float(r["vy_km_s"]), float(r["vz_km_s"])])
    return np.concatenate([RX @ y[:3], RX @ y[3:6]])

print("=" * 78)
print("(1) tau 复核：flyby_entry 状态二体积分到近星点（r·v=0 事件）")
for r in rows:
    if r["event"] != "flyby_entry":
        continue
    b = r["body"]; t_in = float(r["epoch_mjd"]); tau_csv = float(r["tau_s"])
    y = state_of(r)
    rb, vb = bstate(b, t_in)
    rr, vv = y[:3] - rb, y[3:6] - vb
    mu = GM[b]
    def rhs(t, s):
        rn = np.linalg.norm(s[:3])
        return [s[3], s[4], s[5], *(-mu * s[:3] / rn**3)]
    def peri(t, s):
        return float(np.dot(s[:3], s[3:6]))
    peri.direction = -1.0    # r·v 由正变负->不对：接近段 r·v<0，过近点后>0，取升穿
    peri.direction = 1.0
    sol = solve_ivp(rhs, (0, 60 * DAY), np.concatenate([rr, vv]),
                    method="DOP853", rtol=1e-12, atol=1e-9, events=[peri])
    t_peri_true = sol.t_events[0][0]
    tau_true = t_peri_true
    # 代码公式重算（v5 _b_bplane 的 H 计算）
    rn = np.linalg.norm(rr); vn = np.linalg.norm(vv)
    ev = ((vn**2 - mu / rn) * rr - np.dot(rr, vv) * vv) / mu
    e = max(np.linalg.norm(ev), 1.0 + 1e-6)
    vinf = math.sqrt(max(1e-9, vn**2 - 2 * mu / rn))
    cosnu = np.dot(ev, rr) / (e * rn)
    nu = math.acos(max(-1, min(1, cosnu)))
    if np.dot(rr, vv) < 0:
        nu = 2 * math.pi - nu
    H_code = math.asinh((e + math.cos(nu)) / (1 + e * math.cos(nu)))
    a_ = -mu / vinf**2
    tau_code = math.sqrt(-a_**3 / mu) * (e * math.sinh(H_code) - H_code)
    if np.dot(rr, vv) > 0:
        tau_code = -tau_code
        tau_true = -t_peri_true
    print(f"  {b}:  CSV tau={tau_csv:.1f}s | 2体积分真值={tau_true:.1f}s | "
          f"公式复算={tau_code:.1f}s | 公式-CSV={tau_code-tau_csv:+.2f}s | "
          f"公式-真值={tau_code-tau_true:+.2f}s ({(tau_code-tau_true)/tau_true*100:+.3f}%)")

print()
print("(1b) flyby_periapsis 历元一致性（t_in + tau 真值 vs CSV 行）")
peris = {r["body"]: float(r["epoch_mjd"]) for r in rows if r["event"] == "flyby_periapsis"}
entries = {r["body"]: float(r["epoch_mjd"]) for r in rows if r["event"] == "flyby_entry"}
for b, t_e in entries.items():
    print(f"  {b}: CSV(t_peri)-CSV(t_in) = {(peris[b]-t_e)*DAY:.1f} s（与 tau 列同源）")

print()
print("(2) B 平面 T/R 基：黄道 N vs 赤道 N（GMAT SunMJ2000Eq）")
nodes = __import__("json").load(open("/home/z/my-project/emej-audit/data/emej_nodes.json"))
for r in rows:
    if r["event"] != "flyby_entry":
        continue
    b = r["body"]; t_in = float(r["epoch_mjd"])
    y = state_of(r); rb, vb = bstate(b, t_in)
    rr, vv = y[:3] - rb, y[3:6] - vb
    mu = GM[b]
    h = np.cross(rr, vv); nh = h / np.linalg.norm(h)
    vinf = math.sqrt(np.dot(vv, vv) - 2 * mu / np.linalg.norm(rr))
    # S：入射渐近线方向（进动方向）
    ev = ((np.dot(vv, vv) - mu / np.linalg.norm(rr)) * rr - np.dot(rr, vv) * vv) / mu
    e = np.linalg.norm(ev)
    beta = math.acos(1 / e)
    nx = np.cross(nh, ev); nx /= np.linalg.norm(nx)
    S = math.cos(beta) * ev / e + math.sin(beta) * nx
    bmag = np.linalg.norm(h) / vinf
    B = bmag * np.cross(S, nh)
    for tag, N in (("黄道N(6.py)", np.array([0., 0., 1.])),
                   ("赤道N(GMAT SunMJ2000Eq)", RX @ np.array([0., 0., 1.]))):
        T = np.cross(S, N); T /= np.linalg.norm(T)
        Rv = np.cross(S, T)
        print(f"  {b} [{tag}]: BT={np.dot(B,T):9.1f}  BR={np.dot(B,Rv):9.1f}  |B|={bmag:8.1f}")
    print(f"  {b} CSV: BT={float(r['BT_km']):9.1f}  BR={float(r['BR_km']):9.1f}")

print()
print("(3) 时间尺度：若按真 TDB=UTC+69.184s 取星历，行星位置差")
for b, t in (("mars", 60733.031), ("earth", 61390.375), ("jupiter", 62596.524)):
    r1, _ = bstate(b, t); r2, _ = bstate(b, t + 69.184 / DAY)
    d = np.linalg.norm(r2 - r1)
    print(f"  {b} @MJD {t}: 69.184s 位移 = {d:8.1f} km")

print()
print("(4) 出发/到达状态与 nodes 一致性")
dep = next(r for r in rows if r["event"] == "departure")
t0 = nodes["departure"]["epoch_mjd"]
rE, vE = bstate("earth", t0)
y = state_of(dep)
vinf_dep_csv = y[3:6] - vE
vd = np.array(nodes["departure"]["vinf_km_s"])
print(f"  出发 vinf: nodes={np.round(vd,4)}  CSV(地球中心,同历元)={np.round(vinf_dep_csv,4)}")
print(f"  出发 SOI 出口离地心 = {np.linalg.norm(y[:3]-rE):.0f} km (SOI=9.25e5)")
arr = next(r for r in rows if r["event"] == "arrival")
rJ, _ = bstate("jupiter", float(arr["epoch_mjd"]))
ya = state_of(arr)
print(f"  到达 miss(CSV)={float(arr['miss_km'])} km | 复算={np.linalg.norm(ya[:3]-rJ):.3f} km")
# 无动力飞越能量守恒抽查：entry->exit |v_rel| 应基本不变（设计口径）
for b in ("mars", "earth"):
    en = next(r for r in rows if r["event"] == "flyby_entry" and r["body"] == b)
    ex = next(r for r in rows if r["event"] == "flyby_exit" and r["body"] == b)
    ye, yx = state_of(en), state_of(ex)
    te, tx = float(en["epoch_mjd"]), float(ex["epoch_mjd"])
    _, vbe = bstate(b, te); _, vbx = bstate(b, tx)
    v1 = np.linalg.norm(ye[3:6] - vbe); v2 = np.linalg.norm(yx[3:6] - vbx)
    print(f"  {b} 飞越: |v_rel(entry)|={v1:.4f} |v_rel(exit)|={v2:.4f} km/s "
          f"(nodes 设计 {nodes['meta']['vinf_in_mag'][0 if b=='mars' else 1]:.4f})")
