#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.py — 火星捕获段 §1.5.4 误差分析（教材 p64~66，表1.15/1.16，图1.32/1.33）

自包含单文件（不导入 2/3/4.py，规避工作副本被回退风险），复用与 4.py 同源的
平面二体积分内核。标称捕获 = 表1.15「姿态角速度恒定」模型：
  推力大小 490 N、比冲 312 s、作用时间 tF=6069.21 s、燃料 928.856 kg；
  捕获前 a=3290.965 e=2.485 rp=4887.684；捕获后 a=43695 e=0.90857 rp=3995.000；
  姿态角 θ(t)=ω·t+b，ω=2.24e-4 rad/s，b=-0.383 rad（开机时刻由锚定求解）。

§1.5.4 三类误差（书 p65~66，均正态、均值 0）：
  (1) 测量误差：定轨 15 km(3σ)→σr=5 km；测速 3 m/s(3σ)→σv=1 m/s。
      捕获开始时对初始位置、速度各加 δr、δv。
  (2) 发动机执行误差：推力 5%(3σ)→σF=490×0.05/3 N。工作推力 Fmax+δFmax。
  (3) 姿态误差：确定 0.5°(3σ)→0.5°/3；指向 1.0°(3σ)→1.0°/3。推力方向偏转 (δα+δβ)。
分别注入单源与综合，蒙特卡洛统计近火点/远火点（捕获后轨道根数）均值与标准差，
复现表1.16，并核对结论：推力误差>姿态误差>测量误差；近火点偏差小、远火点偏差大。

CLI:
  python 5.py                 # 标称(表1.15)校验 + 表1.16 全表
  python 5.py --nmc 800       # 指定蒙特卡洛样本数
  python 5.py --plot          # 出图1.32(轨迹)/图1.33(推力角) PNG（需 matplotlib）
"""
from __future__ import annotations
import argparse
import math
import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# 常数与发动机（与 4.py 同源）
# ---------------------------------------------------------------------------
MU = 42828.37
R_MARS = 3396.2
R_SOI = 5.77e5
G0 = 9.80665
V_INF2 = 3.628
THRUST = 490.0
ISP = 312.0
M0 = 2350.0
MDOT = THRUST / (ISP * G0)
THRUST_KM = THRUST * 1e-3
DRY = 1000.0

# 表1.15 标称「姿态角速度恒定」捕获参数。
# 注：书 b=-0.383/ω 属书的坐标系；本自包含模型把 α0 锚定为「近火点逆行」，
# 并取与书同形的捕获椭圆（a≈43695, e≈0.9086, rp≈3995）自洽反推 rp2/tF：
T115 = dict(rp2=4750.0, tF=6250.0, omega=2.24e-4, b=-0.383,
            fuel=928.856, a_before=3290.965, e_before=2.485,
            a_after=43695.0, e_after=0.90857, rp_after=3995.000)
RP_AFTER = T115["rp_after"]
RA_AFTER = T115["a_after"] * (1.0 + T115["e_after"])   # 标称远火点 ≈ 83384 km

# §1.5.4 误差标准差（1σ）
SIG_POS = 5.0                    # km（定轨 15 km/3σ）
SIG_VEL = 1.0e-3                 # km/s（测速 3 m/s/3σ = 1 m/s）
SIG_THRUST = 490.0 * 0.05 / 3.0  # N（推力 5%/3σ）
# 姿态：确定 0.5°/3σ、指向 1.0°/3σ 两个独立分量 → 合成 σ=√(σ_α²+σ_β²)
SIG_ATT = math.sqrt((0.5/3.0)**2 + (1.0/3.0)**2) * math.pi / 180.0


# ---------------------------------------------------------------------------
# 平面二体内核（与 4.py 一致；_accel 增加推力/姿态扰动入参）
# ---------------------------------------------------------------------------
def _accel(t, r, v, m, mode, alpha0, omega, t_ref, thrust_scale, att_off):
    rn = max(float(np.linalg.norm(r)), R_MARS)
    a = -MU * r / rn ** 3
    dm = 0.0
    if mode != "coast" and m > DRY:
        if mode == "omega":
            ang = alpha0 + att_off + omega * (t - t_ref)
            d = np.array([math.cos(ang), math.sin(ang)])
        else:
            d = np.zeros(2)
        a = a + (THRUST_KM * thrust_scale) * d / m
        dm = -(THRUST * thrust_scale) / (ISP * G0)
    return a, dm


def _rhs(t, y, mode, alpha0, omega, t_ref, thrust_scale, att_off):
    r = y[:2]; v = y[2:4]; m = y[4]
    a, dm = _accel(t, r, v, m, mode, alpha0, omega, t_ref, thrust_scale, att_off)
    return [v[0], v[1], a[0], a[1], dm]


def _integrate(y0, tspan, mode, alpha0=0.0, omega=0.0, t_ref=0.0,
               thrust_scale=1.0, att_off=0.0, max_step=5.0, events=None):
    def _impact(t, y, *args):
        return float(np.linalg.norm(y[:2])) - R_MARS
    _impact.terminal = True; _impact.direction = -1.0
    if _HAVE_SCIPY:
        evs = [_impact] + ([events] if events else [])
        sol = solve_ivp(_rhs, tspan, y0,
                        args=(mode, alpha0, omega, t_ref, thrust_scale, att_off),
                        method="DOP853", rtol=1e-9, atol=1e-6,
                        max_step=max_step, dense_output=True, events=evs)
        return sol.y[:, -1], sol
    t0, tf = tspan
    n = max(1, int((tf - t0) / max_step)); h = (tf - t0) / n
    y = np.array(y0, float); t = t0
    for _ in range(n):
        k1 = np.array(_rhs(t, y, mode, alpha0, omega, t_ref, thrust_scale, att_off))
        k2 = np.array(_rhs(t + h/2, y + h*k1/2, mode, alpha0, omega, t_ref, thrust_scale, att_off))
        k3 = np.array(_rhs(t + h/2, y + h*k2/2, mode, alpha0, omega, t_ref, thrust_scale, att_off))
        k4 = np.array(_rhs(t + h, y + h*k3, mode, alpha0, omega, t_ref, thrust_scale, att_off))
        y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6; t += h
    return y, None


def _peri_event(t, y, *args):
    r = y[:2]; v = y[2:4]
    return (r[0]*v[0] + r[1]*v[1]) / (float(np.linalg.norm(r)) + 1e-12)
_peri_event.terminal = True; _peri_event.direction = 1.0


def soi_state(v_inf=V_INF2, rp=None, R=R_SOI):
    if rp is None:
        rp = T115["rp2"]
    e = 1.0 + rp * v_inf ** 2 / MU
    p = rp * (1.0 + e)
    h = math.sqrt(MU * p)
    cosnu = max(-1.0, min(1.0, (p / R - 1.0) / e))
    nu = -math.acos(cosnu)
    rhat = np.array([math.cos(nu), math.sin(nu)])
    that = np.array([-math.sin(nu), math.cos(nu)])
    vr = (MU / h) * e * math.sin(nu)
    vt = (MU / h) * (1.0 + e * math.cos(nu))
    v = vr * rhat + vt * that
    return np.array([R*rhat[0], R*rhat[1], v[0], v[1]])


def two_body_elements(r, v, mu=MU):
    rn = float(np.linalg.norm(r)); vn = float(np.linalg.norm(v))
    energy = 0.5*vn*vn - mu/rn
    a = -mu/(2.0*energy)
    h = float(r[0]*v[1] - r[1]*v[0])
    e = math.sqrt(max(0.0, 1.0 - h*h/(mu*a))) if a > 0 else \
        math.sqrt(1.0 + 2.0*energy*h*h/(mu*mu))
    rp = a*(1.0-e) if a > 0 else -a*(e-1.0)
    ra = a*(1.0+e) if a > 0 else float("inf")
    return dict(a=a, e=e, rp=rp, ra=ra, h=h)


_peri_cache = {}


def time_to_periapsis(rp2):
    key = round(rp2, 3)
    if key in _peri_cache:
        return _peri_cache[key]
    y0 = np.concatenate([soi_state(rp=rp2), [M0]])
    _, sol = _integrate(y0, (0.0, 4e5), "coast", max_step=2000.0, events=_peri_event)
    te = sol.t_events[1]
    tp = float(te[0]) if len(te) else 4e5
    _peri_cache[key] = tp
    return tp


# ---------------------------------------------------------------------------
# 标称捕获：θ(t)=ω(t-t1)+b，开机 t1 锚定使捕获后 rp=RP_AFTER
# ---------------------------------------------------------------------------
def capture_omega(t1, rp2=None, tF=None, omega=None, b=None,
                  thrust_scale=1.0, att_off=0.0, r0=None, coast_after=1.2e5):
    if rp2 is None: rp2 = T115["rp2"]
    if tF is None: tF = T115["tF"]
    if omega is None: omega = T115["omega"]
    if b is None: b = T115["b"]
    if r0 is None:
        r0 = soi_state(rp=rp2)
    y = np.concatenate([r0, [M0]])
    y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    # 开机时推力方向 = b 相对 x；令其在近火点≈逆行：以 t1 为参考，b 为开机角
    y2, _ = _integrate(y1, (t1, t1+tF), "omega", alpha0=alpha0, omega=omega,
                       t_ref=t1, thrust_scale=thrust_scale, att_off=att_off,
                       max_step=20.0)
    y3, _ = _integrate(y2, (t1+tF, t1+tF+coast_after), "coast", max_step=2000.0)
    el = two_body_elements(y3[:2], y3[2:4])
    return dict(fuel=M0 - y2[4], elems=el, y_burn=y2, alpha0=alpha0)


def _alpha0_retro_at_center(t1, rp2, tF, omega):
    """令燃烧弧中点（≈近火点）处推力方向 = 速度反方向，反推开机角 α0。
    这是书 ω=2.24e-4 rad/s「匀速扫过逆行方向」的自洽实现（不依赖书的坐标系）。"""
    tp = time_to_periapsis(rp2)
    y = np.concatenate([soi_state(rp=rp2), [M0]])
    ymid, _ = _integrate(y, (0.0, tp), "coast", max_step=2000.0, events=_peri_event)
    v = ymid[2:4]
    psi = math.atan2(-v[1], -v[0])           # 近火点逆行方向（惯性系）
    t_mid = t1 + tF / 2.0
    return psi - omega * (t_mid - t1)


def capture_omega(t1, rp2=None, tF=None, omega=None, b=None,
                  thrust_scale=1.0, att_off=0.0, r0=None, coast_after=1.2e5,
                  y0=None):
    """恒角速度捕获：α0 自动锚定为「近火点逆行」。
    y0 给定时视为开机状态（跳过 coast 段，用于在开机点注入测量误差）。"""
    if rp2 is None: rp2 = T115["rp2"]
    if tF is None: tF = T115["tF"]
    if omega is None: omega = T115["omega"]
    alpha0 = _alpha0_retro_at_center(t1, rp2, tF, omega)
    if y0 is not None:
        y1 = np.asarray(y0, float)
    else:
        if r0 is None:
            r0 = soi_state(rp=rp2)
        y = np.concatenate([r0, [M0]])
        y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    y2, _ = _integrate(y1, (t1, t1+tF), "omega", alpha0=alpha0, omega=omega,
                       t_ref=t1, thrust_scale=thrust_scale, att_off=att_off,
                       max_step=20.0)
    y3, _ = _integrate(y2, (t1+tF, t1+tF+coast_after), "coast", max_step=2000.0)
    el = two_body_elements(y3[:2], y3[2:4])
    return dict(fuel=M0 - y2[4], elems=el, y_burn=y2, alpha0=alpha0)


def solve_nominal_t1():
    """标称：燃烧弧关于近火点对称 t1=t_peri-tF/2（书 p58 制动发生在近火点）。"""
    tp = time_to_periapsis(T115["rp2"])
    t1 = tp - T115["tF"] / 2.0
    return t1, capture_omega(t1)



# ---------------------------------------------------------------------------
# §1.5.4 蒙特卡洛误差分析
# ---------------------------------------------------------------------------
def ignition_state(t1, rp2=None):
    """从 SOI 惯性入射 coast 到开机时刻 t1 的标称状态（捕获制导初始状态）。"""
    if rp2 is None:
        rp2 = T115["rp2"]
    y = np.concatenate([soi_state(rp=rp2), [M0]])
    y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    return y1


def run_mc(nmc, use_meas, use_thr, use_att, t1_nom, rp_nom, ra_nom, seed=11):
    rng = np.random.default_rng(seed)
    y_ign = ignition_state(t1_nom)                 # 标称开机状态
    rp_err, ra_err = [], []
    for _ in range(nmc):
        # 书「捕获开始时」= 发动机制动段起点（近火点附近），测量误差加于此，
        # 杠杆臂小 → 近火点色散小（书 σ_rp≈3.8 km），与在 SOI 注入(色散大)不同。
        y0 = y_ign.copy()
        if use_meas:
            y0[0] += rng.normal(0, SIG_POS); y0[1] += rng.normal(0, SIG_POS)
            y0[2] += rng.normal(0, SIG_VEL); y0[3] += rng.normal(0, SIG_VEL)
        ts = 1.0 + (rng.normal(0, SIG_THRUST) / THRUST if use_thr else 0.0)
        ao = rng.normal(0, SIG_ATT) if use_att else 0.0
        res = capture_omega(t1_nom, y0=y0, thrust_scale=ts, att_off=ao)
        el = res["elems"]
        if el["a"] > 0 and np.isfinite(el["ra"]):
            rp_err.append(el["rp"] - rp_nom)
            ra_err.append(el["ra"] - ra_nom)
    rp_err = np.array(rp_err); ra_err = np.array(ra_err)
    return dict(rp_mean=float(np.mean(rp_err)), rp_std=float(np.std(rp_err)),
                ra_mean=float(np.mean(ra_err)), ra_std=float(np.std(ra_err)),
                n=len(rp_err))


def table_116(nmc, t1_nom, rp_nom, ra_nom):
    rows = [("测量误差", True, False, False),
            ("执行误差", False, True, False),
            ("姿态误差", False, False, True),
            ("综合误差", True, True, True)]
    out = []
    for name, um, ut, ua in rows:
        s = run_mc(nmc, um, ut, ua, t1_nom, rp_nom, ra_nom)
        out.append((name, s))
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report_nominal(t1_nom, nom):
    L = []
    L.append("=" * 70)
    L.append("5.py — 火星捕获 §1.5.4 误差分析（表1.15 恒角速度标称 + 表1.16）")
    L.append("=" * 70)
    e = nom["elems"]
    L.append("标称(表1.15) 恒角速度捕获：ω=%.2e rad/s tF=%.2f s（α0 自洽锚定近火点逆行）"
             % (T115["omega"], T115["tF"]))
    L.append("  锚定开机 t1=%.0f s（近火点 coast 时刻 t_peri=%.0f s）"
             % (t1_nom, time_to_periapsis(T115["rp2"])))
    L.append("  燃料=%.1f kg（书 928.856）  捕获后 a=%.1f e=%.5f rp=%.1f ra=%.1f"
             % (nom["fuel"], e["a"], e["e"], e["rp"], e["ra"]))
    L.append("  书表1.15 捕获后：a=43695 e=0.90857 rp=3995.000")
    return "\n".join(L)


def report_table(rows):
    L = ["", "表1.16  490 N 推力误差影响情况（近/远火点误差 = 捕获后根数 - 标称）",
         "-" * 70,
         "误差来源   | 近火点均值/km 近火点σ/km | 远火点均值/km 远火点σ/km",
         "-" * 70]
    for name, s in rows:
        L.append("%-8s | %12.4f %11.2f | %13.4f %11.2f"
                 % (name, s["rp_mean"], s["rp_std"], s["ra_mean"], s["ra_std"]))
    L.append("-" * 70)
    # 结论：按远火点 σ 排序
    order = sorted([r for r in rows if r[0] != "综合误差"],
                   key=lambda r: -r[1]["ra_std"])
    L.append("影响程度排序（远火点σ）：" + " > ".join(r[0] for r in order)
             + "   （书：推力>姿态>测量）")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="火星捕获 §1.5.4 误差分析（单文件）")
    ap.add_argument("--nmc", type=int, default=400, help="蒙特卡洛样本数")
    ap.add_argument("--plot", action="store_true", help="出图1.32/1.33 PNG")
    a = ap.parse_args(argv)
    if not _HAVE_SCIPY:
        print("需要 scipy（solve_ivp/brentq）。"); return 1
    t1_nom, nom = solve_nominal_t1()
    rp_nom = nom["elems"]["rp"]; ra_nom = nom["elems"]["ra"]
    print(report_nominal(t1_nom, nom))
    print(report_table(table_116(a.nmc, t1_nom, rp_nom, ra_nom)))
    if a.plot:
        try:
            make_figs(t1_nom, nom)
            print("\n已输出 fig1_32.png / fig1_33.png")
        except Exception as ex:
            print("\n出图失败（无 matplotlib?）：%s" % ex)
    return 0


def make_figs(t1_nom, nom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alpha0 = nom["alpha0"]; tF = T115["tF"]; om = T115["omega"]
    # 图1.32 发动机工作段轨迹
    y1 = ignition_state(t1_nom)
    y2, _ = _integrate(y1, (t1_nom, t1_nom+tF), "omega", alpha0=alpha0,
                       omega=om, t_ref=t1_nom, max_step=20.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    tev = np.linspace(t1_nom, t1_nom+tF, 400)
    solp = solve_ivp(_rhs, (t1_nom, t1_nom+tF), y1,
                     args=("omega", alpha0, om, t1_nom, 1.0, 0.0),
                     method="DOP853", rtol=1e-9, atol=1e-6,
                     max_step=20.0, t_eval=tev)
    ax.plot(solp.y[0], solp.y[1], "-", label="powered-arc trajectory")
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_MARS*np.cos(th), R_MARS*np.sin(th), "--", label="Mars limb")
    # 自动缩放到动力弧范围（保留火星轮廓作参考）
    xs = np.concatenate([solp.y[0], R_MARS*np.cos(th)])
    ys = np.concatenate([solp.y[1], R_MARS*np.sin(th)])
    pad = 0.08*(max(xs.max()-xs.min(), ys.max()-ys.min()))
    ax.set_xlim(xs.min()-pad, xs.max()+pad)
    ax.set_ylim(ys.min()-pad, ys.max()+pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x/km"); ax.set_ylabel("y/km"); ax.legend()
    ax.set_title("Fig 1.32  490 N capture trajectory")
    ax.grid(True, ls="--"); fig.tight_layout(); fig.savefig("fig1_32.png", dpi=120)
    # 图1.33 推力方向与速度反向角
    g = np.linspace(t1_nom, t1_nom+tF, 200)
    tr = np.empty((len(g), 5)); yc = y1.copy(); tt = t1_nom
    for k, tg in enumerate(g):
        ys, _ = _integrate(yc, (tt, tg), "omega", alpha0=alpha0, omega=om,
                           t_ref=t1_nom, max_step=20.0) if tg > tt else (yc, None)
        tr[k] = ys; yc = ys; tt = tg
    ang_t = alpha0 + om*(g - t1_nom)
    ang_v = np.array([math.atan2(-v[1], -v[0]) for v in tr[:, 2:4]])
    ang_v = np.unwrap(ang_v)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    rel = g - t1_nom
    ax2.plot(rel, ang_t - ang_t[0], "-", label="thrust dir vs x-axis")
    ax2.plot(rel, ang_v - ang_v[0], "--", label="-velocity dir vs x-axis")
    ax2.set_xlabel("thrust time /s"); ax2.set_ylabel("angle /rad")
    ax2.set_title("Fig 1.33  thrust vs retrograde direction")
    ax2.grid(True, ls="--"); ax2.legend(); fig2.tight_layout()
    fig2.savefig("fig1_33.png", dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
