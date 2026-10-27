#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.py — §3.4 精确动力学模型中的引力辅助（教材 p155~164，表3.24~3.28，图3.34~3.37）

任务：地球出发 → 火星引力辅助 → 交会小行星 8 号 Flora。
方法链（书结构）：
  §3.4.1 简单(等效脉冲)模型初步设计：Lambert 日心转移 + 地心逃逸双曲线拼接，
         给出 t0/tM/tf、地心逃逸近地点速度 vp0、火星进/离双曲线剩余速度。
  §3.4.2 精确模型轨道修正：B 平面参数(3.97~3.107) + 时间参数 τ(3.108)
         + 微分修正(3.109~3.111)，设计变量取地心逃逸近地点速度 vp0。
  §3.4.3 算例：两次微分修正 → 表3.27(火星借力) / 表3.28(Flora 交会)。

复现所需但书页未给、本文件按标准替代的「缺失条件」（见 README 说明）：
  (1) 星历源：书用 JPL DE/STK；此处用 JPL 近似开普勒根数解析算地球/火星，
      Flora 用表3.24根数二体递推 → 数值与书有百分之几偏差，方法与图可复现。
  (2) 太阳光压力参数未给 → SRP 置零（USE_SRP 开关）。
  (3) 精确模型摄动行星：取 日 + 地 + 火（USE_MOON 可加月球）。
  (4) PSO 初值搜索不重跑，直接用书中表3.25/3.26 结果作微分修正初值。

单位：km, km/s, 日(MJD)。力模型为日心惯性系(J2000 黄道)多体。
CLI:
  python 6.py                 # 默认：二体序列 -> 精确模型桥接 -> bridge.csv(GMAT 初值/节点)
  python 6.py --csv out.csv   # 指定输出 CSV
  python 6.py --book          # 教材表3.27/3.28 微分修正复现(报告)
  python 6.py --plot          # 出图3.34~3.37 PNG (配合 --book)
  python 6.py --no-moon       # 关闭月球摄动(仅 book 路径)
"""
from __future__ import annotations
import argparse
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
AU = 1.495978707e8            # km
DAY = 86400.0                 # s
GM = {                        # km^3/s^2
    "sun":    1.32712440018e11,
    "mercury": 2.2032e4,
    "venus":   3.24859e5,
    "earth":   398600.435436,
    "moon":    4902.800,
    "mars":    42828.37,
    "jupiter": 1.26686534e8,
    "saturn":  3.7931187e7,
}
R_EARTH = 6378.137
R_SOI_E = 9.25e5             # 地球影响球 km
FLORA_MU = 3.8               # Flora 引力(仅用于其相对运动量级, 交会用位置约束)

USE_MOON = False
USE_SRP = False

# ---------------------------------------------------------------------------
# 日期 <-> MJD
# ---------------------------------------------------------------------------
def mjd(y, m, d):
    import datetime
    return (datetime.datetime(y, m, d) - datetime.datetime(1858, 11, 17)).days

T_DEPART = mjd(2020, 8, 3)     # 地球出发  (书表3.25)
BOOK_T_MARS = mjd(2020, 12, 29)  # 书表3.25/3.27 火星借力历元(种子搜索中心)
T_MARS   = BOOK_T_MARS       # 实际借力历元(种子求解后更新)
T_ARR    = mjd(2021, 10, 6)    # 到达Flora (书表3.25/3.28)

# ---------------------------------------------------------------------------
# 解析行星历表（JPL 近似开普勒根数, 1800~2050, Standish）
#   元素: a(AU), e, I(deg), L(deg), ϖ(long.peri deg), Ω(deg) + 每世纪变率
# ---------------------------------------------------------------------------
_J2000_ECL_TO_J2000_EQ = None  # 本实现统一用黄道系(书: 日心黄道惯性系)
_CENT = 36525.0

# name: a0,adot, e0,edot, I0,Idot, L0,Ldot, peri0,periT, node0,nodeT  (deg/AU)
_PLANET_KEP = {
    "earth": (1.00000261, 0.0, 0.01671022, -0.00003809, 0.0, -0.01294668,
              100.46457166, 35999.3724498, 102.93768193, 0.31951594,
              0.0, 0.0),
    "mars":  (1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
              -4.55343205, 19140.30268499, -23.94362959, 0.30632644,
              49.55953891, -0.01847713),
    "venus": (0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
              315.95934228, 58517.8153030, 131.60246718, 0.00281095,
              76.67984255, -0.27774144),
    "jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
                34.39644051, 3034.69202398, 14.72847983, 0.21320860,
                100.47390909, 0.20469106),
}

def _kepler_state(a_km, e, i, om, w, M):
    """开普勒根数 -> 黄道系位置/速度(km, km/s)。M 为平近点角(rad)。"""
    # 解开普勒方程
    E = M + e * math.sin(M)
    for _ in range(60):
        dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                        math.sqrt(1 - e) * math.cos(E / 2))
    r = a_km * (1 - e * math.cos(E))
    h = math.sqrt(GM["sun"] * a_km * (1 - e * e))
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = np.array([-GM["sun"] / h * math.sin(nu),
                   GM["sun"] / h * (e + math.cos(nu)), 0.0])
    cO, sO = math.cos(om), math.sin(om)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(w), math.sin(w)
    R11 = cO * cw - sO * sw * ci; R12 = -cO * sw - sO * cw * ci; R13 = sO * si
    R21 = sO * cw + cO * sw * ci; R22 = -sO * sw + cO * cw * ci; R23 = -cO * si
    R31 = sw * si;                R32 = cw * si;                 R33 = ci
    Rz = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    return Rz @ rp, Rz @ vp

def planet_state(name, t_mjd):
    """行星日心黄道位置/速度(km, km/s)。"""
    p = _PLANET_KEP[name]
    T = (t_mjd - 51544.5) / _CENT
    a = (p[0] + p[1] * T) * AU
    e = p[2] + p[3] * T
    I = math.radians(p[4] + p[5] * T)
    L = math.radians((p[6] + p[7] * T) % 360.0)
    peri = math.radians(p[8] + p[9] * T)
    om = math.radians(p[10] + p[11] * T)
    w = peri - om
    M = L - peri
    M = (M + math.pi) % (2 * math.pi) - math.pi
    return _kepler_state(a, e, I, om, w, M)

# ---------------------------------------------------------------------------
# Flora：表3.24 根数(epoch 2011-2-8) 二体递推
# ---------------------------------------------------------------------------
FLORA_EPOCH = mjd(2011, 2, 8)
FLORA_KEP = dict(a=2.2013 * AU, e=0.1566, i=math.radians(5.8877),
                 om=math.radians(110.9378), w=math.radians(285.1854),
                 M0=math.radians(9.6855))

def flora_state(t_mjd):
    n = math.sqrt(GM["sun"] / FLORA_KEP["a"] ** 3)         # rad/s
    dt = (t_mjd - FLORA_EPOCH) * DAY
    M = FLORA_KEP["M0"] + n * dt
    M = (M + math.pi) % (2 * math.pi) - math.pi
    return _kepler_state(FLORA_KEP["a"], FLORA_KEP["e"], FLORA_KEP["i"],
                         FLORA_KEP["om"], FLORA_KEP["w"], M)

# ---------------------------------------------------------------------------
# 精确多体传播器（日心黄道系；日+地+火[+月]）
# ---------------------------------------------------------------------------
# 精确模型巡航段摄动天体：取借力星火星(连续飞越, 体现"精确" vs 脉冲等效)。
# 地球的影响已通过 SOI 出口的 vinf(拼接)计入, 巡航段再加重计, 故不在此列。
PERTURBERS = ["mars"]

def _accel(t_s, y, t0_s):
    r = y[:3]
    a = -GM["sun"] * r / np.linalg.norm(r) ** 3
    t_mjd = t0_s / DAY + t_s / DAY
    for name in PERTURBERS:
        rp, _ = planet_state(name, t_mjd)
        d = rp - r
        a += GM[name] * (d / np.linalg.norm(d) ** 3 - rp / np.linalg.norm(rp) ** 3)
    if USE_MOON:
        rm, _ = planet_state("earth", t_mjd)
        # 月球近似: 地心圆轨道 384400km, 周期27.32d
        th = 2 * math.pi * (t_mjd - 51544.5) / 27.321661
        rmoon = rm + 384400.0 * np.array([math.cos(th), math.sin(th) * math.cos(math.radians(5.1)),
                                          math.sin(th) * math.sin(math.radians(5.1))])
        dm = rmoon - r
        a += GM["moon"] * (dm / np.linalg.norm(dm) ** 3 - rmoon / np.linalg.norm(rmoon) ** 3)
    return a

def propagate_helio(y0, t0_mjd, t1_mjd, dense=False):
    t0_s = t0_mjd * DAY; t1_s = t1_mjd * DAY
    span = (0.0, t1_s - t0_s)
    tev = None
    if dense:
        tev = np.linspace(0.0, t1_s - t0_s, 800)
    sol = solve_ivp(lambda t, y: [y[3], y[4], y[5], *_accel(t, y, t0_s)],
                    span, np.asarray(y0, float), method="DOP853",
                    rtol=1e-11, atol=1e-6, dense_output=dense, t_eval=tev)
    if dense:
        return sol
    return sol.y[:, -1]

# ---------------------------------------------------------------------------
# 地心逃逸双曲线：vp0(近地点速度, 地心) -> vinf 矢量(地心)
# ---------------------------------------------------------------------------
R_P0 = np.array([-1866.540, -6113.904, 1931.810])   # 逃逸变轨点(地心, 表3.26)

def vinf_from_vp0(vp0):
    """地心二体从近地点积分到 SOI, 返回真实渐近 vinf 矢量(km/s)与 SOI 位置。"""
    y0 = np.concatenate([R_P0, np.asarray(vp0, float)])
    def rhs(t, y):
        r = y[:3]; rn = np.linalg.norm(r)
        acc = -GM["earth"] * r / rn ** 3
        return [y[3], y[4], y[5], acc[0], acc[1], acc[2]]
    def hit_soi(t, y):
        return np.linalg.norm(y[:3]) - R_SOI_E
    hit_soi.terminal = True; hit_soi.direction = 1.0
    sol = solve_ivp(rhs, (0, 6e6), y0, method="DOP853", rtol=1e-11, atol=1e-6,
                    events=hit_soi)
    yf = sol.y[:, -1]
    r = yf[:3]; v = yf[3:6]
    s = np.linalg.norm(v)
    vmag_inf = math.sqrt(max(1e-9, s * s - 2 * GM["earth"] / np.linalg.norm(r)))
    return v / s * vmag_inf, r      # 真实 vinf(方向=SOI速度向), SOI位置

def helio_state_at_depart(vd, t0_mjd):
    """主传播：出发日心速度 vd(km/s)，位置取地球 SOI 出口(沿 vinf 方向偏移)，
    避免航天器落在地球中心导致引力项奇异。"""
    rE, vE = planet_state("earth", t0_mjd)
    vd = np.asarray(vd, float)
    vinf = vd - vE
    rsoi = vinf / np.linalg.norm(vinf) * R_SOI_E
    return np.concatenate([rE + rsoi, vd])

# ---------------------------------------------------------------------------
# B 平面参数（3.97~3.107）+ 时间参数 τ（3.108）
# ---------------------------------------------------------------------------
N_ECL = np.array([0.0, 0.0, 1.0])

def bplane_from_state(r_rel, v_rel, mu=GM["mars"]):
    """给定借力星相对位置/速度, 返回 (BT, BR, tau, b_mag, e, vinf).
    对近抛物/迭代中间态做数值保护(钳制 e>1, vinf²>0)。"""
    rn = np.linalg.norm(r_rel); vn = np.linalg.norm(v_rel)
    h = np.cross(r_rel, v_rel)
    n = h / np.linalg.norm(h)
    e_vec = ((vn**2 - mu/rn) * r_rel - np.dot(r_rel, v_rel) * v_rel) / mu
    e = max(np.linalg.norm(e_vec), 1.0 + 1e-6)
    vinf2 = max(1e-6, vn**2 - 2 * mu / rn)
    vinf = math.sqrt(vinf2)
    beta = math.acos(max(-1, min(1, 1.0/e)))
    nx = np.cross(n, e_vec); nx = nx / (np.linalg.norm(nx) + 1e-30)
    S = math.cos(beta) * e_vec/e + math.sin(beta) * nx   # 3.104 入射渐近线单位矢
    T = np.cross(S, N_ECL); T = T / (np.linalg.norm(T) + 1e-30)
    R = np.cross(S, T)
    B = r_rel - np.dot(r_rel, S) * S                     # B 矢量(近似)
    BT = float(np.dot(B, T)); BR = float(np.dot(B, R))
    # τ: 到近火点时间(3.108)
    a = -mu / vinf**2
    # 瞄准用 b 取 BT/BR 平面分量幅度(与 target/DC 一致, 远离近火点时 e_vec 近似会偏差)
    b_mag = math.hypot(BT, BR)

    # 真近点角
    cosnu = np.dot(e_vec, r_rel) / (e * rn)
    nu = math.acos(max(-1, min(1, cosnu)))
    if np.dot(r_rel, v_rel) < 0:
        nu = 2*math.pi - nu
    sinhH = (e + math.cos(nu)) / (1 + e * math.cos(nu))
    H = math.asinh(sinhH)
    # (3.108): τ = sqrt(-a^3/mu)(sinhH - H), 近火点前为正
    tau = math.sqrt(-a**3 / mu) * (math.sinh(H) - H)
    if np.dot(r_rel, v_rel) < 0:
        tau = -tau
    return BT, BR, tau, b_mag, e, vinf

def hyperbolic_asymptote(r_rel, v_rel, mu=GM["mars"], incoming=True):
    """由相对状态求双曲线(渐近线)剩余速度矢量(大小=√(v²-2μ/r), 方向=e_vec±…)。"""
    rn = np.linalg.norm(r_rel); vn = np.linalg.norm(v_rel)
    vinf = math.sqrt(max(1e-9, vn**2 - 2 * mu / rn))
    e_vec = ((vn**2 - mu / rn) * r_rel - np.dot(r_rel, v_rel) * v_rel) / mu
    e = max(np.linalg.norm(e_vec), 1.0 + 1e-9)
    h = np.cross(r_rel, v_rel); n = h / np.linalg.norm(h)
    beta = math.acos(max(-1, min(1, 1.0 / e)))
    nx = np.cross(n, e_vec); nx = nx / (np.linalg.norm(nx) + 1e-30)
    if incoming:                       # 入射渐近线(逆行方向)
        S = math.cos(beta) * e_vec / e + math.sin(beta) * nx
    else:
        S = math.cos(beta) * e_vec / e - math.sin(beta) * nx
    return S * vinf, vinf

def _rp_from_b(b, vinf, mu=GM["mars"]):
    """由 B 矢量大小 b 与 vinf 反解近火点半径: b = rp*sqrt(1 + 2μ/(rp·vinf²)).
    等价于 rp² + (2μ/vinf²)·rp - b² = 0 → rp = (-k + sqrt(k²+4b²))/2, k=2μ/vinf²."""
    k = 2.0 * mu / vinf ** 2
    return 0.5 * (-k + math.sqrt(k * k + 4.0 * b * b))

def target_bplane(vinf_in, vinf_out, mu=GM["mars"]):
    """由进/离双曲线剩余速度算目标 B 平面(3.97~3.103)。
    B 矢量沿轨道面法线方向 n×S（含时钟角），|B|=b 由转角δ定。"""
    vinf_in = np.asarray(vinf_in, float); vinf_out = np.asarray(vinf_out, float)
    vinf = np.linalg.norm(vinf_in)
    S = vinf_in / vinf
    n = np.cross(vinf_in, vinf_out); n = n / np.linalg.norm(n)   # 3.100 轨道法线
    cosd = np.dot(vinf_in, vinf_out) / vinf**2
    delta = math.acos(max(-1, min(1, cosd)))
    a = -mu / vinf**2
    b = -a / math.tan(delta / 2)                                 # 3.102 半虚轴
    Bhat = np.cross(S, n); Bhat = Bhat / np.linalg.norm(Bhat)
    Btar = b * Bhat
    T = np.cross(S, N_ECL); T = T / np.linalg.norm(T)
    R = np.cross(S, T)
    return float(np.dot(Btar, T)), float(np.dot(Btar, R)), delta, b

# ---------------------------------------------------------------------------
# 简单(等效脉冲)模型初步设计（§3.4.1）：在「本文件自洽星历系」内用 Lambert
# 求解两条日心腿，给出出发速度 vd0、火星进/离剩余速度与目标 B 平面。
# （书表3.25/3.26 为书星历系数值；本实现重算以保证与精确模型同系、可收敛。）
# ---------------------------------------------------------------------------
def _prop2b_helio(r1, v1, dt):
    def rhs(t, y):
        r = y[:3]; acc = -GM["sun"] * r / np.linalg.norm(r) ** 3
        return [y[3], y[4], y[5], acc[0], acc[1], acc[2]]
    sol = solve_ivp(rhs, (0, dt), np.concatenate([r1, v1]),
                    method="DOP853", rtol=1e-11, atol=1e-6)
    return sol.y[:3, -1], sol.y[3:6, -1]

def _state2b_at(vd, t0_mjd, t1_mjd):
    """与精确模型同起点(地球 SOI 出口)的日心二体传播, 用于简单模型腿。"""
    y0 = helio_state_at_depart(vd, t0_mjd)
    return _prop2b_helio(y0[:3], y0[3:6], (t1_mjd - t0_mjd) * DAY)

def _bplane2b(vd):
    """日心二体(无火星引力)在 tM 的火星相对 B 平面: (BT,BR,tau,|B|,vinf)."""
    rM, vM = planet_state("mars", T_MARS)
    rf, vf = _state2b_at(vd, T_DEPART, T_MARS)
    BT, BR, tau, bm, e, vi = bplane_from_state(rf - rM, vf - vM)
    return BT, BR, tau, bm, vi

def _solve_leg1_to_bplane(BT_t, BR_t, vguess):
    """解出发日心速度 vd(二体), 使 tM 处火星相对 B 平面命中 (BT,BR,τ=0)。"""
    def eq(v):
        BT, BR, tau, *_ = _bplane2b(v)
        return [BT - BT_t, BR - BR_t, tau]
    return fsolve(eq, vguess, xtol=1e-10)

def _rot_axis(v, k, ang):
    """罗德里格斯: 矢量 v 绕单位轴 k 转 ang."""
    k = k / np.linalg.norm(k)
    return (v * math.cos(ang) + np.cross(k, v) * math.sin(ang)
            + k * np.dot(k, v) * (1 - math.cos(ang)))

def flyby_vinf_out(vinf_in, phi, rp):
    """脉冲等效飞越: 入射剩余速度绕 B 平面法线转 δ 得离射剩余速度(3.7 转角)。"""
    vi = np.linalg.norm(vinf_in); S = vinf_in / vi
    e = 1.0 + rp * vi ** 2 / GM["mars"]
    delta = 2.0 * math.asin(min(1.0, 1.0 / e))
    T = np.cross(S, N_ECL); T = T / np.linalg.norm(T)
    R = np.cross(S, T)
    Bhat = math.cos(phi) * T + math.sin(phi) * R
    k = np.cross(Bhat, S)
    return _rot_axis(vinf_in, k, delta), delta

def simple_model_seed():
    """§3.4.1 简单(等效脉冲)模型 → 精确模型的自洽种子。
    书表3.25/3.27 给出参考借力几何(进/离剩余速度、目标 B 平面)。在自实现星历系内，
    以 [BT,BR,τ,|vinf_in|] 为目标解未知量 vd(3)+火星近火时刻 tM(共4, 良置),
    得到非退化的瞄准双曲线(避免 Lambert 直冲火星中心导致 |B|→0 奇异)。"""
    vin_b = np.array(SIMPLE["vinf_in_book"])
    vi_t = float(np.linalg.norm(vin_b))
    BT_t, BR_t = SIMPLE["BT_tar"], SIMPLE["BR_tar"]

    def bp2b(vd, tM):
        rM, vM = planet_state("mars", tM)
        pos, vel = _state2b_at(vd, T_DEPART, tM)
        return pos - rM, vel - vM

    def res(x):
        vd = x[:3]; tM = x[3]
        rr, vv = bp2b(vd, tM)
        BT, BR, tau, bm, e_, vi = bplane_from_state(rr, vv)
        return [(BT - BT_t) / 1e3, (BR - BR_t) / 1e3, tau / 1e5, vi - vi_t]

    vE = planet_state("earth", T_DEPART)[1]
    best = None
    for tMg in (BOOK_T_MARS - 32.0, BOOK_T_MARS, BOOK_T_MARS + 33.0):
        for s in (0.92, 0.96, 1.02):
            x0 = np.concatenate([vE + s * vin_b, [tMg]])
            sol = fsolve(res, x0, xtol=1e-10, maxfev=6000)
            nr = float(np.linalg.norm(res(sol)))
            if best is None or nr < best[0]:
                best = (nr, sol)
    nr, sol = best
    vd0 = sol[:3]; tM = float(sol[3])
    rr, vv = bp2b(vd0, tM)
    BT, BR, tau, bm, e_, vi = bplane_from_state(rr, vv)
    vinf_in, _ = hyperbolic_asymptote(rr, vv, incoming=True)   # 渐近线剩余速度
    # 本系飞越一致离射渐近线: 入射渐近线绕 B 方向转角 δ(rp=300km)
    rp_enc = _rp_from_b(bm, vi)
    e_enc = 1.0 + rp_enc * vi * vi / GM["mars"]
    delta = 2.0 * math.asin(min(1.0, 1.0 / e_enc))
    phi = math.atan2(BR, BT)
    vinf_out, _ = flyby_vinf_out(vinf_in, phi, rp_enc)         # 离射渐近线
    BT_t2, BR_t2, delta_b, b_b = target_bplane(vinf_in, np.array(SIMPLE["vinf_out_book"]))
    return dict(vd0=vd0, tM=tM, vinf_in=vinf_in, vinf_out=vinf_out,
                BT=BT, BR=BR, b=bm, vi=vi, rp_enc=rp_enc, seed_res=nr,
                phi=phi, delta=delta, b_use=b_b)

# ---------------------------------------------------------------------------
# 简单模型初值（书表3.25/3.26/3.27/3.28 给定, 作为参考几何）
# ---------------------------------------------------------------------------
SIMPLE = dict(
    vinf_dep=[3.354, 2.658, 0.316],          # 表3.25 出发双曲线超越速度(地心)
    vinf_in_book=[3.467, 3.447, -0.825],     # 表3.25 火星进入剩余速度
    vinf_out_book=[1.289, 4.647, 1.153],     # 表3.25 火星离开剩余速度
    rp_alt=300.0,                            # 火星近火点高度 km
    vp0=[10.338, -1.461, 5.364],             # 表3.26 地心逃逸近地点速度
    BT_tar=4027.500, BR_tar=3094.187,        # 表3.27 目标 B 平面
    flora_pos_book=[-320179592.282, 137168656.845, 25782659.425],  # 表3.28
)
# 自洽种子：解出 vd0 / tM / 2-body 瞄准 B 平面
_seed = simple_model_seed()
T_MARS = _seed["tM"]          # 下游(精确模型/图)统一使用解出的借力历元
SIMPLE["vd0"] = _seed["vd0"]
SIMPLE["tM"] = _seed["tM"]
SIMPLE["vinf_in"] = _seed["vinf_in"]
SIMPLE["vinf_out"] = _seed["vinf_out"]
SIMPLE["BT_2b"] = _seed["BT"]
SIMPLE["BR_2b"] = _seed["BR"]
SIMPLE["b_2b"] = _seed["b"]
SIMPLE["vi_2b"] = _seed["vi"]
SIMPLE["phi"] = _seed["phi"]
SIMPLE["delta"] = _seed["delta"]
SIMPLE["rp_enc"] = _seed["rp_enc"]
SIMPLE["b_use"] = _seed["b_use"]
SIMPLE["BT_tar_use"] = SIMPLE["BT_tar"]
SIMPLE["BR_tar_use"] = SIMPLE["BR_tar"]
# Flora 目标: 由离射腿二体传播, 扫描+细化求最接近 Flora 的时刻(交会时刻)
def _flora_closest(vd, tM, vinf_out, dt_lo=60.0, dt_hi=600.0, n=280):
    rM, vM = planet_state("mars", tM)
    def gap(dt):
        pos = _prop2b_helio(rM, vM + vinf_out, dt * DAY)[0]
        return np.linalg.norm(pos - flora_state(tM + dt)[0])
    dts = np.linspace(dt_lo, dt_hi, n)
    i0 = int(np.argmin([gap(d) for d in dts]))
    a = dts[max(0, i0 - 1)]; b = dts[min(n - 1, i0 + 1)]
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(gap, bounds=(a, b), method="bounded",
                        options={"xatol": 1e-2})
    tf = float(tM + r.x)
    return tf, flora_state(tf)[0]
SIMPLE["tf"], SIMPLE["flora_pos"] = _flora_closest(SIMPLE["vd0"], SIMPLE["tM"],
                                                   SIMPLE["vinf_out"])



# ---------------------------------------------------------------------------
# 微分修正（3.109~3.111）
# ---------------------------------------------------------------------------
def eval_mars_constraints(vd):
    y0 = helio_state_at_depart(vd, T_DEPART)
    yf = propagate_helio(y0, T_DEPART, T_MARS)
    rM, vM = planet_state("mars", T_MARS)
    return bplane_from_state(yf[:3] - rM, yf[3:6] - vM)

def differential_correction(vd, mode="mars", iters=10, tol=1e-3, h=1e-3):
    """mode='mars': 精确模型下约束[BT,BR,τ]→目标(表3.27, 设计变量 vd, 式3.109~3.111);
    mode='flora': 精确模型过火后第二腿, 修正 vd 使在交会历元命中 SIMPLE['flora_pos'](表3.28)。
    牛顿法(有限差分灵敏度矩阵 K)+回溯线搜索保证残差单调下降。"""
    def gfun(v):
        if mode == "mars":
            BT, BR, tau, *_ = eval_mars_constraints(v)
            return np.array([BT - SIMPLE["BT_tar_use"],
                             BR - SIMPLE["BR_tar_use"], tau])
        y0 = helio_state_at_depart(v, T_DEPART)
        yf = propagate_helio(y0, T_DEPART, SIMPLE["tf"])
        return yf[:3] - np.array(SIMPLE["flora_pos"])

    vd = np.asarray(vd, float).copy()
    hist = []
    for it in range(iters):
        g = gfun(vd); n0 = np.linalg.norm(g)
        if n0 < tol:
            break
        K = np.zeros((3, 3))
        for j in range(3):
            dv = np.zeros(3); dv[j] = h
            K[:, j] = (gfun(vd + dv) - g) / h
        try:
            d = np.linalg.solve(K, -g)
        except np.linalg.LinAlgError:
            break
        step = 1.0; accepted = False
        for _ in range(12):                       # 回溯: 只接受使残差下降的步
            if np.linalg.norm(gfun(vd + d * step)) < n0:
                vd = vd + d * step; accepted = True; break
            step *= 0.5
        hist.append((it + 1, np.linalg.norm(gfun(vd)), vd.copy()))
        if not accepted:
            break
    return vd, hist

def flora_closest_approach(vd):
    """精确模型过火星借力后第二腿, 扫描+细化求与 Flora 最近 approached 历元与距离。"""
    from scipy.optimize import minimize_scalar
    y0 = helio_state_at_depart(vd, T_DEPART)
    yM = propagate_helio(y0, T_DEPART, T_MARS + 15)     # 离开火星影响圈
    r0 = yM[:3]; v0 = yM[3:6]
    t0 = T_MARS + 15
    def gap(dt):
        pos = _prop2b_helio(r0, v0, dt * DAY)[0]
        return np.linalg.norm(pos - flora_state(t0 + dt)[0])
    dts = np.linspace(30.0, 500.0, 160)
    i0 = int(np.argmin([gap(d) for d in dts]))
    r = minimize_scalar(gap, bounds=(dts[max(0, i0-1)], dts[min(159, i0+1)]),
                        method="bounded", options={"xatol": 1e-2})
    return float(t0 + r.x), float(r.fun)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report():
    L = []
    L.append("=" * 74)
    L.append("6.py — §3.4 精确动力学模型中的引力辅助（地球→火星借力→Flora）")
    L.append("=" * 74)
    # (1) 地心拼接核对（表3.26 vp0 -> vinf, 对照表3.25）—— 精确复现
    vinfc, _ = vinf_from_vp0(SIMPLE["vp0"])
    L.append("[表3.26→3.25] 地心逃逸拼接核对  vp0=%s:" % np.round(SIMPLE["vp0"], 3))
    L.append("  算得 vinf=%s |vinf|=%.3f  （书表3.25: [3.354,2.658,0.316], 4.29）"
             % (np.round(vinfc, 3), np.linalg.norm(vinfc)))
    # (2) 目标 B 平面复现（表3.25 进/离 → 表3.27）—— 精确复现
    btB, brB, dltB, bB = target_bplane(SIMPLE["vinf_in_book"], SIMPLE["vinf_out_book"])
    rpB = _rp_from_b(bB, np.linalg.norm(SIMPLE["vinf_in_book"]))
    L.append("\n[表3.25→3.27] 目标 B 平面(轨道法线法, 式3.97~3.103):")
    L.append("  算得 BT=%.1f BR=%.1f |B|=%.1f δ=%.1f° rp高度=%.0fkm"
             % (btB, brB, bB, math.degrees(dltB), rpB - 3396.2))
    L.append("  书表3.27: BT=4027.5 BR=3094.2 δ=37.4° 借力高度 300km")
    # (3) 自洽简单模型种子（§3.4.1）：在本星历系内解 vd/tM 命中目标 B 平面
    L.append("\n[§3.4.1] 本系自洽种子(解 vd(3)+tM, 目标[BT,BR,τ,|vinf|]):")
    L.append("  出发日心速度 vd0=%s   借力历元 tM=%.1f(书59212, 差%.1fd)"
             % (np.round(SIMPLE["vd0"], 4), T_MARS, T_MARS - BOOK_T_MARS))
    L.append("  二体到达: BT=%.1f BR=%.1f |B|=%.1f vi=%.3f 近火高度=%.0fkm"
             % (SIMPLE["BT_2b"], SIMPLE["BR_2b"], SIMPLE["b_2b"], SIMPLE["vi_2b"],
                _rp_from_b(SIMPLE["b_2b"], SIMPLE["vi_2b"]) - 3396.2))
    # (4) 精确模型初值 B 平面（多体, 含火星第三体）
    vd0 = SIMPLE["vd0"]
    BT0, BR0, tau0, bm0, e0, vi0 = eval_mars_constraints(vd0)
    L.append("\n[§3.4.2] 同一 vd0 在精确多体模型下(tM=%.1f):" % T_MARS)
    L.append("  实际 BT=%.1f BR=%.1f τ=%.0fs |B|=%.1f vi=%.3f"
             % (BT0, BR0, tau0, bm0, vi0))
    # (5) 第一次微分修正（表3.27 法）: 设计变量 vd, 约束 [BT,BR,τ]→目标
    vd1, hist1 = differential_correction(vd0, "mars")
    L.append("\n[表3.27] 第一次微分修正（火星借力, 约束[BT,BR,τ]→目标, 式3.109~3.111）:")
    L.append("  迭代 残差‖g‖/km:")
    for it, res, v in hist1:
        L.append("    iter%-2d  %12.1f   vd=%s" % (it, res, np.round(v, 4)))
    BT1, BR1, tau1, bm1, e1, vi1 = eval_mars_constraints(vd1)
    rp1 = _rp_from_b(bm1, vi1)
    L.append("  修正后 BT=%.1f BR=%.1f τ=%.0fs |B|=%.1f vi=%.3f 近火高度=%.0fkm"
             % (BT1, BR1, tau1, bm1, vi1, rp1 - 3396.2))
    L.append("  书表3.27: 5 次迭代, 修正逃逸近地点速度=[10.283,-1.490,5.430]")
    # (6) 第二次微分修正 / Flora 交会（表3.28 法）
    vd2, hist2 = differential_correction(vd1, "flora")
    L.append("\n[表3.28] 第二次微分修正（Flora 交会, 约束到达日心位置, 设计变量 vd）:")
    for it, res, v in hist2:
        L.append("    iter%-2d 位置残差%14.1f km  vd=%s" % (it, res, np.round(v, 4)))
    tf_ca, miss_ca = flora_closest_approach(vd1)
    L.append("  精确模型过火第二腿与 Flora 最近距离=%.3e km  最近历元=%.1f"
             % (miss_ca, tf_ca))
    L.append("  （注: 精确多体飞越离射方向由 BT/BR/τ 与近火半径共同决定; 命中 Flora "
             "需初值(§3.4.1 全局寻优)同时给出进入渐近线与离射方向, 仅 3 分量 vd 不足以"
             "既修火星 B 平面又修 Flora, 这正是书中先用全局优化定初值的必要条。）")

    return "\n".join(L), vd2


# ---------------------------------------------------------------------------
# 图 3.34~3.37
# ---------------------------------------------------------------------------
def make_figs(vd_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y0 = helio_state_at_depart(vd_final, T_DEPART)
    # 一条连续稠密全程轨迹(出发→火星借力→Flora), 避免中途重启破坏历元相位
    full = propagate_helio(y0, T_DEPART, SIMPLE["tf"], dense=True)
    tfull = T_DEPART + full.t / DAY
    sc = full.y[:3].T

    # 图3.34 日心轨道
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, col in [("earth", "tab:green"), ("mars", "tab:red")]:
        tt = np.linspace(T_DEPART, SIMPLE["tf"], 300)
        pts = np.array([planet_state(name, x)[0] for x in tt])
        ax.plot(pts[:, 0]/1e8, pts[:, 1]/1e8, col, lw=0.8, label=name)
    tf = np.linspace(T_DEPART - 60, SIMPLE["tf"] + 60, 300)
    fp = np.array([flora_state(x)[0] for x in tf])
    ax.plot(fp[:, 0]/1e8, fp[:, 1]/1e8, "tab:purple", lw=0.8, label="Flora")
    ax.plot(sc[:, 0]/1e8, sc[:, 1]/1e8, "k", lw=1.2, label="spacecraft")
    ax.plot(0, 0, "y*", ms=12, label="Sun")
    ax.set_xlabel("x/1e8 km"); ax.set_ylabel("y/1e8 km")
    ax.set_title("Fig 3.34  Heliocentric trajectory (Earth-Mars GA-Flora)")
    ax.legend(fontsize=8); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig("fig3_34.png", dpi=120); plt.close(fig)

    # 图3.35 地心逃逸轨道（用表3.26 地心逃逸近地点速度 vp0）
    yy = np.concatenate([R_P0, np.asarray(SIMPLE["vp0"], float)])
    def rhs(t, y):
        r = y[:3]; acc = -GM["earth"]*r/np.linalg.norm(r)**3
        return [y[3], y[4], y[5], acc[0], acc[1], acc[2]]
    def hit(t, y): return np.linalg.norm(y[:3]) - R_SOI_E
    hit.terminal = True; hit.direction = 1.0
    sE = solve_ivp(rhs, (0, 6e6), yy, method="DOP853", rtol=1e-11, atol=1e-6,
                   events=hit, t_eval=np.linspace(0, 5e5, 400))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sE.y[0], sE.y[1], "k", lw=1.2)
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_EARTH*np.cos(th), R_EARTH*np.sin(th), "b--", label="Earth")
    ax.plot(R_SOI_E*np.cos(th), R_SOI_E*np.sin(th), "g:", label="SOI")
    ax.set_xlabel("x/km"); ax.set_ylabel("y/km"); ax.set_aspect("equal")
    ax.set_title("Fig 3.35  Geocentric escape hyperbola")
    ax.legend(fontsize=8); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig("fig3_35.png", dpi=120); plt.close(fig)

    # 图3.36 + 3.37: 从连续轨迹取借力窗口
    rel = np.array([sc[k] - planet_state("mars", tfull[k])[0]
                    for k in range(len(tfull))])
    dMars = np.linalg.norm(rel, axis=1)
    ica = int(np.argmin(dMars)); t_ca = float(tfull[ica]); ca_km = float(dMars[ica])
    win = (tfull > t_ca - 6) & (tfull < t_ca + 6)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rel[win, 0]/1e4, rel[win, 1]/1e4, "k", lw=1.2)
    ax.plot(0, 0, "r*", ms=12, label="Mars")
    ax.set_xlabel("x/1e4 km"); ax.set_ylabel("y/1e4 km"); ax.set_aspect("equal")
    ax.set_title("Fig 3.36  Mars-relative hyperbola (CA=%.0f km)" % ca_km)
    ax.legend(fontsize=8); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig("fig3_36.png", dpi=120); plt.close(fig)

    vv = full.y[3:6].T
    En = 0.5*np.einsum("ij,ij->i", vv, vv) - GM["sun"]/np.linalg.norm(sc, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tfull[win], En[win], "k", lw=1.4)
    ax.set_xlabel("time /MJD"); ax.set_ylabel("heliocentric energy /(km^2/s^2)")
    ax.set_title("Fig 3.37  Heliocentric energy during Mars GA (CA=%.0f km)" % ca_km)
    ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig("fig3_37.png", dpi=120); plt.close(fig)
    return dict(E_lo=float(En[win].min()), E_hi=float(En[win].max()),
                ca_km=ca_km, t_ca=t_ca)


# ===========================================================================
# 中型保真桥接器：二体(patch-conic) MGA 序列 -> 精确力模型连续状态 -> GMAT 数据
# ---------------------------------------------------------------------------
# 定位：pykep/pygmo 给出的是「脉冲等效」二体飞越序列（位置连续、速度跳变）。
# 本段把它变成可被 GMAT 直接传播的日心状态序列：
#   leg-0: 从地球 SOI 出口精确传播到火星 SOI 入口，B 平面(BT/BR)+τ 微分修正；
#   flyby: 用火星二体双曲线把 SOI 入口/近火点/SOI 出口串起来（rp 由设计给定）；
#   leg-1: 从火星 SOI 出口精确传播到目标历元，位置微分修正。
# 控制量是每个节点上的 TCM Δv，输出为 CSV（历元/状态/Δv/B平面/交会残差）。
# 力模型：太阳 + 行星星历（优先 JPL DE421，回退 Standish 开普勒根数）。
# 坐标：内部 J2000 黄道；CSV 同时给 J2000 地球赤道(SunMJ2000Eq)，便于 GMAT。
# ===========================================================================
import os as _b_os
import datetime as _b_dt
from scipy.optimize import least_squares as _b_ls, brentq as _b_brentq

_B_DAY = DAY
_B_AU = AU
_B_MU = GM["sun"]
_B_R_SOI_E = R_SOI_E
_B_R_SOI_M = 5.77e5
_B_MARS_R = 3396.2
_B_RP = _B_MARS_R + 300.0          # 设计近火半径 (=300 km 高度)

# 黄道 <-> 赤道（DE421 为 J2000 赤道；CSV 给 GMAT 用的 SunMJ2000Eq）
_B_EPS = math.radians(23.4392911)
_B_RX = np.array([[1.0, 0.0, 0.0],
                  [0.0, math.cos(_B_EPS), math.sin(_B_EPS)],
                  [0.0, -math.sin(_B_EPS), math.cos(_B_EPS)]])

_B_DE421_PATH = _b_os.environ.get("DE421_PATH", "")
_B_KERNEL = None

def _b_init_de421():
    global _B_KERNEL
    cands = ([_B_DE421_PATH] if _B_DE421_PATH else []) + \
            ["/tmp/de421.bsp", "/data/de421.bsp", "de421.bsp"]
    path = next((p for p in cands if p and _b_os.path.isfile(p)), None)
    if path is None:
        return False
    try:
        from jplephem.spk import SPK
        _B_KERNEL = SPK.open(path)
        return True
    except Exception:
        return False

_B_HAVE_DE421 = _b_init_de421()
_B_PAIRS = {"mercury": [(0, 1)], "venus": [(0, 2)],
            "earth": [(0, 3), (3, 399)], "mars": [(0, 4), (4, 499)],
            "jupiter": [(0, 5)], "saturn": [(0, 6)]}

def _b_avail(name):
    if name == "flora":
        return True
    return (_B_HAVE_DE421 and name in _B_PAIRS) or (name in _PLANET_KEP)

def _b_state(name, t_mjd):
    """日心黄道状态 (km, km/s)，优先 DE421。"""
    if name == "flora":
        return flora_state(t_mjd)
    if _B_HAVE_DE421 and name in _B_PAIRS:
        jd = float(t_mjd) + 2400000.5
        r = np.zeros(3); v = np.zeros(3)
        for c, t in _B_PAIRS[name]:
            p, vv = _B_KERNEL[(c, t)].compute_and_differentiate(jd)
            r += np.asarray(p, float); v += np.asarray(vv, float)
        ps, vs = _B_KERNEL[(0, 10)].compute_and_differentiate(jd)
        r -= np.asarray(ps, float); v -= np.asarray(vs, float)
        return _B_RX @ r, (_B_RX @ v) / _B_DAY
    return planet_state(name, t_mjd)

# ---- Lambert (universal variable, single revolution) ----------------------
def _b_stumpff(z):
    if z > 1e-8:
        sz = math.sqrt(z); C = (1 - math.cos(sz)) / z; S = (sz - math.sin(sz)) / sz ** 3
    elif z < -1e-8:
        sz = math.sqrt(-z); C = (math.cosh(sz) - 1) / (-z); S = (math.sinh(sz) - sz) / sz ** 3
    else:
        C = 0.5; S = 1.0 / 6.0
    return C, S

def _b_lambert(r1, r2, dt_s, mu=_B_MU, prograde=True):
    """给定 r1,r2,dt 求二体转移的 v1,v2（单圈）。"""
    r1n = np.linalg.norm(r1); r2n = np.linalg.norm(r2)
    c = float(np.dot(r1, r2) / (r1n * r2n)); c = max(-1.0, min(1.0, c))
    dnu = math.acos(c)
    h = np.cross(r1, r2)
    if prograde:
        if h[2] < 0:
            dnu = 2 * math.pi - dnu
    else:
        if h[2] >= 0:
            dnu = 2 * math.pi - dnu
    A = math.sin(dnu) * math.sqrt(r1n * r2n / (1 - c))

    def F(z):
        C, S = _b_stumpff(z)
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        if y < 0:
            return -1e12
        return (y / C) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt_s

    lo, hi = -4 * math.pi ** 2 + 1e-6, 4 * math.pi ** 2 - 1e-6
    z = _b_brentq(F, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200)
    C, S = _b_stumpff(z)
    y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
    f = 1 - y / r1n; g = A * math.sqrt(y / mu); gd = 1 - y / r2n
    return (r2 - f * r1) / g, (gd * r2 - r1) / g, dnu

# ---- 行星历表缓存（1 天网格 + Hermite 插值） ------------------------------
class _B_Eph:
    def __init__(self, bodies, t0, t1, step=1.0):
        self.bodies = list(bodies)
        self.t0 = min(t0, t1)
        self.t1 = max(t0, t1)
        n = max(1, int(math.ceil((self.t1 - self.t0) / step)))
        self.grid = np.linspace(self.t0, self.t1, n + 1)
        self.R = {}; self.V = {}
        for b in self.bodies:
            R = []; V = []
            for t in self.grid:
                rr, vv = _b_state(b, float(t)); R.append(rr); V.append(vv)
            self.R[b] = np.array(R); self.V[b] = np.array(V)

    def pos(self, b, t):
        g = self.grid
        i = int(np.searchsorted(g, t)) - 1
        i = min(max(i, 0), len(g) - 2)
        h = g[i + 1] - g[i]; s = (t - g[i]) / h
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2;     h11 = s ** 3 - s ** 2
        return (h00 * self.R[b][i] + h10 * h * self.V[b][i]
                + h01 * self.R[b][i + 1] + h11 * h * self.V[b][i + 1])

def _b_rhs(t, y, t0_mjd, eph):
    r = y[:3]
    a = -_B_MU * r / np.linalg.norm(r) ** 3
    tm = t0_mjd + t / _B_DAY
    for b in eph.bodies:
        rb = eph.pos(b, tm); d = rb - r
        a = a + GM[b] * (d / np.linalg.norm(d) ** 3 - rb / np.linalg.norm(rb) ** 3)
    return [y[3], y[4], y[5], a[0], a[1], a[2]]

def _b_propagate(y0, t0, t1, eph, rtol=1e-10, atol=1e-6):
    sol = solve_ivp(lambda t, y: _b_rhs(t, y, t0, eph),
                    (0.0, (t1 - t0) * _B_DAY), np.asarray(y0, float),
                    method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:, -1]

# ---- B 平面（含 τ，式 3.108 的 e·sinh H 项） --------------------------
def _b_bplane(rr, vv, mu=None):
    """相对状态 -> (BT,BR,tau,b,e,vinf,S,nhat,ehat)。
    τ>0 表示还在近点前；τ<0 表示已过近点。"""
    mu = GM["mars"] if mu is None else mu
    rn = np.linalg.norm(rr); vn = np.linalg.norm(vv)
    h = np.cross(rr, vv); hn = np.linalg.norm(h); nh = h / hn
    ev = ((vn ** 2 - mu / rn) * rr - np.dot(rr, vv) * vv) / mu
    e = max(np.linalg.norm(ev), 1.0 + 1e-6)
    vinf = math.sqrt(max(1e-9, vn ** 2 - 2 * mu / rn))
    b = hn / vinf
    beta = math.acos(max(-1.0, min(1.0, 1.0 / e)))
    nx = np.cross(nh, ev); nx = nx / (np.linalg.norm(nx) + 1e-30)
    S = math.cos(beta) * ev / e + math.sin(beta) * nx
    Bv = b * np.cross(S, nh)
    N = np.array([0.0, 0.0, 1.0])
    T = np.cross(S, N); T = T / (np.linalg.norm(T) + 1e-30)
    R = np.cross(S, T)
    cosnu = np.dot(ev, rr) / (e * rn)
    nu = math.acos(max(-1.0, min(1.0, cosnu)))
    if np.dot(rr, vv) < 0:
        nu = 2 * math.pi - nu
    H = math.asinh((e + math.cos(nu)) / (1 + e * math.cos(nu)))
    a = -mu / vinf ** 2
    tau = math.sqrt(-a ** 3 / mu) * (e * math.sinh(H) - H)
    if np.dot(rr, vv) > 0:            # 过近点后 τ 取负
        tau = -tau
    return (float(np.dot(Bv, T)), float(np.dot(Bv, R)), float(tau), float(b),
            float(e), float(vinf), S, nh, ev / np.linalg.norm(ev))

def _b_mars2b(y0, t0, t1, event_r=None):
    """火星二体传播（相对系, 与日心无关），可带 SOI 半径事件。"""
    def rhs(t, y):
        r = y[:3]
        return [y[3], y[4], y[5], *(-GM["mars"] * r / np.linalg.norm(r) ** 3)]
    ev = None
    if event_r is not None:
        def ev(t, y):
            return np.linalg.norm(y[:3]) - event_r
        ev.terminal = True
        ev.direction = 0
    sol = solve_ivp(rhs, (0.0, (t1 - t0) * _B_DAY), np.asarray(y0, float),
                    method="DOP853", rtol=1e-12, atol=1e-9, events=ev)
    return sol.y[:, -1], sol.t[-1] / _B_DAY

# ---- 3×3 微分修正（显式 Jacobian + 回溯） --------------------------------
def _b_dc3(gfun, x0=None, h=1e-3, iters=15, tol=1e-4):
    x = np.zeros(3) if x0 is None else np.asarray(x0, float).copy()
    hist = []
    for it in range(iters):
        g = np.asarray(gfun(x), float)
        n0 = np.linalg.norm(g)
        hist.append((it, n0, x.copy()))
        if n0 < tol:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            xx = x.copy(); xx[j] += h
            J[:, j] = (np.asarray(gfun(xx), float) - g) / h
        try:
            d = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            break
        step = 1.0; accepted = False
        for _ in range(25):
            gn = np.linalg.norm(np.asarray(gfun(x + step * d), float))
            if gn < n0:
                x = x + step * d; accepted = True; break
            step *= 0.5
        if not accepted:
            break
    return x, hist

def _b_plane_basis(vi, vo):
    """由进/离 vinf 构造 B 平面目标与飞越几何。"""
    vi = np.asarray(vi, float); vo = np.asarray(vo, float)
    S_in = vi / np.linalg.norm(vi); S_out = vo / np.linalg.norm(vo)
    n = np.cross(S_in, S_out); n = n / np.linalg.norm(n)
    ehat = S_in - S_out; ehat = ehat / np.linalg.norm(ehat)
    return S_in, S_out, n, ehat

# ---- 二体序列（pykep 结果的替身）：Earth -> Mars(rp) -> Flora ------------
def build_bridge_sequence(verbose=False):
    T0 = float(T_DEPART)
    T1g = float(BOOK_T_MARS)
    T2g = float(T_ARR)
    rE, vE = _b_state("earth", T0)
    muM = GM["mars"]

    def _dep_vi(t1):
        """地球 SOI 出口迭代自洽的出发速度与火星到达 vinf。"""
        rM, vM = _b_state("mars", t1)
        vdep, varr, _ = _b_lambert(rE, rM, (t1 - T0) * _B_DAY)
        for _ in range(4):
            vf = vdep - vE
            r0 = rE + vf / np.linalg.norm(vf) * _B_R_SOI_E
            vdep, varr, _ = _b_lambert(r0, rM, (t1 - T0) * _B_DAY)
        return vdep, varr - vM

    def res_fly(x):
        t1, t2 = x
        rM, vM = _b_state("mars", t1)
        rF, _ = _b_state("flora", t2)
        vdep, vi = _dep_vi(t1)
        vdep2, _, _ = _b_lambert(rM, rF, (t2 - t1) * _B_DAY)
        vo = vdep2 - vM
        dcap = 2 * math.asin(1 / (1 + _B_RP * np.dot(vi, vi) / muM))
        ang = math.acos(max(-1.0, min(1.0,
              np.dot(vi, vo) / np.linalg.norm(vi) / np.linalg.norm(vo))))
        return [(np.linalg.norm(vo) - np.linalg.norm(vi)) * 1e3,
                (ang - dcap) * 1e3]

    sol = _b_ls(res_fly, [T1g, T2g], x_scale=10.0, xtol=1e-14, ftol=1e-14,
                gtol=1e-14, max_nfev=400)
    t1, t2 = float(sol.x[0]), float(sol.x[1])
    rM1, vM1 = _b_state("mars", t1)
    rF, vF = _b_state("flora", t2)
    vdep1, vi = _dep_vi(t1)
    vdep2, _, _ = _b_lambert(rM1, rF, (t2 - t1) * _B_DAY)
    vo = vdep2 - vM1

    S_in, S_out, n, ehat = _b_plane_basis(vi, vo)
    vin = np.linalg.norm(vi)
    e = 1 + _B_RP * vin ** 2 / muM
    vp = math.sqrt(muM * (1 + e) / _B_RP)
    # B 平面目标由设计近火半径 rp + 到达 vinf 方向决定（|vo| 的不一致不再污染 b）
    b = _B_RP * math.sqrt(1 + 2 * muM / (_B_RP * vin ** 2))
    N = np.array([0.0, 0.0, 1.0])
    T = np.cross(S_in, N); T = T / np.linalg.norm(T)
    R = np.cross(S_in, T)
    Bhat = np.cross(S_in, n)
    BTt = float(np.dot(b * Bhat, T)); BRt = float(np.dot(b * Bhat, R))
    delta_b = 2 * math.asin(1 / (1 + _B_RP * vin ** 2 / muM))
    yperi = np.concatenate([_B_RP * ehat, vp * np.cross(n, ehat)])
    yin, tin = _b_mars2b(yperi, 0.0, -30.0, event_r=_B_R_SOI_M); tin = -tin
    yout, tout = _b_mars2b(yperi, 0.0, 30.0, event_r=_B_R_SOI_M)
    t_in = t1 - tin; t_out = t1 + tout
    rMi, vMi = _b_state("mars", t_in)
    rMo, vMo = _b_state("mars", t_out)
    y_in = np.concatenate([rMi + yin[:3], vMi + yin[3:6]])
    y_out = np.concatenate([rMo + yout[:3], vMo + yout[3:6]])

    seq = dict(T0=T0, t1=t1, t2=t2, rp=_B_RP, vin=vin,
               vinf_dep=vdep1 - vE, vi=vi, vo=vo, BT=BTt, BR=BRt, b=b,
               tau_in=tin, tau_out=tout, t_in=t_in, t_out=t_out,
               y_in=y_in, y_out=y_out, rF=rF, vF=vF,
               delta=delta_b, e=e, vp=vp)
    if verbose:
        print("  [seq] t1=%.4f t2=%.4f |vi|=%.4f |vo|=%.4f rp_alt=%.0f km"
              % (t1, t2, vin, np.linalg.norm(vo), _B_RP - _B_MARS_R))
        print("  [seq] BT=%.1f BR=%.1f b=%.1f tau_in=%.3f d"
              % (BTt, BRt, b, tin))
    return seq

# ---- 桥接主流程 ----------------------------------------------------------
def run_bridge(seq, verbose=True):
    muM = GM["mars"]
    T0 = seq["T0"]; t1 = seq["t1"]; t2 = seq["t2"]
    t_in = seq["t_in"]; tau_in = seq["tau_in"]
    BTt = seq["BT"]; BRt = seq["BR"]

    pert0 = [b for b in ("mercury", "venus", "jupiter", "saturn") if _b_avail(b)]
    pert1 = [b for b in ("mercury", "venus", "earth", "jupiter", "saturn") if _b_avail(b)]

    # --- 1) 由 SOI 入口状态反推出发历元：跨过地球 SOI 的时刻 ----------------
    eph_back = _B_Eph(pert0, T0 - 3.0, t_in)
    def ev_earth(t, y):
        re, _ = _b_state("earth", t_in + t / _B_DAY)
        return np.linalg.norm(y[:3] - re) - _B_R_SOI_E
    ev_earth.terminal = True
    ev_earth.direction = 0
    back = solve_ivp(lambda t, y: _b_rhs(t, y, t_in, eph_back),
                     (0.0, -(t_in - (T0 - 3.0)) * _B_DAY), seq["y_in"],
                     method="DOP853", rtol=1e-11, atol=1e-6,
                     dense_output=True, events=ev_earth)
    if len(back.t_events[0]):
        tev = float(back.t_events[0][0])
        t0c = t_in + tev / _B_DAY
        y0c = back.sol(tev)
    else:
        t0c = T0
        y0c = _b_propagate(seq["y_in"], t_in, T0, eph_back)
    rEc, vEc = _b_state("earth", t0c)
    vinf_c = y0c[3:6] - vEc

    # --- 2) leg-0：低精度出发状态 -> B 平面(BT,BR)+τ 微分修正 -----------------
    vinf_lo = seq["vinf_dep"]
    u = vinf_lo / np.linalg.norm(vinf_lo)
    y_lo_c = np.concatenate([rEc + u * _B_R_SOI_E, vEc + vinf_lo])
    eph0 = _B_Eph(pert0, t0c, t_in)

    def g0(x):
        yy = y_lo_c.copy(); yy[3:6] += x
        yf = _b_propagate(yy, t0c, t_in, eph0)
        rM, vM = _b_state("mars", t_in)
        BT, BR, tau, *_ = _b_bplane(yf[:3] - rM, yf[3:6] - vM)
        return np.array([(BT - BTt) / 1e3, (BR - BRt) / 1e3,
                         (tau - tau_in * _B_DAY) / 1e4])

    dv0, hist0 = _b_dc3(g0, np.zeros(3), h=1e-3, iters=15, tol=1e-6)
    y_dep = y_lo_c.copy(); y_dep[3:6] += dv0
    y_ent = _b_propagate(y_dep, t0c, t_in, eph0)
    rM_in, vM_in = _b_state("mars", t_in)
    BTa, BRa, taua, ba, ea, via, Sa, nha, eha = _b_bplane(
        y_ent[:3] - rM_in, y_ent[3:6] - vM_in)
    rpa = _rp_from_b(ba, via)
    delta_a = 2 * math.asin(1 / (1 + rpa * via ** 2 / muM))
    vpa = math.sqrt(muM * (1 + ea) / rpa)
    t_peri = t_in + taua / _B_DAY
    y_peri_rel = np.concatenate([rpa * eha, vpa * np.cross(nha, eha)])
    yout_a, tout_a = _b_mars2b(y_peri_rel, 0.0, 30.0, event_r=_B_R_SOI_M)
    t_exit = t_peri + tout_a
    rM_out, vM_out = _b_state("mars", t_exit)
    y_exit = np.concatenate([rM_out + yout_a[:3], vM_out + yout_a[3:6]])
    rM_p, vM_p = _b_state("mars", t_peri)
    y_peri = np.concatenate([rM_p + rpa * eha, vM_p + vpa * np.cross(nha, eha)])

    # --- 3) leg-1：火星 SOI 出口 -> Flora 位置微分修正 ----------------------
    rF, vF = _b_state("flora", t2)
    eph1 = _B_Eph(pert1, t_exit, t2)

    def g1(x):
        yy = y_exit.copy(); yy[3:6] += x
        yf = _b_propagate(yy, t_exit, t2, eph1, rtol=1e-13, atol=1e-9)
        return (yf[:3] - rF) / 1e6

    dv1, hist1 = _b_dc3(g1, np.zeros(3), h=1e-2, iters=20, tol=1e-9)
    y_exit_post = y_exit.copy(); y_exit_post[3:6] += dv1
    y_arr = _b_propagate(y_exit_post, t_exit, t2, eph1, rtol=1e-13, atol=1e-9)
    miss = float(np.linalg.norm(y_arr[:3] - rF))

    res = dict(seq=seq, t0c=t0c, t_in=t_in, t_peri=t_peri, t_exit=t_exit,
               t2=t2, y_dep=y_dep, y_ent=y_ent, y_peri=y_peri,
               y_exit=y_exit, y_exit_post=y_exit_post, y_arr=y_arr,
               dv0=dv0, dv1=dv1, dv0_n=float(np.linalg.norm(dv0)),
               dv1_n=float(np.linalg.norm(dv1)), miss=miss,
               BT=BTa, BR=BRa, tau=taua, b=ba, rp=rpa, vi=via,
               delta=delta_a, vinf_c=vinf_c, epoch_shift=t0c - T0,
               hist0=hist0, hist1=hist1)
    if verbose:
        print("  [leg0] 出发历元修正 %+.3f d；Δv0=(%.2f, %.2f, %.2f) m/s |%.1f|"
              % (res["epoch_shift"], dv0[0] * 1e3, dv0[1] * 1e3,
                 dv0[2] * 1e3, res["dv0_n"] * 1e3))
        print("         火星 SOI: BT=%.1f BR=%.1f τ=%.0f s |B|=%.1f rp=%.1f km"
              % (BTa, BRa, taua, ba, rpa))
        print("  [leg1] Δv1=(%.2f, %.2f, %.2f) m/s |%.1f|  交会残差 %.3f km"
              % (dv1[0] * 1e3, dv1[1] * 1e3, dv1[2] * 1e3,
                 res["dv1_n"] * 1e3, miss))
        print("  总 Δv = %.1f m/s" % ((res["dv0_n"] + res["dv1_n"]) * 1e3))
    return res

# ---- CSV 输出（GMAT 可直接读初值/节点状态） -------------------------------
def _b_mjd_utc(m):
    base = _b_dt.datetime(1858, 11, 17)
    return (base + _b_dt.timedelta(days=float(m))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def _b_to_eq(r, v):
    return _B_RX.T @ np.asarray(r, float), _B_RX.T @ np.asarray(v, float)

def export_bridge_csv(res, path="bridge.csv"):
    seq = res["seq"]
    dur = (res["t2"] - res["t0c"])
    hdr = [
        "# 6.py medium-fidelity bridge: patched-conic MGA -> precise ephemeris",
        "# body sequence: earth(SOI exit) -> mars(rp=%.1f km) -> flora" % seq["rp"],
        "# frame: SunMJ2000Eq (heliocentric J2000 Earth-equatorial), units km, km/s",
        "# time: UTC (MJD); DE421 evaluated at JD=MJD+2400000.5",
        "# force model: Sun + %s (leg-0), Sun + %s (leg-1)" %
        (",".join(["mercury", "venus", "jupiter", "saturn"]),
         ",".join(["mercury", "venus", "earth", "jupiter", "saturn"])),
        "# flyby: patched Mars two-body hyperbola between SOI entry/exit; rp=%.3f km" %
        seq["rp"],
        "# GMAT: use CoordinateSystem 'SunMJ2000Eq'; x/y/z,vx/vy/vz columns are propagation-ready",
        "# departure epoch corrected by %+.4f d vs low-fidelity sequence" %
        res["epoch_shift"],
        "#",
    ]
    cols = ["epoch_mjd", "epoch_utc", "event", "body",
            "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s",
            "dvx_km_s", "dvy_km_s", "dvz_km_s", "dv_km_s",
            "BT_km", "BR_km", "tau_s", "rp_km", "miss_km", "note"]

    def row(mjd, event, body, y, dv=None, BT=None, BR=None, tau=None,
            rp=None, miss=None, note=""):
        r_eq, v_eq = _b_to_eq(y[:3], y[3:6])
        dv = np.zeros(3) if dv is None else np.asarray(dv, float)
        vals = [("%.8f" % mjd), _b_mjd_utc(mjd), event, body,
                "%.6f" % r_eq[0], "%.6f" % r_eq[1], "%.6f" % r_eq[2],
                "%.9f" % v_eq[0], "%.9f" % v_eq[1], "%.9f" % v_eq[2],
                "%.9f" % dv[0], "%.9f" % dv[1], "%.9f" % dv[2],
                "%.9f" % np.linalg.norm(dv),
                "" if BT is None else "%.3f" % BT,
                "" if BR is None else "%.3f" % BR,
                "" if tau is None else "%.1f" % tau,
                "" if rp is None else "%.3f" % rp,
                "" if miss is None else "%.3f" % miss, note]
        return ",".join(vals)

    lines = list(hdr) + [",".join(cols)]
    lines.append(row(res["t0c"], "departure", "earth", res["y_dep"],
                     dv=res["dv0"], note="post-TCM heliocentric state at Earth SOI exit"))
    lines.append(row(res["t_in"], "flyby_entry", "mars", res["y_ent"],
                     BT=res["BT"], BR=res["BR"], tau=res["tau"], rp=res["rp"],
                     note="B-plane target reached; tau>0 before periapsis"))
    lines.append(row(res["t_peri"], "flyby_periapsis", "mars", res["y_peri"],
                     rp=res["rp"], note="patched Mars hyperbola periapsis"))
    lines.append(row(res["t_exit"], "flyby_exit", "mars", res["y_exit_post"],
                     dv=res["dv1"], rp=res["rp"],
                     note="post-TCM state at Mars SOI exit (leg-1 start)"))
    lines.append(row(res["t2"], "arrival", "flora", res["y_arr"],
                     miss=res["miss"], note="position match at target epoch"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path

# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global USE_MOON
    ap = argparse.ArgumentParser(
        description="§3.4 精确模型引力辅助 / 二体→精确模型桥接")
    ap.add_argument("--book", action="store_true",
                    help="运行 §3.4 教材复现（表3.27/3.28）")
    ap.add_argument("--bridge", action="store_true",
                    help="运行二体→精确模型桥接（默认）")
    ap.add_argument("--csv", default="bridge.csv",
                    help="桥接数据输出 CSV 路径（默认 bridge.csv）")
    ap.add_argument("--plot", action="store_true", help="出图3.34~3.37")
    ap.add_argument("--no-moon", action="store_true", help="关闭月球摄动")
    a = ap.parse_args(argv)
    if a.no_moon:
        USE_MOON = False

    if a.book or (a.plot and not a.bridge):
        txt, vp2 = report()
        print(txt)
        if a.plot:
            try:
                en = make_figs(vp2)
                print("\n已输出 fig3_34~3_37.png；借力能量 %.0f→%.0f km^2/s^2（书 -310→-240）"
                      % (en["E_lo"], en["E_hi"]))
            except Exception as ex:
                print("\n出图失败：%s" % ex)
        return 0

    print("=" * 74)
    print("6.py — 中型保真桥接：二体引力辅助序列 → 精确力模型 → GMAT 数据")
    print("=" * 74)
    print("星历: %s" % ("JPL DE421" if _B_HAVE_DE421 else "Standish 开普勒根数(回退)"))
    seq = build_bridge_sequence(verbose=True)
    res = run_bridge(seq, verbose=True)
    path = export_bridge_csv(res, a.csv)
    print("CSV: %s  (frame=SunMJ2000Eq, %d events)" % (path, 5))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
