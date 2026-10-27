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
  python 6.py                 # 表3.27/3.28 微分修正 + 标称校验
  python 6.py --plot          # 出图3.34~3.37 PNG
  python 6.py --no-moon       # 关闭月球摄动
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


# ---------------------------------------------------------------------------

# ===========================================================================
# 中型保真桥接器：二体(patch-conic) MGA 序列 -> 精确力模型连续状态 -> GMAT 数据
# ---------------------------------------------------------------------------
# 定位：pykep/pygmo 给出「脉冲等效」二体飞越序列（位置连续、速度跳变）。
# 本段把它变成可被 GMAT 直接传播的日心状态序列，每个节点给一条 TCM Δv：
#   leg-i : 从上一节点(出发 SOI 出口 / 飞越 SOI 出口)精确传播到下一飞越 SOI 入口，
#           用 3 维速度增量把 B 平面 (BT,BR) 与时间参数 τ 修正到设计值；
#   flyby : 用该借力体二体双曲线把 SOI 入口 → 近点 → SOI 出口串起来(rp 设计值)；
#   末段  : 精确传播到目标历元，位置微分修正(交会/命中)。
# 输入：nodes 字典（见 --dump-nodes 生成的模板），飞越个数不限；
#       可选 "dsms":[ {epoch_mjd,dv_km_s} | null, ... ] 与各段一一对应（MGA-1DSM 结果）。
# 输出：CSV（历元/状态/Δv/B 平面/近火半径/交会残差），供 GMAT 直接读初值与节点。
# 力模型：太阳 + 行星星历（优先 JPL DE421，回退 Standish 开普勒根数）。
# 坐标：内部 J2000 黄道；CSV 给 J2000 地球赤道(SunMJ2000Eq)，便于 GMAT。
# ===========================================================================
import os as _b_os
import json as _b_json
import datetime as _b_dt
from scipy.optimize import least_squares as _b_ls, brentq as _b_brentq

_B_DAY = DAY
_B_AU = AU
_B_MU = GM["sun"]
_B_R_SOI_E = R_SOI_E

_B_RADII = {"mercury": 2439.7, "venus": 6051.8, "earth": 6378.137,
            "mars": 3396.2, "jupiter": 71492.0, "saturn": 60268.0}
_B_SOI = {"mercury": 1.12e5, "venus": 6.16e5, "earth": 9.25e5,
          "mars": 5.77e5, "jupiter": 4.82e7, "saturn": 5.46e7}
_B_PLANETS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn")
_B_RP_ALT = 300.0

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
    if name in ("flora",):
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

# ---- B 平面（含 τ，式 3.108 的 e·sinh H 项） ------------------------------
def _b_bplane(rr, vv, mu=None):
    """相对状态 -> (BT,BR,tau,b,e,vinf,S,nhat,ehat)。τ>0 表示还在近点前。"""
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

def _b_prop2b(y0, t0, t1, mu, event_r=None):
    """二体传播（相对系），可带 SOI 半径终止事件；返回末状态与末时刻(日, 带符号)。"""
    def rhs(t, y):
        r = y[:3]
        return [y[3], y[4], y[5], *(-mu * r / np.linalg.norm(r) ** 3)]
    ev = None
    if event_r is not None:
        def ev(t, y):
            return np.linalg.norm(y[:3]) - event_r
        ev.terminal = True
        ev.direction = 0
    sol = solve_ivp(rhs, (0.0, (t1 - t0) * _B_DAY), np.asarray(y0, float),
                    method="DOP853", rtol=1e-12, atol=1e-9, events=ev)
    return sol.y[:, -1], sol.t[-1] / _B_DAY

def _b_prop_leg(y0, t0, t1, eph, dsm=None, rtol=1e-10, atol=1e-6):
    """整段传播：中间若给 DSM(固定历元+固定 Δv)则先漂到 DSM 再打脉冲后继续。"""
    if not dsm:
        return _b_propagate(y0, t0, t1, eph, rtol, atol)
    td = float(dsm["epoch_mjd"]); dv = np.asarray(dsm["dv_km_s"], float)
    if not (min(t0, t1) < td < max(t0, t1)):
        raise ValueError("DSM 历元 %.3f 不在本段 [%.3f, %.3f] 内" % (td, t0, t1))
    y = _b_propagate(y0, t0, td, eph, rtol, atol).copy()
    y[3:6] += dv
    return _b_propagate(y, td, t1, eph, rtol, atol)

# ---- 3×3 微分修正（显式 Jacobian + 回溯） --------------------------------
def _b_dc3(gfun, x0=None, h=1e-3, iters=15, tol=1e-4):
    x = np.zeros(3) if x0 is None else np.asarray(x0, float).copy()
    hist = []; stall = 0
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
        step = 1.0; accepted = False; gn = n0
        for _ in range(25):
            gn = np.linalg.norm(np.asarray(gfun(x + step * d), float))
            if gn < n0:
                x = x + step * d; accepted = True; break
            step *= 0.5
        if not accepted:
            break
        # 停滞检测：残差下降过慢(接近积分噪声底)时提前收手
        if gn > 0.5 * n0:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
    return x, hist

# ---- 飞越几何：由进/离 vinf 与 rp 定 B 平面目标、双曲线近点状态 ------------
def _b_flyby_geom(body, vi, vo, rp=None, alt_min=200.0):
    mu = GM[body]; rb = _B_RADII[body]
    vi = np.asarray(vi, float); vo = np.asarray(vo, float)
    svi = np.linalg.norm(vi); svo = np.linalg.norm(vo)
    S_in = vi / svi; S_out = vo / svo
    n = np.cross(S_in, S_out); nn = np.linalg.norm(n)
    if nn < 1e-9:
        raise ValueError("vinf_in 与 vinf_out 近平行：飞越平面退化，无法定 B 平面")
    n = n / nn
    ehat = S_in - S_out; ehat = ehat / np.linalg.norm(ehat)
    ang = math.acos(max(-1.0, min(1.0, float(np.dot(S_in, S_out)))))
    vinf = 0.5 * (svi + svo)
    if rp is None:                        # 由转角反解近点半径
        e = 1.0 / math.sin(max(ang, 1e-4) / 2.0)
        rp = mu * (e - 1.0) / vinf ** 2
    rp = float(max(float(rp), rb + alt_min))
    e = 1.0 + rp * vinf ** 2 / mu
    delta = 2.0 * math.asin(min(1.0, 1.0 / e))
    vp = math.sqrt(mu * (1.0 + e) / rp)
    b = rp * math.sqrt(1.0 + 2.0 * mu / (rp * vinf ** 2))
    N = np.array([0.0, 0.0, 1.0])
    T = np.cross(S_in, N); T = T / np.linalg.norm(T); R = np.cross(S_in, T)
    Bhat = np.cross(S_in, n)
    return dict(S_in=S_in, S_out=S_out, n=n, ehat=ehat, vinf=float(vinf),
                rp=rp, e=float(e), delta=float(delta), vp=float(vp), b=float(b),
                BT=float(np.dot(b * Bhat, T)), BR=float(np.dot(b * Bhat, R)),
                turn=float(ang))

# ---- 低精度序列替身（pykep 输出格式）：Earth -> Mars(rp) -> Flora --------
def build_bridge_nodes(verbose=False):
    """生成 nodes 字典（等价于 pykep 两级 MGA 的输出）：
    departure{body,epoch_mjd,vinf_km_s} / flybys[{body,epoch_mjd,vinf_in/out,rp_km}]
    / arrival{epoch_mjd,target}。"""
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
        dcap = 2 * math.asin(1 / (1 + _B_RP_ALT_ref() * np.dot(vi, vi) / muM))
        ang = math.acos(max(-1.0, min(1.0,
              np.dot(vi, vo) / np.linalg.norm(vi) / np.linalg.norm(vo))))
        return [(np.linalg.norm(vo) - np.linalg.norm(vi)) * 1e3,
                (ang - dcap) * 1e3]

    sol = _b_ls(res_fly, [T1g, T2g], x_scale=10.0, xtol=1e-14, ftol=1e-14,
                gtol=1e-14, max_nfev=400)
    t1, t2 = float(sol.x[0]), float(sol.x[1])
    rM1, vM1 = _b_state("mars", t1)
    rF, _ = _b_state("flora", t2)
    vdep1, vi = _dep_vi(t1)
    vdep2, _, _ = _b_lambert(rM1, rF, (t2 - t1) * _B_DAY)
    vo = vdep2 - vM1
    rp = _B_RADII["mars"] + _B_RP_ALT
    nodes = dict(
        departure=dict(body="earth", epoch_mjd=T0,
                       vinf_km_s=[float(x) for x in (vdep1 - vE)]),
        flybys=[dict(body="mars", epoch_mjd=t1,
                     vinf_in_km_s=[float(x) for x in vi],
                     vinf_out_km_s=[float(x) for x in vo],
                     rp_km=float(rp))],
        arrival=dict(epoch_mjd=t2, target="flora"))
    if verbose:
        print("  [seq] t1=%.4f t2=%.4f |vi|=%.4f |vo|=%.4f rp_alt=%.0f km"
              % (t1, t2, np.linalg.norm(vi), np.linalg.norm(vo), rp - _B_RADII["mars"]))
    return nodes

def _B_RP_ALT_ref():
    return _B_RADII["mars"] + _B_RP_ALT

def _b_hyper_exit_days(rp, vinf, mu, r_end):
    """从近星点沿双曲线飞到 r_end 的时间（天），用于汇报出发 SOI 出口历元。"""
    if vinf <= 1e-9:
        return 0.0
    A = mu / (vinf * vinf)                 # -a > 0
    e = 1.0 + rp * vinf * vinf / mu
    c = (r_end / A + 1.0) / e
    if c <= 1.0000001:
        return 0.0
    H = math.acosh(c)
    return math.sqrt(A ** 3 / mu) * (e * math.sinh(H) - H) / _B_DAY

# ---- nodes -> 桥接序列（飞越双曲线 + SOI 入/出口状态） --------------------
def seq_from_nodes(nodes, verbose=False):
    dep = nodes["departure"]; body0 = dep.get("body", "earth")
    t0 = float(dep["epoch_mjd"])
    r0, v0 = _b_state(body0, t0)
    vinf0 = np.asarray(dep["vinf_km_s"], float)
    flies = []
    for f in nodes.get("flybys", []):
        b = f["body"]; t = float(f["epoch_mjd"]); mu = GM[b]; SOI = _B_SOI[b]
        g = _b_flyby_geom(b, f["vinf_in_km_s"], f["vinf_out_km_s"], f.get("rp_km"))
        yperi = np.concatenate([g["rp"] * g["ehat"],
                                g["vp"] * np.cross(g["n"], g["ehat"])])
        tlim = 2.5 * SOI / g["vinf"] / _B_DAY
        yin, tin = _b_prop2b(yperi, 0.0, -tlim, mu, event_r=SOI)
        yout, tout = _b_prop2b(yperi, 0.0, +tlim, mu, event_r=SOI)
        tin = -tin
        if tin <= 0 or tout <= 0:          # 事件未触发（数值兜底）
            tin = tlim; tout = tlim
        rb, vb = _b_state(b, t - tin)
        y_in = np.concatenate([rb + yin[:3], vb + yin[3:]])
        rb2, vb2 = _b_state(b, t + tout)
        y_out = np.concatenate([rb2 + yout[:3], vb2 + yout[3:]])
        flies.append(dict(body=b, epoch=t, t_in=t - tin, t_out=t + tout,
                          tau_in=tin, tau_out=tout, y_in=y_in, y_out=y_out, **g))
    arr = nodes["arrival"]; t2 = float(arr["epoch_mjd"])
    target = arr.get("target", "flora")
    rT, _ = _b_state(target, t2)
    dsms = list(nodes.get("dsms") or [None] * (len(flies) + 1))
    if len(dsms) != len(flies) + 1:
        raise ValueError("dsms 长度应为 %d (每段一个，可为 null)" % (len(flies) + 1))
    seq = dict(t0=t0, body0=body0, r0=r0, v0=v0, vinf_dep=vinf0,
               flies=flies, t2=t2, target=target, rT=rT, nodes=nodes, dsms=dsms)
    if verbose:
        for i, f in enumerate(flies):
            print("  [flyby %d/%d] %s t=%.4f rp=%.1f km vi=%.3f vo=%.3f δ=%.2f° "
                  "tin=%.2f d tout=%.2f d"
                  % (i + 1, len(flies), f["body"], f["epoch"], f["rp"],
                     np.linalg.norm(f["S_in"]) * f["vinf"],
                     np.linalg.norm(f["S_out"]) * f["vinf"],
                     math.degrees(f["delta"]), f["tau_in"], f["tau_out"]))
    return seq

# ---- 桥接主流程：逐段 TCM 打靶 -------------------------------------------
def run_bridge(seq, verbose=True):
    body0 = seq["body0"]; t0 = seq["t0"]; flies = seq["flies"]
    t2 = seq["t2"]; target = seq["target"]; SOI0 = _B_SOI.get(body0, _B_R_SOI_E)
    allp = [p for p in _B_PLANETS if _b_avail(p)]

    # --- 1) 出发：与 pykep 完全一致，t0 时刻从"天体中心 + vinf"起算 ----------
    #     （旧做法从首个飞越 SOI 入口反向积分上百年，103~10^3 天的段误差被放大
    #      到 1e7 km；改为直接用节点历元，TCM 只吸收星历差。）
    vinf_lo = seq["vinf_dep"]
    vn0 = float(np.linalg.norm(vinf_lo))
    rb0, vb0 = _b_state(body0, t0)
    t0c = t0
    vinf_c = vinf_lo.copy()

    # --- 2) 出发状态 --------------------------------------------------------
    u = vinf_lo / np.linalg.norm(vinf_lo)
    y_cur = np.concatenate([rb0, vb0 + vinf_lo])
    t_cur = t0c

    # --- 3) 逐段：B 平面(BT,BR)+τ 打靶 -> 飞越双曲线 -> 下一段 --------------
    legs = []
    for i, f in enumerate(flies):
        b = f["body"]; mu = GM[b]; t_in = f["t_in"]; SOI = _B_SOI[b]
        excl = {b} | ({body0} if i == 0 else set())
        pert = [p for p in allp if p not in excl]
        eph = _B_Eph(pert, t_cur, t_in)
        BTt, BRt, taut = f["BT"], f["BR"], f["tau_in"] * _B_DAY
        dsm = seq["dsms"][i]

        # 两级打靶：
        #   (1) 打到设计给出的 SOI 入口日心位置 —— 良态，先把偏差压到公里级；
        #   (2) 再用 (BT,BR,τ) 精修 —— 锁死飞越几何，避免借力出口速度跑偏。
        y_tgt = np.asarray(f["y_in"][:3], float)

        def gpos(x, _e=eph, _t0=t_cur, _t1=t_in, _yt=y_tgt, _dsm=dsm):
            yy = y_cur.copy(); yy[3:6] += x
            yf = _b_prop_leg(yy, _t0, _t1, _e, dsm=_dsm)
            return (yf[:3] - _yt) / 1e3

        def gfun(x, _e=eph, _t0=t_cur, _t1=t_in, _b=b, _mu=mu,
                 _BT=BTt, _BR=BRt, _tau=taut, _dsm=dsm):
            yy = y_cur.copy(); yy[3:6] += x
            yf = _b_prop_leg(yy, _t0, _t1, _e, dsm=_dsm)
            rb, vb = _b_state(_b, _t1)
            BT, BR, tau, *_ = _b_bplane(yf[:3] - rb, yf[3:6] - vb, _mu)
            return np.array([(BT - _BT) / 1e3, (BR - _BR) / 1e3,
                             (tau - _tau) / 1e4])

        x0 = _b_dc3(gpos, np.zeros(3), h=1e-3, iters=8, tol=1e-2)[0]
        dv, hist = _b_dc3(gfun, x0, h=1e-3, iters=10, tol=1e-6)
        g_end = float(np.linalg.norm(gfun(dv)))
        conv = g_end < 1e-2            # BT/BR 残差 < 10 km 视为收敛
        y_dep = y_cur.copy(); y_dep[3:6] += dv
        y_dsm = (_b_propagate(y_dep, t_cur, float(dsm["epoch_mjd"]), eph)
                 if dsm else None)
        # 出发段额外给出"离开出发天体 SOI"的时刻与状态，供 GMAT 取初值
        t_dep_soi, y_dep_soi = None, None
        if i == 0:
            rp0 = _B_RADII[body0] + _B_RP_ALT
            tau_e = _b_hyper_exit_days(rp0, vn0, GM[body0], SOI0)
            t_dep_soi = t_cur + tau_e
            y_dep_soi = (_b_propagate(y_dep, t_cur, t_dep_soi, eph)
                         if tau_e > 0 else y_dep)
        y_ent = _b_prop_leg(y_dep, t_cur, t_in, eph, dsm=dsm)
        rb, vb = _b_state(b, t_in)
        BTa, BRa, taua, ba, ea, via, Sa, nha, eha = _b_bplane(
            y_ent[:3] - rb, y_ent[3:6] - vb, mu)
        rpa = f["rp"]
        # 用设计的 rp 与设计的 vinf 反算近星点速度，保证 |v∞out| = |v∞in| = 设计值；
        # 旧式 sqrt(mu(1+e_actual)/rp) 混用实际 e 与设计 rp，会破坏能量。
        vpa = math.sqrt(f["vinf"] ** 2 + 2.0 * mu / rpa)
        t_peri = t_in + taua / _B_DAY
        yperi_rel = np.concatenate([rpa * eha, vpa * np.cross(nha, eha)])
        rbp, vbp = _b_state(b, t_peri)
        y_peri = np.concatenate([rbp + yperi_rel[:3], vbp + yperi_rel[3:]])
        tlim = 2.5 * SOI / max(1e-6, np.linalg.norm(yperi_rel[3:6])) / _B_DAY
        yout, tout = _b_prop2b(yperi_rel, 0.0, tlim, mu, event_r=SOI)
        t_exit = t_peri + tout
        rbo, vbo = _b_state(b, t_exit)
        y_exit = np.concatenate([rbo + yout[:3], vbo + yout[3:]])

        legs.append(dict(i=i, body=b, t_start=t_cur, y_start=y_cur, dv=dv,
                         y_start_post=y_dep, t_in=t_in, y_ent=y_ent,
                         BT=BTa, BR=BRa, tau=taua, b=ba, e=ea, vinf=via,
                         rp=rpa, t_peri=t_peri, y_peri=y_peri,
                         t_exit=t_exit, y_exit=y_exit, hist=hist,
                         conv=conv, g_end=g_end, dsm=dsm, y_dsm=y_dsm,
                         t_dep_soi=t_dep_soi, y_dep_soi=y_dep_soi))
        if verbose:
            if dsm:
                print("         (leg%d DSM @%.3f, |Δv_dsm|=%.1f m/s)"
                      % (i, dsm["epoch_mjd"], np.linalg.norm(dsm["dv_km_s"]) * 1e3))
            print("  [leg%d] %s->%s  Δv=(%.2f, %.2f, %.2f) m/s |%.1f|  "
                  "BT=%.1f BR=%.1f τ=%.0f s rp=%.1f km"
                  % (i, body0 if i == 0 else flies[i - 1]["body"], b,
                     dv[0] * 1e3, dv[1] * 1e3, dv[2] * 1e3,
                     np.linalg.norm(dv) * 1e3, BTa, BRa, taua, rpa))
            if not conv:
                print("  [!] leg%d 打靶未收敛：残差 %.2e (BT/BR 偏差≈%.0f km, τ≈%.0f s)。"
                      "该低精度种子在精确模型下不可达，建议在 pykep/pygmo 侧重设节点历元。"
                      % (i, g_end, math.hypot(g_end, 0) * 1e3, g_end * 1e4))
        y_cur = y_exit; t_cur = t_exit

    # --- 4) 末段：精确传播到目标历元，位置微分修正 --------------------------
    rT = seq["rT"]
    dsm_f = seq["dsms"][-1]
    pert = [p for p in allp if p != target]
    eph = _B_Eph(pert, t_cur, t2)

    def gfin(x):
        yy = y_cur.copy(); yy[3:6] += x
        yf = _b_prop_leg(yy, t_cur, t2, eph, dsm=dsm_f, rtol=1e-13, atol=1e-9)
        return (yf[:3] - rT) / 1e6

    dvf, histf = _b_dc3(gfin, np.zeros(3), h=1e-2, iters=10, tol=1e-7)
    y_end = y_cur.copy(); y_end[3:6] += dvf
    y_arr = _b_prop_leg(y_end, t_cur, t2, eph, dsm=dsm_f, rtol=1e-13, atol=1e-9)
    miss = float(np.linalg.norm(y_arr[:3] - rT))
    y_dsm_f = (_b_propagate(y_end, t_cur, float(dsm_f["epoch_mjd"]), eph)
               if dsm_f else None)
    fin = dict(t_start=t_cur, dv=dvf, y_start_post=y_end, t_end=t2,
               y_arr=y_arr, miss=miss, hist=histf, dsm=dsm_f, y_dsm=y_dsm_f)

    dv_tot = sum(float(np.linalg.norm(l["dv"])) for l in legs) + \
             float(np.linalg.norm(dvf))
    res = dict(seq=seq, t0c=t0c, legs=legs, final=fin, miss=miss,
               dv_tot=dv_tot, vinf_c=vinf_c, epoch_shift=t0c - t0)
    if verbose:
        if dsm_f:
            print("         (leg%d DSM @%.3f, |Δv_dsm|=%.1f m/s)"
                  % (len(legs), dsm_f["epoch_mjd"],
                     np.linalg.norm(dsm_f["dv_km_s"]) * 1e3))
        print("  [leg%d] ->%s  Δv=(%.2f, %.2f, %.2f) m/s |%.1f|  交会残差 %.3f km"
              % (len(legs), target, dvf[0] * 1e3, dvf[1] * 1e3, dvf[2] * 1e3,
                 np.linalg.norm(dvf) * 1e3, miss))
        print("  出发历元修正 %+.3f d；总 Δv = %.1f m/s"
              % (res["epoch_shift"], dv_tot * 1e3))
    return res


# ---- CSV 输出（GMAT 可直接读初值/节点状态） -------------------------------
def _b_mjd_utc(m):
    base = _b_dt.datetime(1858, 11, 17)
    return (base + _b_dt.timedelta(days=float(m))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def _b_to_eq(r, v):
    return _B_RX.T @ np.asarray(r, float), _B_RX.T @ np.asarray(v, float)

def export_bridge_csv(res, path="bridge.csv"):
    """逐事件写出历元/日心状态/该节点 TCM/飞越 B 平面量，供 GMAT 读取。"""
    seq = res["seq"]; legs = res["legs"]; fin = res["final"]
    flies = seq["flies"]
    chain = [seq["body0"]] + [f["body"] for f in flies] + [seq["target"]]
    hdr = [
        "# 6.py medium-fidelity bridge: patched-conic MGA -> precise ephemeris",
        "# body sequence: " + " -> ".join(
            ["%s(SOI exit)" % chain[0]] +
            ["%s(rp=%.1f km)" % (f["body"], f["rp"]) for f in flies] +
            [chain[-1]]),
        "# frame: SunMJ2000Eq (heliocentric J2000 Earth-equatorial), units km, km/s",
        "# time: UTC (MJD); DE421 evaluated at JD=MJD+2400000.5",
        "# TCM: one 3-axis burn at each leg start (SOI exit); dv columns are that burn",
        "# dsm rows: seed-model deep-space burn, state given is PRE-burn, add dv at that row",
        "# force model per leg excludes the flyby body (patched conic); leg-0 also excludes",
        "#   the departure body; final leg includes all available planets",
        "# GMAT: use CoordinateSystem 'SunMJ2000Eq'; x/y/z,vx/vy/vz are propagation-ready",
        "# departure epoch corrected by %+.4f d vs low-fidelity sequence" % res["epoch_shift"],
        "#",
    ]
    cols = ["leg", "epoch_mjd", "epoch_utc", "event", "body",
            "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s",
            "dvx_km_s", "dvy_km_s", "dvz_km_s", "dv_km_s",
            "BT_km", "BR_km", "tau_s", "rp_km", "miss_km", "note"]

    def fmt(leg, t, event, body, y, dv=None, BT=None, BR=None, tau=None,
            rp=None, miss=None, note=""):
        r_eq, v_eq = _b_to_eq(y[:3], y[3:6])
        dv = np.zeros(3) if dv is None else np.asarray(dv, float)
        vals = ["%d" % leg, "%.8f" % t, _b_mjd_utc(t), event, body,
                "%.6f" % r_eq[0], "%.6f" % r_eq[1], "%.6f" % r_eq[2],
                "%.9f" % v_eq[0], "%.9f" % v_eq[1], "%.9f" % v_eq[2],
                "%.9f" % dv[0], "%.9f" % dv[1], "%.9f" % dv[2],
                "%.9f" % np.linalg.norm(dv),
                "" if BT is None else "%.3f" % BT,
                "" if BR is None else "%.3f" % BR,
                "" if tau is None else "%.1f" % tau,
                "" if rp is None else "%.3f" % rp,
                "" if miss is None else "%.6f" % miss, note]
        return ",".join(vals)

    def dsm_row(leg, L):
        d = L.get("dsm")
        if not d or L.get("y_dsm") is None:
            return None
        return fmt(leg, float(d["epoch_mjd"]), "dsm", "-", L["y_dsm"],
                   dv=d["dv_km_s"],
                   note="deep-space manoeuvre (seed model), pre-burn state")

    rows = []
    dv0 = legs[0]["dv"] if legs else np.zeros(3)
    L0 = legs[0] if legs else None
    t_dep = (L0.get("t_dep_soi") if L0 else None) or res["t0c"]
    y_dep = (L0.get("y_dep_soi") if L0 else None)
    if y_dep is None:
        y_dep = L0["y_start_post"] if L0 else fin["y_start_post"]
        note = "post-TCM heliocentric state at departure epoch"
    else:
        note = "post-TCM state at %s SOI exit (burn applied at departure epoch)" % seq["body0"]
    rows.append(fmt(0, t_dep, "departure", seq["body0"], y_dep, dv=dv0, note=note))
    r = dsm_row(0, legs[0]) if legs else None
    if r:
        rows.append(r)
    for i, L in enumerate(legs):
        rows.append(fmt(i, L["t_in"], "flyby_entry", L["body"], L["y_ent"],
                        BT=L["BT"], BR=L["BR"], tau=L["tau"], rp=L["rp"],
                        note="B-plane target reached; tau>0 before periapsis"))
        rows.append(fmt(i, L["t_peri"], "flyby_periapsis", L["body"], L["y_peri"],
                        rp=L["rp"], note="patched %s two-body hyperbola periapsis"
                        % L["body"]))
        dv_next = legs[i + 1]["dv"] if i + 1 < len(legs) else fin["dv"]
        y_exit_post = L["y_exit"].copy(); y_exit_post[3:6] += dv_next
        rows.append(fmt(i + 1, L["t_exit"], "flyby_exit", L["body"], y_exit_post,
                        dv=dv_next, rp=L["rp"],
                        note="post-TCM state at %s SOI exit (next leg start)" % L["body"]))
        if i + 1 < len(legs):
            r = dsm_row(i + 1, legs[i + 1])
        else:
            r = dsm_row(i + 1, fin)
        if r:
            rows.append(r)
    rows.append(fmt(len(legs), fin["t_end"], "arrival", seq["target"], fin["y_arr"],
                    miss=fin["miss"], note="position match at target epoch"))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hdr) + "\n")
        fh.write(",".join(cols) + "\n")
        fh.write("\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global USE_MOON
    ap = argparse.ArgumentParser(
        description="§3.4 精确模型引力辅助 / 二体→精确模型桥接")
    ap.add_argument("--book", action="store_true",
                    help="运行 §3.4 教材复现（表3.27/3.28）")
    ap.add_argument("--bridge", action="store_true",
                    help="运行二体→精确模型桥接（默认）")
    ap.add_argument("--nodes", default=None,
                    help="低精度序列 JSON（departure/flybys/arrival；可用 --dump-nodes 生成模板）")
    ap.add_argument("--dump-nodes", default=None,
                    help="把当前序列写成 JSON 模板后退出前保存")
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

    if a.nodes:
        with open(a.nodes, "r", encoding="utf-8") as fh:
            nodes = _b_json.load(fh)
        print("序列: %s" % a.nodes)
    else:
        nodes = build_bridge_nodes(verbose=True)

    if a.dump_nodes:
        with open(a.dump_nodes, "w", encoding="utf-8") as fh:
            _b_json.dump(nodes, fh, indent=2, ensure_ascii=False)
        print("nodes 模板: %s" % a.dump_nodes)

    seq = seq_from_nodes(nodes, verbose=True)
    res = run_bridge(seq, verbose=True)
    path = export_bridge_csv(res, a.csv)
    n_dsm = sum(1 for L in res["legs"] if L.get("dsm")) + (1 if res["final"].get("dsm") else 0)
    print("CSV: %s  (frame=SunMJ2000Eq, %d events)"
          % (path, 2 + 3 * len(seq["flies"]) + n_dsm))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
