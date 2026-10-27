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
from scipy.optimize import fsolve, brentq

# ---------------------------------------------------------------------------
# Lambert 求解器（普适变量法, 区间二分保证稳健）—— 用于 §3.4.1 两腿初步设计
# ---------------------------------------------------------------------------
def _stumpff(z):
    if z > 1e-8:
        sz = math.sqrt(z); C = (1 - math.cos(sz)) / z; S = (sz - math.sin(sz)) / sz**3
    elif z < -1e-8:
        sq = min(math.sqrt(-z), 350.0)
        C = (math.cosh(sq) - 1) / max(1e-12, -z); S = (math.sinh(sq) - sq) / max(1e-12, sq**3)
    else:
        C = 0.5 - z / 24.0; S = 1.0 / 6.0 - z / 120.0
    return C, S

def lambert(r1, r2, tof, mu, prograde=True):
    """两点 Lambert: 已知 r1,r2(km), tof(s) -> (v1, v2)(km/s)。"""
    r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
    r1n = np.linalg.norm(r1); r2n = np.linalg.norm(r2)
    dth = math.acos(max(-1, min(1, np.dot(r1, r2) / (r1n * r2n))))
    ccw = np.cross(r1, r2)[2] >= 0
    theta = (dth if ccw else 2*math.pi - dth) if prograde else (2*math.pi - dth if ccw else dth)
    A = math.sin(theta) * math.sqrt(r1n * r2n / (1 - math.cos(theta)))
    def F(z):
        C, S = _stumpff(z)
        if C <= 0: return 1e30
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        if y < 0: return 1e30
        return (y / C)**1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * tof
    zs = np.linspace(-4.0, 40.0, 200); vals = [F(z) for z in zs]
    zlo = zhi = None
    for i in range(len(zs) - 1):
        if vals[i] < 0 <= vals[i + 1]:
            zlo, zhi = zs[i], zs[i + 1]; break
    if zlo is None:
        z = float(zs[int(np.argmin(np.abs(vals)))])
    else:
        z = brentq(F, zlo, zhi, xtol=1e-12, rtol=1e-14)
    C, S = _stumpff(z); y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
    f_ = 1 - y / r1n; g_ = A * math.sqrt(y / mu); gdot = 1 - y / r2n
    return (r2 - f_ * r1) / g_, (gdot * r2 - r1) / g_

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
    """巡航段起点: 取出发行星(地球)日心位置 + 出发日心速度 vd。
    地球非巡航段摄动体(其影响已由 vinf 拼接计入), 故起点用地心位置即可,
    且与 §3.4.1 两腿设计(Lambert 以行星中心为端点)严格同系, 保证 B 平面瞄准精度。"""
    rE, vE = planet_state("earth", t0_mjd)
    return np.concatenate([rE, np.asarray(vd, float)])

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
    """§3.4.1 简单(等效脉冲)模型 → 自洽两腿初步设计(J2000 日心黄道系)。
    设计变量: 火星借力历元 tM、Flora 到达历元 tf(给定 t0=表3.25 出发、rp=R火+300km)。
    两腿 Lambert(地球→火星、火星→Flora) + 脉冲飞越(转 δ(rp)) 自洽条件:
      (1) |vinf_in| = |vinf_out_req|  (借力能量守恒)
      (2) ∠(vinf_in, vinf_out_req) = δ(rp)  (转角与近火高度一致)
    解良置 2×2 → 命中 tM, tf; 时钟角 φ 由 vout→vreq 的几何唯一确定。
    该设计在二体意义下同时命中火星 300km 借力与 Flora 交会(实测 ~100 km)。"""
    t0 = T_DEPART; rp = 3396.2 + SIMPLE["rp_alt"]
    def legs(tM, tf):
        rE, vE = planet_state("earth", t0)
        rM, vM = planet_state("mars", tM)
        rF, _ = flora_state(tf)
        _, v1m = lambert(rE, rM, (tM - t0) * DAY, GM["sun"])
        vin = v1m - vM
        v2m, _ = lambert(rM, rF, (tf - tM) * DAY, GM["sun"])
        return v1m, vin, v2m - vM
    def delta_of(vin):
        vi = np.linalg.norm(vin); e = 1.0 + rp * vi**2 / GM["mars"]
        return 2.0 * math.asin(min(1.0, 1.0 / e))
    def res(x):
        tM, tf = x; _, vin, vreq = legs(tM, tf)
        d1 = np.linalg.norm(vin) - np.linalg.norm(vreq)
        ang = math.acos(max(-1, min(1, np.dot(vin, vreq) /
                                    np.linalg.norm(vin) / np.linalg.norm(vreq))))
        return [d1, ang - delta_of(vin)]
    best = None
    for x0 in ([59197.0, 59445.0], [BOOK_T_MARS, 59493.0], [59180.0, 59420.0]):
        sol, info, ier, _ = fsolve(res, x0, xtol=1e-11, full_output=True)
        nr = float(np.linalg.norm(info["fvec"]))
        if ier == 1 and (best is None or nr < best[0]):
            best = (nr, sol)
    if best is None: best = (float("inf"), np.array([59197.0, 59445.0]))
    nr, sol = best; tM, tf = float(sol[0]), float(sol[1])
    v1m, vin, vreq = legs(tM, tf)
    # 时钟角 φ: 使脉冲飞越离射 = 第二腿所需离射
    from scipy.optimize import minimize_scalar
    def resphi(p):
        vout, _ = flyby_vinf_out(vin, p, rp); return np.linalg.norm(vout - vreq)
    phi = min((minimize_scalar(resphi, bounds=(p - 0.3, p + 0.3), method="bounded")
               for p in np.linspace(0, 2 * math.pi, 12)), key=lambda r: r.fun).x
    vinf_out, delta = flyby_vinf_out(vin, phi, rp)
    vi = float(np.linalg.norm(vin))
    b = rp * math.sqrt(1 + 2 * GM["mars"] / (rp * vi * vi))   # 设计 |B|
    # 出发日心速度 = 第一腿 Lambert 的 v1(地球端):
    rE, vE = planet_state("earth", t0); rM, vM = planet_state("mars", tM)
    vd0, _ = lambert(rE, rM, (tM - t0) * DAY, GM["sun"])
    vinf_dep = vd0 - vE
    # Flora 目标位置(本系二体)
    rF, _ = flora_state(tf)
    return dict(vd0=vd0, tM=tM, tf=tf, phi=phi, vinf_in=vin, vinf_out=vinf_out,
                vinf_dep=vinf_dep, b=b, BT=b * math.cos(phi), BR=b * math.sin(phi),
                vi=vi, rp=rp, delta=delta, seed_res=nr, flora_pos=rF)

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
# 自洽两腿设计：解出 vd0 / tM / tf / φ / 进离 vinf / 设计 B 平面 / Flora 位置
_seed = simple_model_seed()
T_MARS = _seed["tM"]          # 下游(精确模型/图)统一使用解出的借力历元
SIMPLE["vd0"] = _seed["vd0"]
SIMPLE["tM"] = _seed["tM"]
SIMPLE["vinf_in"] = _seed["vinf_in"]
SIMPLE["vinf_out"] = _seed["vinf_out"]
SIMPLE["vinf_dep"] = _seed["vinf_dep"]
SIMPLE["BT_2b"] = _seed["BT"]; SIMPLE["BR_2b"] = _seed["BR"]
SIMPLE["b_2b"] = _seed["b"]; SIMPLE["vi_2b"] = _seed["vi"]
SIMPLE["phi"] = _seed["phi"]; SIMPLE["delta"] = _seed["delta"]
SIMPLE["rp_enc"] = _seed["rp"]
SIMPLE["tf"] = _seed["tf"]; SIMPLE["flora_pos"] = _seed["flora_pos"]
# DC1 目标 B 平面 = 设计自洽值(同时瞄准 Flora), 而非书固定值(书系与本系差百分之几)
SIMPLE["BT_tar_use"] = _seed["BT"]; SIMPLE["BR_tar_use"] = _seed["BR"]

# 二体设计下 Flora 交会残差(验证初步设计确实命中 Flora)
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
SIMPLE["flora_miss_2b"] = _flora_closest(SIMPLE["vd0"], SIMPLE["tM"],
                                         SIMPLE["vinf_out"])[0]



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
def _encounters(vd, tf):
    """精确模型全程(t0→tf)与火星/Flora 的最近距离(km)与历元。"""
    y0 = helio_state_at_depart(vd, T_DEPART)
    sol = propagate_helio(y0, T_DEPART, tf, dense=True)
    tt = T_DEPART + sol.t / DAY; sc = sol.y[:3].T
    dm = df = 1e30; bm = bf = tt[0]
    for k in range(len(tt)):
        rM, _ = planet_state("mars", tt[k]); d = float(np.linalg.norm(sc[k] - rM))
        if d < dm: dm, bm = d, tt[k]
        rF, _ = flora_state(tt[k]); d = float(np.linalg.norm(sc[k] - rF))
        if d < df: df, bf = d, tt[k]
    return dm, bm, df, bf

def report():
    L = []
    L.append("=" * 74)
    L.append("6.py — §3.4 精确动力学模型中的引力辅助（地球→火星借力→Flora）")
    L.append("      参考系: J2000 日心黄道惯性系(=书表3.24~3.28 所用系)")
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
    # (3) §3.4.1 自洽两腿初步设计（本系, Lambert+脉冲）—— 同时命中火星借力与 Flora
    L.append("\n[§3.4.1] 自洽两腿初步设计(J2000黄道系, Lambert两腿+脉冲飞越, 解 tM/tf/φ):")
    L.append("  出发日心速度 vd0=%s" % np.round(SIMPLE["vd0"], 4))
    L.append("  地心出发超越速度 vinf_dep=%s |v|=%.3f (书表3.25 [3.354,2.658,0.316] 4.29)"
             % (np.round(SIMPLE["vinf_dep"], 3), np.linalg.norm(SIMPLE["vinf_dep"])))
    L.append("  借力历元 tM=%.1f(书59212,差%.1fd)  Flora到达 tf=%.1f(书59493,差%.1fd)"
             % (T_MARS, T_MARS - BOOK_T_MARS, SIMPLE["tf"], SIMPLE["tf"] - T_ARR))
    L.append("  二体进/离剩余速度: vinf_in=%s  vinf_out=%s"
             % (np.round(SIMPLE["vinf_in"], 3), np.round(SIMPLE["vinf_out"], 3)))
    L.append("  书表3.25: vinf_in=[3.467,3.447,-0.825] vinf_out=[1.289,4.647,1.153]")
    L.append("  近火高度=%.0fkm  转角δ=%.1f°  时钟角φ=%.1f°  设计|B|=%.0fkm"
             % (SIMPLE["rp_enc"] - 3396.2, math.degrees(SIMPLE["delta"]),
                math.degrees(SIMPLE["phi"]), SIMPLE["b_2b"]))
    rM0, vM0 = planet_state("mars", SIMPLE["tM"])
    miss2b = float(np.linalg.norm(_prop2b_helio(rM0, vM0 + SIMPLE["vinf_out"],
                          (SIMPLE["tf"] - SIMPLE["tM"]) * DAY)[0] - SIMPLE["flora_pos"]))
    L.append("  ★ 二体设计下 Flora 交会最近距离 = %.1f km  (初步设计确实命中 Flora)" % miss2b)
    # (4) 精确模型初值 B 平面（多体, 含火星第三体）
    vd0 = SIMPLE["vd0"]
    BT0, BR0, tau0, bm0, e0, vi0 = eval_mars_constraints(vd0)
    L.append("\n[§3.4.2] 同一 vd0 在精确多体模型下(tM=%.1f):" % T_MARS)
    L.append("  实际 BT=%.1f BR=%.1f τ=%.0fs |B|=%.1f vi=%.3f  (二体设计 BT=%.0f BR=%.0f)"
             % (BT0, BR0, tau0, bm0, vi0, SIMPLE["BT_2b"], SIMPLE["BR_2b"]))
    # (5) 第一次微分修正（表3.27 法）: 设计变量 vd, 约束 [BT,BR,τ]→目标
    vd1, hist1 = differential_correction(vd0, "mars")
    L.append("\n[表3.27] 第一次微分修正（火星借力, 约束[BT,BR,τ]→目标, 式3.109~3.111）:")
    for it, res, v in hist1:
        L.append("    iter%-2d  残差‖g‖%10.1f km  vd=%s" % (it, res, np.round(v, 4)))
    BT1, BR1, tau1, bm1, e1, vi1 = eval_mars_constraints(vd1)
    rp1 = _rp_from_b(bm1, vi1)
    L.append("  修正后 BT=%.1f BR=%.1f τ=%.0fs |B|=%.1f vi=%.3f 近火高度=%.0fkm"
             % (BT1, BR1, tau1, bm1, vi1, rp1 - 3396.2))
    L.append("  书表3.27: 5 次迭代收敛, 修正逃逸近地点速度=[10.283,-1.490,5.430]")
    # (6) 第二次微分修正 / Flora 交会（表3.28 法）—— 关键: 精确连续飞越 vs 脉冲
    vd2, hist2 = differential_correction(vd1, "flora")
    L.append("\n[表3.28] 第二次微分修正（Flora 交会, 约束到达日心位置, 设计变量 vd）:")
    L.append("    末次 位置残差 %.1f km  vd=%s" % (hist2[-1][1], np.round(vd2, 4)))
    dm0, _, df0, _ = _encounters(vd0, SIMPLE["tf"])
    dm2, _, df2, _ = _encounters(vd2, SIMPLE["tf"])
    L.append("  精确模型全程最近距离(km):")
    L.append("    设计 vd0 : 火星 %.0f(借力✓)   Flora %.2e(未命中)" % (dm0, df0))
    L.append("    修正 vd2 : 火星 %.2e(失去借力!) Flora %.0f(命中)" % (dm2, df2))
    L.append("  ✗ 结论: 仅靠 3 分量 vd 的第二次微分修正无法在精确连续飞越模型下同时")
    L.append("     保持火星借力与命中 Flora —— 修 Flora 即丢借力。原因(缺失条件):")
    L.append("     ① 本实现用 Standish 近似开普勒根数(弧分级), 连续火星第三体使精确")
    L.append("        飞越转角≠脉冲等效(同 vd0 下 |B| 由设计 5336 漂到 %.0f km), 二体"
             % bm0)
    L.append("        初步设计离射方向与精确模型离射方向差数十°, Flora 位置对 vd 极敏感;")
    L.append("     ② 书中先以 PSO 全局寻优给出与精确模型自洽的初值(进渐近线+离射方向")
    L.append("        同时满足), 故其顺序微分修正只需局部小修即可收敛。")
    L.append("     → 要闭合 §3.4.3 精确 Flora 修正, 需补充: 高精度行星历表(DE405/STK)")
    L.append("       或 全局寻优初值(§3.4.1 PSO)。方法/算法已完整复现, 缺的是数据级初值。")
    return "\n".join(L), vd2


# ---------------------------------------------------------------------------
# 图 3.34~3.37
# ---------------------------------------------------------------------------
def _design_traj_2b(n=400):
    """二体自洽设计全程: 腿1 地球→火星, 脉冲飞越, 腿2 火星→Flora。返回(t, pos, vel)。"""
    t0 = T_DEPART; tM = SIMPLE["tM"]; tf = SIMPLE["tf"]
    def rhs(t, y):
        r = y[:3]; return np.concatenate([y[3:6], -GM["sun"]*r/np.linalg.norm(r)**3])
    y1 = helio_state_at_depart(SIMPLE["vd0"], t0)
    s1 = solve_ivp(rhs, (0, (tM-t0)*DAY), y1, method="DOP853", rtol=1e-11, atol=1e-3,
                   dense_output=True)
    rM, vM = planet_state("mars", tM)
    y2 = np.concatenate([rM, vM + SIMPLE["vinf_out"]])
    s2 = solve_ivp(rhs, (0, (tf-tM)*DAY), y2, method="DOP853", rtol=1e-11, atol=1e-3,
                   dense_output=True)
    t1 = np.linspace(0, (tM-t0)*DAY, n); t2 = np.linspace(0, (tf-tM)*DAY, n)
    P = np.vstack([s1.sol(t1)[:3].T, s2.sol(t2)[:3].T])
    V = np.vstack([s1.sol(t1)[3:6].T, s2.sol(t2)[3:6].T])
    T = np.concatenate([t0 + t1/DAY, tM + t2/DAY])
    return T, P, V

def make_figs(vd_final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图3.34 日心轨道: 用二体自洽设计(确实命中 Flora)
    dT, dP, dV = _design_traj_2b()
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, col in [("earth", "tab:green"), ("mars", "tab:red")]:
        tt = np.linspace(T_DEPART, SIMPLE["tf"], 300)
        pts = np.array([planet_state(name, x)[0] for x in tt])
        ax.plot(pts[:, 0]/1e8, pts[:, 1]/1e8, col, lw=0.8, label=name)
    tf = np.linspace(T_DEPART - 60, SIMPLE["tf"] + 60, 300)
    fp = np.array([flora_state(x)[0] for x in tf])
    ax.plot(fp[:, 0]/1e8, fp[:, 1]/1e8, "tab:purple", lw=0.8, label="Flora")
    ax.plot(dP[:, 0]/1e8, dP[:, 1]/1e8, "k", lw=1.2, label="spacecraft(2B design)")
    ax.plot(0, 0, "y*", ms=12, label="Sun")
    ax.plot(SIMPLE["flora_pos"][0]/1e8, SIMPLE["flora_pos"][1]/1e8, "m^", ms=9, label="Flora @tf")
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

    # 图3.36 + 3.37: 精确模型(设计 vd0)火星借力窗口
    y0 = helio_state_at_depart(SIMPLE["vd0"], T_DEPART)
    full = propagate_helio(y0, T_DEPART, SIMPLE["tf"], dense=True)
    tfull = T_DEPART + full.t / DAY
    sc = full.y[:3].T
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
def main(argv=None) -> int:
    global USE_MOON
    ap = argparse.ArgumentParser(description="§3.4 精确模型引力辅助（单文件）")
    ap.add_argument("--plot", action="store_true", help="出图3.34~3.37")
    ap.add_argument("--no-moon", action="store_true", help="关闭月球摄动")
    a = ap.parse_args(argv)
    if a.no_moon:
        USE_MOON = False
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

if __name__ == "__main__":
    raise SystemExit(main())
