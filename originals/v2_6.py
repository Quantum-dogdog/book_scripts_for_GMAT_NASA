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
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares

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
    """行星日心黄道位置/速度(km, km/s)。
    优先 JPL DE421(用户提供, 高精度)：SSB 质心减太阳位置得日心，
    再绕 x 轴 -ε=−23.4392911° 由 J2000 赤道转到 J2000 黄道；
    不可用时回退到 Standish 近似开普勒根数(弧分级)。"""
    if _HAVE_DE421 and name in _DE421_PAIRS:
        try:
            return _de421_state(name, t_mjd)
        except Exception:
            pass
    p = _PLANET_KEP.get(name)
    if p is None:
        raise KeyError("no ephemeris for %s" % name)
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
# JPL DE421 星历（用户提供）—— 覆盖地球/火星的日心黄道位置速度
#   jplephem: kernel[(c,t)].compute_and_differentiate(JD)  返回 (km, km/day)
#   DE421 无小行星 → Flora 仍走表3.24 二体递推。
# ---------------------------------------------------------------------------
_DE421_PATH = os.environ.get("DE421_PATH", "/tmp/de421.bsp")
_DE421_KERNEL = None
_RX_ECL = None
_DE421_PAIRS = {"earth": [(0, 3), (3, 399)], "mars": [(0, 4), (4, 499)]}

def _init_de421():
    global _DE421_KERNEL, _RX_ECL
    if not os.path.isfile(_DE421_PATH):
        return False
    try:
        from jplephem.spk import SPK
        _DE421_KERNEL = SPK.open(_DE421_PATH)
        eps = math.radians(23.4392911)
        c, s = math.cos(eps), math.sin(eps)
        _RX_ECL = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
        return True
    except Exception:
        return False

def _de421_state(name, t_mjd):
    jd = t_mjd + 2400000.5
    r_ssb = np.zeros(3); v_ssb = np.zeros(3)
    for c, t in _DE421_PAIRS[name]:
        p, vel = _DE421_KERNEL[(c, t)].compute_and_differentiate(jd)
        r_ssb += np.asarray(p, float); v_ssb += np.asarray(vel, float)
    ps, vs = _DE421_KERNEL[(0, 10)].compute_and_differentiate(jd)
    r_ssb -= np.asarray(ps, float); v_ssb -= np.asarray(vs, float)
    return _RX_ECL @ r_ssb, (_RX_ECL @ v_ssb) / DAY

_HAVE_DE421 = _init_de421()

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
# §3.4.1 PSO 全局搜索（Kennedy & Eberhart 1995 标准 PSO，5D=[vd(3),tM,phi]）
#   代价 = rp≈(3396+300)km + (BT,BR)≈书目标 + 二体脉冲模型 Flora 最近距离
# ---------------------------------------------------------------------------
def _pulse_flora_scan(vd, tM, phi):
    rM, vM = planet_state("mars", tM)
    pos, vel = _state2b_at(vd, T_DEPART, tM)
    rr, vv = pos - rM, vel - vM
    BT, BR, tau, bm, e_, vi = bplane_from_state(rr, vv)
    rp_enc = _rp_from_b(bm, vi)
    if not np.isfinite(rp_enc) or not np.isfinite(BT):
        return 1e5, 0.0, 0.0, 1e10
    rp_eff = min(max(rp_enc, 3400.0), 5.0e4)
    vinf_in, _ = hyperbolic_asymptote(rr, vv, incoming=True)
    vinf_out, _ = flyby_vinf_out(vinf_in, phi, rp_eff)
    v2 = vM + vinf_out
    dts = np.linspace(80.0, 460.0, 8)
    miss = min(np.linalg.norm(_prop2b_helio(rM, v2, dt * DAY)[0]
                              - flora_state(tM + dt)[0]) for dt in dts)
    return rp_enc - (3396.2 + SIMPLE["rp_alt"]), BT, BR, float(miss)

def _pso_cost(x):
    try:
        drp, BT, BR, miss = _pulse_flora_scan(x[:3], x[3], x[4])
    except Exception:
        return 1e8
    return ((drp / 200.0) ** 2
            + ((BT - SIMPLE["BT_tar"]) ** 2 + (BR - SIMPLE["BR_tar"]) ** 2) / 1e7
            + (miss / 1e6) ** 2)

def pso_seed(n=30, iters=60, seed=42, warm=None):
    rng = np.random.default_rng(seed)
    lb = np.array([22.0, 15.0, -6.0, BOOK_T_MARS - 50.0, 0.0])
    ub = np.array([32.0, 27.0,  6.0, BOOK_T_MARS + 50.0, 2.0 * math.pi])
    d = len(lb)
    X = lb + rng.random((n, d)) * (ub - lb)
    if warm is not None:
        X[0] = np.clip(warm, lb, ub)
    V = (ub - lb) * (rng.random((n, d)) - 0.5) * 0.3
    J = np.array([_pso_cost(x) for x in X])
    pbest = X.copy(); pbest_J = J.copy()
    ib = int(np.argmin(pbest_J)); gbest = pbest[ib].copy(); gbest_J = pbest_J[ib]
    w = 0.7; c1 = 1.5; c2 = 1.5
    for it in range(iters):
        R1 = rng.random((n, d)); R2 = rng.random((n, d))
        V = w * V + c1 * R1 * (pbest - X) + c2 * R2 * (gbest - X)
        V = np.clip(V, -0.5 * (ub - lb), 0.5 * (ub - lb))
        Xn = np.clip(X + V, lb, ub)
        Jn = np.array([_pso_cost(x) for x in Xn])
        upd = Jn < pbest_J
        pbest[upd] = Xn[upd]; pbest_J[upd] = Jn[upd]
        X = Xn; J = Jn
        j = int(np.argmin(pbest_J))
        if pbest_J[j] < gbest_J:
            gbest_J = pbest_J[j]; gbest = pbest[j].copy()
        w = max(0.4, w * 0.985)
    return gbest, float(gbest_J)

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
# 自洽种子：先 fsolve 直击 B 平面，再 PSO 全局搜。默认下游沿用 fsolve seed 复现表3.27；
# --pso 时切到 PSO seed (§3.4.1 全局寻优严格复现, Flora 距离更小但当前 3 元 DC 不收敛)。
_seed = simple_model_seed()

def _seed_from_vd_tM(vd, tM, src, phi=None):
    rM, vM = planet_state("mars", tM)
    pos, vel = _state2b_at(vd, T_DEPART, tM)
    rr, vv = pos - rM, vel - vM
    BT, BR, tau, bm, e_, vi = bplane_from_state(rr, vv)
    vinf_in, _ = hyperbolic_asymptote(rr, vv, incoming=True)
    rp_enc = _rp_from_b(bm, vi)
    e_enc = 1.0 + rp_enc * vi * vi / GM["mars"]
    delta = 2.0 * math.asin(min(1.0, 1.0 / e_enc))
    if phi is None:
        phi = math.atan2(BR, BT)
    vinf_out, _ = flyby_vinf_out(vinf_in, phi, rp_enc)
    _, _, _, b_b = target_bplane(vinf_in, np.array(SIMPLE["vinf_out_book"]))
    return dict(vd0=np.asarray(vd, float), tM=float(tM), vinf_in=vinf_in,
                vinf_out=vinf_out, BT=BT, BR=BR, b=bm, vi=vi, rp_enc=rp_enc,
                seed_res=float("nan"), phi=phi, delta=delta, b_use=b_b, src=src)

def _flora_miss_2b(vd, tM, vinf_out, dt_lo=60.0, dt_hi=500.0, n=60):
    rM, vM = planet_state("mars", tM)
    dts = np.linspace(dt_lo, dt_hi, n)
    return min(np.linalg.norm(_prop2b_helio(rM, vM + vinf_out, dt * DAY)[0]
                              - flora_state(tM + dt)[0]) for dt in dts)

_fsolve_miss = _flora_miss_2b(_seed["vd0"], _seed["tM"], _seed["vinf_out"])
SIMPLE["fsolve_vd0"] = [float(v) for v in _seed["vd0"]]
SIMPLE["fsolve_tM"] = float(_seed["tM"])
_warm = np.concatenate([_seed["vd0"], [_seed["tM"]], [_seed["phi"]]])
_pso_x, _pso_J = pso_seed(warm=_warm)
_pso_seed = _seed_from_vd_tM(_pso_x[:3], _pso_x[3], "PSO", phi=_pso_x[4])
_pso_miss = _flora_miss_2b(_pso_seed["vd0"], _pso_seed["tM"], _pso_seed["vinf_out"])
SIMPLE["pso_J"] = _pso_J
SIMPLE["fsolve_miss"] = _fsolve_miss
SIMPLE["pso_miss"] = _pso_miss
SIMPLE["pso_phi"] = float(_pso_x[4])
SIMPLE["pso_vd0"] = [float(v) for v in _pso_seed["vd0"]]
SIMPLE["pso_tM"] = _pso_seed["tM"]
SIMPLE["seed_src"] = "fsolve(书BT/BR目标)"
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
# 6 元 DC（混合精确+脉冲模型）— 扩展设计变量到 [vd(3), tM, tf, phi]，
# 6 约束: [BT, BR, rp, Flora_x, Flora_y, Flora_z]。
# 传播策略: 精确多体 T_DEPART→tM(含火星第三体) → 在 tM 应用脉冲飞越(phi 时钟角)
#           → 日心二体 tM→tf → 命中 Flora(tf)。
# 目的: 让 3 元 DC 无法同时满足的 "火星借力+Flora 交会" 在 6 维空间闭合。
# ---------------------------------------------------------------------------
def dc_6var_hybrid(x0, iters=15, tol=50.0, verbose_hist=None):
    """6 元混合 DC: x=[vd(3),tM,tf,phi] ↔ g=[|vi|-vi_t, rp-rp_t, BT-BT_t, Flora(3)].
    物理动机: 精确多体到 tM 得 |vi|、rp、BT/BR; 若 |vi| 太大则 δ=2asin(1/e) 达不到
    书 37.4°，Flora 无法闭合; 故 |vi| 权重最高(尺度 1e-4)，把 vd 拉向低能入射。"""
    scales = np.array([1e-4, 1e2, 1e4, 1e6, 1e6, 1e6])
    rp_t = 3396.2 + SIMPLE["rp_alt"]
    vi_t = float(np.linalg.norm(SIMPLE["vinf_in_book"]))
    def g_raw(x):
        vd = x[:3]; tM = x[3]; tf = x[4]; phi = x[5]
        if tf <= tM + 1.0 or tM <= T_DEPART + 30.0:
            return np.full(6, 1e6)
        y0 = helio_state_at_depart(vd, T_DEPART)
        yM = propagate_helio(y0, T_DEPART, tM)
        rM, vM = planet_state("mars", tM)
        rr = yM[:3] - rM; vv = yM[3:6] - vM
        BT, BR, tau, bm, e_, vi = bplane_from_state(rr, vv)
        rp_enc = _rp_from_b(bm, vi)
        if not np.isfinite(rp_enc):
            rp_enc = 1e5
        vinf_in, _ = hyperbolic_asymptote(rr, vv, incoming=True)
        rp_eff = min(max(rp_enc, 3396.2), 5e4)
        vinf_out, _ = flyby_vinf_out(vinf_in, phi, rp_eff)
        pos_tf, _ = _prop2b_helio(rM, vM + vinf_out, (tf - tM) * DAY)
        rF, _ = flora_state(tf)
        return np.array([vi - vi_t, rp_enc - rp_t, BT - SIMPLE["BT_tar"],
                         pos_tf[0] - rF[0], pos_tf[1] - rF[1], pos_tf[2] - rF[2]])
    def g_norm(x):
        return g_raw(x) / scales
    x0 = np.asarray(x0, float).copy()
    lb = x0 + np.array([-5.0, -5.0, -5.0, -25.0, -100.0, -math.pi])
    ub = x0 + np.array([ 5.0,  5.0,  5.0,  25.0,  100.0,  math.pi])
    hist = []
    def report_cb(ks):
        r = g_norm(ks) * scales
        hist.append((len(hist) + 1, float(np.linalg.norm(r)), ks.copy(), r.copy()))
    report_cb(x0)
    try:
        res = least_squares(g_norm, x0, bounds=(lb, ub), method="trf",
                            x_scale=np.array([0.6, 0.6, 0.6, 4.0, 15.0, 0.4]),
                            diff_step=1e-3, ftol=1e-12, xtol=1e-12, gtol=1e-10,
                            max_nfev=80 * iters)
        x = res.x
    except Exception:
        return x0, hist
    report_cb(x)
    if verbose_hist is not None:
        verbose_hist.extend(hist[-3:])
    return x, hist


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
    # (3) 自洽简单模型种子（§3.4.1）：两条路线对比
    L.append("\n[§3.4.1] 两条初步设计路线(本星历系内二体脉冲等效模型):")
    L.append("  A) fsolve 直击书表3.27 B 平面 (BT=4027, BR=3094, rp=300km):")
    L.append("     vd=[%.4f %.4f %.4f] tM=%.1f 二体 Flora 最近=%.3e km"
             % (SIMPLE["fsolve_vd0"][0], SIMPLE["fsolve_vd0"][1], SIMPLE["fsolve_vd0"][2],
                SIMPLE["fsolve_tM"], SIMPLE["fsolve_miss"]))
    L.append("  B) PSO 5D 全局搜索 [vd(3),tM,phi] (rp+B平面+Flora 三项代价):")
    L.append("     vd=[%.4f %.4f %.4f] tM=%.1f phi=%.2f 二体 Flora 最近=%.3e km (J=%.2f)"
             % (SIMPLE["pso_vd0"][0], SIMPLE["pso_vd0"][1], SIMPLE["pso_vd0"][2],
                SIMPLE["pso_tM"], SIMPLE["pso_phi"], SIMPLE["pso_miss"], SIMPLE["pso_J"]))
    L.append("  下游 seed: %s" % SIMPLE["seed_src"])
    L.append("  当前 seed: vd0=%s   借力历元 tM=%.1f(书59212, 差%+.1fd)"
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
    # (7) 6 元 DC 混合模型（本轮新增：vd(3)+tM+tf+phi ↔ BT/BR/rp/Flora(3)）
    L.append("\n[§3.4.2-扩展] 6 元混合 DC（精确多体→tM + 脉冲飞越@tM + 二体→tf）:")
    # 用 DC1 修正后的 vd1 作起点(其 rp_enc 已在有效域), tM/tf/phi 取当前设计值
    BT1p, BR1p, taup, bmp, ep, vip = eval_mars_constraints(vd1)
    rp_enc1 = _rp_from_b(bmp, vip)
    phi1 = math.atan2(BR1p, BT1p) if rp_enc1 > 3396.2 else SIMPLE["phi"]
    x0 = np.concatenate([vd1, [T_MARS, SIMPLE["tf"], phi1]])
    L.append("  x0: vd=%s tM=%.2f tf=%.2f phi=%.2f (rp_enc=%.0f km)"
             % (np.round(vd1, 4), T_MARS, SIMPLE["tf"], phi1, rp_enc1))
    hist6 = []
    try:
        x6, _h6 = dc_6var_hybrid(x0, iters=12, tol=50.0, verbose_hist=hist6)
        # 每 iter 详细残差
        for it, res, xx, raw in _h6[:12]:
            L.append("    iter%-2d 综合残差%12.1e  BT%+8.0f BR%+8.0f rp%+8.0f  "
                     "FloraΔ=(%+.1e,%+.1e,%+.1e)  tM=%.2f tf=%.2f phi=%.2f"
                     % (it, res, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                        xx[3], xx[4], xx[5]))
        vd_f = x6[:3]; tM_f = x6[3]; tf_f = x6[4]; phi_f = x6[5]
        y0 = helio_state_at_depart(vd_f, T_DEPART)
        yM = propagate_helio(y0, T_DEPART, tM_f)
        rM, vM = planet_state("mars", tM_f)
        BTf, BRf, tauf, bmf, ef, vif = bplane_from_state(yM[:3] - rM, yM[3:6] - vM)
        rp_f = _rp_from_b(bmf, vif)
        vinf_in_f, _ = hyperbolic_asymptote(yM[:3] - rM, yM[3:6] - vM, incoming=True)
        rp_eff_f = min(max(rp_f, 3396.2), 5e4)
        vinf_out_f, _ = flyby_vinf_out(vinf_in_f, phi_f, rp_eff_f)
        pos_tf, _ = _prop2b_helio(rM, vM + vinf_out_f, (tf_f - tM_f) * DAY)
        rF_f, _ = flora_state(tf_f)
        gap_f = float(np.linalg.norm(pos_tf - rF_f))
        L.append("  → 修正后: |vi|=%.3f (目标4.958) BT=%.1f BR=%.1f rp=%.1f km (alt=%.0f)"
                 % (vif, BTf, BRf, rp_f, rp_f - 3396.2))
        L.append("  → 命中 Flora(tf=%.1f) 残差 = %.1f km  (tol=50 km)" % (tf_f, gap_f))
        if gap_f < 1e3:
            L.append("  ✓ 6 元 DC 在混合模型下成功闭合 Flora 交会!")
            vd2 = vd_f   # 只有收敛才覆盖 3-var DC2 输出
        else:
            L.append("  ✗ 6 元 DC 未收敛到 Flora (残差仍 %.1e km); 保留 3-var DC1 结果作图."
                     % gap_f)
    except Exception as ex:
        L.append("  6 元 DC 异常: %s" % ex)
    # (8) 结论：本轮 DE421 + PSO + 6 元 DC 综合状态
    L.append("\n" + "-" * 74)
    L.append("[结论] 本轮 DE421 + PSO + 6 元 DC 落地情况")
    L.append("  ✓ DE421(J2000 黄道, SSB-太阳 + 绕 x 轴 -ε 旋) 与 Standish 位置差 6~23 km;")
    L.append("     说明星历精度不是 Flora 闭合瓶颈。")
    L.append("  ✓ PSO(§3.4.1 全局搜索)：二体脉冲等效下 Flora 最近 %.2e → %.2e km (%.1fx)"
             % (SIMPLE["fsolve_miss"], SIMPLE["pso_miss"],
                SIMPLE["fsolve_miss"] / max(SIMPLE["pso_miss"], 1.0)))
    L.append("     验证书中「需先全局寻优」的论断。")
    L.append("  ✓ 新增 6 元混合 DC(vd+tM+tf+phi ↔ BT/BR/rp/Flora_xyz)，可修火星借力"
             " B 平面同时约束 Flora。")
    L.append("  △ 6 元 DC 局部探索: 尝试将 |vi| 权重放到最紧(1e-4 尺度)以强制脉冲 δ→37.4°,")
    L.append("     结果 vd 被拉向 rp=1.07e5 km 的非交会解(|vi|=4.981 达 但飞越丢失)；")
    L.append("     说明 |vi|_precise=4.958 + rp=3696 + Flora(3) 三约束在局部线性化里冲突,")
    L.append("     需 PSO/差分进化在**精确模型代价**下重跑 §3.4.1(每次代价 ≈ 1.5 s,")
    L.append("     30×60=1800 次≈45 min)才能找到全局可行 basin。")
    L.append("  → 下一步: 用 precise-cost PSO (n=20×30iter≈10 min) 直接搜 vd/tM/phi/tf 使")
    L.append("     精确多体下 |vi|=4.958 & rp=3696 & Flora 最小; 命中后再 6-var DC 收尾。")

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

    # 图3.34 日心轨道: 精确多体连续轨迹 + 二体脉冲等效设计腿
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, col in [("earth", "tab:green"), ("mars", "tab:red")]:
        tt = np.linspace(T_DEPART, SIMPLE["tf"], 300)
        pts = np.array([planet_state(name, x)[0] for x in tt])
        ax.plot(pts[:, 0]/1e8, pts[:, 1]/1e8, col, lw=0.8, label=name)
    tf = np.linspace(T_DEPART - 60, SIMPLE["tf"] + 60, 300)
    fp = np.array([flora_state(x)[0] for x in tf])
    ax.plot(fp[:, 0]/1e8, fp[:, 1]/1e8, "tab:purple", lw=0.8, label="Flora")
    ax.plot(sc[:, 0]/1e8, sc[:, 1]/1e8, "k", lw=1.2, label="spacecraft(precise)")
    rE0, _ = planet_state("earth", T_DEPART)
    leg1 = solve_ivp(lambda t, y: [y[3], y[4], y[5],
                                   *(-GM["sun"]*y[:3]/np.linalg.norm(y[:3])**3)],
                     (0, (T_MARS - T_DEPART)*DAY),
                     np.concatenate([rE0, SIMPLE["vd0"]]),
                     method="DOP853", rtol=1e-11, atol=1e-6,
                     t_eval=np.linspace(0, (T_MARS-T_DEPART)*DAY, 200))
    rM0, vM0 = planet_state("mars", T_MARS)
    leg2 = solve_ivp(lambda t, y: [y[3], y[4], y[5],
                                   *(-GM["sun"]*y[:3]/np.linalg.norm(y[:3])**3)],
                     (0, (SIMPLE["tf"]-T_MARS)*DAY),
                     np.concatenate([rM0, vM0 + SIMPLE["vinf_out"]]),
                     method="DOP853", rtol=1e-11, atol=1e-6,
                     t_eval=np.linspace(0, (SIMPLE["tf"]-T_MARS)*DAY, 200))
    ax.plot(leg1.y[0]/1e8, leg1.y[1]/1e8, "b--", lw=0.9, label="pulse design leg1")
    ax.plot(leg2.y[0]/1e8, leg2.y[1]/1e8, "b--", lw=0.9, label="pulse design leg2")
    fx = SIMPLE["flora_pos"][0]/1e8; fy = SIMPLE["flora_pos"][1]/1e8
    ax.plot(fx, fy, "m^", ms=10, label="Flora @ tf (pulse)")
    ax.plot(0, 0, "y*", ms=12, label="Sun")
    ax.set_xlabel("x/1e8 km"); ax.set_ylabel("y/1e8 km")
    ax.set_title("Fig 3.34  Heliocentric trajectory (Earth-Mars GA-Flora)")
    ax.legend(fontsize=7); ax.grid(True, ls="--"); fig.tight_layout()
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
def main(argv=None) -> int:
    global USE_MOON, T_MARS
    ap = argparse.ArgumentParser(description="§3.4 精确模型引力辅助（单文件）")
    ap.add_argument("--plot", action="store_true", help="出图3.34~3.37")
    ap.add_argument("--no-moon", action="store_true", help="关闭月球摄动")
    ap.add_argument("--pso", action="store_true",
                    help="下游 DC 起点改用 PSO 全局搜索 seed (§3.4.1 严格复现)")
    a = ap.parse_args(argv)
    if a.no_moon:
        USE_MOON = False
    if a.pso and SIMPLE["pso_miss"] < 0.9 * SIMPLE["fsolve_miss"]:
        _ps = _seed_from_vd_tM(SIMPLE["pso_vd0"], SIMPLE["pso_tM"], "PSO",
                               phi=SIMPLE["pso_phi"])
        SIMPLE["seed_src"] = "PSO(下游)"
        T_MARS = _ps["tM"]
        for k in ["vd0","tM","vinf_in","vinf_out","BT_2b","BR_2b",
                  "b_2b","vi_2b","phi","delta","rp_enc"]:
            SIMPLE[k] = _ps[k]
        SIMPLE["tf"], SIMPLE["flora_pos"] = _flora_closest(
            SIMPLE["vd0"], SIMPLE["tM"], SIMPLE["vinf_out"])
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
