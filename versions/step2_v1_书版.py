#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火星探测器 · 火心段 B 平面设计与精确模型差分修正（单文件）
==========================================================

对应书 §1.2.3（火心段参数设计 / B 平面）与 §1.3（精确模型下轨道参数修正）。

链路：
  A. 初步设计（圆锥拼接，§1.2.3，式 1.13~1.17）
       由表1.3 的 V∞2 与目标倾角 i2、近火点半径 r_p2 → 标称 B^nom = [BT,BR,τ]
  B. 精确模型（§1.3，式 1.18~1.33）
       日心 N 体传播（太阳 + 金/地/火/木/土，间接第三体项）
       终端状态 → 实际 B 平面 (式1.21~1.30)
       数值雅可比 K = ∂[BT,BR,τ]/∂v0 (式1.31)
       差分修正 Δv0 = K⁻¹(B^nom − F(v0)) (式1.33)，牛顿迭代到残差→0

用法：
    python 2.py                      # 内置 2015 窗口1（表1.3/1.4 数据）
    python 2.py --i2 95 --rp2 3800  # 改目标火心倾角/近火点高度(km)
    python 2.py --no-correct         # 只出初步设计，不做差分修正

只依赖 numpy（差分修正需 scipy 的 solve_ivp；无 scipy 时自动降级为
固定步长 RK4 积分）。
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
MU_SUN = 1.32712440018e11          # km^3/s^2
MU_MARS = 42828.37
AU = 1.495978707e8
J2000 = datetime(2000, 1, 1, 12, 0, 0)
R_MARS = 3396.2                    # 火星平均半径 km

# 火星自转轴（北极）方向：书式 1.14~1.16 的 N̂ 与火心倾角 i 均以"火星赤道系"
# 为基准，不是黄道系。IAU 火星北极 α=327.8°, δ=+17.64°（J2000 赤道系），
# 绕 x 轴转 ε=23.43929° 换到 J2000 黄道系，与本机日心传播同一坐标系。
_OBLIQUITY = math.radians(23.43929)
_MARS_POLE_RA, _MARS_POLE_DEC = math.radians(327.8), math.radians(17.64)
_pole_eq = np.array([math.cos(_MARS_POLE_DEC) * math.cos(_MARS_POLE_RA),
                     math.cos(_MARS_POLE_DEC) * math.sin(_MARS_POLE_RA),
                     math.sin(_MARS_POLE_DEC)])
# v_ecl = Rx(ε)·v_eq
N_MARS = np.array([_pole_eq[0],
                   _pole_eq[1] * math.cos(_OBLIQUITY) - _pole_eq[2] * math.sin(_OBLIQUITY),
                   _pole_eq[1] * math.sin(_OBLIQUITY) + _pole_eq[2] * math.cos(_OBLIQUITY)])
N_MARS = N_MARS / np.linalg.norm(N_MARS)
# 书 §1.2.3：B 平面基的参考矢量 N̂ 取该坐标系 Z 轴（黄极）→ 火心倾角以黄道为基准
N_REF = np.array([0.0, 0.0, 1.0])

# Standish(1992) 日心黄道根数 [1800-2050]：(a,e,I,L,wbar,Om) + 每世纪速率
# 用于 N 体摄动（金、地、火、木、土）
_PLANETS = {
    "venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
              324859.0),
    "earth": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0),
              398600.4),
    "mars": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
             MU_MARS),
    "jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
                126686534.0),
    "saturn": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.60405805, 113.66344615),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794),
               37931187.0),
}


# ---------------------------------------------------------------------------
# 日心历表（自 J2000 的秒）
# ---------------------------------------------------------------------------
def helio_state(name, t_sec):
    (a, e, I, L, wb, Om), (da, de, dI, dL, dw, dO), _mu = _PLANETS[name]
    Tc = (t_sec / 86400.0) / 36525.0
    a += da * Tc; e += de * Tc; I += dI * Tc
    L += dL * Tc; wb += dw * Tc; Om += dO * Tc
    a *= AU
    M = math.radians((L - wb) % 360.0)
    E = M
    for _ in range(60):
        d = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-13:
            break
    n = math.sqrt(MU_SUN / a ** 3)
    Ed = n / (1 - e * math.cos(E))
    xv = a * (math.cos(E) - e); yv = a * math.sqrt(1 - e * e) * math.sin(E)
    vx = -a * math.sin(E) * Ed; vy = a * math.sqrt(1 - e * e) * math.cos(E) * Ed
    w = math.radians(wb - Om)
    cw, sw = math.cos(w), math.sin(w)
    cO, sO = math.cos(math.radians(Om)), math.sin(math.radians(Om))
    cI, sI = math.cos(math.radians(I)), math.sin(math.radians(I))
    P = np.array([cw * cO - sw * sO * cI, cw * sO + sw * cO * cI, sw * sI])
    Q = np.array([-sw * cO - cw * sO * cI, -sw * sO + cw * cO * cI, cw * sI])
    return xv * P + yv * Q, vx * P + vy * Q


def _stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z); return (1 - math.cos(s)) / z, (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z); return (1 - math.cosh(s)) / z, (math.sinh(s) - s) / s ** 3
    return 1 / 2 - z / 24 + z * z / 720, 1 / 6 - z / 120 + z * z / 5040


def lambert(r1, r2, tof, mu=MU_SUN):
    R1 = float(np.linalg.norm(r1)); R2 = float(np.linalg.norm(r2))
    cd = float(r1 @ r2) / (R1 * R2)
    cz = float(np.cross(r1, r2)[2])
    dnu = math.acos(max(-1.0, min(1.0, cd)))
    if cz < 0:
        dnu = 2 * math.pi - dnu
    A = math.sin(dnu) * math.sqrt(R1 * R2 / (1 - cd))
    lo, hi = -4 * math.pi ** 2 + 1e-8, 4 * math.pi ** 2
    for _ in range(220):
        z = 0.5 * (lo + hi)
        C, S = _stumpff(z)
        y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
        if y < 0:
            lo = z; continue
        x = math.sqrt(y / C)
        tt = (x ** 3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        if tt < tof:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    C, S = _stumpff(z)
    y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
    f = 1 - y / R1
    g = A * math.sqrt(y / mu)
    gd = 1 - y / R2
    return (np.array(r2) - f * np.array(r1)) / g, (gd * np.array(r2) - np.array(r1)) / g


# ---------------------------------------------------------------------------
# (B) 日心 N 体传播：探测器受太阳 + 第三体（金/木/土）间接摄动
#     地球、火星不计入：出发/到达剩余速度 V∞1、V∞2 已含两体逃逸，
#     且探测器起点在地球、终点在火星附近，施其直接引力项会奇异。
#     这正是 §1.3 的"日心段只受太阳引力 + 行星球引力"模型。
# ---------------------------------------------------------------------------
_PERTURBERS = [p for p in _PLANETS if p not in ("earth", "mars")]


def _accel(t, y):
    r = y[:3]; v = y[3:]
    a = -MU_SUN * r / np.linalg.norm(r) ** 3
    for name in _PERTURBERS:
        rp, _ = helio_state(name, t)
        d = rp - r
        rn = max(float(np.linalg.norm(d)), 1e-3)
        rr = float(np.linalg.norm(rp))
        mu = _PLANETS[name][2]
        a += mu * (d / rn ** 3 - rp / rr ** 3)
    return np.concatenate([v, a])


def _rk4(t0, y0, t1, n_steps):
    h = (t1 - t0) / n_steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = _accel(t, y)
        k2 = _accel(t + h / 2, y + h / 2 * k1)
        k3 = _accel(t + h / 2, y + h / 2 * k2)
        k4 = _accel(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def propagate(t0, y0, t1):
    """日心状态 [r;v] 从 t0 积分到 t1（自 J2000 秒），返回 (终态, dense_sol)"""
    if _HAVE_SCIPY:
        sol = solve_ivp(_accel, (t0, t1), y0, rtol=1e-11, atol=1e-6,
                        method="DOP853", dense_output=True)
        return sol.y[:, -1], sol
    n = int((t1 - t0) / (6 * 3600)) + 1
    y = _rk4(t0, y0, t1, n)
    return y, None


R_SOI_MARS = 5.7e5   # 火星影响球半径 km


def _rel_mars(sol, t):
    """t 时刻相对火星的位置/速度（日心系）"""
    if sol is not None:
        y = sol.sol(t)
    else:
        raise RuntimeError("无 dense 解")
    rm, vm = helio_state("mars", t)
    return y[:3] - rm, y[3:] - vm


def arrival_bplane(t0, r0, v0, t1):
    """传播到 t1 附近，在接近段火星相对距离最小处评估 B 平面。

    BT、BR 沿整条接近双曲线为常量，故在任意非退化点评估都一样；取
    最近接近点最稳（不依赖 SOI 下穿求根，扰动轨迹也不会失效）。
    τ 换算成"从终端时刻 t1 到近火点"的时间（书 τ^nom=0 即近火点=t1）。
    """
    # 接近段窗口 [t1-10d, t1+2d]：积分需覆盖到窗口上界，否则 dense 外插失真
    dt = 24 * 3600.0
    t_scan_end = t1 + 2 * dt
    _, sol = propagate(t0, np.concatenate([r0, v0]), t_scan_end)
    if sol is None:
        y = propagate(t0, np.concatenate([r0, v0]), t1)[0]
        rm, vm = helio_state("mars", t1)
        return bplane_from_state(y[:3] - rm, y[3:] - vm)
    # 在接近段窗口内粗扫找最近接近点，再局部细化
    lo, hi = t1 - 10 * dt, t1 + 2 * dt
    ts = np.linspace(lo, hi, 400)
    dmin = 1e30
    t_ca = t1
    for t in ts:
        rr = float(np.linalg.norm(_rel_mars(sol, t)[0]))
        if rr < dmin:
            dmin = rr
            t_ca = t
    a, b = t_ca - (ts[1] - ts[0]), t_ca + (ts[1] - ts[0])
    for _ in range(60):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if float(np.linalg.norm(_rel_mars(sol, m1)[0])) < \
                float(np.linalg.norm(_rel_mars(sol, m2)[0])):
            b = m2
        else:
            a = m1
    t_ca = 0.5 * (a + b)
    dmin = float(np.linalg.norm(_rel_mars(sol, t_ca)[0]))

    # 评估点：真实掠过(最近距<SOI)取入段 SOI 穿越，避免 r→0 退化；
    # 否则(脱靶)取最近接近点。BT/BR 沿整条双曲线为常量，二者等价。
    if dmin < R_SOI_MARS:
        # 在 [t_ca-窗口, t_ca] 内找 r=R_SOI 的入段穿越
        aa, bb = lo, t_ca
        if float(np.linalg.norm(_rel_mars(sol, aa)[0])) < R_SOI_MARS:
            t_eval = t_ca
        else:
            for _ in range(60):
                mm = 0.5 * (aa + bb)
                if float(np.linalg.norm(_rel_mars(sol, mm)[0])) > R_SOI_MARS:
                    aa = mm
                else:
                    bb = mm
            t_eval = 0.5 * (aa + bb)
    else:
        t_eval = t_ca
    rf, vf = _rel_mars(sol, t_eval)
    bp = bplane_from_state(rf, vf)
    tau_from_t1 = bp.tau + (t_eval - t1)
    return BPlane(bp.BT, bp.BR, tau_from_t1, bp.r_p, bp.v_inf, bp.i, bp.Bv, bp.S)


# ---------------------------------------------------------------------------
# (B) 实际 B 平面参数（式 1.21~1.30），全部在火心 J2000 系
# ---------------------------------------------------------------------------
@dataclass
class BPlane:
    BT: float
    BR: float
    tau: float          # 距近火点时间 (s)，>0 表示尚未到近火点
    r_p: float          # 实际近火点半径 km
    v_inf: float
    i: float            # 轨道倾角 deg（相对火星赤道）
    Bv: np.ndarray = None   # B 矢量（日心黄道直角系）
    S: np.ndarray = None     # 渐近线方向 Ŝ


def bplane_from_state(rf, vf, mu=MU_MARS, ref=None):
    """式 1.21~1.30：由火心相对状态求实际 B 平面。

    ref 为参考面法向 N̂，默认取坐标系 Z 轴 N_REF（书§1.2.3：火心倾角以黄道为基准）。
    """
    if ref is None:
        ref = N_REF
    rf = np.asarray(rf, float); vf = np.asarray(vf, float)
    r = float(np.linalg.norm(rf)); v2 = float(vf @ vf)
    energy = 0.5 * v2 - mu / r
    if energy <= 0:
        raise ValueError("相对火星为椭圆/抛物线，无 B 平面")
    v_inf = math.sqrt(2 * energy)
    a = -mu / (2 * energy)
    h = np.cross(rf, vf)
    n2 = h / np.linalg.norm(h)
    e2 = ((v2 - mu / r) * rf - (rf @ vf) * vf) / mu
    e = float(np.linalg.norm(e2))
    p = float(h @ h) / mu
    b = -a * math.sqrt(e * e - 1)
    # 渐近线半角 θb：cos = 1/e，取 (0,90)（书表3 θ1=acos(1/e1)=36.8827° 印证）
    theta_b = math.acos(1.0 / e)
    ct, st = math.cos(theta_b), math.sin(theta_b)
    cross = np.cross(n2, e2)
    S = ct * e2 / np.linalg.norm(e2) + st * cross / np.linalg.norm(cross)
    S = S / np.linalg.norm(S)
    Bvec = b * np.cross(S, n2)
    T = np.cross(S, ref)
    nT = float(np.linalg.norm(T))
    if nT < 1e-12:                      # Ŝ ∥ N̂：BT/BR 退化，给个正交占位基
        tmp = np.array([1.0, 0.0, 0.0]) if abs(ref[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        T = np.cross(S, tmp); nT = float(np.linalg.norm(T))
    T = T / nT
    R = np.cross(S, T)
    BT = float(Bvec @ T); BR = float(Bvec @ R)
    # 真近点角 f
    cosf = (1 / e) * (p / r - 1)
    sinf = math.sqrt(p / mu) * (rf @ vf) / (e * r)
    f = math.atan2(sinf, cosf)
    # 双曲偏近点角 E： cosE=(e+cosf)/(1+e cosf)
    cosE = (e + cosf) / (1 + e * cosf)
    cosE = max(1.0, cosE)
    sign = 1.0 if math.sin(f) >= 0 else -1.0
    E = sign * math.acosh(cosE)
    sinhH = math.sinh(E)
    # 双曲平近点角 M = e·sinh H − H（入段 H<0 ⇒ M<0），且 M = n(t − t_p)
    # ⇒ 距近火点时间 τ = t_p − t = −M/n = −√(−a³/μ)(e·sinhH − H)，入段为正
    tau = -math.sqrt(-a ** 3 / mu) * (e * sinhH - E)
    i = math.degrees(math.acos(max(-1.0, min(1.0, float(n2 @ np.asarray(ref, float))))))
    return BPlane(BT, BR, tau, abs(a) * (e - 1), v_inf, i, Bvec, S)


# ---------------------------------------------------------------------------
# (B) 差分修正：牛顿迭代 Δv0 = K⁻¹ (B^nom − F(v0))（式 1.31~1.33）
# ---------------------------------------------------------------------------
@dataclass
class Correction:
    v0: np.ndarray
    F: BPlane
    Bnom: tuple
    iters: int
    resid: float


def differential_correction(t0, r0, v0, t1, Bnom_vec, dv_eps=1e-3,
                            tol=1e-6, max_iter=60):
    """牛顿-莱文伯格迭代求 Δv0 = K⁻¹(B^nom − F(v0))（式1.31~1.33）。

    目标 [BT, BR, τ]，BT/BR 按式 1.21~1.30 在"当前 Ŝ 的 B 平面基"上定义，
    因 Bv ⟂ Ŝ 恒成立，故 BT²+BR²=‖Bv‖²=b²=‖B‖²，等价于同时锁定 ‖B‖ → r_p。
    若改用固定标称基做投影，则 Bv 沿 Ŝ0 的分量不受约束，‖B‖ 可以任意大，
    解不物理（近火点半径失控），这也是常见 TPBVIP 实现的一个坑。
    BT/BR(~1e3 km) 与 τ(~1e5 s) 量级差两个数量级：归一化 + LM 阻尼 + 回溯。
    """
    v0 = np.array(v0, float)
    target = np.array(Bnom_vec, float)
    scale = np.array([1e3, 1e3, 1e5])          # BT, BR: km; τ: s

    def F(v):
        return arrival_bplane(t0, r0, v, t1)

    def vec(bp):
        return np.array([bp.BT, bp.BR, bp.tau])

    def cost(v):
        return float(np.linalg.norm((target - vec(F(v))) / scale))

    lam = 1e-2
    c = cost(v0)
    for it in range(1, max_iter + 1):
        f0 = vec(F(v0))
        resid_n = (target - f0) / scale
        if np.linalg.norm(resid_n) < tol:
            return Correction(v0, F(v0), tuple(target), it,
                              float(np.linalg.norm(target - f0)))
        # 归一化数值雅可比
        K = np.zeros((3, 3))
        for j in range(3):
            dp = v0.copy(); dp[j] += dv_eps
            dm = v0.copy(); dm[j] -= dv_eps
            K[:, j] = ((vec(F(dp)) - vec(F(dm))) / scale) / (2 * dv_eps)
        r = (target - f0) / scale
        A = K.T @ K + lam * np.eye(3)
        try:
            dv = np.linalg.solve(A, K.T @ r)
        except np.linalg.LinAlgError:
            dv = np.linalg.lstsq(K, r, rcond=None)[0]
        # 限幅：单步日心速度改动不超过 0.2 km/s
        step = float(np.linalg.norm(dv))
        if step > 0.2:
            dv *= 0.2 / step
        # 回溯线搜索
        ok = False
        for _ in range(20):
            vtry = v0 + dv
            ctry = cost(vtry)
            if ctry < c:
                v0, c = vtry, ctry
                lam = max(lam * 0.5, 1e-8)
                ok = True
                break
            dv *= 0.5
        if not ok:
            lam *= 4.0
            if lam > 1e6:
                break
    bp = F(v0)
    return Correction(v0, bp, tuple(target), max_iter,
                      float(np.linalg.norm(target - vec(bp))))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _jd(dt: datetime) -> float:
    return (dt - J2000).total_seconds() / 86400.0 + 2451545.0


def compute(i2=95.0, rp2_alt=400.0, correct=True,
            dep=None, arr=None):
    # 窗口1（表1.2）：2015-12-25 → 2016-09-29，转移 278 天
    if dep is None:
        dep = datetime(2015, 12, 25, 0, 0, 0)
    if arr is None:
        arr = datetime(2016, 9, 29, 0, 0, 0)
    t0 = (_jd(dep) - 2451545.0) * 86400.0
    t1 = (_jd(arr) - 2451545.0) * 86400.0
    tof = t1 - t0

    # 全程日心黄道 J2000，自洽：
    #   Lambert 给出转移弧两端速度 → V∞1(出发剩余)、V∞2(到达剩余)
    re, ve = helio_state("earth", t0)
    rm, vm = helio_state("mars", t1)
    v1, v2 = lambert(re, rm, tof)
    V_inf1 = v1 - ve                 # 出发地心剩余速度(黄道)
    V_inf2 = v2 - vm                 # 到达火心剩余速度(黄道)
    v2m = float(np.linalg.norm(V_inf2))

    r_p2 = R_MARS + rp2_alt
    # 标称 ||B|| 由 r_p2 与 V∞2（式1.17 第一式）
    Bmag = math.sqrt(r_p2 ** 2 + 2 * r_p2 * MU_MARS / v2m ** 2)
    # 标称 θb：cos i2 = cosθb · cosδ∞2（式1.16）
    # δ∞2 为 Ŝ（≈V∞2 反方向）的黄纬：与 Z 轴 N_REF 夹角的余角（asin(Vz/|V|)=-27.2° 印证）
    dec2 = math.degrees(math.asin(float(np.dot(V_inf2 / v2m, N_REF))))
    cos_th = math.cos(math.radians(i2)) / math.cos(math.radians(dec2))
    cos_th = max(-1.0, min(1.0, cos_th))
    th_deg = math.degrees(math.acos(cos_th))
    BT_nom = Bmag * math.cos(math.radians(th_deg))
    BR_nom = Bmag * math.sin(math.radians(th_deg))

    # 精确模型：以 Lambert 初值 N 体传播 → 实际 B 平面（修正前，SOI 处评估）
    r0 = re
    v0_helio = v1
    bp_unc = arrival_bplane(t0, r0, v0_helio, t1)

    corr = None
    v0_nom = None
    if correct:
        v0_nom = v0_helio.copy()
        corr = differential_correction(t0, r0, v0_helio, t1,
                                       [BT_nom, BR_nom, 0.0])
    return dict(V_inf1=V_inf1, V_inf2=V_inf2, v2=v2m, dec2=dec2, i2=i2,
                r_p2=r_p2, rp2_alt=rp2_alt, Bmag=Bmag, th_deg=th_deg,
                BT_nom=BT_nom, BR_nom=BR_nom, bp_unc=bp_unc, corr=corr,
                v0_nom=v0_nom, tof_days=tof / 86400.0)


def report(r) -> str:
    L = []
    L.append("=" * 70)
    L.append("火心段 B 平面设计 + 精确模型差分修正")
    L.append("=" * 70)
    L.append("[到达剩余速度 V∞2（Lambert，日心黄道自洽）]")
    L.append("  V∞2 = [%+.4f, %+.4f, %+.4f]  |V∞2| = %.4f km/s"
             % (*r["V_inf2"], r["v2"]))
    L.append("  火星赤纬 δ∞2 = %.2f°   转移 = %.0f 天" % (r["dec2"], r["tof_days"]))
    L.append("")
    L.append("[A] 初步设计（式1.17）  i2=%.1f°  近火点高度=%.0f km (r_p2=%.1f km)"
             % (r["i2"], r["rp2_alt"], r["r_p2"]))
    L.append("  ||B|| = %.1f km   θb = %.3f°" % (r["Bmag"], r["th_deg"]))
    L.append("  B^nom = [BT=%.1f, BR=%.1f, τ=0] km" % (r["BT_nom"], r["BR_nom"]))
    L.append("")
    bu = r["bp_unc"]
    L.append("[B] 精确模型 · 圆锥拼接初值（未修正）")
    L.append("  实际 B = [BT=%.1f, BR=%.1f, τ=%.1f] km" % (bu.BT, bu.BR, bu.tau))
    L.append("  实际近火点 r_p=%.1f km (高度 %.0f)  i=%.2f°"
             % (bu.r_p, bu.r_p - R_MARS, bu.i))
    L.append("  → 残差 ΔBT=%.1f  ΔBR=%.1f km（圆锥拼接满足不了火心段，需修正）"
             % (bu.BT - r["BT_nom"], bu.BR - r["BR_nom"]))
    if r["corr"] is not None:
        c = r["corr"]
        L.append("")
        L.append("[B] 差分修正（式1.31~1.33，牛顿迭代 %d 次）" % c.iters)
        L.append("  Lambert 初值 v0(日心) = [%+.4f, %+.4f, %+.4f] km/s" % (*r["v0_nom"],))
        L.append("  修正后   v0(日心) = [%+.4f, %+.4f, %+.4f] km/s" % (*c.v0,))
        L.append("  修正量 Δv0 = %.4f km/s" % float(np.linalg.norm(c.v0 - r["v0_nom"])))
        L.append("  修正后 B = [BT=%.2f, BR=%.2f, τ=%.3f] km"
                 % (c.F.BT, c.F.BR, c.F.tau))
        L.append("  终端残差 |ΔB| = %.3e km   近火点 r_p=%.1f km  i=%.2f°"
                 % (c.resid, c.F.r_p, c.F.i))
    L.append("=" * 70)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="火心段 B 平面设计与精确模型差分修正（单文件）")
    ap.add_argument("--i2", type=float, default=95.0, help="目标火心倾角 deg")
    ap.add_argument("--rp2", type=float, default=400.0, help="近火点高度 km")
    ap.add_argument("--no-correct", action="store_true", help="只出初步设计")
    a = ap.parse_args(argv)
    r = compute(i2=a.i2, rp2_alt=a.rp2, correct=not a.no_correct)
    print(report(r))
    if not _HAVE_SCIPY:
        print("注：未检测到 scipy，使用固定步长 RK4（精度略低）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
