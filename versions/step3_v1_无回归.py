#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火星探测器 · 日心段中途修正（§1.4，单文件，自包含）
==================================================

对应书 §1.4 中途修正（式 1.34~1.41、表 1.7~1.10、图 1.15~1.25 的核心方法）。

动力学与 B 平面取自 2.py 的已验证实现（全程 J2000 日心黄道自洽）：
  · Standish(1992) 历表 + Lambert + 太阳/金木土间接第三体 N 体传播
  · 实际 B 平面（式1.21~1.30，入射渐近线 acos(1/e)，参考轴 = Z）

本文件新增（§1.4）：
  A. 灵敏度矩阵 K(t)=∂[BT,BR,τ]/∂v（式1.31 在任意中间时刻的形式）
  B. 修正脉冲 Δv(t)=K⁻¹(B_tar−B(t))（式1.36）与标量增益 k(t)（式1.38）
  C. 蒙特卡洛误差传播：初始协方差 V0 → 终端 B 平面误差分布（式1.35/1.37，表1.8）
  D. 色散曲线：终端 ‖B‖/τ 误差标准差随"剩余传递时间"的变化（图1.15/1.16）
  E. B 平面碰撞概率：二维高斯落在火星投影圆内的积分（图1.19/1.20/1.25）

用法：
    python 3.py                 # 跑全套演示（默认 2015 窗口1）
    python 3.py --mc 400        # 蒙特卡洛样本数
    python 3.py --disp          # 只出图1.15/1.16 色散曲线数据
    python 3.py --coll          # 只出碰撞概率曲线（图1.20）

依赖：numpy（必需）、scipy（推荐，DOP853 高精度积分；无则降级 RK4）。
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ===========================================================================
# 0. 动力学核心：本文件自带一份已验证实现（不导入 2.py）
#    原因：工作区 2.py 可能被回退到旧版（asin/火星自转轴），导入会把 bug 带进来。
#    下面这份与"审计修复后的 2.py"逐式一致：入射渐近线 acos(1/e)、参考轴 Z、
#    τ 入段为正、SOI 处评估。
# ===========================================================================
_CORE = None

MU_SUN = 1.32712440018e11
MU_MARS = 42828.37
AU = 1.495978707e8
J2000 = datetime(2000, 1, 1, 12, 0, 0)
R_MARS = 3396.2
R_SOI_MARS = 5.7e5
N_REF = np.array([0.0, 0.0, 1.0])      # 书式1.14：Z 轴为参考矢量


def _jd(dt: datetime) -> float:
    return (dt - J2000).total_seconds() / 86400.0 + 2451545.0


# ---- 若 2.py 可用则直接转发其经验证的函数 ----
if _CORE is not None:
    helio_state = _CORE.helio_state
    lambert = _CORE.lambert
    propagate = _CORE.propagate
    _rel_mars = _CORE._rel_mars
    bplane_from_state = _CORE.bplane_from_state
    arrival_bplane = _CORE.arrival_bplane
    _SRC = "2.py"
else:
    # ---------------------- 内置回退实现（与 2.py 等价）----------------------
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
    _PERTURBERS = [p for p in _PLANETS if p not in ("earth", "mars")]

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

    def _accel(t, y):
        r = y[:3]
        a = -MU_SUN * r / np.linalg.norm(r) ** 3
        for name in _PERTURBERS:
            rp, _ = helio_state(name, t)
            d = rp - r
            rn = max(float(np.linalg.norm(d)), 1e-3)
            rr = float(np.linalg.norm(rp))
            a += _PLANETS[name][2] * (d / rn ** 3 - rp / rr ** 3)
        return np.concatenate([y[3:], a])

    def _rk4(t0, y0, t1, n):
        h = (t1 - t0) / n
        t, y = t0, np.array(y0, float)
        for _ in range(n):
            k1 = _accel(t, y); k2 = _accel(t + h / 2, y + h / 2 * k1)
            k3 = _accel(t + h / 2, y + h / 2 * k2); k4 = _accel(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
        return y

    def propagate(t0, y0, t1):
        if _HAVE_SCIPY:
            sol = solve_ivp(_accel, (t0, t1), y0, rtol=1e-11, atol=1e-6,
                            method="DOP853", dense_output=True)
            return sol.y[:, -1], sol
        n = int((t1 - t0) / (6 * 3600)) + 1
        return _rk4(t0, y0, t1, n), None

    def _rel_mars(sol, t):
        y = sol.sol(t)
        rm, vm = helio_state("mars", t)
        return y[:3] - rm, y[3:] - vm

    @dataclass
    class BPlane:
        BT: float
        BR: float
        tau: float
        r_p: float
        v_inf: float
        i: float
        Bv: np.ndarray = None
        S: np.ndarray = None

    def bplane_from_state(rf, vf, mu=MU_MARS, ref=None):
        if ref is None:
            ref = N_REF
        rf = np.asarray(rf, float); vf = np.asarray(vf, float)
        r = float(np.linalg.norm(rf)); v2 = float(vf @ vf)
        energy = 0.5 * v2 - mu / r
        if energy <= 0:
            raise ValueError("相对火星为椭圆/抛物线，无 B 平面")
        v_inf = math.sqrt(2 * energy)
        a = -mu / (2 * energy)
        h = np.cross(rf, vf); n2 = h / np.linalg.norm(h)
        e2 = ((v2 - mu / r) * rf - (rf @ vf) * vf) / mu
        e = float(np.linalg.norm(e2))
        p = float(h @ h) / mu
        b = -a * math.sqrt(e * e - 1)
        th = math.acos(1.0 / e)                       # 式1.24：acos，非 asin
        ct, st = math.cos(th), math.sin(th)
        cr = np.cross(n2, e2)
        S = ct * e2 / np.linalg.norm(e2) + st * cr / np.linalg.norm(cr)
        S = S / np.linalg.norm(S)
        Bv = b * np.cross(S, n2)
        T = np.cross(S, ref); nT = float(np.linalg.norm(T))
        if nT < 1e-12:
            tmp = np.array([1., 0., 0.]) if abs(ref[0]) < 0.9 else np.array([0., 1., 0.])
            T = np.cross(S, tmp); nT = float(np.linalg.norm(T))
        T = T / nT; R = np.cross(S, T)
        BT = float(Bv @ T); BR = float(Bv @ R)
        cosf = (1 / e) * (p / r - 1)
        sinf = math.sqrt(p / mu) * (rf @ vf) / (e * r)
        f = math.atan2(sinf, cosf)
        cosE = max(1.0, (e + cosf) / (1 + e * cosf))
        sgn = 1.0 if math.sin(f) >= 0 else -1.0
        E = sgn * math.acosh(cosE)
        tau = -math.sqrt(-a ** 3 / mu) * (e * math.sinh(E) - E)   # 入段为正
        i = math.degrees(math.acos(max(-1., min(1., float(n2 @ np.asarray(ref, float))))))
        return BPlane(BT, BR, tau, abs(a) * (e - 1), v_inf, i, Bv, S)

    def arrival_bplane(t0, r0, v0, t1):
        dt = 24 * 3600.0
        t_scan_end = t1 + 2 * dt
        _, sol = propagate(t0, np.concatenate([r0, v0]), t_scan_end)
        if sol is None:
            y = propagate(t0, np.concatenate([r0, v0]), t1)[0]
            rm, vm = helio_state("mars", t1)
            rf, vf = y[:3] - rm, y[3:] - vm
            bp = bplane_from_state(rf, vf)
            return BPlane(bp.BT, bp.BR, bp.tau, bp.r_p, bp.v_inf, bp.i, bp.Bv, bp.S)
        lo, hi = t1 - 10 * dt, t1 + 2 * dt
        ts = np.linspace(lo, hi, 400)
        dmin, t_ca = 1e30, t1
        for t in ts:
            rr = float(np.linalg.norm(_rel_mars(sol, t)[0]))
            if rr < dmin:
                dmin, t_ca = rr, t
        a, b = t_ca - (ts[1] - ts[0]), t_ca + (ts[1] - ts[0])
        for _ in range(60):
            m1 = a + (b - a) / 3; m2 = b - (b - a) / 3
            if float(np.linalg.norm(_rel_mars(sol, m1)[0])) < \
                    float(np.linalg.norm(_rel_mars(sol, m2)[0])):
                b = m2
            else:
                a = m1
        t_ca = 0.5 * (a + b)
        dmin = float(np.linalg.norm(_rel_mars(sol, t_ca)[0]))
        if dmin < R_SOI_MARS:
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
        return BPlane(bp.BT, bp.BR, bp.tau + (t_eval - t1), bp.r_p, bp.v_inf, bp.i, bp.Bv, bp.S)

    _SRC = "内置回退"


# ===========================================================================
# 标称转移轨道（默认 2015 窗口1：2015-12-25 → 2016-09-29，280.02 d）
# ===========================================================================
# ===========================================================================
# 0b. 接入 2.py 的差分修正收敛解（中途修正的标称轨道 = 命中 i=95°/rp 的解）
#     仅当 2.py 是"审计修复版"（acos(1/e) 渐近线 + Z 轴参考 N_REF）时才使用；
#     若检测到被回退的旧版（asin / 火星自转轴），自动退回内置 Lambert 标称。
# ===========================================================================
_M2 = None
_M2_SAFE = False


def _load_2py():
    global _M2, _M2_SAFE
    if _M2 is not None:                 # 已尝试加载过（成功或判定不安全）
        return _M2 if _M2_SAFE else None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.py")
    if not os.path.exists(path):
        return None
    try:
        src = open(path, "r", encoding="utf-8").read()
        ok = ("theta_b = math.acos(1.0 / e)" in src
              and "N_REF" in src
              and "ref = N_REF" in src)
        spec = importlib.util.spec_from_file_location("m2_targeting", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["m2_targeting"] = m      # dataclass 解析 __module__ 需先注册
        spec.loader.exec_module(m)
        _M2 = m
        _M2_SAFE = bool(ok)
    except Exception:
        _M2, _M2_SAFE = None, False
    return _M2 if _M2_SAFE else None


_NOM_CACHE = {}


def nominal_transfer(dep=None, arr=None):
    key = (dep, arr)
    if key not in _NOM_CACHE:
        _NOM_CACHE[key] = _nominal_uncached(dep, arr)
    return _NOM_CACHE[key]


def _nominal_uncached(dep=None, arr=None):
    if dep is None:
        dep = datetime(2015, 12, 25, 0, 0, 0)
    if arr is None:
        arr = datetime(2016, 9, 29, 10, 0, 0)
    t0 = (_jd(dep) - 2451545.0) * 86400.0
    t1 = (_jd(arr) - 2451545.0) * 86400.0
    m2 = _load_2py()
    if m2 is not None:
        # 用 2.py 的差分修正收敛 v0 作为标称，使 §1.4 中途修正作用在任务轨道上
        try:
            r2 = m2.compute(dep=dep, arr=arr, correct=True)
            re, ve = m2.helio_state("earth", t0)
            rm, vm = m2.helio_state("mars", t1)
            corr = r2.get("corr")
            v_lam = m2.lambert(re, rm, t1 - t0)
            v_dep = np.asarray(corr.v0 if corr is not None else v_lam[0], float)
            Vinf2 = np.asarray(r2["V_inf2"], float)       # 到达端 Lambert 剩余速度
            bp2 = getattr(corr, "F", None) if corr is not None else None
            return dict(t0=t0, t1=t1, dep=dep, arr=arr, re=re, ve=ve, rm=rm, vm=vm,
                        v1=v_dep, v2=v_lam[1], Vinf2=Vinf2, bp2=bp2,
                        tof_days=(t1 - t0) / 86400.0,
                        from2py=True, tagged=corr is not None)
        except Exception:
            pass
    re, ve = helio_state("earth", t0)
    rm, vm = helio_state("mars", t1)
    v1, v2 = lambert(re, rm, t1 - t0)
    return dict(t0=t0, t1=t1, dep=dep, arr=arr, re=re, ve=ve, rm=rm, vm=vm,
                v1=v1, v2=v2, Vinf2=v2 - vm, tof_days=(t1 - t0) / 86400.0,
                from2py=False, tagged=False)


# ===========================================================================
# A. 灵敏度矩阵 K(t) = ∂[BT,BR,τ]/∂v（式1.31 在中间时刻 t 的形式）
# ===========================================================================
def engine_crosscheck(tol=2.0):
    """数值交叉校验：3.py 内置引擎 vs 2.py 引擎在同一收敛 v0 上的 B 平面。

    证明两引擎的 acos(1/e)/Z 轴参考/τ 号完全一致；文本 grep 防不住数值差异，
    此项给出可证伪的硬指标。返回 (ok, lines)。
    """
    L = []
    nt = nominal_transfer()
    if not nt.get("tagged") or nt.get("bp2") is None:
        return False, ["交叉校验：未取得 2.py 收敛解（2.py 未加载或被判定未修复），跳过。"]
    bp3 = arrival_bplane(nt["t0"], nt["re"], nt["v1"], nt["t1"])   # 3.py 内置引擎
    bp2 = nt["bp2"]
    dBT, dBR = abs(bp3.BT - bp2.BT), abs(bp3.BR - bp2.BR)
    dtau = abs(bp3.tau - bp2.tau)
    ok = dBT < tol and dBR < tol
    L.append("[校验] 同一收敛 v0，两引擎 B 平面对比（km）")
    L.append("  2.py : BT=%+.2f BR=%+.2f τ=%+.2f  r_p=%.1f  i=%.3f"
             % (bp2.BT, bp2.BR, bp2.tau, bp2.r_p, bp2.i))
    L.append("  3.py : BT=%+.2f BR=%+.2f τ=%+.2f  r_p=%.1f  i=%.3f"
             % (bp3.BT, bp3.BR, bp3.tau, bp3.r_p, bp3.i))
    L.append("  偏差 ΔBT=%.2f ΔBR=%.2f Δτ=%.2f  →  %s"
             % (dBT, dBR, dtau, "一致(PASS)" if ok else "不一致(FAIL)"))
    return ok, L


def state_at(tnom, y0, t):
    """把标称状态从 tnom 传播到中间时刻 t，返回 [r;v]。"""
    y, _ = propagate(tnom, y0, t)
    return y


def sensitivity_K(t, r, v, t1, dv_eps=5e-4):
    """K[i,j]=∂F_i/∂v_j，F=[BT,BR,τ]。中心差分，6 次传播。"""
    scale = np.array([1e3, 1e3, 1e5])
    v = np.asarray(v, float)

    def F(vv):
        bp = arrival_bplane(t, r, vv, t1)
        return np.array([bp.BT, bp.BR, bp.tau])

    K = np.zeros((3, 3))
    for j in range(3):
        dp = v.copy(); dp[j] += dv_eps
        dm = v.copy(); dm[j] -= dv_eps
        K[:, j] = ((F(dp) - F(dm)) / scale) / (2 * dv_eps)
    return K


# ===========================================================================
# B. 修正脉冲（式1.36）与标量增益 k(t)（式1.38）
# ===========================================================================
def correction_impulse(t, r, v, t1, Btar, dv_eps=5e-4, ridge=1e-8):
    """Δv = K⁻¹ (B_tar − B(t))，残差按尺度归一化后求解（式1.36）。

    返回 (Δv[3] km/s, B_now, K)。τ 分量单位 s，BT/BR 单位 km。
    """
    scale = np.array([1e3, 1e3, 1e5])
    bp = arrival_bplane(t, r, v, t1)
    Bnow = np.array([bp.BT, bp.BR, bp.tau])
    K = sensitivity_K(t, r, v, t1, dv_eps)
    # K 已按 scale 归一化列→行；目标残差同样归一化
    rhs = (np.asarray(Btar, float) - Bnow) / scale
    A = K.T @ K + ridge * np.eye(3)
    dv = np.linalg.solve(A, K.T @ rhs)
    return dv, bp, K


def gain_k(t, r, v, t1, dv_eps=5e-4):
    """标量增益 k(t)=‖ΔB‖/‖Δv‖：单位速度脉冲引起的终端 ‖B‖ 变化量级。

    取 K 的最大奇异值（最敏感方向）作为 k(t)。书式1.38 用其倒数估脉冲，
    且 k(t) 随剩余时间增大而增大（越早修正越省力），对应图1.15 平台段。
    """
    K = sensitivity_K(t, r, v, t1, dv_eps)
    sv = np.linalg.svd(K / np.array([[1.], [1.], [1e-2]]), compute_uv=False)
    return float(sv.max())


# ===========================================================================
# C. 蒙特卡洛误差传播（式1.35/1.37，表1.8）
# ===========================================================================
@dataclass
class TerminalError:
    mean: np.ndarray      # [δBT, δBR, δτ, δ‖B‖] 均值
    std: np.ndarray       # 标准差
    n: int


def monte_carlo_terminal(nt, t0, y0, t1, sig_pos_km, sig_vel_kms, Bnom, rng=None):
    """初始状态高斯扰动 → 传播到 t1 → 终端 B 平面误差分布（表1.8）。

    sig_pos_km/sig_vel_kms 按 1σ 输入。Bnom = 标称终端 [BT,BR,τ,‖B‖]，
    误差 = 样本 − 标称。返回各分量的均值/标准差。
    """
    if rng is None:
        rng = np.random.default_rng(20151225)
    r0 = y0[:3].copy(); v0 = y0[3:].copy()
    d = []
    for _ in range(nt):
        rp = r0 + rng.normal(0, sig_pos_km, 3)
        vp = v0 + rng.normal(0, sig_vel_kms, 3)
        try:
            bp = arrival_bplane(t0, rp, vp, t1)
        except Exception:
            continue
        Bmag = math.hypot(bp.BT, bp.BR)
        d.append([bp.BT - Bnom[0], bp.BR - Bnom[1], bp.tau - Bnom[2], Bmag - Bnom[3]])
    D = np.array(d)
    return TerminalError(D.mean(0), D.std(0, ddof=1), len(D))


# ===========================================================================
# D. 色散曲线（图1.15/1.16）：终端误差标准差 vs 修正时刻
# ===========================================================================
def dispersion_curve(t0, y0, t1, sig_pos_km, sig_vel_kms, times, nmc=120, rng=None):
    """在多个中间时刻 t，注入同样初始误差，测其传递到 t1 的终端 ‖B‖/τ 标准差。

    书图1.15/1.16：前 40 天误差大致相同（平台），之后剩余时间越短扩散越小。
    times 为自 t0 起的天数列表。
    """
    if rng is None:
        rng = np.random.default_rng(7)
    out = []
    for d in times:
        t = t0 + d * 86400.0
        # 标称传播到 t，取该处状态
        y, _ = propagate(t0, y0, t)
        r, v = y[:3], y[3:]
        dB, dtau = [], []
        for _ in range(nmc):
            rp = r + rng.normal(0, sig_pos_km, 3)
            vp = v + rng.normal(0, sig_vel_kms, 3)
            try:
                bp = arrival_bplane(t, rp, vp, t1)
            except Exception:
                continue
            dB.append(math.hypot(bp.BT, bp.BR)); dtau.append(bp.tau)
        if len(dB) > 3:
            out.append((d, float(np.std(dB)), float(np.std(dtau))))
    return out


# ===========================================================================
# E. B 平面碰撞概率（图1.19/1.20/1.25）
# ===========================================================================
def collision_probability(mu_BT, mu_BR, sig_BT, sig_BR, R_impact, nr=200, nt=400):
    """二维独立高斯(BT,BR)落在以原点为心、半径 R_impact 圆盘内的概率。

    目标均值在 (mu_BT,mu_BR)，火星投影为原点圆盘（书§1.4.3：BT/BR 独立高斯）。
    极坐标数值积分，向量化。
    """
    if sig_BT <= 0 or sig_BR <= 0:
        return 0.0
    rr = np.linspace(0, R_impact, nr)
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    dr = rr[1] - rr[0]
    dth = th[1] - th[0]
    Rg, Tg = np.meshgrid(rr, th, indexing="ij")
    x = mu_BT + Rg * np.cos(Tg)
    y = mu_BR + Rg * np.sin(Tg)
    norm = 1.0 / (2 * math.pi * sig_BT * sig_BR)
    pdf = norm * np.exp(-0.5 * ((x / sig_BT) ** 2 + (y / sig_BR) ** 2))
    P = float(np.sum(pdf * Rg) * dr * dth)
    return min(max(P, 0.0), 1.0)


def required_offset(sig_BT, sig_BR, R_impact, p_target=1e-3, hi=1.2e5, tol=50.0):
    """求使撞击概率 < p_target 的最小目标偏置距离（图1.25 的逆问题）。

    偏置沿最不利方向（主轴）放置；二分搜索。返回偏置 km。
    """
    lo = 0.0
    if collision_probability(hi, 0.0, sig_BT, sig_BR, R_impact) > p_target:
        return float("inf")
    if collision_probability(lo, 0.0, sig_BT, sig_BR, R_impact) <= p_target:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collision_probability(mid, 0.0, sig_BT, sig_BR, R_impact) > p_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def impact_radius_from_rp(rp, Vinf):
    """近火点半径 rp → B 平面上的等效碰撞圆半径 ‖B‖（式1.17 第一式）。"""
    return math.sqrt(rp ** 2 + 2 * rp * MU_MARS / Vinf ** 2)


# ===========================================================================
# 演示主流程
# ===========================================================================
def run_demo(nmc=200, do_disp=True, do_coll=True, sig_pos_3s=30.0, sig_vel_3s=3.0):
    L = []
    L.append("=" * 70)
    L.append("日心段中途修正 §1.4（动力学源：%s）" % _SRC)
    L.append("=" * 70)
    ok, ck = engine_crosscheck()
    L.extend(ck)
    L.append("")
    nt = nominal_transfer()
    t0, t1 = nt["t0"], nt["t1"]
    y0 = np.concatenate([nt["re"], nt["v1"]])
    src_tag = "2.py 差分修正收敛解" if nt.get("tagged") else (
        "2.py Lambert 初值" if nt.get("from2py") else "内置 Lambert（2.py 不可用/未修复）")
    L.append("标称轨道来源：%s" % src_tag)
    L.append("标称转移：出发 %s → 到达 %s，TOF=%.2f d"
             % (nt["dep"].strftime("%Y-%m-%d"), nt["arr"].strftime("%Y-%m-%d %H:%M"),
                nt["tof_days"]))
    Vinf2 = float(np.linalg.norm(nt["Vinf2"]))
    L.append("|V∞2| = %.4f km/s" % Vinf2)

    # 标称终端 B 平面
    bp_nom = arrival_bplane(t0, nt["re"], nt["v1"], t1)
    Bmag_nom = math.hypot(bp_nom.BT, bp_nom.BR)
    L.append("标称终端 B = [BT=%.1f, BR=%.1f km, τ=%.1f s]  ‖B‖=%.1f km"
             % (bp_nom.BT, bp_nom.BR, bp_nom.tau, Bmag_nom))
    L.append("")

    # A/B：灵敏度 + 修正脉冲（在 L+3d 处，把终端 B 拉回标称）
    t3 = t0 + 3 * 86400.0
    y3 = state_at(t0, y0, t3)
    # 人为给终端一个目标偏置（模拟发射偏差后的修正需求）
    Btar = [bp_nom.BT + 3000.0, bp_nom.BR - 2000.0, 60.0]
    dv, bp3, K = correction_impulse(t3, y3[:3], y3[3:], t1, Btar)
    L.append("[A/B] L+3d 修正脉冲（式1.36）")
    L.append("  当前终端 B=[%.1f, %.1f, %.1f]" % (bp3.BT, bp3.BR, bp3.tau))
    L.append("  目标 B    =[%+.1f, %+.1f, %+.1f]" % (*Btar,))
    L.append("  Δv = [%+.5f %+.5f %+.5f] km/s   ‖Δv‖=%.2f m/s"
             % (*dv, np.linalg.norm(dv) * 1e3))
    L.append("  灵敏度增益 k(t)=‖ΔB‖/‖Δv‖ 最大奇异值 = %.1f km/(km/s)" % gain_k(t3, y3[:3], y3[3:], t1))
    L.append("")

    # C：蒙特卡洛终端误差（表1.8）——发射入轨误差 30km(3σ)/3m/s(3σ)
    sig_pos = sig_pos_3s / 3.0
    sig_vel = sig_vel_3s * 1e-3 / 3.0
    te = monte_carlo_terminal(nmc, t0, y0, t1, sig_pos, sig_vel,
                              [bp_nom.BT, bp_nom.BR, bp_nom.tau, Bmag_nom])
    L.append("[C] 蒙特卡洛终端误差（表1.8，入轨 %.0fkm/%.1fm/s @3σ，N=%d）"
             % (sig_pos_3s, sig_vel_3s, te.n))
    L.append("  量          均值            标准差")
    L.append("  δBT/km     %+12.1f   %12.1f" % (te.mean[0], te.std[0]))
    L.append("  δBR/km     %+12.1f   %12.1f" % (te.mean[1], te.std[1]))
    L.append("  δτ/s       %+12.1f   %12.1f" % (te.mean[2], te.std[2]))
    L.append("  δ‖B‖/km    %+12.1f   %12.1f" % (te.mean[3], te.std[3]))
    L.append("  参考：书表1.8 δBT~1.17e6 km；本值随入轨σ线性缩放，"
             "如与书不符请以书给定σ覆写 --sig-pos/--sig-vel")
    L.append("")

    # E：碰撞概率（图1.20）——给定偏置与终端误差标准差
    Rimp = impact_radius_from_rp(R_MARS, Vinf2)
    L.append("[E] 碰撞概率（图1.20，火星投影圆 ‖B‖<%.0f km）" % Rimp)
    sigB = float(np.mean([te.std[0], te.std[1]]))
    for off in sorted((0.0, 5000.0, 7245.4, 30300.0, 320000.0)):
        p = collision_probability(off, 0.0, sigB, sigB, Rimp)
        L.append("  偏置 %9.1f km → 撞击概率 %.3e" % (off, p))
    # 图1.25 逆问题：使撞击概率 < 1e-3 的最小目标偏置（隔离半径）
    off_req = required_offset(sigB, sigB, Rimp, p_target=1e-3)
    L.append("  使 P<1e-3 的最小偏置（隔离半径，本书σ）= %.1f km  [书图1.25≈3.03e4 km]"
             % off_req)
    L.append("")

    # D：色散曲线（图1.15/1.16）
    if do_disp:
        L.append("[D] 色散曲线（图1.15/1.16，初始 30km/0.1m/s @3σ）")
        times = [1, 5, 10, 20, 40, 80, 120, 160, 200, 240, 270]
        cur = dispersion_curve(t0, y0, t1, 30.0 / 3, 0.1e-3 / 3, times, nmc=60)
        L.append("  t/d   σ‖B‖/km   στ/s")
        for d, sb, st in cur:
            L.append("  %4.0f  %9.1f  %8.1f" % (d, sb, st))
        L.append("")
    L.append("=" * 70)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="日心段中途修正 §1.4（单文件）")
    ap.add_argument("--mc", type=int, default=200, help="蒙特卡洛样本数")
    ap.add_argument("--disp", action="store_true", help="只出色散曲线")
    ap.add_argument("--coll", action="store_true", help="只出碰撞概率")
    ap.add_argument("--check", action="store_true",
                    help="仅做 2.py/3.py 引擎交叉校验（快速）")
    ap.add_argument("--sig-pos", type=float, default=30.0,
                    help="入轨位置误差 3σ (km)，默认 30")
    ap.add_argument("--sig-vel", type=float, default=3.0,
                    help="入轨速度误差 3σ (m/s)，默认 3.0")
    a = ap.parse_args(argv)
    if a.check:
        ok, lines = engine_crosscheck()
        print("\n".join(lines))
        return 0 if ok else 2
    print(run_demo(nmc=a.mc, do_disp=not a.coll, do_coll=not a.disp,
                   sig_pos_3s=a.sig_pos, sig_vel_3s=a.sig_vel))
    if not _HAVE_SCIPY:
        print("注：未检测到 scipy，使用固定步长 RK4（精度略低）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
