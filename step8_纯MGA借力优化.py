#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_纯MGA借力优化.py
=====================
纯 MGA（Multiple Gravity Assist）全局优化器：**无 DSM、飞越近心点零脉冲**。

为什么需要它
------------
`pykep.trajopt.mga` 的飞越模型是：

    DV_fb = pk.fb_dv(v_rel_in, v_rel_out, planet)

当两段 Lambert 的 v∞ 模长不等、或转角落不进 safe_radius 对应的双曲线时，
它就在双曲线末端补一次脉冲（代码里直接加进目标函数）。染色体里也没有飞越
半径，等效于把飞越半径固定成 safe_radius。这不是"打水漂"式的纯借力。

本脚本把飞越严格写成弹道双曲线（patched conic，Mivovitch 模型）：

    |v∞out| = |v∞in|                       （模长守恒，硬约束）
    δ = angle(v∞in, v∞out)                 （转角由前后两段 Lambert 给定）
    rp = μ/v∞² · (1/sin(δ/2) − 1)          （唯一确定的近心半径）

* δ 必须 ≤ δmax = 2·asin( 1/(1 + rp_min·v∞²/μ) )，否则这个飞越物理上做不到；
* rp 是**输出量**，不是决策变量——给定前后两段方向，弹道飞越的 rp 唯一；
* 决策向量只有 [t0, T1, ..., TN]，没有 DSM，没有飞越脉冲；
* 目标函数里的"飞越缺陷"只是罚函数，收敛后必须为 0，不是可花的 Δv。

优化器：pygmo.sade 全局搜索 + 可选 scipy SLSQP 精修（把
|v∞in|²−|v∞out|²=0 当等式约束、rp−rp_min≥0 当不等式约束）。
导出：与 step6 完全兼容的 nodes JSON（dsms 全为 null）。

决策向量
--------
    x = [t0(MJD2000), T1, T2, ..., TN]      N = 段数 = 行星数-1
    飞越 rp 不进染色体；需要时由几何反解，报告在结果里。

目标函数
--------
    J = w_dep·|v∞_dep| + w_arr·|v∞_arr| + W·Σ (Δv_fb,i)²

    Δv_fb,i = |v∞out − R(δmax)·v∞in|     （模长不等 + 转角不够的合成缺陷）
    W 默认 100；收敛判据 max|Δv_fb| < 1e-3 km/s。

用法示例
--------
# 1) Galileo 1989 VEEGA 纯 MGA（地-金-地-地-木，4 段）
python3 step8_纯MGA借力优化.py --case eveej --rp-min earth=6678,venus=6352 \
        --polish --wenchang --out data/puremga_nodes.json --csv data/puremga.csv

# 2) 换窗口：1995-2005 年每 20 天一个窗口扫一遍
python3 step8_纯MGA借力优化.py --case eveej --scan \
        --scan-start 1995-01-01 --scan-end 2005-01-01 --scan-step 20 \
        --out data/eveej_scan.json --csv data/eveej_scan.csv

# 3) 强制第 3 段用 1 圈 Lambert 的指定分支（默认 auto 自动选）
python3 step8_纯MGA借力优化.py --case eveej --revs 0,0,1,0 --branch 0,0,2,0

# 4) 谐振返回段（skipping-stone 那种 1:1 地-地借力）：第 3 段不走 Lambert，
#    改由内层打靶解 (δ, ψ, τ)，τ/P 强制≈2 圈，真正的 TOF 是输出量
python3 step8_纯MGA借力优化.py --names earth,venus,earth,earth \
        --t0 10896.5 10971.5 --tof 155,170 320,340 700,760 --revs 1,1,3 \
        --resonant 2:2 --rp-min earth=6678,venus=6352 \
        --w-dep 1 --w-arr 0 --w-fb 2000 --pop 60 --gen 120

#    换个谐振阶 / 换颗行星 / 换条腿都行：--resonant 2:3 是 3:1 那一族，
#    --names earth,venus,venus,earth --resonant 1:3 是"金星借力后回金星"，
#    --resonant 1:2,3:2 可以同时声明两段。

注意
----
* 两点 Lambert 在转移角≈0°（谐振返回）时是**退化**的：过同一点、同 a 的椭圆
  有无穷多条，朝向定了才有 |v∞|，pykep 只给共面切向的那几条（δ=0.00°，
  |v∞| 4~7 km/s），于是 1:1 谐振借力在纯 Lambert 模型里根本表达不出来，
  飞越缺陷怎么都压不到 0。这类段要用 --resonant，让内层用"圆锥 + 打靶命中"
  解出 δ/ψ/τ。
* 谐振命中在 TOF 空间上是**孤立点**（容差约 1 分钟），所以 τ 绝不能当随机变量
  去搜——只能解析地解出来，这也是 --resonant 的设计动机。
* --resonant 只替换**指定的那几段**，其余段照常走 Lambert；不声明的段行为与
  以前完全一致。
* 出发/到达脉冲这里按 |v∞| 计（MGA 惯例）。真实任务还要加上从 LEO 到
  双曲线的损失、木星捕获 JOI 等，别把 J 直接当总 Δv。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pykep as pk
import pygmo as pg

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MJD2000 = 51544.5
DAY = 86400.0
MU_SUN = pk.MU_SUN                 # m^3/s^2
KMS = 1e-3                         # m/s -> km/s
OBLIQUITY = math.radians(23.4392911)

# 支持的行星（jpl_lp 低精度星历，与 step7 一致）
PLANET_NAMES = ("mercury", "venus", "earth", "mars",
                "jupiter", "saturn", "uranus", "neptune")

# 默认近心半径下限：半径 + 保守高度裕量 [km]
DEFAULT_RP_ALT = dict(mercury=200.0, venus=300.0, earth=200.0, mars=200.0,
                      jupiter=5000.0, saturn=5000.0, uranus=5000.0,
                      neptune=5000.0)


def _wrap360(x):
    return x % 360.0


def mjd_to_utc(mjd):
    """MJD（UTC 近似）→ datetime。"""
    return datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=float(mjd))


def parse_date(s):
    """接受 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM' / 纯数字（MJD2000）。"""
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp() / DAY + 40587.0 - MJD2000
        except ValueError:
            continue
    raise ValueError("无法解析日期: %s" % s)


def ecl_to_eq(v):
    """黄道 → 赤道（绕 x 轴转 ε）。"""
    x, y, z = np.asarray(v, float)
    return np.array([x,
                     y * math.cos(OBLIQUITY) - z * math.sin(OBLIQUITY),
                     y * math.sin(OBLIQUITY) + z * math.cos(OBLIQUITY)])


# ---------------------------------------------------------------------------
# Lambert：列出所有分支
# ---------------------------------------------------------------------------
def lambert_all(r0, r1, tof_days, mu, revs):
    """返回 [(branch_id, v0, v1), ...]（SI 单位 m, m/s）。

    pykep 在近 0°/360° 转移角时偶有数值退化，这里过滤 NaN/Inf。
    任何异常都返回空列表，由调用方当作高惩罚。
    """
    if tof_days <= 0:
        return []
    try:
        lp = pk.lambert_problem(list(r0), list(r1), tof_days * DAY, mu,
                                cw=False, multi_revs=int(revs))
    except Exception:
        return []
    out = []
    for k in range(len(lp.v0)):
        v0 = np.asarray(lp.v0[k], float)
        v1 = np.asarray(lp.v1[k], float)
        if np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)):
            out.append((k, v0, v1))
    return out


def rotate_toward(vin, vout, dmax):
    """把 vin 在 (vin,vout) 平面内朝 vout 转 min(δ,dmax)，保持模长。"""
    ni = float(np.linalg.norm(vin))
    a = vin / ni
    b = vout / np.linalg.norm(vout)
    c = float(np.clip(a @ b, -1.0, 1.0))
    d = math.acos(c)
    if d < 1e-12:
        return vin.copy()
    ax = np.cross(a, b)
    na = float(np.linalg.norm(ax))
    if na < 1e-12:
        return vin.copy()
    ax /= na
    dd = min(d, dmax)
    return ni * (a * math.cos(dd) + np.cross(ax, a) * math.sin(dd)
                 + ax * (ax @ a) * (1.0 - math.cos(dd)))


def flyby_terms(vin, vout, mu_p, rp_min):
    """飞越几何：返回缺陷 Δv、转角、最大转角、所需 rp（全部 km / km/s）。"""
    vin = np.asarray(vin, float) * KMS
    vout = np.asarray(vout, float) * KMS
    ni = float(np.linalg.norm(vin))
    no = float(np.linalg.norm(vout))
    if ni < 1e-9 or no < 1e-9:
        return None
    a = vin / ni
    b = vout / no
    delta = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    vinf = 0.5 * (ni + no)
    mu_km = mu_p * 1e-9
    emin = 1.0 + rp_min * vinf ** 2 / mu_km
    dmax = 2.0 * math.asin(min(1.0, 1.0 / emin))
    dv = float(np.linalg.norm(vout - rotate_toward(vin, vout, dmax)))
    # rp 必须由**真实转角**反解；转角不够时它自然小于 rp_min，
    # 这样不等式约束才看得见违反（不能用 min(delta,dmax) 截断）。
    sh = math.sin(delta * 0.5)
    rpreq = mu_km / vinf ** 2 * (1.0 / sh - 1.0) if sh > 1e-9 else 1e12
    return dict(vin=vin, vout=vout, vinf_in=ni, vinf_out=no, vinf=vinf,
                delta=delta, dmax=dmax, dv=dv, rp_req=rpreq,
                feasible=(dv < 1e-6 and rpreq >= rp_min * (1 - 1e-6)))


# ---------------------------------------------------------------------------
# 零脉冲飞越的出射方向锥面
# ---------------------------------------------------------------------------
def cone_dir(a_hat, delta_deg):
    """以入射渐近线 a_hat 为轴、半顶角 δ 的锥面；ψ（B 面角，度）→ 单位向量。

    零脉冲飞越只约束 |v∞out| = |v∞in|，方向就是这个锥面上任一点，
    锥面位置由 B 面角 ψ 决定——这正是两点 Lambert 表达不出来的那一个自由度。
    """
    a_hat = np.asarray(a_hat, float)
    a_hat = a_hat / float(np.linalg.norm(a_hat))
    tmp = np.array([0., 0., 1.]) if abs(a_hat[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(a_hat, tmp)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(a_hat, p1)
    d = math.radians(float(delta_deg))

    def f(psi_deg):
        q = math.radians(float(psi_deg))
        return (a_hat * math.cos(d)
                + (p1 * math.cos(q) + p2 * math.sin(q)) * math.sin(d))
    return f


# ---------------------------------------------------------------------------
# 纯 MGA UDP
# ---------------------------------------------------------------------------
class PureMGA:
    """pygmo 用户定义问题：纯 MGA，无 DSM、飞越零脉冲。"""

    def __init__(self, names, t0_bounds, tof_bounds, revs=None, branch=None,
                 rp_min=None, w_dep=1.0, w_arr=1.0, w_fb=100.0,
                 vinf_dep_max=None, vinf_arr_max=None, max_combos=64,
                 resonant=None, res_tol_km=1.0e4):
        self.names = list(names)
        self.seq = [pk.planet(pk.udpla.jpl_lp(n)) for n in names]
        self.mu = float(self.seq[0].mu_central_body)
        self.mu_self = [float(p.mu_self) for p in self.seq]
        self.radius = [float(p.radius) for p in self.seq]
        self.t0_bounds = tuple(float(v) for v in t0_bounds)
        self.tof_bounds = [(float(a), float(b)) for a, b in tof_bounds]
        self.n_legs = len(names) - 1
        if len(self.tof_bounds) != self.n_legs:
            raise ValueError("tof_bounds 长度应为 %d" % self.n_legs)
        self.revs = list(revs) if revs else [0] * self.n_legs
        self.branch = list(branch) if branch else ["auto"] * self.n_legs
        self.rp_min = dict()
        for i, n in enumerate(names):
            if i == 0 or i == len(names) - 1:
                self.rp_min[n] = np.inf       # 出发/到达不算飞越
            else:
                alt = DEFAULT_RP_ALT.get(n, 200.0)
                self.rp_min[n] = self.radius[i] * KMS + alt
        if rp_min:
            self.rp_min.update(rp_min)
        self.w_dep = float(w_dep)
        self.w_arr = float(w_arr)
        self.w_fb = float(w_fb)
        self.vinf_dep_max = vinf_dep_max
        self.vinf_arr_max = vinf_arr_max
        self.max_combos = int(max_combos)
        # 谐振返回段：{leg_index: k} —— 该段不走 Lambert，改由内层打靶解 (δ,ψ,τ)。
        # 腿号从 0 计数、且不能是第 0 段（需要前一段给入射渐近线）；对行星种类
        # 和腿位置都不加限制——任何"借力后绕 k 圈再撞上同一颗行星"的段都能声明。
        self.resonant = {}
        if resonant:
            for kk, vv in dict(resonant).items():
                i, k = int(kk), int(vv)
                if i < 1 or i >= self.n_legs:
                    raise ValueError("谐振段号必须是 1..%d（需要前一段给入射渐近线）"
                                     % (self.n_legs - 1))
                if k < 1:
                    raise ValueError("谐振圈数 k 必须 ≥ 1")
                self.resonant[i] = k
        self.res_tol_km = float(res_tol_km)

    # -- pygmo 接口 ---------------------------------------------------------
    def get_bounds(self):
        lb = [self.t0_bounds[0]] + [a for a, b in self.tof_bounds]
        ub = [self.t0_bounds[1]] + [b for a, b in self.tof_bounds]
        return lb, ub

    def get_nobj(self):
        return 1

    def fitness(self, x):
        return [self.evaluate(x)["J"]]

    # -- 解码 ---------------------------------------------------------------
    def evaluate(self, x, combo=None, detail=True):
        """x = [t0, T1..TN]；返回 J 与全部几何。"""
        if self.resonant:
            return self._evaluate_resonant(x, combo=combo, detail=detail)
        x = np.asarray(x, float)
        t0 = float(x[0])
        T = np.asarray(x[1:], float)
        if len(T) != self.n_legs or np.any(T <= 0):
            return dict(J=1e9, ok=False, reason="bad_x")
        ep = t0 + np.concatenate([[0.0], np.cumsum(T)])
        r, v = [], []
        try:
            for i, p in enumerate(self.seq):
                ri, vi = p.eph(pk.epoch(ep[i]))
                r.append(np.asarray(ri, float))
                v.append(np.asarray(vi, float))
        except Exception:
            return dict(J=1e9, ok=False, reason="eph")

        sols = []
        for i in range(self.n_legs):
            s = lambert_all(r[i], r[i + 1], T[i], self.mu, self.revs[i])
            if not s:
                return dict(J=1e9, ok=False, reason="lambert")
            if self.branch[i] != "auto":
                want = int(self.branch[i])
                s = [q for q in s if q[0] == want] or s
            sols.append(s)

        if combo is not None:
            idx = [min(c, len(s) - 1) for c, s in zip(combo, sols)]
            combos = [idx]
        else:
            ncomb = 1
            for s in sols:
                ncomb *= len(s)
            if ncomb <= self.max_combos:
                combos = list(itertools.product(*[range(len(s)) for s in sols]))
            else:
                combos = [tuple(0 for _ in sols)]

        best = None
        for cb in combos:
            legs = [(sols[i][cb[i]][1], sols[i][cb[i]][2])
                    for i in range(self.n_legs)]
            out = self._score(legs, v)
            if out is None:
                continue
            if best is None or out["J"] < best["J"]:
                best = out
                best["combo"] = tuple(int(c) for c in cb)
                best["branch_ids"] = tuple(int(sols[i][cb[i]][0])
                                           for i in range(self.n_legs))
        if best is None:
            return dict(J=1e9, ok=False, reason="score")
        best["x"] = x.copy()
        best["t0"] = t0
        best["T"] = T.copy()
        best["epoch_mjd2000"] = ep
        return best if detail else dict(J=best["J"])

    def _score(self, legs, v):
        vd = float(np.linalg.norm(legs[0][0] - v[0])) * KMS
        va = float(np.linalg.norm(legs[-1][1] - v[-1])) * KMS
        if not (np.isfinite(vd) and np.isfinite(va)):
            return None
        pen = 0.0
        det = []
        for i in range(1, self.n_legs):
            vin = legs[i - 1][1] - v[i]
            vout = legs[i][0] - v[i]
            ft = flyby_terms(vin, vout, self.mu_self[i],
                             self.rp_min[self.names[i]])
            if ft is None:
                return None
            pen += ft["dv"] ** 2
            ft["body"] = self.names[i]
            ft["alt"] = ft["rp_req"] - self.radius[i] * KMS
            det.append(ft)
        J = self.w_dep * vd + self.w_arr * va + self.w_fb * pen
        if self.vinf_dep_max is not None:
            J += 100.0 * max(0.0, vd - self.vinf_dep_max) ** 2
        if self.vinf_arr_max is not None:
            J += 100.0 * max(0.0, va - self.vinf_arr_max) ** 2
        max_dv = max((f["dv"] for f in det), default=0.0)
        return dict(J=float(J), ok=True, vd=vd, va=va, pen=pen, det=det,
                    legs=legs, max_defect=max_dv,
                    feasible=(max_dv < 1e-3 and
                              all(f["rp_req"] >= self.rp_min[f["body"]]
                                  for f in det)))

    # -- 谐振返回段（内层打靶）------------------------------------------------
    def _leg_branches(self, leg, r0, r1, tof_days):
        """该腿所有可用 Lambert 解（已按 --branch 过滤），序号即 combo 里的位。"""
        s = lambert_all(r0, r1, tof_days, self.mu, self.revs[leg])
        if s and self.branch[leg] != "auto":
            want = int(self.branch[leg])
            s = [q for q in s if q[0] == want] or s
        return s

    def _resonant_leg(self, i, r0, v0, vin, t_node, tof_seed, k):
        """在第 i 个节点做零脉冲飞越，解 (δ, ψ, τ) 让 τ 天后正好再撞上第 i+1 个节点。

        转移角≈0° 的两点 Lambert 是退化的（过同一点、同 a 的椭圆有无穷多条，
        朝向定了才有 |v∞|），所以这里不走 Lambert：
            未知量 δ（转角）、ψ（B 面角）、τ（飞行时间）= 3
            方程 命中行星的 x/y/z                        = 3
        染色体里的 T[i] 只当**种子/选支**用，真正的 τ 是输出量。
        对行星种类没有假设：μ、rp_min、δmax 全按本节点行星取。
        返回 None 表示 ±20% 时间窗内找不到命中。
        """
        try:
            from scipy.optimize import least_squares
        except Exception:
            return None
        nv = float(np.linalg.norm(vin))                      # m/s
        if not (50.0 < nv < 1e5):                           # 0.05 ~ 100 km/s
            return None
        mu_p_km = self.mu_self[i] * 1e-9
        rp_min_km = self.rp_min[self.names[i]]
        vinf_km = nv * KMS
        sd = 1.0 / (1.0 + rp_min_km * vinf_km ** 2 / mu_p_km)
        if sd >= 1.0 or sd <= 0.0:
            return None
        dmax_deg = math.degrees(2.0 * math.asin(sd))
        a_hat = np.asarray(vin, float) / nv
        tgt = self.seq[i + 1]

        def geom(z):
            dd, psi, tau = float(z[0]), float(z[1]), float(z[2])
            u = cone_dir(a_hat, dd)(psi)
            vsc = v0 + u * nv                                # |v∞out|=|v∞in| 由构造保证
            rr, vv = pk.propagate_lagrangian([list(r0), list(vsc)],
                                             tau * DAY, self.mu)
            rb, vb = tgt.eph(pk.epoch(t_node + tau))
            return (np.asarray(rr, float) - np.asarray(rb, float)) * KMS, \
                np.asarray(vv, float), vsc, dd, psi, tau

        def resid(z):
            return geom(z)[0]

        lo, hi = tof_seed * 0.80, tof_seed * 1.20
        best = None
        for fd in (0.25, 0.5, 0.75, 0.95):
            for psi0 in (0.0, 120.0, 240.0):
                z0 = [min(max(fd * dmax_deg, 1.0), 170.0), psi0, tof_seed]
                try:
                    sol = least_squares(resid, z0,
                                        bounds=([0.5, -720.0, lo],
                                                [170.0, 720.0, hi]),
                                        xtol=1e-14, ftol=1e-14, diff_step=1e-7)
                except Exception:
                    continue
                D = float(np.linalg.norm(sol.fun))
                if D > self.res_tol_km:
                    continue
                dd, psi, tau = (float(z) for z in sol.x)
                g = geom(sol.x)
                # 日心周期 → 实际谐振阶
                eps = float(np.dot(g[2], g[2])) / 2.0 - self.mu \
                    / float(np.linalg.norm(r0))
                a_h = -self.mu / (2.0 * eps) if eps < 0 else float("inf")
                P = (2.0 * math.pi * math.sqrt(a_h ** 3 / self.mu) / DAY
                     if math.isfinite(a_h) and a_h > 0 else float("nan"))
                k_eff = tau / P if P and math.isfinite(P) and P > 0 else 1e9
                rank = (abs(k_eff - k), D)
                if best is None or rank < best[0]:
                    best = (rank, dict(D=D, delta=dd, psi=psi, tof=tau,
                                       vsc=g[2], varr=g[1],
                                       a_km=a_h * 1e-3, P=P, k_eff=k_eff))
        return best[1] if best else None

    def _evaluate_resonant(self, x, combo=None, detail=True):
        """带谐振段的解码：节点历元逐步生成（谐振段的 τ 是解出来的）。"""
        x = np.asarray(x, float)
        t0 = float(x[0])
        T = np.asarray(x[1:], float)
        if len(T) != self.n_legs or np.any(T <= 0):
            return dict(J=1e9, ok=False, reason="bad_x")
        nn = self.n_legs + 1
        rs, vs, ep = [None] * nn, [None] * nn, [None] * nn
        try:
            r0, v0 = self.seq[0].eph(pk.epoch(t0))
        except Exception:
            return dict(J=1e9, ok=False, reason="eph")
        rs[0], vs[0], ep[0] = np.asarray(r0, float), np.asarray(v0, float), t0

        # 谐振段之前的节点历元由种子 TOF 直接累加；之后的节点要靠内层解，先不算
        for i in range(1, nn):
            if i - 1 in self.resonant:
                break
            tj = ep[i - 1] + T[i - 1]
            try:
                ri, vi = self.seq[i].eph(pk.epoch(tj))
            except Exception:
                return dict(J=1e9, ok=False, reason="eph")
            ep[i], rs[i], vs[i] = tj, np.asarray(ri, float), np.asarray(vi, float)

        opts = []
        for leg in range(self.n_legs):
            if leg in self.resonant or (leg - 1) in self.resonant:
                opts.append([0])          # 谐振腿 / 谐振段后的腿：只有一个伪分支
                continue
            s = self._leg_branches(leg, rs[leg], rs[leg + 1], T[leg])
            if not s:
                return dict(J=1e9, ok=False, reason="lambert")
            opts.append(list(range(len(s))))
        ncomb = 1
        for o in opts:
            ncomb *= len(o)
        if ncomb > self.max_combos:
            opts = [o[:1] for o in opts]

        best = None
        for cb in itertools.product(*opts):
            legs, info = [], {}
            live_ep, live_r, live_v = [t0], [rs[0]], [vs[0]]
            ok = True
            for leg in range(self.n_legs):
                if leg not in self.resonant and leg - 1 not in self.resonant:
                    s = self._leg_branches(leg, live_r[leg], rs[leg + 1], T[leg])
                    if not s:
                        ok = False
                        break
                    pick = s[int(cb[leg])]
                    legs.append((pick[1], pick[2]))
                    live_ep.append(ep[leg + 1])
                    live_r.append(rs[leg + 1])
                    live_v.append(vs[leg + 1])
                    info[leg] = dict(kind="lambert", branch=int(pick[0]))
                elif leg in self.resonant:
                    if leg == 0 or not legs:
                        ok = False
                        break
                    vin = np.asarray(legs[leg - 1][1], float) - live_v[leg]
                    sol = self._resonant_leg(leg, live_r[leg], live_v[leg], vin,
                                             live_ep[leg], T[leg],
                                             self.resonant[leg])
                    if sol is None:
                        ok = False
                        break
                    legs.append((sol["vsc"], sol["varr"]))
                    tj = live_ep[leg] + sol["tof"]
                    try:
                        ri, vi = self.seq[leg + 1].eph(pk.epoch(tj))
                    except Exception:
                        ok = False
                        break
                    live_ep.append(tj)
                    live_r.append(np.asarray(ri, float))
                    live_v.append(np.asarray(vi, float))
                    info[leg] = dict(kind="resonant", k=self.resonant[leg],
                                     k_eff=sol["k_eff"], delta=sol["delta"],
                                     psi=sol["psi"], tof=sol["tof"],
                                     resid_km=sol["D"], a_km=sol["a_km"],
                                     P=sol["P"])
                else:            # 紧跟在谐振段之后的腿：起点是解出来的节点
                    tj = live_ep[leg] + T[leg]        # 该腿 TOF 仍由外层决定
                    try:
                        ri, vi = self.seq[leg + 1].eph(pk.epoch(tj))
                    except Exception:
                        ok = False
                        break
                    s = self._leg_branches(leg, live_r[leg],
                                           np.asarray(ri, float), T[leg])
                    if not s:
                        ok = False
                        break
                    pick = s[0]
                    legs.append((pick[1], pick[2]))
                    live_ep.append(tj)
                    live_r.append(np.asarray(ri, float))
                    live_v.append(np.asarray(vi, float))
                    info[leg] = dict(kind="lambert", branch=int(pick[0]),
                                     post_resonant=True)
            if not ok or len(legs) != self.n_legs:
                continue
            out = self._score(legs, live_v)
            if out is None:
                continue
            if best is None or out["J"] < best["J"]:
                out["combo"] = tuple(int(c) for c in cb)
                out["branch_ids"] = tuple(
                    -int(info[L]["k"]) if info[L]["kind"] == "resonant"
                    else int(info[L]["branch"]) for L in range(self.n_legs))
                out["leg_info"] = info
                out["ep_live"] = list(live_ep)
                best = out
        if best is None:
            return dict(J=1e9, ok=False, reason="resonant")
        best["x"] = x.copy()
        best["t0"] = t0
        live = np.asarray(best["ep_live"], float)
        best["T"] = live[1:] - live[:-1]
        best["epoch_mjd2000"] = live
        return best if detail else dict(J=best["J"])

    # -- 结果导出 -----------------------------------------------------------
    def export_nodes(self, res, path, wenchang=None):
        """写成 step6 可直接吃的 nodes JSON。"""
        names = self.names
        ep = res["epoch_mjd2000"]
        V = res["legs"]
        arr_vinf = (V[-1][1] - self.seq[-1].eph(pk.epoch(ep[-1]))[1]) * KMS
        nodes = dict(
            departure=dict(
                body=names[0],
                epoch_mjd=float(ep[0] + MJD2000),
                vinf_km_s=[float(q) for q in
                           (V[0][0] - self.seq[0].eph(pk.epoch(ep[0]))[1]) * KMS]),
            flybys=[],
            dsms=[None] * self.n_legs,
            arrival=dict(epoch_mjd=float(ep[-1] + MJD2000),
                         target=names[-1]),
            meta=dict(
                source="step8 pure MGA (ballistic flybys, no DSM)",
                t0_mjd2000=float(ep[0]),
                tof_days=[float(t) for t in res["T"]],
                objective=float(res["J"]),
                vinf_dep_km_s=float(res["vd"]),
                vinf_arr_km_s=float(res["va"]),
                branch_ids=[int(b) for b in res["branch_ids"]],
                max_flyby_defect_m_s=float(res["max_defect"] * 1e3),
                resonant=[dict(leg=L + 1, body=names[L], k=int(q["k"]),
                               tau_days=float(q["tof"]),
                               delta_deg=float(q["delta"]),
                               psi_deg=float(q["psi"]),
                               a_AU=float(q["a_km"] / 1.495978707e8),
                               period_days=float(q["P"]),
                               k_eff=float(q["k_eff"]),
                               resid_km=float(q["resid_km"]))
                          for L, q in sorted((res.get("leg_info") or {}).items())
                          if q["kind"] == "resonant"],
                x=[float(q) for q in res["x"]],
                wenchang=wenchang,
            ),
        )
        for i, ft in enumerate(res["det"], start=1):
            rb, vb = self.seq[i].eph(pk.epoch(ep[i]))
            vi = (V[i - 1][1] - np.asarray(vb, float)) * KMS
            vo = (V[i][0] - np.asarray(vb, float)) * KMS
            nodes["flybys"].append(dict(
                body=names[i],
                epoch_mjd=float(ep[i] + MJD2000),
                vinf_in_km_s=[float(q) for q in vi],
                vinf_out_km_s=[float(q) for q in vo],
                rp_km=float(ft["rp_req"]),
                alt_km=float(ft["alt"]),
                turn_deg=float(math.degrees(ft["delta"])),
                turn_max_deg=float(math.degrees(ft["dmax"])),
                defect_m_s=float(ft["dv"] * 1e3),
            ))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nodes, fh, indent=1, ensure_ascii=False)
        return nodes, path


# ---------------------------------------------------------------------------
# SLSQP 精修：把飞越模长差当等式约束
# ---------------------------------------------------------------------------
def polish(prob: PureMGA, res, maxiter=300):
    """在 DE 冠军附近解一个带约束的 NLP，把缺陷压到机器精度。"""
    try:
        from scipy.optimize import minimize
    except Exception:
        return res
    combo = res.get("combo")
    x0 = np.asarray(res["x"], float)
    lb, ub = prob.get_bounds()
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    def unpack(x):
        r = prob.evaluate(x, combo=combo)
        return r

    def obj(x):
        r = unpack(x)
        if not r.get("ok"):
            return 1e6
        return prob.w_dep * r["vd"] + prob.w_arr * r["va"]

    def eq(x):
        r = unpack(x)
        if not r.get("ok"):
            return np.ones(prob.n_legs - 1) * 1e3
        # 无量纲化：模长平方差 / 模长平方和
        return np.array([(f["vinf_in"] ** 2 - f["vinf_out"] ** 2)
                         / (f["vinf_in"] ** 2 + f["vinf_out"] ** 2 + 1e-12)
                         for f in r["det"]], float)

    def ineq(x):
        r = unpack(x)
        if not r.get("ok"):
            return -np.ones(prob.n_legs - 1) * 1e3
        return np.array([(f["rp_req"] - prob.rp_min[f["body"]])
                         / prob.rp_min[f["body"]] for f in r["det"]], float)

    cons = [dict(type="eq", fun=eq)]
    if prob.n_legs > 1:
        cons.append(dict(type="ineq", fun=ineq))
    try:
        sol = minimize(obj, x0, method="SLSQP", bounds=list(zip(lb, ub)),
                       constraints=cons,
                       options=dict(maxiter=maxiter, ftol=1e-10, disp=False))
        cand = prob.evaluate(sol.x, combo=combo)
        j_phys = lambda r: prob.w_dep * r["vd"] + prob.w_arr * r["va"]
        if (cand.get("ok") and cand.get("feasible")
                and j_phys(cand) <= j_phys(res) * 1.01 + 1e-6):
            return cand
    except Exception as exc:
        print("[polish] SLSQP 失败：%s" % exc)
    return res


# ---------------------------------------------------------------------------
# pygmo 求解
# ---------------------------------------------------------------------------
def run_opt(prob: PureMGA, pop_size=160, gen=600, seed=1, islands=1,
            x_seed=None, verbose=True):
    pgprob = pg.problem(prob)
    best = None
    for k in range(islands):
        pop = pg.population(pgprob, size=pop_size, seed=seed + k)
        if x_seed is not None and k == 0:
            try:
                pop.set_x(0, np.asarray(x_seed, float))
            except Exception:
                pass
        pop = pg.algorithm(pg.sade(gen=gen)).evolve(pop)
        res = prob.evaluate(pop.champion_x)
        if verbose:
            print("  [island %d] J=%.4f  v∞dep=%.3f v∞arr=%.3f  "
                  "缺陷max=%.2f m/s  %s" %
                  (k + 1, res["J"], res.get("vd", np.nan),
                   res.get("va", np.nan), res.get("max_defect", np.nan) * 1e3,
                   "可行" if res.get("feasible") else "不可行"))
        if best is None or res["J"] < best["J"]:
            best = res
    return best


# ---------------------------------------------------------------------------
# 文昌发射几何（复用 step1）
# ---------------------------------------------------------------------------
def load_step1():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "step1_发射窗口与上升段闭合.py")
    spec = importlib.util.spec_from_file_location("step1_geo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step1_geo"] = mod
    spec.loader.exec_module(mod)
    return mod


def wenchang_check(res, prob: PureMGA, rp_launch=6700.0):
    """用 step1 的平面几何判断这个出发 v∞ 文昌能不能打。

    返回：最小倾角、最佳方位角、发射时刻、滑行时间、SOI 相对出发的偏移。
    """
    try:
        s1 = load_step1()
    except Exception as exc:
        return dict(ok=None, err="step1 载入失败: %s" % exc)
    vinf_ecl = (res["legs"][0][0]
                - prob.seq[0].eph(pk.epoch(res["epoch_mjd2000"][0]))[1]) * KMS
    vinf_eq = ecl_to_eq(vinf_ecl)
    t_dep = mjd_to_utc(res["epoch_mjd2000"][0] + MJD2000).replace(tzinfo=None)
    try:
        hyp = s1.build_hyperbola(vinf_eq, rp_launch)
        wins = s1.daily_window(hyp, t_dep, s1.WENCHANG[0], s1.WENCHANG[1],
                               t_dep, step_min=1.0)
    except Exception as exc:
        return dict(ok=False, err=str(exc))
    if not wins:
        return dict(ok=False, reason="±30/42h 内无方位角走廊窗口",
                    vinf_km_s=float(np.linalg.norm(vinf_eq)),
                    dec_deg=float(math.degrees(math.asin(
                        np.clip(vinf_eq[2] / np.linalg.norm(vinf_eq), -1, 1)))),
                    ra_deg=float(_wrap360(math.degrees(math.atan2(
                        vinf_eq[1], vinf_eq[0])))))
    # daily_window 返回的列表已按"SOI 穿越 ≈ 出发历元"排序，wins[0] 是定时闭合最好的
    best = wins[0]
    lowest = min(wins, key=lambda w: w.i_opt)
    return dict(ok=True, vinf_km_s=float(np.linalg.norm(vinf_eq)),
                dec_deg=float(hyp.dec_deg), ra_deg=float(hyp.ra_deg),
                t_launch_utc=best.t_opt.strftime("%Y-%m-%d %H:%M:%S"),
                inc_deg=float(best.i_opt), az_deg=float(best.az_opt),
                coast_min=float(best.coast_opt),
                dt_to_dep_min=float((best.t_opt - t_dep).total_seconds() / 60.0),
                n_windows=len(wins),
                i_min_deg=float(lowest.i_opt),
                az_at_imin_deg=float(lowest.az_opt))


# ---------------------------------------------------------------------------
# 算例配置
# ---------------------------------------------------------------------------
def case_config(case):
    """返回 names, t0_bounds, tof_bounds, revs。"""
    if case == "eveej":
        # Galileo: 地-金-地-地-木
        return dict(
            names=["earth", "venus", "earth", "earth", "jupiter"],
            t0_bounds=(parse_date("1989-06-01"), parse_date("1989-12-31")),
            tof_bounds=[(60, 220), (180, 420), (600, 900), (800, 1500)],
            revs=[0, 0, 1, 0],
        )
    if case == "emej":
        return dict(
            names=["earth", "mars", "earth", "jupiter"],
            t0_bounds=(parse_date("2024-01-01"), parse_date("2025-12-31")),
            tof_bounds=[(120, 320), (400, 900), (900, 1600)],
            revs=[0, 0, 0],
        )
    raise ValueError("未知算例: %s" % case)


def build_prob(args):
    cfg = case_config(args.case)
    names = [s.strip() for s in args.names.split(",")] if args.names \
        else cfg["names"]
    t0b = cfg["t0_bounds"]
    tofb = cfg["tof_bounds"]
    revs = cfg["revs"]
    if args.t0:
        t0b = (parse_date(args.t0[0]), parse_date(args.t0[1]))
    if args.tof:
        tofb = [tuple(float(v) for v in s.split(",")) for s in args.tof]
    if args.revs:
        revs = [int(v) for v in args.revs.split(",")]
    branch = args.branch.split(",") if args.branch else None
    rp_min = None
    if args.rp_min:
        rp_min = {}
        for item in args.rp_min.split(","):
            k, val = item.split("=")
            rp_min[k.strip()] = float(val)
    resonant = None
    if args.resonant:
        resonant = {}
        for item in args.resonant.split(","):
            k, v = item.split(":")
            resonant[int(k)] = int(v)
    return PureMGA(names, t0b, tofb, revs=revs, branch=branch, rp_min=rp_min,
                   w_dep=args.w_dep, w_arr=args.w_arr, w_fb=args.w_fb,
                   vinf_dep_max=args.vinf_dep_max,
                   vinf_arr_max=args.vinf_arr_max,
                   resonant=resonant, res_tol_km=args.res_tol)


# ---------------------------------------------------------------------------
# 窗口扫描
# ---------------------------------------------------------------------------
def scan_windows(args, prob: PureMGA):
    """按 t0 窗口切片，每片跑一个短 DE，汇总所有可行解。"""
    t_start = parse_date(args.scan_start)
    t_end = parse_date(args.scan_end)
    step = float(args.scan_step)
    width = float(args.scan_width)
    rows = []
    t = t_start
    idx = 0
    while t <= t_end:
        idx += 1
        sub = PureMGA(prob.names,
                      (t - width / 2, t + width / 2),
                      prob.tof_bounds, revs=prob.revs, branch=prob.branch,
                      rp_min={k: v for k, v in prob.rp_min.items()
                              if np.isfinite(v)},
                      w_dep=prob.w_dep, w_arr=prob.w_arr, w_fb=prob.w_fb,
                      vinf_dep_max=prob.vinf_dep_max,
                      vinf_arr_max=prob.vinf_arr_max,
                      resonant=prob.resonant, res_tol_km=prob.res_tol_km)
        res = run_opt(sub, pop_size=args.scan_pop, gen=args.scan_gen,
                      seed=args.seed + 17 * idx, islands=1, verbose=False)
        if res.get("ok"):
            row = dict(window=t, t0=res["t0"], J=res["J"], vd=res["vd"],
                       va=res["va"], max_defect=res["max_defect"],
                       feasible=bool(res["feasible"]),
                       T=";".join("%.2f" % q for q in res["T"]),
                       rp=";".join("%.0f" % f["rp_req"] for f in res["det"]),
                       alt=";".join("%.0f" % f["alt"] for f in res["det"]),
                       delta=";".join("%.1f" % math.degrees(f["delta"])
                                      for f in res["det"]),
                       combo=";".join(str(c) for c in res["combo"]))
            rows.append(row)
            print("  [%3d] t0=%8.1f J=%7.3f dep=%5.2f arr=%5.2f "
                  "def=%6.1f m/s %s" %
                  (idx, res["t0"], res["J"], res["vd"], res["va"],
                   res["max_defect"] * 1e3,
                   "OK" if res["feasible"] else ""))
        t += step
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="纯 MGA（无 DSM、飞越零脉冲）pykep+pygmo 优化器")
    ap.add_argument("--case", default="eveej", choices=["eveej", "emej"])
    ap.add_argument("--t0", nargs=2, metavar=("START", "END"),
                    help="出发窗口，日期或 MJD2000")
    ap.add_argument("--tof", nargs="+", metavar="MIN,MAX",
                    help="逐段飞行时间上下界（天）")
    ap.add_argument("--revs", help="逐段 Lambert 圈数，如 0,0,1,0")
    ap.add_argument("--branch", help="逐段 Lambert 分支号；默认 auto")
    ap.add_argument("--names", help="自定义行星序列，如 earth,venus,earth,earth")
    ap.add_argument("--resonant",
                    help="谐振返回段 LEG:K[,...]。LEG 从 0 计数（不能是第 0 段），"
                         "K 是谐振圈数；如 2:2 = 第 3 段绕太阳 2 圈后零脉冲返回，"
                         "该段 TOF 由内层打靶解出。可多段：'1:2,3:3'")
    ap.add_argument("--res-tol", type=float, default=1.0e4,
                    help="谐振命中的位置残差上限 km（默认 1e4）")
    ap.add_argument("--rp-min", help="飞越近心半径下限 km，如 earth=6678,venus=6352")
    ap.add_argument("--w-dep", type=float, default=1.0, help="出发 v∞ 权重")
    ap.add_argument("--w-arr", type=float, default=1.0, help="到达 v∞ 权重")
    ap.add_argument("--w-fb", type=float, default=100.0, help="飞越缺陷罚权重")
    ap.add_argument("--vinf-dep-max", type=float, default=None,
                    help="出发 v∞ 软上限 km/s")
    ap.add_argument("--vinf-arr-max", type=float, default=None)
    ap.add_argument("--pop", type=int, default=180)
    ap.add_argument("--gen", type=int, default=800)
    ap.add_argument("--islands", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x-seed", help="已知染色体 JSON（含 x 字段）")
    ap.add_argument("--polish", action="store_true", help="SLSQP 精修")
    ap.add_argument("--wenchang", action="store_true", help="文昌几何后检")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--scan-start", default="1995-01-01")
    ap.add_argument("--scan-end", default="2005-01-01")
    ap.add_argument("--scan-step", type=float, default=20.0)
    ap.add_argument("--scan-width", type=float, default=40.0)
    ap.add_argument("--scan-pop", type=int, default=80)
    ap.add_argument("--scan-gen", type=int, default=250)
    ap.add_argument("--out", default="data/puremga_nodes.json")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    prob = build_prob(args)

    if args.scan:
        print("扫描 t0 窗口：%s → %s，步长 %.0f 天，窗宽 %.0f 天" %
              (args.scan_start, args.scan_end, args.scan_step, args.scan_width))
        rows = scan_windows(args, prob)
        rows.sort(key=lambda r: (not r["feasible"], r["J"]))
        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                                   ["window"])
                w.writeheader()
                w.writerows(rows)
            print("已写 %s（%d 行）" % (args.csv, len(rows)))
        return 0

    print("序列: %s" % " → ".join(prob.names))
    print("段数: %d  圈数: %s  分支: %s" %
          (prob.n_legs, prob.revs, prob.branch))
    if prob.resonant:
        tag = "、".join("第%d段=%d圈" % (L + 1, k)
                        for L, k in sorted(prob.resonant.items()))
        print("谐振段: %s（该段 TOF 由内层打靶解出，残差上限 %.0f km）"
              % (tag, prob.res_tol_km))
    print("出发窗口: MJD2000 %.2f ~ %.2f" % prob.t0_bounds)
    print("rp_min: %s" % {k: round(v, 1) for k, v in prob.rp_min.items()
                          if np.isfinite(v)})

    x_seed = None
    if args.x_seed:
        with open(args.x_seed, encoding="utf-8") as fh:
            blob = json.load(fh)
        # 既接受 {"x": [...]}，也接受 step8 自己导出的 nodes（x 在 meta 里）
        x_seed = blob.get("x") or blob.get("meta", {}).get("x")
        if x_seed is None:
            ap.error("--x-seed 文件里找不到 x 或 meta.x")
    best = run_opt(prob, pop_size=args.pop, gen=args.gen, seed=args.seed,
                   islands=args.islands, x_seed=x_seed)
    if args.polish and prob.resonant:
        print("（--polish 在谐振模式下跳过：内层打靶已把命中约束硬解掉，"
              "SLSQP 的 Lambert 残差不再适用）")
        args.polish = False
    if args.polish:
        polished = polish(prob, best)
        # 精修点的罚项≈0，J 可能比"带 6 m/s 缺陷的冠军"略高一点点；
        # 只要物理目标（w_dep·v∞dep + w_arr·v∞arr）不显著变差且可行，就采用。
        j_phys = lambda r: prob.w_dep * r["vd"] + prob.w_arr * r["va"]
        if polished.get("ok") and polished.get("feasible") and (
                j_phys(polished) <= j_phys(best) * 1.01 + 1e-6):
            best = polished

    if not best.get("ok"):
        print("优化失败：%s" % best.get("reason"))
        return 1

    print("\n===== 冠军 =====")
    print("t0 = %.6f MJD2000  (%s UTC)" %
          (best["t0"], mjd_to_utc(best["t0"] + MJD2000).strftime("%Y-%m-%d %H:%M:%S")))
    print("T  = %s  [天]" % np.round(best["T"], 3))
    print("J  = %.4f   v∞dep=%.4f  v∞arr=%.4f km/s   缺陷max=%.3f m/s  %s" %
          (best["J"], best["vd"], best["va"], best["max_defect"] * 1e3,
           "可行(零脉冲)" if best["feasible"] else "不可行"))
    for i, f in enumerate(best["det"], start=1):
        print("  飞越%d %-8s t=%10.3f  rp=%9.1f km (h=%8.1f km)  "
              "δ=%6.2f° / δmax=%6.2f°  v∞=%.4f km/s  缺陷=%6.2f m/s" %
              (i, f["body"], best["epoch_mjd2000"][i] + MJD2000,
               f["rp_req"], f["alt"], math.degrees(f["delta"]),
               math.degrees(f["dmax"]), f["vinf"], f["dv"] * 1e3))

    li = best.get("leg_info") or {}
    res_legs = [(L, q) for L, q in sorted(li.items()) if q["kind"] == "resonant"]
    if res_legs:
        print("\n谐振段（τ 是内层解出来的输出量，染色体里只当种子/选支）:")
        for L, q in res_legs:
            au = q["a_km"] / 1.495978707e8 if q["a_km"] > 0 else float("nan")
            print("  段%d %s→%s: δ=%7.3f°  ψ=%8.3f°  τ=%9.4f d  "
                  "a=%.6f AU  P=%8.3f d  τ/P=%.4f（目标 %d 圈）  残差=%.1f km" %
                  (L + 1, prob.names[L], prob.names[L + 1], q["delta"], q["psi"],
                   q["tof"], au, q["P"], q["k_eff"], q["k"], q["resid_km"]))

    if args.wenchang:
        wc = wenchang_check(best, prob)
        print("\n文昌几何: %s" % wc)
        best["wenchang"] = wc

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    prob.export_nodes(best, args.out, wenchang=best.get("wenchang"))
    print("已写 %s" % args.out)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "value"])
            w.writerow(["t0_mjd2000", "%.6f" % best["t0"]])
            w.writerow(["t0_utc",
                        mjd_to_utc(best["t0"] + MJD2000).isoformat()])
            w.writerow(["tof_days", ";".join("%.3f" % q for q in best["T"])])
            w.writerow(["J", "%.4f" % best["J"]])
            w.writerow(["vinf_dep_km_s", "%.6f" % best["vd"]])
            w.writerow(["vinf_arr_km_s", "%.6f" % best["va"]])
            w.writerow(["max_defect_m_s", "%.3f" % (best["max_defect"] * 1e3)])
            w.writerow(["feasible", str(best["feasible"])])
            for i, f in enumerate(best["det"], start=1):
                w.writerow(["flyby%d_body" % i, f["body"]])
                w.writerow(["flyby%d_epoch_mjd" % i,
                            "%.6f" % (best["epoch_mjd2000"][i] + MJD2000)])
                w.writerow(["flyby%d_rp_km" % i, "%.3f" % f["rp_req"]])
                w.writerow(["flyby%d_alt_km" % i, "%.1f" % f["alt"]])
                w.writerow(["flyby%d_delta_deg" % i,
                            "%.4f" % math.degrees(f["delta"])])
                w.writerow(["flyby%d_delta_max_deg" % i,
                            "%.4f" % math.degrees(f["dmax"])])
                w.writerow(["flyby%d_vinf_km_s" % i, "%.6f" % f["vinf"]])
                w.writerow(["flyby%d_defect_m_s" % i,
                            "%.3f" % (f["dv"] * 1e3)])
        print("已写 %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
