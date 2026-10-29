#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9：复现验证 —— Quantum-dogdog/Skipping_stone 的 2029 VEEGA 解

被验证对象（该仓库 README 打印的结果）：

    Launch  2029-12-02 23:45 GMT   C3 = 14.981 km²/s²   mass 5000 kg
    Venus   2030-05-14 10:09       turnangle = 52.303°  DV@peri = 0  CA = 3728.8 km
    Earth   2031-04-08 03:51       turnangle = 24.479°  DV@peri = 0  CA = 6926.2 km
    Earth   2033-04-07 15:59       turnangle = 33.608°  DV@peri = 0  CA = 1991.1 km
    Jupiter 2035-09-03 17:04       arrival V-inf = 6.136 km/s
    Total Delta-V = 0

三段复算，全部用 pykep 自带 jpl_lp（JPL 解析历表）+ 修补圆锥（patched conic）：

  A. 点对点 Lambert —— 直接拿 README 的五个历元解 Lambert，看能量链条对不对。
     关键发现：EE 段（2031-04-08 → 2033-04-07，TOF = 730.5056 d）的转移角只有
     0.007°，是死在 1:1 谐振点上的退化构型。step8 默认 revs=[0,0,1,0] 取的是
     1 圈分支（a=1.587 AU），那不是它用的分支；放开圈数才看得到 |v∞|≈10.5 那一支。

  B. 前向瞄准 —— 复现它真正的构造方式。零脉冲借力只约束
        |v∞out| = |v∞in|（能量不变），转角 δ 由近心点 rp 决定，
        出射方向在以 vin 为轴、半顶角 δ 的锥面上（锥面位置 = B 面角 ψ）。
     未知量 (δ, ψ, TOF) 三个，命中下一颗行星三个方程 → 最小二乘直接解，
     不做“行星中心对中心”的两点边值。

  C. 灵敏度 —— 谐振返回是病态问题：ΔTOF 1 分钟 → 地心距 1,861 km，
     Δψ 0.001° → 13,741 km。README 只打印到分钟、也没给 ψ，
     所以逐位复现不可能，只能验证到同族、同量级。

  D. DE421 复核 —— B 段用的 jpl_lp 是低精度历表，D 段把行星位置换成 step6 的
     DE421（.bsp 或 PyPI de421 包）重打一遍靶。结论（本脚本实际打印）：
        D1 出发 C3 复现到 0.013%（14.979 vs 14.981）；
        D2 金星段几何复现（钉住 δ=52.303° + TOF=328.7375 d，只放开 |v∞| 和 ψ
           → 离差 82 km），但所需 |v∞|=6.22 与 DE421 Lambert 实给的 6.61 差 6%；
        D3 地球 1:1 谐振返回基本逐位复现（钉 δ=24.479° → ψ=104.92°、
           TOF=730.5068 d、a=0.999997 AU、TOF/P=1.99999、残差 1.3e3 km，
           返回历元与打印值差约 2 分钟）；
        D4 地球2→木星仍差 ~573 m/s，且 δ 限制在物理可行范围内时最近木星距
           4.3e8 km → 这一段不是零脉冲飞越。

依赖：pykep 3.x（内部 SI 单位 m / m·s⁻¹）、numpy、scipy；复用同目录 step8 的
      PureMGA / lambert_all / flyby_terms，D 段复用 step6 的 DE421 历表接口。

用法：
    python3 step9_复现验证_skippingstone.py            # 完整（A+B+C+D，~10 min）
    python3 step9_复现验证_skippingstone.py --coarse   # 只跑 A 段，几秒
    python3 step9_复现验证_skippingstone.py --no-de421 # 跳过 D 段（无需 DE421）

注：pykep 3.0.1 解释器退出时偶发 "corrupted double-linked list"（C++ 析构顺序问题，
    不影响已打印结果）。本脚本末尾直接 os._exit(0) 规避。
"""

import argparse
import importlib.util
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import pykep as pk

HERE = os.path.dirname(os.path.abspath(__file__))
STEP8 = os.path.join(HERE, "step8_纯MGA借力优化.py")

AU = 1.495978707e11                    # m
NAMES = ["earth", "venus", "earth", "earth", "jupiter"]

# README 五个历元（GMT）→ MJD2000
T_READ = np.array([10928.4896, 11090.9229, 11419.6604, 12150.1660, 13029.2111])
READ = dict(c3=14.981, v_arr=6.136,
            venus=(52.303, 3728.8), earth1=(24.479, 6926.2), earth2=(33.608, 1991.1))
SOI = {"earth": 9.24e5, "jupiter": 1.9e7, "venus": 1.02e6}   # km，粗判“命中”用


def load_step8():
    spec = importlib.util.spec_from_file_location("step8", STEP8)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step8"] = mod
    spec.loader.exec_module(mod)
    return mod


def eph_all(prob, t):
    r, v = [], []
    for i, p in enumerate(prob.seq):
        ri, vi = p.eph(pk.epoch(float(t[i])))
        r.append(np.asarray(ri, float))
        v.append(np.asarray(vi, float))
    return r, v


def cone_dir(a_hat, delta_deg):
    """以 a_hat 为轴、半顶角 delta 的锥面（ψ = B 面角，度 → 单位向量）。"""
    tmp = np.array([0., 0., 1.]) if abs(a_hat[2]) < 0.9 else np.array([1., 0., 0.])
    p1 = np.cross(a_hat, tmp)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(a_hat, p1)
    d = math.radians(delta_deg)

    def f(psi_deg):
        q = math.radians(psi_deg)
        return (a_hat * math.cos(d)
                + (p1 * math.cos(q) + p2 * math.sin(q)) * math.sin(d))
    return f


def rp_from_turn(vinf_kms, delta_deg, mu_p_kms):
    s = math.sin(math.radians(delta_deg) / 2.0)
    if s <= 0:
        return np.inf
    return mu_p_kms / vinf_kms ** 2 * (1.0 / s - 1.0)


def turn_from_rp(vinf_kms, rp_km, mu_p_kms):
    return 2.0 * math.degrees(math.asin(1.0 / (1.0 + rp_km * vinf_kms ** 2 / mu_p_kms)))


def elements(mu, rr, vv):
    eps = float(np.dot(vv, vv)) / 2.0 - mu / float(np.linalg.norm(rr))
    a = -mu / (2.0 * eps)
    P = 2 * math.pi * math.sqrt(a ** 3 / mu) / 86400.0 if a > 0 else float("nan")
    return a, P, eps


# ---------------------------------------------------------------------------
# A. 点对点 Lambert
# ---------------------------------------------------------------------------
def part_a(m, prob):
    KMS = m.KMS
    r, v = eph_all(prob, T_READ)
    mu = prob.mu
    print("=" * 78)
    print("A. 点对点 Lambert（README 五个历元）")
    print("=" * 78)
    dot = np.dot(r[2], r[3]) / np.linalg.norm(r[2]) / np.linalg.norm(r[3])
    print("  EE 段转移角 = %.4f°，TOF = %.4f d = %.6f 儒略年 → 正好压在 1:1 谐振点上"
          % (math.degrees(math.acos(np.clip(dot, -1, 1))), T_READ[3] - T_READ[2],
             (T_READ[3] - T_READ[2]) / 365.2422))
    leg1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = leg1[2] - v[2]
    print("  |v∞in|@地球1 = %.4f km/s      → README 地球1 (δ,CA) 隐含 %.3f km/s"
          % (np.linalg.norm(vin1) * KMS,
             math.sqrt(prob.mu_self[2] * 1e-9 / (prob.radius[2] * KMS + READ["earth1"][1])
                       * (1 / math.sin(math.radians(READ["earth1"][0]) / 2) - 1))))
    print("\n  EE 段 Lambert 分支（multi_revs=3）：")
    print("   %-3s %-11s %-10s %-9s %-10s %s"
          % ("br", "|v∞out@E1|", "需转角δ", "需rp km", "高度 km", "日心轨道 a / P / TOF÷P"))
    for br, v0l, v1l in m.lambert_all(r[2], r[3], T_READ[3] - T_READ[2], mu, 3):
        ft = m.flyby_terms(vin1, v0l - v[2], prob.mu_self[2], prob.rp_min["earth"])
        a, P, _ = elements(mu, r[2], v0l)
        star = "   <== 与 README 隐含 |v∞| 同一量级" if abs(np.linalg.norm(v0l - v[2]) * KMS - 10.55) < 0.4 else ""
        print("   %-3d %9.3f  %8.2f°  %9.0f  %+10.0f  %.4f AU / %.2f d / %.4f%s"
              % (br, np.linalg.norm(v0l - v[2]) * KMS, math.degrees(ft["delta"]),
                 ft["rp_req"], ft["rp_req"] - prob.radius[2] * KMS, a / AU, P,
                 (T_READ[3] - T_READ[2]) / P, star))
    res = prob.evaluate(np.concatenate([[T_READ[0]], np.diff(T_READ)]), combo=(0, 0, 0, 0))
    print("\n  出发 C3 = %.3f km²/s² (README %.3f, 差 %+.1f%%)   木星到达 |v∞| = %.4f km/s (README %.3f, 差 %+.2f%%)"
          % (res["vd"] ** 2, READ["c3"], 100 * (res["vd"] ** 2 / READ["c3"] - 1),
             res["va"], READ["v_arr"], 100 * (res["va"] / READ["v_arr"] - 1)))
    print("  点对点闭合缺陷 max Δv = %.0f m/s → 说明它不是按“中心对中心 Lambert”设计的。"
          % (res["max_defect"] * 1000))


# ---------------------------------------------------------------------------
# B. 前向瞄准
# ---------------------------------------------------------------------------
def make_hit(m, prob, mu):
    KMS = m.KMS

    def miss(rn, vn, vsc, t_node, tof, i_target):
        rr, vv = pk.propagate_lagrangian([list(rn), list(vsc)], tof * 86400.0, mu)
        rb, vb = prob.seq[i_target].eph(pk.epoch(t_node + tof))
        rb = np.asarray(rb, float)
        vb = np.asarray(vb, float)
        return ((np.asarray(rr, float) - rb) / 1e3,
                float(np.linalg.norm(np.asarray(vv, float) - vb)) * KMS)

    def hit(rn, vn, t_node, vin, i_target, tof_nom, delta_fix=None,
            d_grid=None, psi_step=2.0):
        """粗扫 (δ,ψ,TOF) 网格找最贴近，再用最小二乘精修到命中。"""
        KMS_ = m.KMS
        nv = float(np.linalg.norm(vin))
        a_hat = vin / nv
        deltas = [delta_fix] if delta_fix is not None else list(d_grid)
        tof_lo, tof_hi = tof_nom * 0.97, tof_nom * 1.03
        best = (np.inf, None, None, None)

        def tof_best(dd, psi):
            """给定出射方向，只调 TOF 找最小离差（TOF-离差曲线又陡又多谷 → 多起点）。"""
            f = cone_dir(a_hat, dd)
            vsc = vn + f(psi) * nv
            out = (np.inf, tof_nom)

            def g(z):
                return miss(rn, vn, vsc, t_node, float(z[0]), i_target)[0]
            for x0 in (tof_lo, tof_nom * 0.98, tof_nom, tof_nom * 1.02, tof_hi):
                try:
                    s = least_squares(g, [x0], bounds=([tof_lo], [tof_hi]),
                                       xtol=1e-14, ftol=1e-14, diff_step=1e-7)
                except Exception:
                    continue
                D = float(np.linalg.norm(s.fun))
                if D < out[0]:
                    out = (D, float(s.x[0]))
            return out

        for dd in deltas:
            for psi in np.arange(0.0, 360.0, psi_step):
                D, x = tof_best(dd, psi)
                if D < best[0]:
                    best = (D, dd, psi, x)
        # 精修（结果与粗扫取优，避免非凸问题里 least_squares 越走越差）

        def res(z):
            if delta_fix is not None:
                psi, tof, dd = z[0], z[1], delta_fix
            else:
                dd, psi, tof = float(z[0]), float(z[1]), float(z[2])
            f = cone_dir(a_hat, dd)
            return miss(rn, vn, vn + f(psi) * nv, t_node, tof, i_target)[0]
        z0 = [best[2], best[3]] if delta_fix is not None else [best[1], best[2], best[3]]
        seeds = [list(z0)]
        for k in (0.99, 1.01):
            z = list(z0)
            z[-1] = z0[-1] * k
            seeds.append(z)
        for z in list(seeds):
            try:
                s = least_squares(res, z, xtol=1e-14, ftol=1e-14, diff_step=1e-7)
                seeds.append([float(u) for u in s.x])
            except Exception:
                pass
        bz, bd = None, np.inf
        for z in seeds:
            D = float(np.linalg.norm(res(z)))
            if D < bd:
                bz, bd = z, D
        if delta_fix is not None:
            dd, psi, tof = delta_fix, bz[0], bz[1]
        else:
            dd, psi, tof = bz[0], bz[1], bz[2]
        f = cone_dir(a_hat, dd)
        vsc = vn + f(psi) * nv
        Dm, vout = miss(rn, vn, vsc, t_node, tof, i_target)
        return dict(delta=dd, psi=psi, tof=tof, miss=float(np.linalg.norm(Dm)),
                    vinf_in=nv * KMS_, vout=vout, vsc=vsc)

    return hit, miss


def part_b(m, prob):
    KMS = m.KMS
    r, v = eph_all(prob, T_READ)
    mu = prob.mu
    R = {n: prob.radius[i] * KMS for i, n in enumerate(NAMES)}
    MUP = {n: prob.mu_self[i] * 1e-9 for i, n in enumerate(NAMES)}
    hit, miss = make_hit(m, prob, mu)
    print("\n" + "=" * 78)
    print("B. 前向瞄准复现（零脉冲：|v∞out|=|v∞in|，未知量 δ / B 面角 ψ / TOF）")
    print("=" * 78)

    lam0 = m.lambert_all(r[0], r[1], T_READ[1] - T_READ[0], mu, 1)[0]
    vdep = (lam0[1] - v[0]) * KMS
    print("\n  ① 出发段：|v∞| = %.4f km/s → C3 = %.3f km²/s²  [README %.3f，差 %+.1f%%]"
          % (np.linalg.norm(vdep), np.linalg.norm(vdep) ** 2, READ["c3"],
             100 * (np.linalg.norm(vdep) ** 2 / READ["c3"] - 1)))
    vin_v = lam0[2] - v[1]

    qv = hit(r[1], v[1], T_READ[1], vin_v, 2, T_READ[2] - T_READ[1],
             d_grid=np.arange(40.0, 70.01, 2.0), psi_step=4.0)
    rp_v = rp_from_turn(qv["vinf_in"], qv["delta"], MUP["venus"])
    imp_v = math.sqrt(MUP["venus"] / (R["venus"] + READ["venus"][1])
                      * (1 / math.sin(math.radians(READ["venus"][0]) / 2) - 1))
    print("\n  ② 金星飞越 → 地球1（δ, ψ, TOF 全自由）：")
    print("     δ=%.3f°  ψ=%.3f°  TOF=%.4f d  命中残差 %.0f km"
          % (qv["delta"], qv["psi"], qv["tof"], qv["miss"]))
    print("     所需 rp = %.0f km → 高度 %+.0f km；|v∞in|=|v∞out|=%.4f km/s（零脉冲）"
          % (rp_v, rp_v - R["venus"], qv["vinf_in"]))
    print("     [README δ=%.3f° 高度=%.1f km] → δ 差 %.3f°，高度差 %+.1f%%"
          % (READ["venus"][0], READ["venus"][1], qv["delta"] - READ["venus"][0],
             100 * ((rp_v - R["venus"]) / READ["venus"][1] - 1)))
    print("     README 的 (δ,CA) 隐含 |v∞|=%.3f km/s，本算 %.4f km/s（差 %+.1f%%）→ 能量对得上，转角偏大"
          % (imp_v, qv["vinf_in"], 100 * (qv["vinf_in"] / imp_v - 1)))
    qp = hit(r[1], v[1], T_READ[1], vin_v, 2, T_READ[2] - T_READ[1],
             delta_fix=READ["venus"][0], psi_step=2.0)
    print("     把 δ 钉在 README %.3f°（TOF 允许 ±3%%）：最小离差 %.0f km > 地球 SOI %.0e km"
          % (READ["venus"][0], qp["miss"], SOI["earth"]))
    # 一致性检验：把入射渐近线缩到 README 隐含的 |v∞|，看转角/TOF 是否回到打印值
    vin_v2 = vin_v * (imp_v / qv["vinf_in"])
    qv2 = hit(r[1], v[1], T_READ[1], vin_v2, 2, T_READ[2] - T_READ[1],
              d_grid=np.arange(40.0, 70.01, 2.0), psi_step=4.0)
    rp_v2 = rp_from_turn(qv2["vinf_in"], qv2["delta"], MUP["venus"])
    print("     一致性检验：入射 |v∞| 缩到 %.3f → δ=%.3f° TOF=%.4f d 残差 %.0f km 高度 %+.0f km"
          % (imp_v, qv2["delta"], qv2["tof"], qv2["miss"], rp_v2 - R["venus"]))
    print("     ⇒ δ 与 README 差 %.2f°，TOF 差 %+.2f d → %s"
          % (qv2["delta"] - READ["venus"][0], qv2["tof"] - (T_READ[2] - T_READ[1]),
             "转角/历元双双落回打印值附近，说明这一段本身复现，差异主要来自那 1.3% 的能量差"
             if abs(qv2["delta"] - READ["venus"][0]) < 5.0 else "缩放能量并不能解释，需另找原因"))
    print("     （矛盾其实出在金星入射能量本身，见 D 段用 DE421 的复核）")

    lam1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = lam1[2] - v[2]
    q1 = hit(r[2], v[2], T_READ[2], vin1, 3, T_READ[3] - T_READ[2],
             d_grid=np.arange(18.0, 32.01, 1.0), psi_step=5.0)
    rp1 = rp_from_turn(q1["vinf_in"], q1["delta"], MUP["earth"])
    a1, P1, _ = elements(mu, r[2], q1["vsc"])
    dmax1 = 2 * math.degrees(math.asin(1 / (1 + 6678.0 * q1["vinf_in"] ** 2 / MUP["earth"])))
    print("\n  ③ 地球1 飞越 → 谐振返回地球2（δ, ψ, TOF 全自由）：")
    print("     δ=%.3f°  ψ=%.3f°  TOF=%.4f d  命中残差 %.0f km"
          % (q1["delta"], q1["psi"], q1["tof"], q1["miss"]))
    print("     出射日心轨道 a=%.6f AU  P=%.3f d  TOF/P=%.5f → 正好 %.4f 个周期（1:1 谐振）"
          % (a1 / AU, P1, q1["tof"] / P1, q1["tof"] / P1))
    print("     所需 rp = %.0f km → 高度 %+.0f km；|v∞|=%.4f km/s（零脉冲）"
          % (rp1, rp1 - R["earth"], q1["vinf_in"]))
    print("     [README δ=%.3f° 高度=%.1f km] → δ 差 %.3f°，高度差 %+.1f%%"
          % (READ["earth1"][0], READ["earth1"][1], q1["delta"] - READ["earth1"][0],
             100 * ((rp1 - R["earth"]) / READ["earth1"][1] - 1)))
    print("     该 |v∞| 下 300 km 高度允许的 δmax=%.2f° → %s"
          % (dmax1, "可行（真实零脉冲飞越）" if q1["delta"] <= dmax1 else "不可行"))

    tE2 = T_READ[2] + q1["tof"]
    rE2 = np.asarray(prob.seq[3].eph(pk.epoch(tE2))[0], float)
    vE2 = np.asarray(prob.seq[3].eph(pk.epoch(tE2))[1], float)
    arr_r, arr_v = pk.propagate_lagrangian([list(r[2]), list(q1["vsc"])],
                                           q1["tof"] * 86400.0, mu)
    vin2 = np.asarray(arr_v, float) - vE2

    q2f = hit(rE2, vE2, tE2, vin2, 4, T_READ[4] - T_READ[3],
              d_grid=np.arange(20.0, 120.1, 4.0), psi_step=6.0)
    q2r = hit(rE2, vE2, tE2, vin2, 4, T_READ[4] - T_READ[3],
              delta_fix=READ["earth2"][0], psi_step=1.0)
    print("\n  ④ 地球2 飞越 → 木星：")
    imp2 = math.sqrt(MUP["earth"] / (R["earth"] + READ["earth2"][1])
                     * (1 / math.sin(math.radians(READ["earth2"][0]) / 2) - 1))
    print("     到达地球2 |v∞in| = %.4f km/s  [README (δ,CA) 隐含 %.3f，差 %+.1f%%]"
          % (q2f["vinf_in"], imp2, 100 * (q2f["vinf_in"] / imp2 - 1)))
    print("     δ 全自由：δ=%.3f° ψ=%.3f° TOF=%.4f d → 残差 %.0f km（木星 SOI≈%.0e km）"
          % (q2f["delta"], q2f["psi"], q2f["tof"], q2f["miss"], SOI["jupiter"]))
    print("     把 TOF 钉在 README 的 879.045 d 附近、δ 钉在 %.3f°：残差 %.0f km → %s"
          % (READ["earth2"][0], q2r["miss"],
             "命中" if q2r["miss"] < SOI["jupiter"] else "打不到木星"))
    # 按打印历元去木星，地球2 出射需要多大 |v∞|（点对点 Lambert）
    lam3 = m.lambert_all(r[3], r[4], T_READ[4] - T_READ[3], mu, 1)[0]
    need = float(np.linalg.norm(lam3[1] - np.asarray(prob.seq[3].eph(pk.epoch(T_READ[3]))[1], float))) * KMS
    print("     ⇒ 这一段是全链里唯一没复现的：按打印历元 879.045 d 到木星，地球2 出射需要 "
          "|v∞out|=%.4f km/s，" % need)
    print("       而零脉冲（|v∞out|=|v∞in|）只能给 %.4f，缺口 %.0f m/s；README 隐含 %.3f，"
          "居中 → 它的两个地球数据不能同时自洽。"
          % (q2f["vinf_in"], abs(need - q2f["vinf_in"]) * 1000, imp2))
    return q1


# ---------------------------------------------------------------------------
# C. 灵敏度
# ---------------------------------------------------------------------------
def part_c(m, prob, q1):
    KMS = m.KMS
    r, v = eph_all(prob, T_READ)
    mu = prob.mu
    lam1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = lam1[2] - v[2]
    n1 = float(np.linalg.norm(vin1))
    dd, psi, tof = q1["delta"], q1["psi"], q1["tof"]
    f = cone_dir(vin1 / n1, dd)
    print("\n" + "=" * 78)
    print("C. 谐振返回的病态程度（为什么无法逐位复现）")
    print("=" * 78)
    rb0, _ = prob.seq[3].eph(pk.epoch(T_READ[2] + tof))
    rb0 = np.asarray(rb0, float)
    for dt in (0.0, 0.0007, 0.007, 0.01, 0.0731, 0.1):
        x, _ = pk.propagate_lagrangian([list(r[2]), list(v[2] + f(psi) * n1)],
                                       (tof + dt) * 86400.0, mu)
        print("   ΔTOF = %+7.4f d (%+7.1f min)  →  地心距 %12.0f km"
              % (dt, dt * 1440, np.linalg.norm(np.asarray(x, float) - rb0) / 1e3))
    for dps in (0.0, 0.001, 0.01, 0.1, 1.0):
        x, _ = pk.propagate_lagrangian([list(r[2]), list(v[2] + f(psi + dps) * n1)],
                                       tof * 86400.0, mu)
        print("   Δψ   = %+7.3f°              →  地心距 %12.0f km"
              % (dps, np.linalg.norm(np.asarray(x, float) - rb0) / 1e3))
    print("   ⇒ 历元只打印到分钟、ψ 根本没打印 → 不唯一；只能验证到同族同量级。")


def load_step6():
    """step6 里有 DE421（.bsp 或 PyPI de421 包）的日心状态接口，用来复核 B 段。"""
    import warnings
    spec = importlib.util.spec_from_file_location(
        "step6", os.path.join(HERE, "step6_引力辅助修正与EMEJ桥接.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step6"] = mod
    with warnings.catch_warnings():          # step6 导入时会跑自检，警告与本题无关
        warnings.simplefilter("ignore")
        _so, _se = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.stdout, sys.stderr = _so, _se
    return mod


def part_d(m, prob):
    """DE421 复核：B 段用的是 jpl_lp 低精度历表，这里换成 DE421 再打一遍靶。"""
    s6 = load_step6()
    if not getattr(s6, "_B_HAVE_DE421", False):
        print("\n  [D] 没找到 DE421（de421.bsp 或 pip install de421 jplephem），跳过。")
        return
    GM = s6._B_MU * 1e9                    # m³/s²
    OFF = 51544.5                          # MJD2000 → MJD

    def st(name, tm):
        r, v = s6._b_state(name, float(tm) + OFF)
        return np.asarray(r, float) * 1e3, np.asarray(v, float) * 1e3

    R, V = [], []
    for i, n in enumerate(NAMES):
        rr, vv = st(n, T_READ[i])
        R.append(rr)
        V.append(vv)
    MUP = {"earth": 398600.4418, "venus": 324859.0}       # km³/s²
    RAD = {"earth": 6371.0, "venus": 6051.8}              # km

    def miss(rn, vn, vsc, t_node, tof, body):
        rr, vv = pk.propagate_lagrangian([list(rn), list(vsc)], tof * 86400.0, GM)
        rb, vb = st(body, t_node + tof)
        return (np.asarray(rr, float) - rb) / 1e3, \
            float(np.linalg.norm(np.asarray(vv, float) - vb)) / 1e3

    def cone(a_hat, delta_deg):
        tmp = np.array([0., 0., 1.]) if abs(a_hat[2]) < 0.9 else np.array([1., 0., 0.])
        p1 = np.cross(a_hat, tmp)
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(a_hat, p1)
        d = math.radians(delta_deg)

        def f(psi_deg):
            q = math.radians(psi_deg)
            return (a_hat * math.cos(d)
                    + (p1 * math.cos(q) + p2 * math.sin(q)) * math.sin(d))
        return f

    print("\n" + "=" * 78)
    print("D. DE421 复核（step6 的高精度历表，替代 jpl_lp）")
    print("=" * 78)

    # D1 出发能量
    lam0 = m.lambert_all(R[0], R[1], T_READ[1] - T_READ[0], GM, 1)[0]
    vd = float(np.linalg.norm(lam0[1] - V[0])) / 1e3
    vin_v = lam0[2] - V[1]
    nv_v = float(np.linalg.norm(vin_v)) / 1e3
    print("   D1 出发 C3 = %.4f km²/s²   [README %.3f → 差 %+.3f%%]（jpl_lp 给 %.3f）"
          % (vd ** 2, READ["c3"], 100 * (vd ** 2 / READ["c3"] - 1), 14.772))
    print("      金星入射 |v∞| = %.4f km/s  [README (δ,CA) 隐含 6.492]" % nv_v)

    # D2 金星段：δ 与 TOF 都钉在 README，只解 |v∞| 和 ψ
    a_v = vin_v / float(np.linalg.norm(vin_v))
    f_v = cone(a_v, READ["venus"][0])

    def res_v(z):
        s = abs(float(z[0])) * 1e3
        return miss(R[1], V[1], V[1] + f_v(float(z[1])) * s,
                    T_READ[1], T_READ[2] - T_READ[1], "earth")
    def g_v(z):
        return res_v(z)[0]
    best = (np.inf, None, None)
    for s0 in (5.5, 6.0, 6.492, 7.0):
        for p0 in np.arange(0.0, 360.0, 15.0):
            try:
                s = least_squares(g_v, [s0, p0], xtol=1e-15, ftol=1e-15, diff_step=1e-8)
            except Exception:
                continue
            D = float(np.linalg.norm(s.fun))
            if D < best[0]:
                best = (D, abs(float(s.x[0])), float(s.x[1]))
    rp_v = rp_from_turn(best[1], READ["venus"][0], MUP["venus"])
    print("   D2 钉 δ=%.3f° + TOF=%.4f d，只放开 |v∞| 与 ψ → 离差 %.0f km（命中）"
          % (READ["venus"][0], T_READ[2] - T_READ[1], best[0]))
    print("      需要 |v∞|=%.4f km/s（Lambert 实给 %.4f，差 %+.1f%%）、ψ=%.2f° → 近金高度 %+.0f km"
          " [README %.1f km]"
          % (best[1], nv_v, 100 * (nv_v / best[1] - 1), best[2] % 360,
             rp_v - RAD["venus"], READ["venus"][1]))

    # D3 地球1 谐振返回：δ 钉在 README，解 ψ 与 TOF
    lam1 = m.lambert_all(R[1], R[2], T_READ[2] - T_READ[1], GM, 1)[0]
    vin1 = lam1[2] - V[2]
    n1 = float(np.linalg.norm(vin1))
    a1 = vin1 / n1
    f1 = cone(a1, READ["earth1"][0])
    tot = T_READ[3] - T_READ[2]
    best = (np.inf, None, None)
    for p0 in np.arange(0.0, 360.0, 3.0):
        vsc = V[2] + f1(p0) * n1
        g = lambda z: miss(R[2], V[2], vsc, T_READ[2], float(z[0]), "earth")[0]
        out = (np.inf, tot)
        for x0 in (tot * 0.97, tot, tot * 1.03):
            try:
                s = least_squares(g, [x0], bounds=([tot * 0.94], [tot * 1.06]),
                                  xtol=1e-14, ftol=1e-14, diff_step=1e-7)
            except Exception:
                continue
            D = float(np.linalg.norm(s.fun))
            if D < out[0]:
                out = (D, float(s.x[0]))
        if out[0] < best[0]:
            best = (out[0], p0, out[1])
    def g_ee(z):
        return miss(R[2], V[2], V[2] + f1(float(z[0])) * n1,
                    T_READ[2], float(z[1]), "earth")[0]
    bb = best
    for kp in (-0.5, 0.0, 0.5):
        for kt in (0.999, 1.0, 1.001):
            try:
                s = least_squares(g_ee, [best[1] + kp, best[2] * kt],
                                  xtol=1e-15, ftol=1e-15, diff_step=1e-8)
            except Exception:
                continue
            D = float(np.linalg.norm(s.fun))
            if D < bb[0]:
                bb = (D, float(s.x[0]), float(s.x[1]))
    vsc1 = V[2] + f1(bb[1]) * n1
    a_h, P, _ = elements(GM, R[2], vsc1)
    rp1 = rp_from_turn(n1 / 1e3, READ["earth1"][0], MUP["earth"])
    print("   D3 钉 δ=%.3f°（README 地球1 转角），解 ψ + TOF："
          % READ["earth1"][0])
    print("      ψ=%.4f°  TOF=%.4f d  命中残差 %.0f km  |  TOF 与打印值差 %+.4f d"
          % (bb[1], bb[2], bb[0], bb[2] - tot))
    print("      出射日心轨道 a=%.6f AU  P=%.3f d  TOF/P=%.5f → 1:1 谐振；近地高度 %+.0f km"
          " [README %.1f km]"
          % (a_h / AU, P, bb[2] / P, rp1 - RAD["earth"], READ["earth1"][1]))
    print("      返回历元 %s  [README 2033-04-07 15:59 GMT]" % m.mjd_to_utc(T_READ[2] + bb[2] + OFF))

    # D4 地球2 → 木星：把 δ 限制在物理可行范围内
    tE2 = T_READ[2] + bb[2]
    rE2, vE2 = st("earth", tE2)
    arr, av = pk.propagate_lagrangian([list(R[2]), list(vsc1)], bb[2] * 86400.0, GM)
    vin2 = np.asarray(av, float) - vE2
    n2 = float(np.linalg.norm(vin2))
    a2 = vin2 / n2
    dmax2 = 2 * math.degrees(math.asin(1 / (1 + prob.rp_min["earth"] * (n2 / 1e3) ** 2 / MUP["earth"])))
    totj = T_READ[4] - T_READ[3]
    bestj = (np.inf, None, None, None)
    for dd in np.arange(16.0, dmax2, 4.0):
        fj = cone(a2, dd)
        for p0 in np.arange(0.0, 360.0, 12.0):
            vsc = vE2 + fj(p0) * n2
            g = lambda z: miss(rE2, vE2, vsc, tE2, float(z[0]), "jupiter")[0]
            out = (np.inf, totj)
            for x0 in totj * np.array([0.85, 1.0, 1.15]):
                try:
                    s = least_squares(g, [x0], bounds=([totj * 0.85], [totj * 1.15]),
                                      xtol=1e-13, ftol=1e-13, diff_step=1e-7)
                except Exception:
                    continue
                D = float(np.linalg.norm(s.fun))
                if D < out[0]:
                    out = (D, float(s.x[0]))
            if out[0] < bestj[0]:
                bestj = (out[0], dd, p0, out[1])
    lam3 = m.lambert_all(R[3], R[4], T_READ[4] - T_READ[3], GM, 1)[0]
    need = float(np.linalg.norm(lam3[1] - V[3])) / 1e3
    print("   D4 地球2 入射 |v∞| = %.4f km/s（1:1 谐振的对称性 ⇒ 与地球1 出射同值）"
          % (n2 / 1e3))
    print("      按打印历元到木星需要 |v∞out| = %.4f km/s → 零脉冲缺口 %.0f m/s"
          % (need, (need - n2 / 1e3) * 1e3))
    print("      在物理可行的转角范围内（δ ≤ δmax=%.2f°，rp≥%.0f km），扫 ψ/TOF 后"
          "最近木星距 %.0f km（SOI %.0e km）→ 无法命中"
          % (dmax2, prob.rp_min["earth"], bestj[0], SOI["jupiter"]))
    print("   ⇒ 换成 DE421 后 D1/D3 基本逐位复现，金星段只差能量，地球2→木星段依然差 ~0.6 km/s。")


def main():
    ap = argparse.ArgumentParser(description="复现验证 Skipping_stone 的 2029 VEEGA 解")
    ap.add_argument("--coarse", action="store_true", help="只跑 A 段（点对点 Lambert + 分支表）")
    ap.add_argument("--no-de421", action="store_true", help="跳过 D 段（不用 DE421 复核）")
    args = ap.parse_args()
    m = load_step8()
    prob = m.PureMGA(NAMES, (10896.5, 10971.5),
                     [(60, 220), (180, 420), (600, 900), (800, 1500)],
                     revs=[1, 1, 3, 1], rp_min={"earth": 6678.0, "venus": 6352.0},
                     w_fb=2000.0)
    print("序列 %s" % " → ".join(NAMES))
    print("历元 MJD2000 = %s   TOF = %s d" % (T_READ, np.round(np.diff(T_READ), 4)))
    part_a(m, prob)
    if not args.coarse:
        q1 = part_b(m, prob)
        part_c(m, prob, q1)
        if not args.no_de421:
            part_d(m, prob)
    sys.stdout.flush()
    os._exit(0)          # 绕开 pykep 3.0.1 退出时的析构缺陷


if __name__ == "__main__":
    main()
