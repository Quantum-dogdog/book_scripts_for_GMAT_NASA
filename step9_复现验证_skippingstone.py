#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9：Skipping_stone 2029 VEEGA 轨道复算

输入：Skipping_stone 仓库 README 打印的五个节点历元（GMT）——

    Launch  2029-12-02 23:45     Earth  2031-04-08 03:51     Earth  2033-04-07 15:59
    Venus   2030-05-14 10:09     Jupiter 2035-09-03 17:04

序列 earth → venus → earth → earth → jupiter，飞越零脉冲（|v∞out| = |v∞in|）。

四段计算（星历 pykep jpl_lp，日心黄道系）：

  A. 点对点 Lambert —— 五个历元原样代入，解各段 Lambert，给出能量链条
     与地球-地球谐振段的全部多圈分支。
  B. 前向瞄准 —— 零脉冲借力的真实构造方式：出射方向在以 vin 为轴、
     半顶角 δ 的锥面上（锥面位置角 = ψ），未知量 (δ, ψ, TOF)，
     命中下一颗行星三个方程，最小二乘求解。
  C. 灵敏度 —— 谐振返回是病态问题：历元差 1 分钟 → 地心距放大约 2e3 km，
     ψ 差 0.001° → 放大约 1e4 km（量级见打印表）。
  D. 自由搜索 —— 只给序列和窗口，SADE 自己搜同族解（同体腿自动带圈数基因，
     直飞与谐振同场竞争；搜完打一张分支族谱，看谐振那一支在不在场）。

依赖：pykep 3.x（内部 SI 单位）、numpy、scipy；复用同目录 step8 的
      PureMGA / lambert_all / flyby_terms / run_opt。

用法：
    python3 step9_复现验证_skippingstone.py            # A + B + C + 自由搜索
    python3 step9_复现验证_skippingstone.py --coarse   # 只跑 A 段，几秒

注：pykep 3.0.1 解释器退出时偶发堆析构错误（C++ 析构顺序问题，不影响已打印
    结果）。本脚本在出口统一 os._exit() 规避，--help / 异常路径同样覆盖。
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
SOI = {"earth": 9.24e5, "jupiter": 1.9e7, "venus": 1.02e6}   # km，粗判“命中”用
MJD2000_OFF = 51544.5                 # MJD2000 → MJD


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
    print("\n" + "=" * 78)
    print("一、点对点 Lambert（五个历元原样代入）")
    print("=" * 78)
    dot = np.dot(r[2], r[3]) / np.linalg.norm(r[2]) / np.linalg.norm(r[3])
    print("  地球1→地球2 段：转移角 %.4f°，TOF = %.4f d = %.6f 儒略年 —— 压在 1:1 谐振点上"
          % (math.degrees(math.acos(np.clip(dot, -1, 1))), T_READ[3] - T_READ[2],
             (T_READ[3] - T_READ[2]) / 365.2422))
    leg1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = leg1[2] - v[2]
    print("  地球1 入射 |v∞| = %.4f km/s" % (np.linalg.norm(vin1) * KMS))
    print("\n  地球1→地球2 段 Lambert 分支（k = 航天器圈数 0..3）：")
    print("   %-3s %-13s %-9s %-9s %-10s %s"
          % ("k", "|v∞out| km/s", "需转角δ", "需rp km", "近地高度 km",
             "日心 a(AU) / P(d) / TOF÷P"))
    for br, v0l, v1l in m.lambert_all(r[2], r[3], T_READ[3] - T_READ[2], mu, 3):
        ft = m.flyby_terms(vin1, v0l - v[2], prob.mu_self[2], prob.rp_min["earth"])
        a, P, _ = elements(mu, r[2], v0l)
        print("   %-3d %11.3f  %9.2f°  %9.0f  %+10.0f  %.4f / %.2f / %.4f"
              % (br, np.linalg.norm(v0l - v[2]) * KMS, math.degrees(ft["delta"]),
                 ft["rp_req"], ft["rp_req"] - prob.radius[2] * KMS, a / AU, P,
                 (T_READ[3] - T_READ[2]) / P))
    res = prob.evaluate(np.concatenate([[T_READ[0]], np.diff(T_READ)]), combo=(0, 0, 0, 0))
    print("\n  出发 C3 = %.3f km²/s²   木星到达 |v∞| = %.4f km/s"
          % (res["vd"] ** 2, res["va"]))
    print("  全链点对点闭合缺陷 max Δv = %.0f m/s —— 该轨迹不是按中心对中心 Lambert 设计的"
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
    print("二、前向瞄准（零脉冲：|v∞out| = |v∞in|，未知量 δ / ψ / TOF）")
    print("=" * 78)

    lam0 = m.lambert_all(r[0], r[1], T_READ[1] - T_READ[0], mu, 1)[0]
    vdep = (lam0[1] - v[0]) * KMS
    print("\n  ① 出发段（Lambert）：|v∞| = %.4f km/s → C3 = %.3f km²/s²"
          % (np.linalg.norm(vdep), np.linalg.norm(vdep) ** 2))
    vin_v = lam0[2] - v[1]

    qv = hit(r[1], v[1], T_READ[1], vin_v, 2, T_READ[2] - T_READ[1],
             d_grid=np.arange(40.0, 70.01, 2.0), psi_step=4.0)
    rp_v = rp_from_turn(qv["vinf_in"], qv["delta"], MUP["venus"])
    print("\n  ② 金星飞越（δ, ψ, TOF 全自由）：")
    print("     δ = %.3f°   ψ = %.3f°   TOF = %.4f d   命中残差 %.0f km"
          % (qv["delta"], qv["psi"], qv["tof"], qv["miss"]))
    print("     所需 rp = %.0f km（近金高度 %+.0f km），|v∞| = %.4f km/s"
          % (rp_v, rp_v - R["venus"], qv["vinf_in"]))

    lam1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = lam1[2] - v[2]
    q1 = hit(r[2], v[2], T_READ[2], vin1, 3, T_READ[3] - T_READ[2],
             d_grid=np.arange(18.0, 32.01, 1.0), psi_step=5.0)
    rp1 = rp_from_turn(q1["vinf_in"], q1["delta"], MUP["earth"])
    a1, P1, _ = elements(mu, r[2], q1["vsc"])
    dmax1 = 2 * math.degrees(math.asin(1 / (1 + 6678.0 * q1["vinf_in"] ** 2 / MUP["earth"])))
    print("\n  ③ 地球1 飞越 → 谐振返回地球2（δ, ψ, TOF 全自由）：")
    print("     δ = %.3f°   ψ = %.3f°   TOF = %.4f d   命中残差 %.0f km"
          % (q1["delta"], q1["psi"], q1["tof"], q1["miss"]))
    print("     出射日心轨道 a = %.6f AU，P = %.3f d，TOF÷P = %.5f → 1:1 谐振"
          % (a1 / AU, P1, q1["tof"] / P1))
    print("     所需 rp = %.0f km（近地高度 %+.0f km），|v∞| = %.4f km/s"
          % (rp1, rp1 - R["earth"], q1["vinf_in"]))
    print("     该 |v∞| 下 300 km 高度允许的 δmax = %.2f° → %s"
          % (dmax1, "可行（真实零脉冲飞越）" if q1["delta"] <= dmax1 else "不可行"))

    tE2 = T_READ[2] + q1["tof"]
    rE2 = np.asarray(prob.seq[3].eph(pk.epoch(tE2))[0], float)
    vE2 = np.asarray(prob.seq[3].eph(pk.epoch(tE2))[1], float)
    arr_r, arr_v = pk.propagate_lagrangian([list(r[2]), list(q1["vsc"])],
                                           q1["tof"] * 86400.0, mu)
    vin2 = np.asarray(arr_v, float) - vE2

    q2f = hit(rE2, vE2, tE2, vin2, 4, T_READ[4] - T_READ[3],
              d_grid=np.arange(20.0, 120.1, 4.0), psi_step=6.0)
    print("\n  ④ 地球2 飞越 → 木星：")
    print("     地球2 入射 |v∞| = %.4f km/s" % q2f["vinf_in"])
    print("     δ 全自由：δ = %.3f°   TOF = %.4f d   最近木星距 %.2e km（木星 SOI ≈ %.0e km）"
          % (q2f["delta"], q2f["tof"], q2f["miss"], SOI["jupiter"]))
    # 按输入历元到木星，地球2 出射需要多大 |v∞|（点对点 Lambert）
    lam3 = m.lambert_all(r[3], r[4], T_READ[4] - T_READ[3], mu, 1)[0]
    need = float(np.linalg.norm(
        lam3[1] - np.asarray(prob.seq[3].eph(pk.epoch(T_READ[3]))[1], float))) * KMS
    print("     按输入历元 %.3f d 到木星需要 |v∞out| = %.4f km/s，"
          "零脉冲只能给 %.4f km/s → 缺口 %.0f m/s"
          % (T_READ[4] - T_READ[3], need, q2f["vinf_in"],
             abs(need - q2f["vinf_in"]) * 1000))
    print("     （输入历元组自身不自洽：地球2 的到达能量与到达历元不能同时满足）")
    return q1


# ---------------------------------------------------------------------------
# C. 灵敏度
# ---------------------------------------------------------------------------
def part_c(m, prob, q1):
    r, v = eph_all(prob, T_READ)
    mu = prob.mu
    lam1 = m.lambert_all(r[1], r[2], T_READ[2] - T_READ[1], mu, 1)[0]
    vin1 = lam1[2] - v[2]
    n1 = float(np.linalg.norm(vin1))
    dd, psi, tof = q1["delta"], q1["psi"], q1["tof"]
    f = cone_dir(vin1 / n1, dd)
    print("\n" + "=" * 78)
    print("三、谐振返回的灵敏度（历元 / ψ 的小误差 → 地心距放大）")
    print("=" * 78)
    rb0, _ = prob.seq[3].eph(pk.epoch(T_READ[2] + tof))
    rb0 = np.asarray(rb0, float)
    for dt in (0.0, 0.0007, 0.007, 0.01, 0.0731, 0.1):
        x, _ = pk.propagate_lagrangian([list(r[2]), list(v[2] + f(psi) * n1)],
                                       (tof + dt) * 86400.0, mu)
        print("   ΔTOF = %+7.4f d（%+7.1f min）→ 地心距 %12.0f km"
              % (dt, dt * 1440, np.linalg.norm(np.asarray(x, float) - rb0) / 1e3))
    for dps in (0.0, 0.001, 0.01, 0.1, 1.0):
        x, _ = pk.propagate_lagrangian([list(r[2]), list(v[2] + f(psi + dps) * n1)],
                                       tof * 86400.0, mu)
        print("   Δψ   = %+7.3f°            → 地心距 %12.0f km"
              % (dps, np.linalg.norm(np.asarray(x, float) - rb0) / 1e3))
    print("   ⇒ 历元打印到分钟、ψ 未打印 → 同族解不唯一，只能复算到同族同量级。")


# ---------------------------------------------------------------------------
# D. 自由搜索：只给序列和窗口，能否自己搜出这条 2 年地球谐振返回
# ---------------------------------------------------------------------------
LABELS = ["出发", "金星", "地球1", "地球2", "木星"]


def part_d(m, pop=20, gen=20, seed=1, islands=None):
    print("\n" + "=" * 78)
    print("四、自由搜索（SADE，同体腿自动带圈数基因 k=0..3，直飞与谐振同场竞争）")
    print("=" * 78)
    prob = m.PureMGA(NAMES, (T_READ[0] - 20.0, T_READ[0] + 20.0),
                     [(140, 220), (290, 380), (690, 780), (820, 950)],
                     revs=[0, 0, 0, 0],
                     rp_min={"earth": 6678.0, "venus": 6352.0},
                     # kmax 交给 auto（起止同一颗行星的段才给基因）：写死 [0,0,3,0]
                     # 等于替这一段序列做特调，换成 EEJ / EVVEJ 就错位了。
                     kmax=None, dsm_max_mps=200.0)
    # x 的尾巴是圈数基因，个数由 kmax 决定，不手数
    x0 = np.concatenate([[T_READ[0]], np.diff(T_READ), [2.0] * len(prob.k_gene)])
    q = prob.evaluate(x0)
    if not q.get("ok"):
        print("  输入历元点在零脉冲模型下解不开：%s %s"
              % (q.get("reason"), q.get("hint", "")))
        return
    print("\n  ① 输入历元原样代入（第 3 段 k=2，τ 为打靶输出量）：")
    print("     v∞dep = %.4f km/s（C3 = %.3f km²/s²）   v∞arr = %.4f km/s   命中残差 %.0f km"
          % (q["vd"], q["vd"] ** 2, q["va"], q.get("hit_max_km", 0.0)))
    print("     %-8s %-8s %9s %9s %9s %9s %10s"
          % ("飞越", "行星", "δ°", "δmax°", "rp km", "近心高度 km", "v∞ km/s"))
    for i, f in enumerate(q["det"], start=1):
        print("     %-8s %-8s %9.3f %9.2f %9.0f %9.0f %10.4f"
              % ("第%d次" % i, f["body"], math.degrees(f["delta"]),
                 math.degrees(f["dmax"]), f["rp_req"], f["alt"], f["vinf"]))
    rd = (q.get("res_diag") or [{}])[0]
    if rd:
        print("     段3 earth→earth: k=%d  τ=%.4f d  a=%.4f AU  k_sc=%.4f  "
              "行星 j=%.4f  ⇒ j:k = %.3f : %.0f"
              % (rd.get("k", -1), rd.get("tof", float("nan")),
                 rd.get("a_AU", float("nan")), rd.get("k_sc", float("nan")),
                 rd.get("j_planet", float("nan")),
                 rd.get("j_planet", float("nan")), rd.get("k", 0)))
    print("     零脉冲闭合缺陷 max = %.0f m/s（地球2 处）——输入历元组不能零脉冲闭合"
          % (q["max_defect"] * 1e3))

    nisl = islands or min(os.cpu_count() or 1, 8)
    print("\n  ② 从输入点出发的自由搜索（SADE，pop=%d × gen=%d × %d 岛并行，seed=%d）："
          % (pop, gen, nisl, seed))
    best = m.run_opt(prob, pop_size=pop, gen=gen, seed=seed, islands=nisl,
                     x_seed=x0, verbose=False, workers=nisl,
                     spec=m.prob_spec(prob), families=True)
    if not best.get("ok"):
        print("     搜索失败：%s" % best.get("reason"))
        return
    print("     冠军 J = %.4f   v∞dep = %.4f   v∞arr = %.4f km/s   "
          "缺陷 max = %.1f m/s   k = %s   %s"
          % (best["J"], best["vd"], best["va"], best["max_defect"] * 1e3,
             best.get("k"), "可行" if best.get("feasible") else "不可行"))
    print("     关闭脉冲合计 %.1f m/s（每处 ≤ %.0f m/s 判据）"
          % (best.get("dsm_total", 0.0) * 1e3, prob.dsm_max_mps))
    print("\n     冠军节点历元 MJD2000：")
    for i, b in enumerate(best["epoch_mjd2000"]):
        print("       %-8s %14.4f" % (LABELS[i], b))
    for rd in best.get("res_diag") or []:
        print("     谐振段：段%d %s→%s  k=%d  τ=%.3f d  a=%.4f AU  k_sc=%.4f  "
              "j=%.4f  ⇒ j:k = %.3f : %.0f"
              % (rd["leg"] + 1, prob.names[rd["leg"]], prob.names[rd["leg"] + 1],
                 rd["k"], rd["tof"], rd["a_AU"], rd["k_sc"], rd["j_planet"],
                 rd["j_planet"], rd["k"]))
    # 冠军只说明 GA 选了哪一支；这张表才回答"谐振那一支这次有没有被搜过"。
    fams = best.get("families") or []
    if fams:
        print("     分支族谱（各岛末种群按圈数 k 归族，×后面是种群里还剩几个）：")
        for line in m.format_families(fams, indent="       "):
            print(line)


def main():
    ap = argparse.ArgumentParser(
        description="Skipping_stone 2029 VEEGA 轨道复算（零脉冲借力）")
    ap.add_argument("--coarse", action="store_true",
                    help="只跑 A 段（点对点 Lambert + 分支表）")
    ap.add_argument("--no-search", action="store_true", help="跳过自由搜索段")
    ap.add_argument("--search-pop", type=int, default=24)
    ap.add_argument("--search-gen", type=int, default=40)
    ap.add_argument("--search-seed", type=int, default=1)
    ap.add_argument("--search-islands", type=int, default=None,
                    help="D 段独立 SADE 的岛数（默认 auto = min(CPU 核数, 8)，"
                         "一岛一进程；跑 1 就退回只看一个起点）")
    args = ap.parse_args()
    m = load_step8()
    prob = m.PureMGA(NAMES, (10896.5, 10971.5),
                     [(60, 220), (180, 420), (600, 900), (800, 1500)],
                     revs=[1, 1, 3, 1], rp_min={"earth": 6678.0, "venus": 6352.0},
                     w_fb=2000.0,
                     # A~C 段是"点对点 Lambert / 前向瞄准"诊断，必须关掉圈数基因，
                     # 否则 x 里多一个 k，evaluate 会直接判 bad_x。
                     kmax=[0, 0, 0, 0])
    print("=" * 78)
    print("Skipping_stone 2029 VEEGA 复算：earth → venus → earth → earth → jupiter")
    print("星历 pykep jpl_lp（日心黄道系）；飞越零脉冲：|v∞out| = |v∞in|")
    print("=" * 78)
    dates = [m.mjd_to_utc(t + MJD2000_OFF).strftime("%Y-%m-%d %H:%M") for t in T_READ]
    print("输入历元：%s" % " / ".join(
        "%s %s" % (lb, d) for lb, d in zip(LABELS, dates)))
    print("段飞行时间（d）：%s" % " / ".join("%.4f" % q for q in np.diff(T_READ)))
    part_a(m, prob)
    if not args.coarse:
        q1 = part_b(m, prob)
        part_c(m, prob, q1)
        if not args.no_search:
            part_d(m, pop=args.search_pop, gen=args.search_gen,
                   seed=args.search_seed, islands=args.search_islands)


if __name__ == "__main__":
    # pykep 3.0.1 的 C++ 析构在解释器退出时偶发堆损坏（含 --help 的 SystemExit
    # 路径），统一在这里收口：正常退出码透传，异常打印 traceback 后以 1 退出，
    # 最后 os._exit 跳过扩展模块的析构。
    _code = 0
    try:
        main()
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
