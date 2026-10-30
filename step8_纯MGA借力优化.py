#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_纯MGA借力优化.py
=====================
纯 MGA（Multiple Gravity Assist）全局优化器：**飞越近心点零脉冲**。
同体腿的谐振返回（EEJ、EVVEJ…）是搜索"涌现"出来的，不需要预先声明。

模型
----
pykep 的 mga 飞越模型在双曲线末端补一次脉冲（直接进目标函数），那不是"打水漂"式
借力。本脚本把飞越严格写成弹道双曲线（patched conic）：

    |v∞out| = |v∞in|                       （模长守恒）
    δ = angle(v∞in, v∞out)                 （转角由前后两段 Lambert 给定）
    rp = μ/v∞² · (1/sin(δ/2) − 1)          （唯一确定的近心半径，输出量）
    δmax = 2·asin( 1/(1 + rp_min·v∞²/μ) )  （rp_min 下能转的最大角）

超出 δmax 的部分记成"合成缺陷" Δv_fb = |v_out − rotate(v_in, δmax)|，
默认只当罚函数（收敛后必须为 0）；--dsm-max 允许它变成一次真脉冲。

谐振返回（同体腿绕 k 圈）
--------------------------
k 是**整数基因**（kmax>0 的段由 GA 自己选 k=0..kmax），直飞与 1:1 / 2:1 / 3:1
在同一次搜索里一起竞争，不用声明；行星那边走了 j 圈是**输出量**，不是约束——
只要到的时候行星在那儿，j 是多少都成立。于是同一个序列能同时搜 EEJ、EVVEJ……

分两条路径算，原因是几何而不是实现口味（step9 的 A 段、C 段给了实测）：

* **普通腿**（不同天体，或同体但 i=0 的出发段）→ 两点 Lambert。
* **同体返回腿**（i>0 且 kmax>0）→ 转移角≈0° 时两点 Lambert 退化（Skipping_stone
  的 E→E 段转移角只有 0.0071°，要的那一支在绕视线的 B 面角 ψ 方向上，两点
  Lambert 表达不出来）。所以这一段走 (δ,ψ,τ) 打靶：未知量 δ(转角)、ψ(B 面角)、
  τ(飞行时间)，方程=命中行星 x/y/z；T[i] 只当种子兼选支，真 τ 是输出量。τ 直接
  定了下一个节点的历元，因此只能顺着节点顺序算。
* 打靶在 θ≈0° 时的位置残差有一个几百 km 的地板（这一支的固有性质，不是精度问题），
  记成软罚，不当硬门——落地靠飞越后 SOI 出口 1~3 m/s 的关闭脉冲吸收。

决策向量
--------
    x = [t0(MJD2000), T1..TN] + [k_i for 段 where kmax>0]
    飞越 rp 不进染色体（由几何反解）；k 按四舍五入取整。
    同体腿的 T_i 只是打靶种子，返回的 T 是解出来的 τ。

目标函数
--------
    J = w_dep·|v∞_dep| + w_arr·|v∞_arr| + W·Σ (Δv_fb,i)² + 10·Σ(D_i/1e4 km)²
    W 默认 100；零脉冲判据 max|Δv_fb| < 1e-3 km/s（--dsm-max 可放宽）。

用法示例
--------
# 1) Galileo 1989 VEEGA 纯 MGA（地-金-地-地-木，4 段，k 自动）
python3 step8_纯MGA借力优化.py --case eveej --rp-min earth=6678,venus=6352 \
        --polish --wenchang --out puremga_nodes_run.json --csv puremga_run.csv

# 2) 天问四号构型 E-V-E-E-J，用公开发射窗口 2029-09-24 ~ 10-09（窗口是输入不是事后
#    校验）；k 自动，地-地段自己试 0/1/2/3 圈；允许飞越后 200 m/s 的关闭脉冲
python3 step8_纯MGA借力优化.py --case eveej2029 --rp-min earth=6678,venus=6352 \
        --pop 60 --gen 120 --seed 5 --dsm-max 200 --wenchang \
        --out tw4_nodes_run.json --csv tw4_run.csv

# 3) 逐位复放一条已经算过的染色体（不跑 SADE，回归对拍用）；强制圈数用 --kmax/--revs
python3 step8_纯MGA借力优化.py --case eveej --x-seed data/eveej_puremga_1989_nodes.json \
        --replay --out replay_nodes.json
python3 step8_纯MGA借力优化.py --case eveej --kmax 0,0,3,2 --revs 0,0,0,0

# 4) 换窗口扫描：1995-2005 每 20 天一片
python3 step8_纯MGA借力优化.py --case eveej --scan \
        --scan-start 1995-01-01 --scan-end 2005-01-01 --scan-step 20

# 文昌相关：--wenchang 给出发走廊；--wc-max-coast 是停泊滑行上限 min（默认 30，同
# step1），结果里 ok=走廊能进、launchable=走廊+滑行都过；--chain-close 定时闭合
#（"滑完就发"），现历元不满足就沿发射窗口往后找闭合日。
# --eph de421 让整个搜索用 DE421 历表（约慢一个量级，桥接时省掉历元修正）。

并行与"漏解"
------------
* 岛之间互相独立，默认一岛一进程（Python 适应度回调受 GIL 限制，线程并行无效）。
  `--islands auto` = min(CPU 核数, 8)；`--serial` 退回单进程。同一 seed 下并行与
  串行逐位一致，可以放心拿来对拍。
* 冠军只代表 GA 选了哪一支（直飞 / 1:1 / 2:1…），不代表另一支没被搜过。
  所以每次跑完会打一张"分支族谱"：把各岛末种群按圈数 k 归族，每族取 J 最小的一条
  再精算。谐振族要是从表上消失了，说明这次是真的没搜到，而不是"搜到了但更贵"。
  `--no-families` 关掉这张表。

注意
----
* 同体腿的"命中残差地板"是这一支的固有性质（step9 的 C 段：Δτ 1 min ≈ 1861 km、
  Δψ 0.001° ≈ 13741 km），对应飞越后 1~3 m/s 的关闭脉冲。
* --dsm-max 的脉冲放在**飞越后 SOI 出口**（势能≈0，Δv 直接改 v∞），
  近心点仍然是 0，不违反"飞越近心点不点火"。
* 出发/到达脉冲按 |v∞| 计（MGA 惯例），别把 J 直接当总 Δv。
* 打印/导出的 T 是**物理 TOF**；同体腿的种子值另存在 T_seed 里。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import multiprocessing
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
AU = 1.495978707e11                # m
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
    # 缺陷的**矢量**（km/s）：朝向 Lambert 要求、而飞越给不出来的那一段。
    # step6 的 DSMS 是三分量脉冲，只写模长会被当成“三轴同时加”而跑偏。
    dv_vec = vout - rotate_toward(vin, vout, dmax)
    # rp 必须由**真实转角**反解；转角不够时它自然小于 rp_min，
    # 这样不等式约束才看得见违反（不能用 min(delta,dmax) 截断）。
    sh = math.sin(delta * 0.5)
    rpreq = mu_km / vinf ** 2 * (1.0 / sh - 1.0) if sh > 1e-9 else 1e12
    return dict(vin=vin, vout=vout, vinf_in=ni, vinf_out=no, vinf=vinf,
                delta=delta, dmax=dmax, dv=dv, dv_vec=dv_vec, rp_req=rpreq,
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
                 kmax=None, dsm_max_mps=0.0):
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
        # 先按"飞越节点"（1..n-2）填，再给从没当过飞越的星补 inf。
        # 旧写法按位置判"出发/到达 = inf"，序列末尾又是同一颗星时
        # （earth,venus,earth,earth —— 末段就是谐振自由返回）会把地球
        # 先写 6578 再被 inf 覆盖，打靶层于是认为地球不能转 → 全族 nan。
        for i in range(1, len(names) - 1):
            n = names[i]
            alt = DEFAULT_RP_ALT.get(n, 200.0)
            self.rp_min[n] = self.radius[i] * KMS + alt
        for n in names:
            self.rp_min.setdefault(n, np.inf)
        if rp_min:
            self.rp_min.update(rp_min)
        self.w_dep = float(w_dep)
        self.w_arr = float(w_arr)
        self.w_fb = float(w_fb)
        self.vinf_dep_max = vinf_dep_max
        self.vinf_arr_max = vinf_arr_max
        self.max_combos = int(max_combos)
        # ---- 圈数 k 作为整数基因（P0/P1）----------------------------------
        # 学的就是 FEX-Skipping_stone 的做法：同体腿的"谐振返回"不是一种特殊腿，
        # 只是一条 TOF 自由、绕 k 圈的**普通 Lambert 腿**；行星那边走了 j 圈是
        # 解完才知道的输出量，不是约束。于是三件事同时解决：
        #   * 位置命中由 Lambert 保证（不再有 400 km 的地板）；
        #   * 直飞（k=0）与谐振返回（k≥1）在同一次搜索里竞争，不用声明；
        #   * 任何序列、任何腿（包括第 0 段）都能谐振，没有"第 0 段不许谐振"的补丁。
        # kmax[i] = 该段允许的最大圈数；=0 表示该段沿用 --revs 的固定值。
        # 默认 auto：起止同一颗行星的段给 3，其余段 0（不给基因，省维度）。
        if kmax is None:
            kmax = [3 if self.names[i] == self.names[i + 1] else 0
                    for i in range(self.n_legs)]
        self.kmax = [int(v) for v in kmax]
        if len(self.kmax) != self.n_legs:
            raise ValueError("kmax 长度应为 %d" % self.n_legs)
        glegs = [i for i in range(self.n_legs) if self.kmax[i] > 0]
        # k 基因排在 [t0, T...] 之后：x[1+n_legs+j] 是第 glegs[j] 段的圈数
        self.k_gene = [(1 + self.n_legs + j, i) for j, i in enumerate(glegs)]
        # ---- 飞越后的关闭脉冲（P2）-----------------------------------------
        # dsm_max_mps>0 时，|缺陷| ≤ 它的飞越算"序列成立"，缺陷本身当成一次脉冲。
        # 位置放在飞越后 SOI 出口（r≈9e5 km 量级，势能≈0），不是近心点：
        # 那里 Δv 直接改 v∞ 的方向和模长，成本恰好等于合成缺陷
        # |v_out − rotate(v_in, min(δ, δmax))|，而近心点仍然零脉冲——
        # "飞越近心点不点火"这条硬约束没有被破坏。
        self.dsm_max_mps = float(dsm_max_mps)
        self.dsm_tol = self.dsm_max_mps * 1e-3        # km/s
        # 可行性判据用的容差，单位 km/s。零脉冲那一档不是"≤0"而是给了 1 m/s
        # 的绝对容差：飞越缺陷由两次 Lambert 的速度差算出，天然带 1e-13 量级
        # 的浮点噪声，写 `<= 0` 会让同一条轨迹一会儿判可行一会儿判不可行。
        self.feas_tol = self.dsm_tol if self.dsm_max_mps > 0 else 1e-3
        # 打靶热启动缓存（键 = (段号, 圈数基因)）：DE 的相邻个体在 x 空间里很近，
        # 上一次的 (δ,ψ,τ) 通常 30 次迭代内就能收敛，省掉 12 个冷启动。
        # 键里必须带 k：否则 k 基因在种群间来回翻时缓存会互相冲掉（实测慢 4 倍）。
        self._shot_mem = {}

    # -- pygmo 接口 ---------------------------------------------------------
    def get_bounds(self):
        lb = [self.t0_bounds[0]] + [a for a, b in self.tof_bounds]
        ub = [self.t0_bounds[1]] + [b for a, b in self.tof_bounds]
        for gi, leg in self.k_gene:
            lb.append(0.0)
            ub.append(float(self.kmax[leg]))
        return lb, ub

    def get_nobj(self):
        return 1

    def fitness(self, x):
        return [self.evaluate(x)["J"]]

    # -- 解码 ---------------------------------------------------------------
    def decode_k(self, x):
        """从染色体解出每段的圈数：基因段四舍五入，其余段用 --revs 的固定值。"""
        k = list(self.revs)
        for gi, leg in self.k_gene:
            k[leg] = int(min(max(round(float(x[gi])), 0), self.kmax[leg]))
        return k

    def evaluate(self, x, combo=None, detail=True):
        """x = [t0, T1..TN, k...]；k 只出现在 kmax>0 的段上，且按整数解码。

        两条路径，学的是 FEX-Skipping_stone「同一序列里直飞与谐振一起搜」：

        * 快路径（`_evaluate_lambert`）：没有「同体 + i>0 + kmax>0」的腿时，
          全是普通 Lambert，历元先由 cumsum(T) 定死，再一次枚举所有分支组合；
        * 顺序路径（`_evaluate_sequential`）：出现同体腿。该段转移角≈0°
          （Skipping_stone 的 E→E 只有 0.0071°），两点 Lambert 在这一支
          是退化的——只给切向解（a=1 AU 但 |v∞|≈0.5 km/s），而打印解要的是
          |v∞|≈10.5 km/s，那一支必须靠绕视线的 B 面角 ψ 才能转到。
          所以这一段走 `_shot_leg` 的 (δ,ψ,τ) 打靶，τ 是输出量，
          后面每个节点的历元都要等它解出来才能算，只能顺着节点用窄 beam 走。
        """
        x = np.asarray(x, float)
        n_need = 1 + self.n_legs + len(self.k_gene)
        if len(x) != n_need:
            return dict(J=1e9, ok=False, reason="bad_x",
                        hint="染色体长度应为 %d = 1 + %d 段 TOF + %d 个圈数基因"
                             % (n_need, self.n_legs, len(self.k_gene)))
        t0 = float(x[0])
        T = np.asarray(x[1:1 + self.n_legs], float)
        if np.any(T <= 0):
            return dict(J=1e9, ok=False, reason="bad_x")
        k = self.decode_k(x)
        if self.has_shot_leg():
            out = self._evaluate_sequential(t0, T, k, combo)
        else:
            out = self._evaluate_lambert(t0, T, k, combo)
        if not out.get("ok"):
            return out
        out["x"] = x.copy()
        out["t0"] = t0
        out["T_seed"] = T.copy()
        out["k"] = list(k)
        # 顺序路径里 T 只是打靶种子，真正的 TOF 是解出来的 τ；对外统一报物理 TOF，
        # 这样 epoch/导出/CSV/复核都不用再分两条路径。
        true_T = out.get("true_T")
        if true_T is not None:
            out["T"] = np.asarray(true_T, float)
        else:
            out["T"] = T.copy()
        out["epoch_mjd2000"] = t0 + np.concatenate([[0.0], np.cumsum(out["T"])])
        return out if detail else dict(J=out["J"])

    def has_shot_leg(self):
        """是否需要打靶层：i>0 的「同体 + 给了圈数基因」腿必须打靶。"""
        for i in range(1, self.n_legs):
            if self.kmax[i] > 0 and self.names[i] == self.names[i + 1]:
                return True
        return False

    def _evaluate_lambert(self, t0, T, k, combo=None):
        """快路径：全部是普通 Lambert，历元由 cumsum(T) 直接给出。"""
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
            s = lambert_all(r[i], r[i + 1], T[i], self.mu, k[i])
            if not s:
                return dict(J=1e9, ok=False, reason="lambert",
                            hint=self._lambert_hint(i, T[i], k[i]))
            if self.branch[i] != "auto":
                want = int(self.branch[i])
                s = [q for q in s if q[0] == want] or s
            sols.append(s)

        if combo is not None:
            idx = [min(c, len(s) - 1) for c, s in zip(combo, sols)]
            combos = [idx]
            cut = False
        else:
            ncomb = 1
            for s in sols:
                ncomb *= len(s)
            cut = ncomb > self.max_combos
            if not cut:
                combos = list(itertools.product(*[range(len(s)) for s in sols]))
            else:
                combos = [tuple(0 for _ in sols)]

        best = None
        for cb in combos:
            legs = [(sols[i][cb[i]][1], sols[i][cb[i]][2])
                    for i in range(self.n_legs)]
            out = self._score(legs, r, v, k, T)
            if out is None:
                continue
            if best is None or out["J"] < best["J"]:
                best = out
                best["combo"] = tuple(int(c) for c in cb)
                best["branch_ids"] = tuple(int(sols[i][cb[i]][0])
                                           for i in range(self.n_legs))
        if best is not None and cut:
            # 分支组合太多，只试了每段第 0 支：报给调用方，免得把"没搜到"
            # 当成"不存在"（族谱表、--max-combos 全靠这个标记提示）。
            best["combo_space"] = ncomb
            best["combo_truncated"] = True
        return best if best is not None else dict(J=1e9, ok=False, reason="score")

    def _evaluate_sequential(self, t0, T, k, combo=None):
        """顺序路径：同体腿走 (δ,ψ,τ) 打靶，历元逐节点推进，分支用窄 beam 搜。"""
        try:
            r0, v0 = self.seq[0].eph(pk.epoch(t0))
        except Exception:
            return dict(J=1e9, ok=False, reason="eph")
        states = [dict(ep=[t0], r=[np.asarray(r0, float)],
                       v=[np.asarray(v0, float)], legs=[], true_T=[],
                       hit=[], branches=[], pen=0.0)]
        width = 3
        for i in range(self.n_legs):
            shot = (i > 0 and self.kmax[i] > 0
                    and self.names[i] == self.names[i + 1])
            nxt = []
            for st in states:
                ri, vi = st["r"][i], st["v"][i]
                if shot:
                    vin = st["legs"][-1][1] - vi
                    rec = self._shot_leg(i, ri, vi, vin, st["ep"][i],
                                         float(T[i]), k[i])
                    if rec is None:
                        continue
                    epn = st["ep"][i] + rec["tof"]
                    try:
                        rb, vb = self.seq[i + 1].eph(pk.epoch(epn))
                    except Exception:
                        continue
                    cands = [(0, rec["vsc"], rec["varr"], rec["tof"], rec["D"])]
                else:
                    epn = st["ep"][i] + T[i]
                    try:
                        rb, vb = self.seq[i + 1].eph(pk.epoch(epn))
                    except Exception:
                        continue
                    sols = lambert_all(ri, rb, T[i], self.mu, k[i])
                    if not sols:
                        continue
                    if self.branch[i] != "auto":
                        want = int(self.branch[i])
                        sols = [q for q in sols if q[0] == want] or sols
                    if combo is not None and i < len(combo):
                        want = int(combo[i])
                        sols = [q for q in sols if q[0] == want] or sols
                    cands = [(q[0], q[1], q[2], T[i], 0.0) for q in sols]
                for bid, vs, va_, tau, D in cands:
                    legs = st["legs"] + [(vs, va_)]
                    pen = st["pen"]
                    if i > 0:
                        ft = flyby_terms(legs[i - 1][1] - vi, vs - vi,
                                         self.mu_self[i],
                                         self.rp_min[self.names[i]])
                        if ft is None:
                            continue
                        pen += ft["dv"] ** 2
                    nxt.append(dict(ep=st["ep"] + [epn], r=st["r"] + [rb],
                                    v=st["v"] + [vb], legs=legs,
                                    true_T=st["true_T"] + [float(tau)],
                                    hit=st["hit"] + ([float(D)] if shot else []),
                                    branches=st["branches"] + [int(bid)],
                                    pen=pen))
            if not nxt:
                if shot:
                    return dict(
                        J=1e9, ok=False, reason="resonant",
                        hint="第 %d 段（%s→%s）零脉冲飞越打靶在 T=%.1f d ±20%% "
                             "内找不到命中：入射 v∞ 的可行转角锥里没有能按时回到 "
                             "%s 的那一支——动 T、动 k，或者换一段/降 kmax"
                             % (i, self.names[i], self.names[i + 1],
                                float(T[i]), self.names[i + 1]))
                return dict(J=1e9, ok=False, reason="lambert",
                            hint=self._lambert_hint(i, T[i], k[i]))
            nxt.sort(key=lambda s: s["pen"])
            states = nxt[:width]
        best = None
        for st in states:
            out = self._score(st["legs"], st["r"], st["v"], k, st["true_T"],
                              hit=st["hit"])
            if out is None:
                continue
            if best is None or out["J"] < best["J"]:
                best = out
                best["combo"] = tuple(st["branches"])
                best["branch_ids"] = tuple(st["branches"])
                best["true_T"] = list(st["true_T"])
        return best if best is not None else dict(J=1e9, ok=False, reason="score")

    def _body_period_days(self, r, v):
        """把行星当前状态换算成日心轨道周期（天），用来报"行星走了几圈"。"""
        eps = 0.5 * float(np.dot(v, v)) - self.mu / float(np.linalg.norm(r))
        if eps >= 0:
            return float("nan")
        a = -self.mu / (2.0 * eps)
        return 2.0 * math.pi * math.sqrt(a ** 3 / self.mu) / DAY

    def _shot_leg(self, i, r0, v0, vin, t_node, tof_seed, k):
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
        bnds = ([0.5, -720.0, lo], [170.0, 720.0, hi])

        def finish(sol):
            """把 least_squares 的结果折算成 (排序键, 记录)。"""
            dd, psi, tau = (float(z) for z in sol.x)
            g = geom(sol.x)
            D = float(np.linalg.norm(sol.fun))
            # 日心周期 → 实际谐振阶
            eps = float(np.dot(g[2], g[2])) / 2.0 - self.mu \
                / float(np.linalg.norm(r0))
            a_h = -self.mu / (2.0 * eps) if eps < 0 else float("inf")
            P = (2.0 * math.pi * math.sqrt(a_h ** 3 / self.mu) / DAY
                 if math.isfinite(a_h) and a_h > 0 else float("nan"))
            k_eff = tau / P if P and math.isfinite(P) and P > 0 else 1e9
            return ((abs(k_eff - k), D),
                    dict(D=D, delta=dd, psi=psi, tof=tau,
                         vsc=g[2], varr=g[1], k_eff=k_eff,
                         a_km=a_h * 1e-3, P=P))

        best = None
        # 0) 热启动：上一轮同一个段的解。收敛得比冷启动快一个数量级，
        #    只接受"圈数还对得上"的结果，避免把 k 搜索锁死在一个错支上。
        warm = self._shot_mem.get((i, k))
        if warm is not None and lo <= warm["tof"] <= hi:
            try:
                sol = least_squares(resid, [warm["delta"], warm["psi"],
                                            warm["tof"]],
                                    bounds=bnds, xtol=1e-14, ftol=1e-14,
                                    diff_step=1e-7, max_nfev=40)
                rank, rec = finish(sol)
                if rec["D"] < 2.0e3 and abs(rec["k_eff"] - k) < 0.35:
                    self._shot_mem[(i, k)] = rec
                    return rec
            except Exception:
                pass
        for fd in (0.25, 0.5, 0.75, 0.95):
            for psi0 in (0.0, 120.0, 240.0):
                z0 = [min(max(fd * dmax_deg, 1.0), 170.0), psi0, tof_seed]
                try:
                    sol = least_squares(resid, z0, bounds=bnds,
                                        xtol=1e-14, ftol=1e-14,
                                        diff_step=1e-7, max_nfev=80)
                except Exception:
                    continue
                rank, rec = finish(sol)
                if rec["D"] > 1.0e5:      # 5 位数量级都没打中，直接丢掉
                    continue
                if best is None or rank < best[0]:
                    best = (rank, rec)
        if best is None:
            return None
        # 多起点只负责"找对支"，选中之后再收紧一次（FEX 的全局→局部分层）。
        # 注意：转移角≈0° 时这一支的 ‖f‖ 有一个几百 km 的地板（光滑、对预算
        # 不敏感），所以命中残差记进目标函数当软罚，不再当硬门——
        # 真要落地就靠 --dsm-max 那 1~3 m/s 的关闭脉冲吸收。
        try:
            sol = least_squares(resid, [best[1]["delta"], best[1]["psi"],
                                        best[1]["tof"]],
                                bounds=bnds, xtol=1e-14, ftol=1e-14,
                                diff_step=1e-7,
                                max_nfev=300)
            rank, rec = finish(sol)
            if rec["D"] <= best[1]["D"]:
                best = (rank, rec)
        except Exception:
            pass
        self._shot_mem[(i, k)] = best[1]
        return best[1]

    def _score(self, legs, r, v, k, T, hit=None):
        vd = float(np.linalg.norm(legs[0][0] - v[0])) * KMS
        va = float(np.linalg.norm(legs[-1][1] - v[-1])) * KMS
        if not (np.isfinite(vd) and np.isfinite(va)):
            return None
        pen = 0.0
        det = []
        # 谐振诊断：同一颗行星重复出现（或 k>0）的段，报"航天器绕了几圈 /
        # 行星那边走了几圈"。比例 (j:k) 是**输出**，不是约束——这正是
        # Skipping_stone 的做法：只要到的时候行星在那儿，j 是多少都成立。
        res_diag = []
        for i in range(self.n_legs):
            if self.names[i] != self.names[i + 1] and k[i] == 0:
                continue
            eps = (0.5 * float(np.dot(legs[i][0], legs[i][0]))
                   - self.mu / float(np.linalg.norm(r[i])))
            a_sc = -self.mu / (2.0 * eps) if eps < 0 else float("inf")
            P_sc = (2.0 * math.pi * math.sqrt(a_sc ** 3 / self.mu) / DAY
                    if math.isfinite(a_sc) else float("nan"))
            P_pl = self._body_period_days(r[i], v[i])
            res_diag.append(dict(
                leg=i, body=self.names[i], k=int(k[i]),
                same_body=(self.names[i] == self.names[i + 1]),
                tof=float(T[i]), a_AU=float(a_sc / AU) if math.isfinite(a_sc) else float("nan"),
                P_sc=P_sc, P_planet=P_pl,
                k_sc=float(T[i] / P_sc) if P_sc > 0 else float("nan"),
                j_planet=float(T[i] / P_pl) if P_pl > 0 else float("nan")))
        for i in range(1, self.n_legs):
            vin = legs[i - 1][1] - v[i]
            vout = legs[i][0] - v[i]
            ft = flyby_terms(vin, vout, self.mu_self[i],
                             self.rp_min[self.names[i]])
            if ft is None:
                return None
            dv = float(ft["dv"])
            pen += dv ** 2
            # 缺陷的"门"要能被目标函数看见，否则 v∞ 项会把 30 m/s 的缺陷
            # 当成零（w_fb=100 时 33 m/s 只值 0.11，而 v∞ 是 4~6 km/s）。
            #   --dsm-max>0：阈值内不加罚（允许的脉冲），越界按二次重罚；
            #   --dsm-max=0：直接按 1 m/s 归一化重罚，逼到机器精度。
            tol = self.feas_tol
            pen += 2.0 * (max(0.0, dv - tol) / tol) ** 2 / self.w_fb
            ft["body"] = self.names[i]
            ft["alt"] = ft["rp_req"] - self.radius[i] * KMS
            det.append(ft)
        J = self.w_dep * vd + self.w_arr * va + self.w_fb * pen
        if self.vinf_dep_max is not None:
            J += 100.0 * max(0.0, vd - self.vinf_dep_max) ** 2
        if self.vinf_arr_max is not None:
            J += 100.0 * max(0.0, va - self.vinf_arr_max) ** 2
        # 谐振返回腿的"命中残差"软罚：转移角≈0° 的那一支有个几百 km 的
        # 地板，物理上由飞越后 1~3 m/s 的关闭脉冲吸收，所以不能当硬门。
        # 1e4 km 归一化 → 400 km 只值 0.016，远小于 1 m/s 的飞越缺陷
        # （w_fb=100 → 1 m/s 值 100）；它只在"两条支一样好"时起到排序作用。
        hit_max = max(hit, default=0.0) if hit else 0.0
        if hit:
            J += 10.0 * sum((float(d) / 1.0e4) ** 2 for d in hit)
        max_dv = max((f["dv"] for f in det), default=0.0)
        dsm_total = float(sum(f["dv"] for f in det))
        # 可行判据：
        #   dsm_max=0（默认）  严格零脉冲：缺陷 < 1 m/s 且每个飞越的 rp 都够大；
        #   dsm_max>0          允许把缺陷当成"飞越后 SOI 出口的一次脉冲"，
        #                      只要每一处都不超过 dsm_max 就算序列成立。
        if self.dsm_max_mps > 0:
            ok_fb = max_dv <= self.feas_tol
        else:
            ok_fb = (max_dv < self.feas_tol and
                     all(f["rp_req"] >= self.rp_min[f["body"]] for f in det))
        return dict(J=float(J), ok=True, vd=vd, va=va, pen=pen, det=det,
                    legs=legs, max_defect=max_dv, dsm_total=dsm_total,
                    dsm_max=max_dv, res_diag=res_diag, feasible=ok_fb,
                    hit_max_km=float(hit_max), hit=list(hit or []))

    def _lambert_hint(self, leg, tof_days, k=0):
        """Lambert 无解时告诉用户是"哪一段、什么圈数"，而不是只报一个 lambert。"""
        extra = ""
        if self.kmax[leg] > 0:
            extra = ("；这一段是圈数基因（kmax=%d），说明当前解码到 k=%d 这一支"
                     "在 T=%.1f d 下不存在——让 TOF 或 k 换一支，"
                     "或者把 kmax 降下来" % (self.kmax[leg], k, tof_days))
        return ("第 %d 段（%s→%s，T=%.2f d）在 %d 圈下没有 Lambert 解："
                "多半是这个圈数对这段 TOF 物理上不存在（TOF 太短、或转移角不合适）"
                "%s" % (leg, self.names[leg], self.names[leg + 1], tof_days,
                        k, extra))

    # -- 结果导出 -----------------------------------------------------------
    # 行星影响球半径（km），只用于把"关闭脉冲"放到真实 SOI 出口时刻
    SOI_KM = dict(mercury=1.12e5, venus=6.16e5, earth=9.25e5,
                  mars=5.77e5, jupiter=4.82e7, saturn=5.46e7)

    def _soi_exit_days(self, i, rp_km, vinf_kms):
        """近心点 → SOI 出口的双曲线飞行时间（天）。

        step6 的桥接按"飞越体 SOI 出口"切段，关闭脉冲如果硬写 +1 d，
        遇到金星（tout≈1.19 d）就会落在段起点之前、直接报错。
        这里按 r = A(e·cosh H − 1)、t = √(A³/μ)(e·sinh H − H) 反解。
        """
        A = self.seq[i].mu_self / (vinf_kms * 1.0e3) ** 2      # m
        e = 1.0 + rp_km * 1.0e3 / A
        r_soi = self.SOI_KM.get(self.names[i], 1.0e6) * 1.0e3
        ch = (r_soi / A + 1.0) / e
        if ch <= 1.0:
            return 1.0
        H = math.acosh(ch)
        t = math.sqrt(A ** 3 / self.seq[i].mu_self) * (e * math.sinh(H) - H)
        return t / 86400.0

    def export_nodes(self, res, path, wenchang=None, extra_meta=None):
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
                source="step8 pure MGA (ballistic flybys; DSM only if --dsm-max)",
                t0_mjd2000=float(ep[0]),
                tof_days=[float(t) for t in res["T"]],
                objective=float(res["J"]),
                vinf_dep_km_s=float(res["vd"]),
                vinf_arr_km_s=float(res["va"]),
                branch_ids=[int(b) for b in res["branch_ids"]],
                k_revs=[int(q) for q in res.get("k", self.revs)],
                kmax=[int(q) for q in self.kmax],
                max_flyby_defect_m_s=float(res["max_defect"] * 1e3),
                dsm_total_m_s=float(res.get("dsm_total", 0.0) * 1e3),
                dsm_max_allowed_m_s=float(self.dsm_max_mps),
                feasible=bool(res.get("feasible")),
                # 谐振不是被声明的：同体腿/多圈腿上报"航天器 k 圈 / 行星 j 圈"，
                # j:k 是输出量（Skipping_stone 的判据就是这个：到的时候行星在那儿）
                resonance=[dict(leg=q["leg"] + 1, body=q["body"], k=q["k"],
                                same_body=bool(q["same_body"]),
                                tof_days=float(q["tof"]),
                                a_AU=float(q["a_AU"]),
                                period_days=float(q["P_sc"]),
                                k_sc=float(q["k_sc"]),
                                planet_period_days=float(q["P_planet"]),
                                j_planet=float(q["j_planet"]))
                           for q in res.get("res_diag", [])],
                x=[float(q) for q in res["x"]],
                wenchang=wenchang,
                wenchang_optimum=res.get("wenchang_optimum"),
                chain_scan=res.get("chain_scan"),
            ),
        )
        # extra_meta 用 update 而不是 **：同名键会被覆盖，
        # 而不是抛 "dict() got multiple values for keyword argument"。
        if extra_meta:
            nodes["meta"].update(extra_meta)
        # 关闭脉冲的位置：飞越之后的 SOI 出口（不是近心点）。没有缺陷的飞越写 None。
        # 与 step6 的约定一致：dsms[i] 属于第 i 段（节点 i → 节点 i+1），
        # 所以"飞越 i 之后"落在索引 i，出发节点（索引 0）永远是 None。
        # 历元必须是真实 SOI 出口（金星 ≈1.19 d、地球 ≈1.10 d），
        # 写死 +1 d 会落在 step6 的段起点之前，桥接直接抛 "不在本段内"。
        def _dsm_node(i, ft):
            if ft["dv"] <= 1e-6:
                return None
            tout = self._soi_exit_days(i, float(ft["rp_req"]),
                                       float(np.linalg.norm(vi_ref[i])))
            return dict(epoch_mjd=float(ep[i] + MJD2000 + tout),
                        dv_km_s=[float(q) for q in ft["dv_vec"]],
                        dv_mag_m_s=float(ft["dv"] * 1e3),
                        soi_exit_days=float(tout),
                        note="飞越%d 后 SOI 出口（近心点后 %.2f d）的关闭脉冲，"
                             "近心点仍为 0" % (i, tout))

        vi_ref = {}
        for i in range(1, self.n_legs):
            vb = self.seq[i].eph(pk.epoch(ep[i]))[1]
            vi_ref[i] = (V[i - 1][1] - np.asarray(vb, float)) * KMS
        nodes["dsms"] = [None] + [_dsm_node(i, ft)
                                  for i, ft in enumerate(res["det"], start=1)]
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
    lb = np.asarray(lb, float).copy()
    ub = np.asarray(ub, float).copy()
    # 圈数基因是整数，SLSQP 对它们没有意义：钉死在冠军值上（上下界都设成它）。
    for gi, _leg in prob.k_gene:
        lb[gi] = ub[gi] = float(x0[gi])

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
def prob_spec(prob):
    """把 prob 拆成一堆纯数据，好送进子进程（pykep 的行星对象不能 pickle）。"""
    return dict(
        eph=getattr(prob, "eph_kind", "jpl_lp"),
        t0_window=(list(prob.t0_window)
                   if getattr(prob, "t0_window", None) else None),
        kwargs=dict(names=list(prob.names),
                    t0_bounds=tuple(prob.t0_bounds),
                    tof_bounds=[tuple(q) for q in prob.tof_bounds],
                    revs=list(prob.revs), branch=list(prob.branch),
                    rp_min={k: v for k, v in prob.rp_min.items()
                            if np.isfinite(v)},
                    w_dep=prob.w_dep, w_arr=prob.w_arr, w_fb=prob.w_fb,
                    vinf_dep_max=prob.vinf_dep_max,
                    vinf_arr_max=prob.vinf_arr_max,
                    max_combos=prob.max_combos, kmax=list(prob.kmax),
                    dsm_max_mps=prob.dsm_max_mps))


def prob_from_spec(spec):
    """prob_spec 的逆运算：子进程里按原样重建 prob。"""
    p = PureMGA(**spec["kwargs"])
    if spec.get("t0_window"):
        p.t0_window = tuple(spec["t0_window"])
    if spec.get("eph") == "de421":
        p = make_de421_prob(p, _step6_mod())
    p.eph_kind = spec.get("eph", "jpl_lp")
    return p


def _pin_blas_threads():
    """一进程一岛，BLAS 再各自开满线程只会互相抢核（还会把小矩阵拖慢）。"""
    for k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"


def _evolve_island(prob, pop_size, gen, seed_k, x_seed=None):
    """一个岛 = 一次独立的 SADE。返回 (冠军结果, 末种群 x, 末种群 f)。"""
    pgprob = pg.problem(prob)
    pop = pg.population(pgprob, size=pop_size, seed=seed_k)
    if x_seed is not None:
        try:
            pop.set_x(0, np.asarray(x_seed, float))
        except Exception:
            pass
    # seed 必须同时给 sade：pygmo 的 SADE 有自己的 RNG，不给 seed 时每次运行
    # 都不同，`--seed` 只固定了初始种群 → 冠军不可复现。
    pop = pg.algorithm(pg.sade(gen=gen, seed=seed_k)).evolve(pop)
    res = prob.evaluate(pop.champion_x)
    return res, [np.asarray(q, float) for q in pop.get_x()], \
        [float(np.ravel(q)[0]) for q in pop.get_f()]


def _island_task(payload):
    """子进程入口：重建 prob → 跑一个岛。"""
    _pin_blas_threads()
    spec, pop_size, gen, seed_k, x_seed = payload
    return _evolve_island(prob_from_spec(spec), pop_size, gen, seed_k, x_seed)


def _collect_families(prob, runs):
    """把各岛末种群按"圈数签名"归族，每族取 f 最小的一条再精算一次。

    为什么要这张表：SADE 只报冠军，冠军只说明它选了哪一支（直飞 / 1:1 / 2:1…），
    不说明另一支没被搜过。谐振族一旦在种群里绝迹，日志上看不出来——
    等换个子窗口它又冒出来，就会误判成"上次算错了"。族谱表就是防这个的。
    """
    best, count = {}, {}
    for _res, xs, fs in runs:
        for x, f in zip(xs, fs):
            key = tuple(int(q) for q in prob.decode_k(x))
            count[key] = count.get(key, 0) + 1
            if key not in best or f < best[key][0]:
                best[key] = (float(f), np.asarray(x, float))
    fams = []
    for key, (f, x) in best.items():
        r = prob.evaluate(x)
        fams.append(dict(k=list(key), n=count[key], f=f,
                         J=float(r.get("J", f)),
                         vd=float(r.get("vd", np.nan)),
                         va=float(r.get("va", np.nan)),
                         defect_m_s=float(r.get("max_defect", np.nan)) * 1e3,
                         feasible=bool(r.get("feasible")),
                         x=[float(q) for q in x]))
    fams.sort(key=lambda q: q["J"])
    return fams


def format_families(fams, champ_k=None, top=8, indent=""):
    """_collect_families 的结果 → 紧凑几行（step8 冠军段、step9 D 段共用）。

    一族一行，直飞族和各谐振族并排列着，谁被搜过、种群里还剩几个，一眼看得见。
    J 是带罚项的目标值，一旦跳到 1e5 以上就说明这一支被罚项打爆、不通。
    """
    out = []
    for q in fams[:top]:
        out.append("%sk=%-12s J=%9.4g  v∞dep=%6.3f  v∞arr=%6.3f  "
                   "缺陷=%8.1f m/s  %-3s ×%4d%s"
                   % (indent, "[" + ",".join(str(v) for v in q["k"]) + "]",
                      q["J"], q["vd"], q["va"], q["defect_m_s"],
                      "可行" if q["feasible"] else "—", q["n"],
                      "  ← 冠军" if champ_k is not None
                      and q["k"] == champ_k else ""))
    if len(fams) > top:
        out.append("%s其余 %d 族 J 更大，完整表见导出 JSON 的 meta.families"
                   % (indent, len(fams) - top))
    return out


def _prepare_multiprocessing():
    """让子进程按名字 import 回定义 worker 的这个模块。

    Windows 的 spawn 不像 fork 那样把内存照抄一份，它是重新 `import` 一遍：
    pickle 只记 worker 的 `__module__` + 函数名。直接跑本脚本时 `__name__` 是
    `__main__`（spawn 会按路径重跑主脚本，没问题）；但 step9 是 importlib 按文件
    路径把本脚本装成 `step8` 的，这个名字在子进程里既不在 sys.path、文件名也对
    不上，于是 `ModuleNotFoundError: No module named 'step8'` + 进程池直接崩。
    这里统一改写成"文件名的 stem"，并把脚本目录塞进 sys.path（spawn 会把
    sys.path 原样传给子进程），谁调用都能还原。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        sys.modules.setdefault(stem, sys.modules[__name__])
    except Exception:
        pass
    for fn in (_island_task, _scan_task):
        fn.__module__ = stem


def run_opt(prob, pop_size=160, gen=600, seed=1, islands=1,
            x_seed=None, verbose=False, workers=1, spec=None, families=False):
    """跑 islands 次互相独立的 SADE，取 J 最小的一支。

    workers>1 时一岛一进程：适应度是本脚本的 Python 回调，受 GIL 限制，
    线程并行等于没并。每个岛都用 seed+k，且各自从空的热启动缓存出发，
    所以并行和 --serial 逐位一致。--x-seed 只塞进第 0 个岛（当种子用，
    不是复放；要逐位复放某条染色体用 --replay）。
    """
    seeds = [seed + k for k in range(max(1, int(islands)))]
    if workers > 1 and len(seeds) > 1:
        if spec is None:
            spec = prob_spec(prob)
        _pin_blas_threads()          # 父进程先设好，spawn 出来的子进程才继承得到
        _prepare_multiprocessing()
        tasks = [(spec, pop_size, gen, sk, x_seed if k == 0 else None)
                 for k, sk in enumerate(seeds)]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(int(workers), len(seeds)), mp_context=ctx) as ex:
            runs = list(ex.map(_island_task, tasks))
    else:
        runs = []
        for k, sk in enumerate(seeds):
            prob._shot_mem = {}       # 岛之间不共享热启动缓存，才叫互相独立
            runs.append(_evolve_island(
                prob, pop_size, gen, sk, x_seed if k == 0 else None))

    if verbose:
        for k, (res, _xs, _fs) in enumerate(runs):
            print("  [island %d] J=%.4f  v∞dep=%.3f v∞arr=%.3f  "
                  "缺陷max=%.2f m/s  %s" %
                  (k + 1, res["J"], res.get("vd", np.nan),
                   res.get("va", np.nan), res.get("max_defect", np.nan) * 1e3,
                   "可行" if res.get("feasible") else "不可行"))
    best = min((r[0] for r in runs), key=lambda r: r["J"])
    if families:
        best["families"] = _collect_families(prob, runs)
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


def wenchang_check(res, prob: PureMGA, rp_launch=6700.0, coast_max=30.0):
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
    # step1 的"能发"口径除了走廊还要滑行短：默认 ≤30 min。
    # 这里把两条都写进结果，ok=几何可行，launchable=几何+滑行都过。
    short = [w for w in wins if w.coast_opt <= coast_max]
    # 定时闭合残差：发射 → 滑行 → 出 SOI 是否正好落在出发历元 t_dep。
    # 这一项就是 step1 里"低倾角发射时出 SOI 比 departure 早/晚几个小时"的那个差，
    # step8 的 t0 是行星中心历元（纯 MGA 的约定），到 SOI 出口那 ~2.5 d 不建模，
    # 所以必须单独报出来，别让"走廊 ok"冒充"能发"。
    gap_min = float((best.t_opt + timedelta(minutes=best.coast_opt)
                     + timedelta(hours=float(getattr(hyp, "t_soi_h", 0.0)))
                     - t_dep).total_seconds() / 60.0)
    return dict(ok=True, vinf_km_s=float(np.linalg.norm(vinf_eq)),
                dec_deg=float(hyp.dec_deg), ra_deg=float(hyp.ra_deg),
                t_launch_utc=best.t_opt.strftime("%Y-%m-%d %H:%M:%S"),
                inc_deg=float(best.i_opt), az_deg=float(best.az_opt),
                coast_min=float(best.coast_opt),
                coast_max_min=float(coast_max),
                coast_ok=bool(best.coast_opt <= coast_max),
                n_windows_coast_ok=len(short),
                coast_min_min=float(min(w.coast_opt for w in wins)),
                soi_transit_h=float(getattr(hyp, "t_soi_h", 0.0)),
                soi_gap_min=gap_min,
                chain_ok=bool(abs(gap_min) <= 30.0),
                launchable=bool(best.coast_opt <= coast_max),
                dt_to_dep_min=float((best.t_opt - t_dep).total_seconds() / 60.0),
                n_windows=len(wins),
                i_min_deg=float(lowest.i_opt),
                az_at_imin_deg=float(lowest.az_opt))


def launch_closure(res, prob: PureMGA, coast_max=30.0, back_days=3.0,
                   step_min=1.0):
    """定时闭合：**解** t_L + coast(t_L) + t_SOI = t_dep。

    和 `wenchang_check` 的分工：
      · wenchang_check 取"窗口里最省倾角"的那一刻，那一刻的 SOI 出口一般
        不在 t_dep 上（实测差 +1.3~2.0 d）——它回答的是"这个 v∞ 能不能打"；
      · 这里换一个问法：在窗口里找**恰好**让 SOI 出口落在 t_dep 的发射时刻。
        滑行 coast 是发射时刻的函数（入轨点→近心点的停泊轨道相位），所以
        方程 f(t) = (t − t_TMI) + coast(t) = 0 每小时有唯一根，扫描+二分即可。
    这正是 step2 里 `tmi_hit` 做的事，只是把锚点从 TMI 换成"SOI 出口"。

    返回 dict，字段与 wenchang_check 对齐（可直接写进 nodes/报告），另外多
    `n_roots / closure_residual_min / method`。
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
        hyp = s1.build_hyperbola(vinf_eq, 6700.0)
    except Exception as exc:
        return dict(ok=False, err=str(exc))
    az_lo, az_hi = float(getattr(s1, "AZ_LO", 90.0)), float(getattr(s1, "AZ_HI", 175.0))
    t_tmi = t_dep - timedelta(hours=hyp.t_soi_h)
    step = float(step_min)
    n = int(max(1.0, float(back_days) * 1440.0) / step)
    t_start = t_tmi - timedelta(days=float(back_days))

    roots, prev = [], None
    for k in range(n + 1):
        t = t_start + timedelta(minutes=k * step)
        g = s1.prograde_at(hyp, t, s1.WENCHANG[0], s1.WENCHANG[1])
        if g is None:
            prev = None
            continue
        f = (t - t_tmi).total_seconds() / 60.0 + g.coast_min
        # coast 每次过近心点会跳一个停泊周期，跳变处不是根，用 |Δf| 筛掉
        if prev is not None and prev[1] * f <= 0 and abs(f - prev[1]) < hyp.period_min * 0.4:
            a, b, fa = prev[0], t, prev[1]
            for _ in range(50):
                mm = a + (b - a) / 2
                gm = s1.prograde_at(hyp, mm, s1.WENCHANG[0], s1.WENCHANG[1])
                if gm is None:
                    break
                fm = (mm - t_tmi).total_seconds() / 60.0 + gm.coast_min
                if fa * fm <= 0:
                    b = mm
                else:
                    a, fa = mm, fm
            ts = a + (b - a) / 2
            gm = s1.prograde_at(hyp, ts, s1.WENCHANG[0], s1.WENCHANG[1])
            if gm:
                soi = ts + timedelta(minutes=gm.coast_min) + timedelta(hours=hyp.t_soi_h)
                roots.append(dict(
                    t_launch_utc=ts.strftime("%Y-%m-%d %H:%M:%S"),
                    coast_min=float(gm.coast_min), inc_deg=float(gm.inc_deg),
                    az_deg=float(gm.az_deg),
                    in_corridor=bool(az_lo <= gm.az_deg <= az_hi),
                    soi_exit_utc=soi.strftime("%Y-%m-%d %H:%M:%S"),
                    residual_min=float((soi - t_dep).total_seconds() / 60.0)))
            prev = None
            continue
        prev = (t, f)

    in_corr = [q for q in roots if q["in_corridor"]]
    short = [q for q in in_corr if q["coast_min"] <= coast_max]
    pool = short or in_corr or roots
    best = None
    if pool:
        # 有短滑行解就取倾角最小；只能长滑行时取滑行最短
        best = min(pool, key=(lambda q: (q["inc_deg"], q["coast_min"]))
                   if short else (lambda q: (q["coast_min"], q["inc_deg"])))
    base = dict(ok=True, method="timer closure: lift+coast+t_SOI == t_dep",
                vinf_km_s=float(np.linalg.norm(vinf_eq)),
                dec_deg=float(hyp.dec_deg), ra_deg=float(hyp.ra_deg),
                soi_transit_h=float(hyp.t_soi_h),
                coast_max_min=float(coast_max),
                corridor_deg=[az_lo, az_hi],
                n_roots=len(roots), n_roots_in_corridor=len(in_corr),
                n_roots_short=len(short),
                t_dep_utc=t_dep.strftime("%Y-%m-%d %H:%M:%S"),
                roots=roots)
    if best is None:
        base.update(ok=False, reason="%.1f d 内没有走廊内的定时闭合根" % back_days,
                    chain_ok=False, launchable=False)
        return base
    gap = best["residual_min"]
    base.update(
        t_launch_utc=best["t_launch_utc"], inc_deg=best["inc_deg"],
        az_deg=best["az_deg"], coast_min=best["coast_min"],
        coast_ok=bool(best["coast_min"] <= coast_max),
        in_corridor=bool(best["in_corridor"]),
        soi_exit_utc=best["soi_exit_utc"],
        soi_gap_min=float(gap),
        closure_residual_min=float(gap),
        chain_ok=bool(abs(gap) <= 1.0),
        launchable=bool(best["coast_min"] <= coast_max and best["in_corridor"]),
        dt_to_dep_min=float((datetime.strptime(best["t_launch_utc"], "%Y-%m-%d %H:%M:%S")
                             - t_dep).total_seconds() / 60.0),
        i_min_deg=float(min(q["inc_deg"] for q in roots)),
        az_at_imin_deg=float(min(roots, key=lambda q: q["inc_deg"])["az_deg"]),
        coast_min_min=float(min(q["coast_min"] for q in roots)),
        n_windows_coast_ok=len(short))
    return base


def chain_close(prob: PureMGA, best, coast_max=30.0, pop=24, gen=30,
                iters=3, seed=1, verbose=True):
    """像 step2 那样，把出发历元换成"滑完就发"的那一瞬。

    step2 的做法是**解方程**而不是"事后对表"：
        先算出发 v∞ → 反解出 TMI/近心点时刻 → 在当日窗口里解出满足
        t_L + coast = t_TMI 的发射时刻。
    这里完全一样，只是锚点从近心点换成 SOI 出口（纯 MGA 的 departure）：
        t_L + coast(t_L) + t_SOI = t_dep
    当前 t0 若解不出"滑行 ≤ coast_max"的走廊根，就沿着窗口把 t0 挪到
    能解出来的那一天（每挪一格都重收轨迹，缺陷仍压到 0）。
    返回新的 best；闭合表挂在 best["chain_scan"] / best["wenchang"] 上。
    """
    q0 = launch_closure(best, prob, coast_max=coast_max)
    log = [dict(t0=float(best["t0"]), J=float(best["J"]),
                defect_m_s=float(best.get("max_defect", 0.0) * 1e3),
                **{k: q0.get(k) for k in ("t_launch_utc", "coast_min", "inc_deg",
                                          "az_deg", "in_corridor", "soi_gap_min",
                                          "launchable")})]
    if verbose:
        print("  [定时闭合] t0=%.4f  %s  滑行 %s min  倾角 %s°  方位 %s°  "
              "残差 %s min"
              % (best["t0"],
                 q0.get("t_launch_utc", "—"),
                 "%.1f" % q0["coast_min"] if q0.get("coast_min") is not None else "—",
                 "%.2f" % q0["inc_deg"] if q0.get("inc_deg") is not None else "—",
                 "%.2f" % q0["az_deg"] if q0.get("az_deg") is not None else "—",
                 "%+.1f" % q0["soi_gap_min"] if q0.get("soi_gap_min") is not None else "—"))
    if q0.get("launchable"):
        best["chain_scan"] = log
        best["wenchang"] = q0
        if verbose:
            print("  [定时闭合] 现历元即闭合：滑行 %.1f min ≤ %.0f min，"
                  "倾角 %.2f°，SOI 出口精确落在 t0" %
                  (q0["coast_min"], coast_max, q0["inc_deg"]))
        return best

    window = getattr(prob, "t0_window", None) or prob.t0_bounds
    span = float(getattr(prob, "_cc_span", 3.0))
    stepd = float(getattr(prob, "_cc_step", 0.05))
    # 扫描只允许往"发射窗口"里挪：窗口是输入（P3），不能挪到窗口外去凑闭合
    t_hi = min(float(window[1]), float(best["t0"]) + span)
    n = int(round(span / stepd))
    cur = best
    closed = []            # 所有"滑完就发"的格点：最后按倾角挑，不要见好就收
    for k in range(1, n + 1):
        tg = float(best["t0"]) + k * stepd
        if tg > t_hi:
            break
        old = prob.t0_bounds
        prob.t0_bounds = (tg - 0.02, tg + 0.02)
        try:
            xx = list(cur["x"]); xx[0] = tg
            cand = polish(prob, dict(cur, x=xx))
            if (not cand.get("ok")) or abs(cand["x"][0] - tg) > 0.03:
                cand = prob.evaluate(xx)
        finally:
            prob.t0_bounds = old
        if not cand.get("ok"):
            continue
        q = launch_closure(cand, prob, coast_max=coast_max)
        log.append(dict(t0=float(cand["t0"]), J=float(cand["J"]),
                        defect_m_s=float(cand.get("max_defect", 0.0) * 1e3),
                        **{kk: q.get(kk) for kk in ("t_launch_utc", "coast_min",
                                                    "inc_deg", "az_deg", "in_corridor",
                                                    "soi_gap_min", "launchable")}))
        if q.get("launchable") and cand.get("feasible"):
            closed.append((float(q["inc_deg"]), float(cand["J"]), cand))
        # 没闭合的格子也保留：只要物理目标没跑掉，就继续拿它当下一格的起点
        if cand["J"] < cur["J"] + 0.5:
            cur = cand
    if closed:
        # 低倾角优先（step1 的"低倾角 + 短滑行"口径），同倾角逐 J
        closed.sort(key=lambda z: (z[0], z[1]))
        best = closed[0][2]
    best["chain_scan"] = log
    qf = launch_closure(best, prob, coast_max=coast_max)
    best["wenchang"] = qf
    if verbose:
        print("  [定时闭合] 扫描 %d 格（%.2f d/格）→ t0=%.4f，%s 滑行 %s min，"
              "倾角 %s°，方位 %s°，残差 %+.2f min，chain_ok=%s"
              % (len(log) - 1, stepd, best["t0"],
                 qf.get("t_launch_utc", "—"),
                 "%.1f" % qf["coast_min"] if qf.get("coast_min") is not None else "—",
                 "%.2f" % qf["inc_deg"] if qf.get("inc_deg") is not None else "—",
                 "%.2f" % qf["az_deg"] if qf.get("az_deg") is not None else "—",
                 qf.get("soi_gap_min", float("nan")), qf.get("chain_ok")))
    return best


# ---------------------------------------------------------------------------
# DE421 复核：把低精度历表下搜到的解拿到高精度历表下重打一遍靶
# ---------------------------------------------------------------------------
def load_step6():
    """step6 里有 DE421（.bsp 或 PyPI de421 包）的日心状态接口，复用它。"""
    import importlib.util
    import warnings
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "step6_引力辅助修正与EMEJ桥接.py")
    spec = importlib.util.spec_from_file_location("step6_de421", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step6_de421"] = mod
    so, se = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = open(os.devnull, "w"), open(os.devnull, "w")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(mod)
    finally:
        sys.stdout, sys.stderr = so, se
    return mod


_STEP6_CACHE = []


def _step6_mod():
    """load_step6 的进程内缓存（pygmo 会反复 deepcopy，别每次都重载星历）。"""
    if not _STEP6_CACHE:
        _STEP6_CACHE.append(load_step6())
    return _STEP6_CACHE[0]


class _EphDE421:
    """把 step6 的 DE421 接口伪装成 pykep.planet 需要的最小面。
    只需要 `.eph(epoch) → (r[m], v[m/s])`、`.mu_self`、`.radius`、
    `.mu_central_body`。step6 的 `_b_state` 返回 (km, km/s) 的日心黄道量，
    这里统一转成 SI，和 PureMGA 内部（jpl_lp 也是 SI）一致。
    """

    def __init__(self, name, s6=None):
        self.name = name
        self._s6 = s6
        if s6 is not None:
            self.mu_self = float(s6.GM[name]) * 1e9
            rad = float(getattr(s6, "_B_RADII", {}).get(name, 0.0))
            self.radius = rad * 1e3
            self.mu_central_body = float(s6._B_MU) * 1e9
        else:
            self._fill()

    def _fill(self):
        s6 = _step6_mod()
        self._s6 = s6
        self.mu_self = float(s6.GM[self.name]) * 1e9
        rad = float(getattr(s6, "_B_RADII", {}).get(self.name, 0.0))
        self.radius = rad * 1e3
        self.mu_central_body = float(s6._B_MU) * 1e9

    # pygmo 会 deepcopy 整个 UDP；module 对象不可 pickle，落地时丢掉、用时再取
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_s6"] = None
        return d

    def __setstate__(self, st):
        self.__dict__.update(st)
        if self._s6 is None:
            self._fill()

    def eph(self, epoch):
        # pykep.epoch 的 mjd2000 是属性（有些版本是方法），两种都兼容
        if hasattr(epoch, "mjd2000"):
            t = epoch.mjd2000
            t = t() if callable(t) else t
        else:
            t = float(epoch)
        r, v = self._s6._b_state(self.name, float(t) + MJD2000)
        return (np.asarray(r, float) * 1e3, np.asarray(v, float) * 1e3)


def make_de421_prob(prob, s6):
    """复制一个 prob，但把所有行星换成 DE421 历表（含 μ、半径）。"""
    p2 = PureMGA(prob.names, prob.t0_bounds, prob.tof_bounds, revs=prob.revs,
                 branch=prob.branch,
                 rp_min={k: v for k, v in prob.rp_min.items()},
                 w_dep=prob.w_dep, w_arr=prob.w_arr, w_fb=prob.w_fb,
                 vinf_dep_max=prob.vinf_dep_max, vinf_arr_max=prob.vinf_arr_max,
                 max_combos=prob.max_combos, kmax=prob.kmax,
                 dsm_max_mps=prob.dsm_max_mps)
    p2.seq = [_EphDE421(n, s6) for n in prob.names]
    # 真实发射窗口要跟着复制：定时闭合只许在窗口里挪 t0，丢了这个属性
    # 就会退回"CLI 子窗口"那一小段，扫描被立刻截断。
    if getattr(prob, "t0_window", None):
        p2.t0_window = tuple(prob.t0_window)
    p2.mu = float(s6._B_MU) * 1e9
    p2.mu_self = [float(p.mu_self) for p in p2.seq]
    for i, n in enumerate(prob.names):          # 半径同步成 DE421 的值
        if p2.seq[i].radius > 0:
            p2.radius[i] = p2.seq[i].radius
    return p2


# ---------------------------------------------------------------------------
# 算例配置
# ---------------------------------------------------------------------------
def case_config(case):
    """返回 names, t0_bounds, tof_bounds, revs, kmax。

    t0_bounds 就是**发射窗口本身**（P3）：真实任务的流程是"窗口是输入、
    弹道在窗口里解"，不是"先随便解一条、事后看文昌打不打得了"。
    """
    if case == "eveej":
        # Galileo: 地-金-地-地-木（1989）
        return dict(
            names=["earth", "venus", "earth", "earth", "jupiter"],
            t0_bounds=(parse_date("1989-06-01"), parse_date("1989-12-31")),
            tof_bounds=[(60, 220), (180, 420), (600, 900), (800, 1500)],
            revs=[0, 0, 1, 0],
        )
    if case == "eveej2029":
        # 天问四号公开构型 E-V-E-E-J + 公开发射窗口 2029-09-24 ~ 10-09
        # （文昌/长征五号，2035-12 到木星系）
        return dict(
            names=["earth", "venus", "earth", "earth", "jupiter"],
            t0_bounds=(parse_date("2029-09-24"), parse_date("2029-10-09")),
            tof_bounds=[(150, 260), (240, 420), (600, 900), (800, 1100)],
            revs=[0, 0, 2, 0],
        )
    if case == "skipping2029":
        # Skipping_stone 打印解的那个 2029 窗口（出发 2029-12-02）
        return dict(
            names=["earth", "venus", "earth", "earth", "jupiter"],
            t0_bounds=(parse_date("2029-11-20"), parse_date("2029-12-15")),
            tof_bounds=[(140, 200), (300, 360), (700, 760), (840, 920)],
            revs=[0, 0, 2, 0],
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
    kmax = None
    if args.kmax and args.kmax.strip().lower() != "auto":
        kmax = [int(v) for v in args.kmax.split(",")]
        if len(kmax) != len(names) - 1:
            raise ValueError("--kmax 要给 %d 个数（段数），收到 %d 个"
                             % (len(names) - 1, len(kmax)))
    prob = PureMGA(names, t0b, tofb, revs=revs, branch=branch, rp_min=rp_min,
                   w_dep=args.w_dep, w_arr=args.w_arr, w_fb=args.w_fb,
                   vinf_dep_max=args.vinf_dep_max,
                   vinf_arr_max=args.vinf_arr_max,
                   max_combos=args.max_combos,
                   kmax=kmax, dsm_max_mps=args.dsm_max)
    # 真实发射窗口（P3）本身：定时闭合扫描只许在它里面挪 t0
    prob.t0_window = tuple(float(v) for v in cfg["t0_bounds"])
    prob.eph_kind = getattr(args, "eph", "jpl_lp")
    if getattr(args, "eph", "jpl_lp") == "de421":
        # 直接在 DE421 下搜（比 jpl_lp 慢，但 step6 桥接时几乎不用再改历元）
        prob = make_de421_prob(prob, load_step6())
        prob.eph_kind = "de421"
        print("历表: JPL DE421（全搜索都用它；step6 桥接可直接接着用）")
    return prob


# ---------------------------------------------------------------------------
# 窗口扫描
# ---------------------------------------------------------------------------
def _scan_row(res):
    """把一个窗口的最优解压成一行（能跨进程 pickle，也只有这些字段进 CSV）。"""
    if not res.get("ok"):
        return None
    return dict(t0=res["t0"], J=res["J"], vd=res["vd"],
                va=res["va"], max_defect=res["max_defect"],
                feasible=bool(res["feasible"]),
                T=";".join("%.2f" % q for q in res["T"]),
                rp=";".join("%.0f" % f["rp_req"] for f in res["det"]),
                alt=";".join("%.0f" % f["alt"] for f in res["det"]),
                delta=";".join("%.1f" % math.degrees(f["delta"])
                               for f in res["det"]),
                combo=";".join(str(c) for c in res["combo"]),
                krevs=";".join(str(q) for q in res.get("k", [])),
                dsm_total="%.1f" % (res.get("dsm_total", 0.0) * 1e3),
                jk=";".join("%.2f:%.0f" % (q["j_planet"], q["k"])
                            for q in res.get("res_diag", [])))


def _scan_task(payload):
    """子进程里跑一片窗口：重建 prob → 单岛 SADE → 只回一行。"""
    _pin_blas_threads()
    spec, pop_size, gen, seed_k = payload
    res = run_opt(prob_from_spec(spec), pop_size=pop_size, gen=gen,
                  seed=seed_k, islands=1, verbose=False)
    return _scan_row(res)


def scan_windows(args, prob: PureMGA, workers: int = 1):
    """按 t0 窗口切片，每片跑一个短 DE，汇总所有可行解。

    切片之间互不知情，所以天生适合并行；打印仍按窗口顺序，不随谁先算完变。
    """
    t_start = parse_date(args.scan_start)
    t_end = parse_date(args.scan_end)
    step = float(args.scan_step)
    width = float(args.scan_width)
    base = prob_spec(prob)
    payloads = []
    t = t_start
    idx = 0
    while t <= t_end:
        idx += 1
        sp = dict(base, kwargs=dict(base["kwargs"],
                                    t0_bounds=(t - width / 2, t + width / 2)))
        payloads.append((sp, args.scan_pop, args.scan_gen, args.seed + 17 * idx,
                         t))
        t += step
    if workers > 1 and len(payloads) > 1:
        _pin_blas_threads()
        _prepare_multiprocessing()
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(int(workers), len(payloads)),
                mp_context=ctx) as ex:
            got = list(ex.map(_scan_task, [p[:4] for p in payloads]))
    else:
        got = [_scan_task(p[:4]) for p in payloads]

    rows = []
    for i, (row, (_sp, _p, _g, _s, tw)) in enumerate(zip(got, payloads), 1):
        if row is None:
            continue
        row = dict(window=tw, **row)
        rows.append(row)
        print("  [%3d] t0=%8.1f J=%7.3f dep=%5.2f arr=%5.2f "
              "def=%6.1f m/s %s" %
              (i, row["t0"], row["J"], row["vd"], row["va"],
               row["max_defect"] * 1e3, "OK" if row["feasible"] else ""))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="纯 MGA（无 DSM、飞越零脉冲）pykep+pygmo 优化器")
    ap.add_argument("--case", default="eveej",
                    choices=["eveej", "eveej2029", "skipping2029", "emej"])
    ap.add_argument("--t0", nargs=2, metavar=("START", "END"),
                    help="出发窗口（= 真实发射窗口），日期或 MJD2000")
    ap.add_argument("--tof", nargs="+", metavar="MIN,MAX",
                    help="逐段飞行时间上下界（天）")
    ap.add_argument("--revs", help="逐段固定 Lambert 圈数，如 0,0,1,0")
    ap.add_argument("--kmax", default="auto",
                    help="逐段允许的最大圈数（整数基因），如 0,0,3,2。"
                         "默认 auto：起止同一颗行星的段给 3，其余段 0。"
                         "同一颗星重复出现的段，k=1/2/3 就是 1:1、2:1、3:1 那些"
                         "谐振返回，k=0 是直飞——它们在同一次搜索里一起竞争")
    ap.add_argument("--branch", help="逐段 Lambert 分支号；默认 auto")
    ap.add_argument("--names", help="自定义行星序列，如 earth,venus,earth,earth")
    ap.add_argument("--dsm-max", type=float, default=0.0,
                    help="允许的飞越后关闭脉冲上限 m/s（默认 0 = 严格零脉冲）。"
                         "填 200 表示每处缺陷 ≤200 m/s 就算序列成立；脉冲位置在"
                         "飞越后 SOI 出口，近心点仍然不点火")
    ap.add_argument("--eph", choices=["jpl_lp", "de421"], default="jpl_lp",
                    help="搜索用的历表。de421 慢一些，但和 step6 桥接同一套星历，"
                         "桥接时的历元修正和残余缺陷小得多")
    ap.add_argument("--rp-min", help="飞越近心半径下限 km，如 earth=6678,venus=6352")
    ap.add_argument("--w-dep", type=float, default=1.0, help="出发 v∞ 权重")
    ap.add_argument("--w-arr", type=float, default=1.0, help="到达 v∞ 权重")
    ap.add_argument("--w-fb", type=float, default=100.0, help="飞越缺陷罚权重")
    ap.add_argument("--vinf-dep-max", type=float, default=None,
                    help="出发 v∞ 软上限 km/s")
    ap.add_argument("--vinf-arr-max", type=float, default=None)
    ap.add_argument("--pop", type=int, default=180)
    ap.add_argument("--gen", type=int, default=800)
    ap.add_argument("--islands", default="auto",
                    help="互相独立的 SADE 跑几次（默认 auto = min(CPU 核数, 8)）。"
                         "岛多只是多几个起点，冠军取其中 J 最小的")
    ap.add_argument("--max-combos", type=int, default=64,
                    help="一条染色体每段 Lambert 分支组合的枚举上限（默认 64）。"
                         "分支数超过它就直接截断，只比每段第 0 支——截断会在冠军"
                         "那行标出来，别把没枚举当成不存在")
    ap.add_argument("--serial", action="store_true",
                    help="单进程串行跑所有岛（对拍/调试用，结果与并行逐位一致）")
    ap.add_argument("--workers", type=int, default=None,
                    help="并行度上限，默认 auto = min(岛数, CPU 核数, 8)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x-seed", help="已知染色体 JSON（含 x 字段）")
    ap.add_argument("--replay", action="store_true",
                    help="跳过 SADE，把 --x-seed 那条染色体原样解码一遍再导出"
                         "（逐位复放，回归对拍用）")
    ap.add_argument("--no-families", action="store_true",
                    help="不打印分支族谱（默认打，一屏以内）")
    ap.add_argument("--polish", action="store_true", help="SLSQP 精修")
    ap.add_argument("--wenchang", action="store_true", help="文昌几何后检")
    ap.add_argument("--wc-max-coast", type=float, default=30.0,
                    help="文昌判据里的停泊轨道滑行上限 min（默认 30，同 step1 "
                         "--max-coast）；超过就标 coast_ok=False / launchable=False")
    ap.add_argument("--chain-close", action="store_true",
                    help="定时闭合：解 t_L + coast + t_SOI = t0（像 step2 的 tmi_hit），"
                         "当前历元解不出短滑行发射就沿窗口往后找，找到能 '滑完就发' 的那天")
    ap.add_argument("--cc-span", type=float, default=3.0,
                    help="往后扫描多少天找可闭合的发射日（默认 3）")
    ap.add_argument("--cc-step", type=float, default=0.05,
                    help="扫描步长，天（默认 0.05 = 72 min）")
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

    ncpu = os.cpu_count() or 1
    islands = (min(ncpu, 8) if str(args.islands).lower() == "auto"
               else max(1, int(args.islands)))
    workers = 1 if args.serial else min(islands, args.workers or min(ncpu, 8))
    prob = build_prob(args)

    if args.scan:
        print("扫描 t0 窗口：%s → %s，步长 %.0f 天，窗宽 %.0f 天（%d 进程）" %
              (args.scan_start, args.scan_end, args.scan_step, args.scan_width,
               workers))
        rows = scan_windows(args, prob, workers=workers)
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

    print("纯 MGA 借力优化：%s（星历 %s，黄道系）"
          % (" → ".join(prob.names), prob.eph_kind))
    if prob.k_gene and not args.replay:
        tag = "、".join("第%d段(kmax=%d)" % (leg + 1, prob.kmax[leg])
                        for _gi, leg in prob.k_gene)
        print("圈数基因：%s —— 同体/多圈段由 GA 自己选 k（k≥1 即谐振返回，"
              "与直飞同场竞争）" % tag)
    if prob.dsm_max_mps > 0:
        print("关闭脉冲：每处飞越后 SOI 出口 ≤ %.0f m/s（近心点仍为 0）"
              % prob.dsm_max_mps)
    print("出发窗口 MJD2000 %.2f ~ %.2f；rp_min %s km"
          % (prob.t0_bounds[0], prob.t0_bounds[1],
             {k: round(v, 1) for k, v in prob.rp_min.items()
              if np.isfinite(v)}))

    x_seed = None
    if args.x_seed:
        with open(args.x_seed, encoding="utf-8") as fh:
            blob = json.load(fh)
        # 既接受 {"x": [...]}，也接受 step8 自己导出的 nodes（x 在 meta 里）
        x_seed = blob.get("x") or blob.get("meta", {}).get("x")
        if x_seed is None:
            ap.error("--x-seed 文件里找不到 x 或 meta.x")
    if args.replay:
        if x_seed is None:
            ap.error("--replay 要配 --x-seed（复放的就是那条染色体）")
        print("复放：%d 维染色体原样解码，不跑 SADE" % len(x_seed))
        if x_seed and len(x_seed) == 1 + prob.n_legs and prob.k_gene:
            # 旧版留档的染色体没有 k 基因（每段只带 TOF），按 k=0 直飞解码。
            prob.kmax = [0] * prob.n_legs
            prob.k_gene = []
            print("圈数基因：该染色体无 k 位，各段圈数按 --revs/算例默认值")
        best = prob.evaluate(np.asarray(x_seed, float))
    else:
        print("SADE 优化：%d 岛 × %d 种群 × %d 代（seed %d），%d 进程"
              % (islands, args.pop, args.gen, args.seed, workers))
        best = run_opt(prob, pop_size=args.pop, gen=args.gen, seed=args.seed,
                       islands=islands, x_seed=x_seed, verbose=False,
                       workers=workers, spec=prob_spec(prob),
                       families=not args.no_families)
    if args.polish:
        polished = polish(prob, best)
        # 精修点的罚项≈0，J 可能比"带 6 m/s 缺陷的冠军"略高一点点；
        # 只要物理目标（w_dep·v∞dep + w_arr·v∞arr）不显著变差且可行，就采用。
        j_phys = lambda r: prob.w_dep * r["vd"] + prob.w_arr * r["va"]
        if polished.get("ok") and polished.get("feasible") and (
                j_phys(polished) <= j_phys(best) * 1.01 + 1e-6):
            best = polished

    if not best.get("ok"):
        print("优化失败：%s%s" % (best.get("reason"),
                                  "（%s）" % best["hint"] if best.get("hint") else ""))
        return 1

    if args.chain_close:
        prob._cc_span, prob._cc_step = args.cc_span, args.cc_step
        best = chain_close(prob, best, args.wc_max_coast)

    print("\n===== 冠军 =====")
    print("t0 = %.6f MJD2000  (%s UTC)" %
          (best["t0"], mjd_to_utc(best["t0"] + MJD2000).strftime("%Y-%m-%d %H:%M:%S")))
    print("T  = %s  [天]" % np.round(best["T"], 3))
    print("J  = %.4f   v∞dep=%.4f  v∞arr=%.4f km/s   缺陷max=%.3f m/s  %s" %
          (best["J"], best["vd"], best["va"], best["max_defect"] * 1e3,
           ("可行(零脉冲)" if prob.dsm_max_mps <= 0 else "可行(≤%g m/s)" % prob.dsm_max_mps)
           if best["feasible"] else "不可行"))
    if best.get("combo_truncated"):
        print("     [!] 分支组合 %d 种 > --max-combos %d：这条只试了每段第 0 支"
              "（没枚举 ≠ 不存在，想看全支请加大 --max-combos）"
              % (best["combo_space"], prob.max_combos))
    if best.get("dsm_total", 0.0) > 1e-9:
        print("     关闭脉冲合计 %.1f m/s（每处 ≤ %.0f m/s 判据）" %
              (best["dsm_total"] * 1e3, prob.dsm_max_mps))
    for i, f in enumerate(best["det"], start=1):
        print("  飞越%d %-8s t=%10.3f  rp=%9.1f km (h=%8.1f km)  "
              "δ=%6.2f° / δmax=%6.2f°  v∞=%.4f km/s  缺陷=%6.2f m/s" %
              (i, f["body"], best["epoch_mjd2000"][i] + MJD2000,
               f["rp_req"], f["alt"], math.degrees(f["delta"]),
               math.degrees(f["dmax"]), f["vinf"], f["dv"] * 1e3))

    # 谐振不是"声明"出来的：同体腿/多圈腿上报 j:k（行星圈数 : 航天器圈数）
    res_diag = best.get("res_diag") or []
    if res_diag:
        print("\n谐振段（j : k = 行星圈数 : 航天器圈数，为解完的输出量）:")
        for q in res_diag:
            # 同体打靶腿的圈数基因只是"选支提示"，真正的圈数是解出来的 k_sc，
            # 所以比值用 k_sc 报，基因另括号注明，免得出现 "2.000 : 0" 这种假象。
            print("  段%d %s→%s  基因k=%d  T=%9.3f d  a=%.6f AU  P=%.3f d  "
                  "k_sc=%.4f  行星周期=%.2f d  j=%.4f   ⇒ %.3f : %.4f"
                  % (q["leg"] + 1, prob.names[q["leg"]], prob.names[q["leg"] + 1],
                     q["k"], q["tof"], q["a_AU"], q["P_sc"], q["k_sc"],
                     q["P_planet"], q["j_planet"], q["j_planet"], q["k_sc"]))

    # 搜索覆盖情况：各岛末种群还剩下哪些圈数族，每族最好的是哪条。
    # 冠军只代表"它选了这一支"，这张表才回答"另一支有没有被搜过"。
    fams = best.get("families") or []
    if fams:
        print("\n分支族谱（各岛末种群按圈数基因 k 归族，同族取 J 最小；"
              "×= 种群里还剩几个；打靶段的 k 只是选支提示，实圈数看上面谐振段）:")
        for line in format_families(fams, champ_k=[int(q) for q in best.get("k", [])]):
            print(line)

    if args.wenchang:
        # 两种口径一起报：
        #   wenchang_check = "窗口里最省倾角那一刻"（不保证 SOI 落在 t0 上）
        #   launch_closure = "滑完就发"那一刻（SOI 精确落在 t0，代价是滑行可能长）
        # --chain-close 已经算过闭合解，就别让旧口径把它盖掉。
        wc = wenchang_check(best, prob, coast_max=args.wc_max_coast)
        print("\n文昌发射几何（窗口内最省倾角那一刻）:")
        if wc.get("ok"):
            print("  发射 %s UTC   滑行 %.1f min   倾角 %.2f°   方位 %.2f°"
                  % (wc.get("t_launch_utc", "—"), wc.get("coast_min") or 0.0,
                     wc.get("inc_deg") if wc.get("inc_deg") is not None else float("nan"),
                     wc.get("az_deg") if wc.get("az_deg") is not None else float("nan")))
        else:
            msg = wc.get("err") or wc.get("reason") or str(wc)
            print("  不可行：%s" % msg)
        if wc.get("ok") and not wc.get("coast_ok"):
            print("  [!] 滑行 %.1f min > %.0f min：走廊能进，但按 step1 口径"
                   "（滑行短）这一天不算能发 → 用 --chain-close 找闭合日"
                   % (wc["coast_min"], args.wc_max_coast))
        if args.chain_close and (best.get("wenchang") or {}).get("method"):
            best["wenchang_optimum"] = wc
        else:
            best["wenchang"] = wc

    extra = dict(k_revs=[int(q) for q in best.get("k", prob.revs)],
                 kmax=[int(q) for q in prob.kmax],
                 dsm_max_allowed_m_s=float(prob.dsm_max_mps))
    if fams:
        extra["families"] = [dict(k=q["k"], n=q["n"], J=q["J"],
                                  feasible=q["feasible"], x=q["x"])
                             for q in fams]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    prob.export_nodes(best, args.out, wenchang=best.get("wenchang"),
                      extra_meta=extra)
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
            w.writerow(["k_revs", ";".join(str(q) for q in best.get("k", []))])
            w.writerow(["kmax", ";".join(str(q) for q in prob.kmax)])
            w.writerow(["dsm_total_m_s", "%.3f" % (best.get("dsm_total", 0.0) * 1e3)])
            w.writerow(["dsm_max_allowed_m_s", "%.1f" % prob.dsm_max_mps])
            for q in (best.get("res_diag") or []):
                L = q["leg"] + 1
                w.writerow(["res_leg%d" % L,
                            "%s→%s" % (prob.names[L - 1], prob.names[L])])
                w.writerow(["res_leg%d_tof_days" % L, "%.6f" % q["tof"]])
                w.writerow(["res_leg%d_k" % L, "%d" % q["k"]])
                w.writerow(["res_leg%d_k_sc" % L, "%.6f" % q["k_sc"]])
                w.writerow(["res_leg%d_j_planet" % L, "%.6f" % q["j_planet"]])
                w.writerow(["res_leg%d_a_AU" % L, "%.6f" % q["a_AU"]])
                w.writerow(["res_leg%d_P_sc_days" % L, "%.6f" % q["P_sc"]])
                w.writerow(["res_leg%d_P_planet_days" % L, "%.6f" % q["P_planet"]])
        print("已写 %s" % args.csv)
    return 0


if __name__ == "__main__":
    # pykep 3.0.1 解释器退出时偶发堆析构错误（C++ 析构顺序问题，不影响已打印
    # 结果），统一在出口 os._exit() 收口：退出码透传，异常打印 traceback 后以 1 退出。
    _code = 0
    try:
        _r = main()
        _code = _r if isinstance(_r, int) else 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
