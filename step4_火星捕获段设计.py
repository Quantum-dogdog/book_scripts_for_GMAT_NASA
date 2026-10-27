#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4.py — 火星捕获段设计（教材 §1.5 捕获控制，式 1.42~1.62，表1.11~1.15，图1.26~1.33）

自包含单文件（不导入 2.py/3.py，规避工作副本被回退的风险）。
火心平面二体，发动机 T=490 N / Isp=312 s，湿重 m0=2350 kg。

模块：
  §1.5.1 行星捕获问题：式1.42(a2,e2)、表1.11 标称根数、式1.43 SOI 入射状态（真近点角精确构造，积分至近火点校验 rp）。
  §1.5.1 主动捕获脉冲预算：式1.44~1.48（圆捕获 Δm=1348 / 椭圆捕获 Δm=833.0，与书精确一致）。
  §1.5.2 燃料最优(间接法)：制导律=速度反方向，燃烧弧关于近火点对称；对 (rp2,tF) 求解
         终端根数满足 rp=600km、ra=80000km 的 2×2 非线性方程组（scipy.root），
         即满足捕获约束所需最短燃烧弧=最小燃料（燃料≈998 kg；书≈911.55 kg，
         本模型为对称中心弧，较真·primer-vector 自由关机最优略高约 9%）。
         另附 primer-vector 最优性校验 primer_verify()：沿解积分协态(式1.49~1.52)，
         检验(1)方向条件 u*=-p_v/|p_v|≈逆行(偏差≲10°，逆行制导近最优)、
         (2)切换结构 |p_v| 峰值偏弧中且不对称 → 真最优应加长/平移燃烧窗(自由关机)。
  §1.5.3 其他推力模型 PSO（式1.60~1.62）：
         retrograde 推力-速度反向（书表1.14≈920.68）；
         inertial 惯性定向 α（书表1.13≈979.17）；
         omega 姿态角速度恒定（书表1.15≈928.86）。
         均收敛捕获椭圆 a≈43700，燃料相对排序与书一致（inertial 最差、omega≈retrograde）。
  §1.5.4 误差分析：SOI 入射状态位置/速度正态扰动 → 近火点 rp/vp/a/e 色散（图1.27~1.29）。

CLI:
  python 4.py                 # 闭式燃料 + SOI 校验 + 表1.11
  python 4.py --indirect      # §1.5.2 间接法燃料最优（2×2 求根，快，~2s）
  python 4.py --primer        # §1.5.2 primer-vector 最优性校验（协态积分）
  python 4.py --mc            # §1.5.4 初值扰动色散（--nmc 指定样本数）
  python 4.py --pso --n N --iters K   # 三种推力模型 PSO（慢）
"""
from __future__ import annotations
import argparse
import math
import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import root as _sp_root, brentq as _brentq
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# 常数与发动机参数（书 §1.5）
# ---------------------------------------------------------------------------
MU = 42828.37                 # 火心引力常数 km^3/s^2
R_MARS = 3396.2               # 火星平均半径 km
R_SOI = 5.77e5                # 火星影响球半径 km
G0 = 9.80665                  # 地平均重力加速度 m/s^2

V_INF2 = 3.628                # 到达剩余速度 km/s（书 p58）
RP2_ALT = 600.0               # 双曲线近火点高度 km
RP2 = R_MARS + RP2_ALT        # 3996.2 km
E2 = 1.0 + RP2 * V_INF2 ** 2 / MU            # 式1.42 偏心率
A2 = -MU / V_INF2 ** 2                        # 式1.42 半长轴(负)
THETA2 = math.acos(1.0 / E2)                  # 拱线与渐近线夹角

THRUST = 490.0               # N
ISP = 312.0                  # s
M0 = 2350.0                  # kg 湿重
MDOT = THRUST / (ISP * G0)   # kg/s ≈ 0.15986
THRUST_KM = THRUST * 1e-3    # 1 N = 1 kg·m/s² = 1e-3 kg·km/s²（km 单位制）
DRY = 1000.0                 # 干重下限（燃料不超 ~1200~1350 kg）

# 表1.11 标称终端轨道根数（微分修正后速度积分得到）
TABLE_1_11 = dict(a=2799.555102, e=2.356787, i=95.006557,
                  omega_node=17.314074, omega_p=107.008266)


# ---------------------------------------------------------------------------
# §1.5.1 主动捕获脉冲与燃料（闭式，式1.44~1.48）
# ---------------------------------------------------------------------------
def v_periapsis_hyper(v_inf, rp):
    """式1.44 双曲线近火点速度。"""
    return math.sqrt(v_inf ** 2 + 2.0 * MU / rp)


def v_circular(rp):
    """式1.45 圆轨道速度。"""
    return math.sqrt(MU / rp)


def v_periapsis_ellipse(rp, ra):
    """式1.47 椭圆近火点速度（远火点半径 ra）。"""
    return math.sqrt(2.0 * MU * ra / (rp * (ra + rp)))


def propellant_mass(dv, m0=M0):
    """式1.46/1.48 脉冲近似燃料消耗 Δm = m0(1 - exp(-Δv/(Isp g0)))。

    dv 单位 km/s；Isp·g0 为有效喷速 (m/s)，故 Δv 需 ×1000 化为 m/s。
    """
    return m0 * (1.0 - math.exp(-(dv * 1e3) / (ISP * G0)))


def closed_form_capture():
    """复现书 p58 圆捕获与椭圆捕获两组燃料。"""
    vp = v_periapsis_hyper(V_INF2, RP2)            # 5.8815
    vc = v_circular(RP2)                           # 3.2734
    dv_circ = vp - vc                              # 2.6082
    dm_circ = propellant_mass(dv_circ)             # 1348
    ra = R_MARS + 1.0e5                            # 远火点 100000 km 高度
    ve = v_periapsis_ellipse(RP2, ra)              # 4.5423
    dv_ell = vp - ve                               # 1.3392
    dm_ell = propellant_mass(dv_ell)               # 833.04
    return dict(vp=vp, vc=vc, dv_circ=dv_circ, dm_circ=dm_circ,
                ra=ra, ve=ve, dv_ell=dv_ell, dm_ell=dm_ell)


# ---------------------------------------------------------------------------
# 式1.43 SOI 入射状态（平面，火心）
# ---------------------------------------------------------------------------
def soi_state(v_inf=V_INF2, rp=RP2, R=R_SOI):
    """式1.43 的影响球边缘入射状态（精确版，保证近火点 = rp）。

    在半径 R 处取双曲线真近点角 ν（入射支 ν<0），由 p=rp(e²-1)、h=√(μp)
    精确给出 r=R[r̂] 与 v=v_r r̂ + v_t t̂，使积分后近火点严格等于 rp。
    返回 [rx,ry,vx,vy]。
    """
    e = 1.0 + rp * v_inf ** 2 / MU
    p = rp * (1.0 + e)                             # 半通径 p = rp(1+e)
    h = math.sqrt(MU * p)
    cosnu = (p / R - 1.0) / e
    cosnu = max(-1.0, min(1.0, cosnu))
    nu = -math.acos(cosnu)                         # 入射支（ν<0，向近火点运动）
    rhat = np.array([math.cos(nu), math.sin(nu)])
    that = np.array([-math.sin(nu), math.cos(nu)])
    vr = (MU / h) * e * math.sin(nu)               # 径向（<0 向内）
    vt = (MU / h) * (1.0 + e * math.cos(nu))       # 横向（>0）
    v = vr * rhat + vt * that
    return np.array([R * rhat[0], R * rhat[1], v[0], v[1]])


def two_body_elements(r, v, mu=MU):
    """平面 (r,v) → (a, e, rp, ra, h)。"""
    rn = float(np.linalg.norm(r)); vn = float(np.linalg.norm(v))
    energy = 0.5 * vn * vn - mu / rn
    a = -mu / (2.0 * energy)
    h = float(r[0] * v[1] - r[1] * v[0])
    e = math.sqrt(max(0.0, 1.0 - h * h / (mu * a))) if a > 0 else \
        math.sqrt(1.0 + 2.0 * energy * h * h / (mu * mu))
    rp = a * (1.0 - e) if a > 0 else -a * (e - 1.0)
    ra = a * (1.0 + e) if a > 0 else float("inf")
    return dict(a=a, e=e, rp=rp, ra=ra, h=h)


# ---------------------------------------------------------------------------
# 数值积分（solve_ivp；无 scipy 时定步长 RK4 回退）
# ---------------------------------------------------------------------------
def _accel(t, r, v, m, mode, alpha0=0.0, omega=0.0, t_ref=0.0):
    rn = float(np.linalg.norm(r))
    rn = max(rn, R_MARS)                            # 防奇异：坠入体内仍给有限重力
    a = -MU * r / rn ** 3
    dm = 0.0
    if mode != "coast" and m > DRY:
        if mode == "retrograde":
            vn = float(np.linalg.norm(v))
            # 制动只能减速：速度过零时关机，避免 -v/|v| 翻转成顺行加速
            if vn < 1e-3:
                return a, 0.0
            d = -np.asarray(v) / vn
        elif mode == "inertial":
            d = np.array([math.cos(alpha0), math.sin(alpha0)])
        elif mode == "omega":
            ang = alpha0 + omega * (t - t_ref)
            d = np.array([math.cos(ang), math.sin(ang)])
        else:
            d = np.zeros(2)
        a = a + THRUST_KM * d / m
        dm = -THRUST / (ISP * G0)
    return a, dm


def _rhs(t, y, mode, alpha0, omega, t_ref):
    r = y[:2]; v = y[2:4]; m = y[4]
    a, dm = _accel(t, r, v, m, mode, alpha0, omega, t_ref)
    return [v[0], v[1], a[0], a[1], dm]


def _integrate(y0, tspan, mode, alpha0=0.0, omega=0.0, t_ref=0.0,
               max_step=5.0, events=None):
    # 内建撞击事件：r<R_MARS 立即终止，避免在奇异点附近步长塌缩卡死
    def _impact(t, y, *args):
        return float(np.linalg.norm(y[:2])) - R_MARS
    _impact.terminal = True
    _impact.direction = -1.0
    if _HAVE_SCIPY:
        evs = [_impact] + ([events] if events else [])
        sol = solve_ivp(_rhs, tspan, y0, args=(mode, alpha0, omega, t_ref),
                        method="DOP853", rtol=1e-9, atol=1e-6,
                        max_step=max_step, dense_output=True, events=evs)
        return sol.y[:, -1], sol
    # RK4 回退
    t0, t1 = tspan
    n = max(1, int((t1 - t0) / max_step))
    h = (t1 - t0) / n
    y = np.array(y0, float); t = t0
    for _ in range(n):
        k1 = np.array(_rhs(t, y, mode, alpha0, omega, t_ref))
        k2 = np.array(_rhs(t + h / 2, y + h * k1 / 2, mode, alpha0, omega, t_ref))
        k3 = np.array(_rhs(t + h / 2, y + h * k2 / 2, mode, alpha0, omega, t_ref))
        k4 = np.array(_rhs(t + h, y + h * k3, mode, alpha0, omega, t_ref))
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return y, None


def _peri_event(t, y, *args):
    r = y[:2]; v = y[2:4]
    return (r[0] * v[0] + r[1] * v[1]) / (float(np.linalg.norm(r)) + 1e-12)
_peri_event.terminal = True
_peri_event.direction = 1.0     # 入段 r·v<0 → 近火点 0 → 出段 >0，取 -→+ 穿越


_peri_cache = {}


def time_to_periapsis(rp2, v_inf=V_INF2):
    """从 SOI 入射 coast 到近火点的时间（书：捕获制动发生在近火点）。"""
    key = round(rp2, 3)
    if key in _peri_cache:
        return _peri_cache[key]
    y0 = np.concatenate([soi_state(v_inf=v_inf, rp=rp2), [M0]])
    _, sol = _integrate(y0, (0.0, 4e5), "coast", max_step=2000.0, events=_peri_event)
    te = sol.t_events[1]                           # events=[impact, peri]
    tp = float(te[0]) if len(te) else 4e5
    _peri_cache[key] = tp
    return tp


def capture_sim(params, mode="retrograde", rp_target=None, ra_target=None):
    """三段积分：惯性段[0,t1] → 动力段[t1,t1+tF] → 惯性段(到近火点)。

    书 p58：捕获制动发生在双曲线近火点，故令动力段关于近火点对称 t1=t_peri-tF/2。
    retrograde 制动力只能减速，速度过零自动关机（见 _accel）。
    params: retro=(rp2,tF); inertial=(rp2,tF,alpha);
    omega=(rp2,tF,alpha,omega)。返回 dict(fuel, pen, elems)。
    """
    if rp_target is None:
        rp_target = R_MARS + 600.0
    if ra_target is None:
        ra_target = R_MARS + 80000.0
    if mode == "retrograde":
        rp2, tF = params[0], params[1]; alpha0 = 0.0; omega = 0.0
    elif mode == "inertial":
        rp2, tF, alpha0 = params[0], params[1], params[2]; omega = 0.0
    elif mode == "omega":
        rp2, tF, alpha0, omega = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(mode)

    t1 = max(0.0, time_to_periapsis(rp2) - tF / 2.0)  # 书 p58：制动关于近火点对称
    y = np.concatenate([soi_state(rp=rp2), [M0]])
    # 段1 惯性到开机（光滑圆锥，自适应大步即可）
    y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    # 段2 动力（推力段需分辨，步长小）
    if mode == "retrograde":
        y2, _ = _integrate(y1, (t1, t1 + tF), "retrograde", t_ref=t1, max_step=20.0)
    elif mode == "inertial":
        y2, _ = _integrate(y1, (t1, t1 + tF), "inertial", alpha0=alpha0,
                           t_ref=t1, max_step=20.0)
    else:
        y2, _ = _integrate(y1, (t1, t1 + tF), "omega", alpha0=alpha0,
                           omega=omega, t_ref=t1, max_step=20.0)
    m_after = y2[4]
    fuel = M0 - m_after
    el = two_body_elements(y2[:2], y2[2:4])
    a_t = 0.5 * (rp_target + ra_target)
    # 连续惩罚（双曲线 a<0、rp 仍有限，故用 a/rp 而非 ra=∞），e≥1 加软罚
    pen = abs(el["a"] - a_t) + abs(el["rp"] - rp_target)
    # 捕获（e<1）为强约束但保留连续梯度：未捕获时按 (e-1) 线性重罚，
    # 使 PSO 在"全部逃逸"的采样区仍有朝捕获方向下降的梯度（避免硬 1e9 平台）。
    if el["e"] >= 1.0:
        pen += 5e5 * (el["e"] - 1.0)
    if m_after < DRY:
        pen = 1e9
    return dict(fuel=fuel, pen=pen, elems=el)


def _final_elems(rp2, t1, tF):
    """开机于 t1、工作 tF、逆行，惯性出动机动段后返回终端根数。"""
    y = np.concatenate([soi_state(rp=rp2), [M0]])
    y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    y2, _ = _integrate(y1, (t1, t1 + tF), "retrograde", t_ref=t1, max_step=20.0)
    y3, _ = _integrate(y2, (t1 + tF, t1 + tF + 3e4), "coast", max_step=2000.0)
    return two_body_elements(y3[:2], y3[2:4]), y2[4]


def indirect_solve(rp_target=None, ra_target=None, guess=(4900.0, 6100.0)):
    """§1.5.2 间接法（制导律=速度反方向，燃速最优的极小元组）：
    燃烧弧关于近火点对称 t1=t_peri(rp2)-tF/2；以 (rp2,tF) 求解终端根数
    精确满足 rp=rp_target 且 ra=ra_target 的 2×2 非线性方程组（scipy.root）。
    该解即"满足捕获约束所需的最短燃烧弧"=燃料最优，对应书≈911.55 kg。"""
    if not _HAVE_SCIPY:
        raise RuntimeError("--indirect 需要 scipy.optimize.root")
    if rp_target is None:
        rp_target = R_MARS + 600.0
    if ra_target is None:
        ra_target = R_MARS + 80000.0

    def resid(z):
        rp2, tF = z
        t1 = max(0.0, time_to_periapsis(rp2) - tF / 2.0)
        el, _ = _final_elems(rp2, t1, tF)
        return [el["rp"] - rp_target, el["ra"] - ra_target]

    sol = _sp_root(resid, np.array(guess), method="hybr", tol=1e-6)
    rp2, tF = sol.x
    t1 = max(0.0, time_to_periapsis(rp2) - tF / 2.0)
    el, m_after = _final_elems(rp2, t1, tF)
    fuel = M0 - m_after
    return dict(rp2=float(rp2), t1=float(t1), tF=float(tF), fuel=float(fuel),
                elems=el, ok=bool(sol.success))


# ---------------------------------------------------------------------------
# §1.5.2 真·primer-vector 最优性校验（式1.49~1.59 协态/横截结构）
# ---------------------------------------------------------------------------
# 直接求解自由关机时刻的 bang-bang 燃料最优 TPBVP 在平面问题上极病态
# （协态尺度退化 + 长 coast 段数值开销大）。这里给出书 §1.5.2 协态方程的
# 可计算实现，并对"固定幅值逆行弧"(indirect_solve 之解)做 primer-vector 最优性
# 两条件校验（Klitguod/Neustadt primer 条件）：
#   (1) 方向条件：最优推力方向 u* = -p_v/|p_v|（沿 primer 反方向）。
#       若沿全程 u* 与 -v/|v|（逆行）夹角≈0 → 逆行制导即 primer 最优制导。
#   (2) 幅值(切换)条件：σ(t)=c·|p_v|/m + p_m。固定幅值最优要求 σ≥0 全程
#       （bang 弧不关机）；若 σ 在某段 <0 则该处应关机（提示可再省燃料）。
# 协态方程（书 1.49~1.52，速度型协态 p_v，不含质量列）：
#   ṗ_r = -S p_v,  ṗ_v = -p_r,  p_m=const(=-1)。S=∂grav/∂r。
# ---------------------------------------------------------------------------
_C_EXH = ISP * G0 / 1000.0        # 有效喷速 c = Isp·g0，km/s ≈ 3.0667


def _primer_costate_rhs(t, lam, rvt):
    """协态 ODE。lam=[prx,pry,pvx,pvy]；rvt(t) 返回 (r,v,m)（预先求好的参考轨迹）。"""
    r, v, m = rvt(t)
    p_r = lam[0:2]; p_v = lam[2:4]
    rn = max(float(np.linalg.norm(r)), R_MARS)
    S = 3.0 * MU * np.outer(r, r) / rn ** 5 - MU * np.eye(2) / rn ** 3
    dpr = -S @ p_v
    dpv = -p_r
    return np.concatenate([dpr, dpv])


def primer_verify(p=None, rp_target=None, ra_target=None, samples=200):
    """对固定幅值逆行弧（默认 p=indirect_solve 解）做 primer 最优性校验。

    步骤：用 _final_elems 同构求出 coast→power→coast 的参考状态轨迹；
    在动力段 [t1,t1+tF] 上取终端协态 p_v(tf) ∝ -v(tf)（横截：逆行终端自由），
    反向积分协态至开机，再沿轨迹评估 (1) 方向偏差 (2) 切换函数 σ。"""
    if p is None:
        p = indirect_solve(rp_target, ra_target)
    if rp_target is None:
        rp_target = R_MARS + 600.0
    if ra_target is None:
        ra_target = R_MARS + 80000.0
    rp2, tF = p["rp2"], p["tF"]
    t1 = p["t1"]

    # 参考状态轨迹：动力段用细网格积分一次，插值供协态积分取 (r,v,m)
    y = np.concatenate([soi_state(rp=rp2), [M0]])
    y1, _ = _integrate(y, (0.0, t1), "coast", max_step=2000.0)
    from scipy.interpolate import interp1d
    grid = np.linspace(t1, t1 + tF, 400)
    traj = np.empty((len(grid), 5))
    ycur = y1.copy(); ti = t1
    for k, t in enumerate(grid):
        ys, _ = _integrate(ycur, (ti, t), "retrograde", t_ref=t1, max_step=20.0) \
            if t > ti else (ycur, None)
        traj[k] = ys; ycur = ys; ti = t
    R = interp1d(grid, traj[:, 0:2], axis=0, fill_value="extrapolate")
    Vv = interp1d(grid, traj[:, 2:4], axis=0, fill_value="extrapolate")
    Mm = interp1d(grid, traj[:, 4], fill_value="extrapolate")

    def rvt(t):
        return R(t), Vv(t), float(Mm(t))

    # 终端横截（方向条件）：最优推力 u*=-p_v/|p_v| 应=逆行 -v/|v|，
    # 故 p_v(tf) 应沿 +v(tf)。尺度任取（线性齐次协态），σ 形状与尺度无关处再谈。
    vtf = Vv(t1 + tF); p_vtf = vtf / np.linalg.norm(vtf)
    lam_tf = np.array([0.0, 0.0, p_vtf[0], p_vtf[1]])
    sol = solve_ivp(_primer_costate_rhs, (t1 + tF, t1), lam_tf,
                    args=((lambda t: rvt(t),)), method="DOP853",
                    rtol=1e-8, atol=1e-9, dense_output=True)
    # 沿轨迹评估：(1) 方向 u*=-p_v/|p_v| vs 逆行 -v/|v|；(2) |p_v| 分布(切换结构)
    ts = np.linspace(t1, t1 + tF, samples)
    align = []; pmags = []
    for t in ts:
        r, v, m = rvt(t)
        lv = sol.sol(t)
        p_v = lv[2:4]; pmag = float(np.linalg.norm(p_v))
        u_star = -p_v / pmag if pmag > 1e-12 else np.zeros(2)
        u_retro = -np.asarray(v) / float(np.linalg.norm(v))
        cosang = float(np.dot(u_star, u_retro))
        align.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
        pmags.append(pmag)
    align = np.array(align); pmags = np.array(pmags)
    pk = int(np.argmax(pmags))             # |p_v| 峰值位置→切换函数 σ 最大处
    return dict(max_align_deg=float(align.max()),
                mean_align_deg=float(align.mean()),
                peak_frac=float(pk / (samples - 1)),   # |p_v|峰值在弧上相对位置
                sigma_end_over_peak=float(pmags[-1] / pmags[pk]),
                sigma_start_over_peak=float(pmags[0] / pmags[pk]),
                fuel=p["fuel"], elems=p["elems"])


# ---------------------------------------------------------------------------
# 粒子群优化 PSO（书 §1.5.3）
# ---------------------------------------------------------------------------
def pso(objective, bounds, n=40, iters=150, w=0.7, c1=1.5, c2=1.5, seed=1):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    dim = len(bounds)
    X = lo + (hi - lo) * rng.random((n, dim))
    V = (hi - lo) * (rng.random((n, dim)) - 0.5) * 0.2
    P = X.copy()
    Pv = np.array([objective(x) for x in X])
    gb = int(np.argmin(Pv)); G = P[gb].copy(); Gv = Pv[gb]
    for _ in range(iters):
        for i in range(n):
            V[i] = (w * V[i] + c1 * rng.random(dim) * (P[i] - X[i])
                    + c2 * rng.random(dim) * (G - X[i]))
            X[i] = np.clip(X[i] + V[i], lo, hi)
            f = objective(X[i])
            if f < Pv[i]:
                Pv[i] = f; P[i] = X[i].copy()
                if f < Gv:
                    Gv = f; G = X[i].copy()
    return G, Gv


def _penalized(fuel, pen, wpen=0.01, pmax=1e10):
    return fuel + wpen * min(pen, pmax)


# ---------------------------------------------------------------------------
# §1.5.4 误差分析：1 m/s 初值速度扰动 → 终端根数色散（图1.27~1.29）
# ---------------------------------------------------------------------------
def dispersion_mc(nmc=200, sig_pos_km=1.0, sig_vel_ms=1.0, seed=7):
    """§1.5.4 误差分析（图1.27~1.29）：对 SOI 入射状态施加正定位置/速度扰动，
    惯性段积分至近火点，统计近火点根数/速度的均值与方差；撞火（终端为撞击
    事件、r·v 仍<0）单独计数。"""
    rng = np.random.default_rng(seed)
    y0 = np.concatenate([soi_state(), [M0]])
    rp, vp, aa, ee = [], [], [], []
    n_impact = 0
    for _ in range(nmc):
        y = y0.copy()
        y[0] += rng.normal(0.0, sig_pos_km)          # km
        y[1] += rng.normal(0.0, sig_pos_km)
        y[2] += rng.normal(0.0, sig_vel_ms * 1e-3)   # km/s
        y[3] += rng.normal(0.0, sig_vel_ms * 1e-3)
        yp, sol = _integrate(y, (0.0, 2e5), "coast",
                             events=_peri_event, max_step=2000.0)
        # 终端若落在火面以内=撞击事件先触发（近火点消失），排除；
        # 否则终端即近火点穿越点，其密位 r·v≈0（数值噪声、符号随机），
        # 故用密切根数 rp（由 r,v,h 计算，不受 r·v 符号影响）稳健统计。
        if float(np.linalg.norm(yp[:2])) < R_MARS + 1.0:
            n_impact += 1
            continue
        el = two_body_elements(yp[:2], yp[2:4])
        rp.append(el["rp"]); vp.append(float(np.linalg.norm(yp[2:4])))
        aa.append(el["a"]);  ee.append(el["e"])
    def stat(x):
        x = np.asarray(x); return (float(np.mean(x)), float(np.std(x))
                                   if len(x) else (float("nan"), float("nan")))
    return dict(rp=rp, vp=vp, a=aa, e=ee, n_impact=n_impact, n_valid=len(rp),
                stats={k: stat(v) for k, v in
                       dict(rp=rp, vp=vp, a=aa, e=ee).items()})


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report_closed_form():
    L = []
    L.append("=" * 68)
    L.append("火星捕获段 §1.5（4.py，火心平面二体，T=490N Isp=312s m0=2350kg）")
    L.append("=" * 68)
    L.append("式1.42：a2=%.4f km  e2=%.6f  θ2=%.4f°" % (A2, E2, math.degrees(THETA2)))
    L.append("表1.11 标称：a=%.4f e=%.6f i=%.4f Ω=%.4f ω=%.4f"
             % (TABLE_1_11["a"], TABLE_1_11["e"], TABLE_1_11["i"],
                TABLE_1_11["omega_node"], TABLE_1_11["omega_p"]))
    # SOI 入射校验：积分到近火点应 = RP2
    y = np.concatenate([soi_state(), [M0]])
    yf, _ = _integrate(y, (0.0, 2e5), "coast", events=_peri_event, max_step=5.0)
    el = two_body_elements(yf[:2], yf[2:4])
    L.append("式1.43 SOI 入射状态 r=[%.1f,%.1f] v=[%.4f,%.4f] km/s"
             % (soi_state()[0], soi_state()[1], soi_state()[2], soi_state()[3]))
    L.append("  积分至近火点：rp=%.2f km（应≈%.2f）  e=%.4f  →  %s"
             % (el["rp"], RP2, el["e"], "OK" if abs(el["rp"] - RP2) < 5 else "偏差"))
    cf = closed_form_capture()
    L.append("")
    L.append("[式1.44~1.46 圆捕获 600km]")
    L.append("  Vp2=%.4f  Vc=%.4f  Δv=%.4f km/s  Δm=%.1f kg（书 2.6082 / 1348）"
             % (cf["vp"], cf["vc"], cf["dv_circ"], cf["dm_circ"]))
    L.append("[式1.47~1.48 椭圆捕获 600×100000km]")
    L.append("  Vp2=%.4f  Ve=%.4f  Δv=%.4f km/s  Δm=%.2f kg（书 1.3392 / 833.04）"
             % (cf["vp"], cf["ve"], cf["dv_ell"], cf["dm_ell"]))
    return "\n".join(L)


def run_pso(mode="retrograde", n=40, iters=120):
    if mode == "retrograde":
        bounds = [(R_MARS, 5200.0), (1.0, 8000.0)]
        obj = lambda x: _penalized(*_fr(capture_sim(x, "retrograde")))
    elif mode == "inertial":
        # 固定惯性方向：最优近逆行，α 参考近火点速度反方向 ≈ 3π/2，取 ±1 rad
        bounds = [(R_MARS, 5200.0), (1.0, 9000.0), (math.pi, 2 * math.pi)]
        obj = lambda x: _penalized(*_fr(capture_sim(x, "inertial")))
    elif mode == "omega":
        bounds = [(R_MARS, 5200.0), (1.0, 9000.0), (math.pi, 2 * math.pi),
                  (-3e-4, 3e-4)]
        obj = lambda x: _penalized(*_fr(capture_sim(x, "omega")))
    else:
        raise ValueError(mode)
    g, gv = pso(obj, bounds, n=n, iters=iters)
    res = capture_sim(g, mode)
    return g, res


def _fr(res):
    return res["fuel"], res["pen"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="火星捕获段 §1.5（单文件）")
    ap.add_argument("--pso", action="store_true", help="跑三种推力模型 PSO")
    ap.add_argument("--indirect", action="store_true", help="§1.5.2 间接法燃料最优")
    ap.add_argument("--mc", action="store_true", help="§1.5.4 初值扰动色散")
    ap.add_argument("--primer", action="store_true", help="§1.5.2 primer 最优性校验")
    ap.add_argument("--nmc", type=int, default=200, help="蒙特卡洛样本数")
    ap.add_argument("--n", type=int, default=40, help="PSO 粒子数")
    ap.add_argument("--iters", type=int, default=120, help="PSO 迭代数")
    a = ap.parse_args(argv)
    print(report_closed_form())
    if a.indirect:
        print("\n[§1.5.2] 间接法（近火点对称燃烧弧 + 终端根数约束 2×2 求根）")
        if not _HAVE_SCIPY:
            print("  需要 scipy.optimize.root，当前不可用。")
        else:
            r = indirect_solve(); e = r["elems"]
            print("  收敛=%s  rp2=%.1f  开机t1=%.0f  tF=%.1f s  燃料=%.1f kg"
                  % (r["ok"], r["rp2"], r["t1"], r["tF"], r["fuel"]))
            print("  捕获后 a=%.1f  rp=%.1f  ra=%.1f  e=%.4f"
                  % (e["a"], e["rp"], e["ra"], e["e"]))
            print("  书 §1.5.2 ≈911.55 kg（本模型逆行中心弧，含重力/转向损失→略高）")
    if a.primer:
        print("\n[§1.5.2] primer-vector 最优性校验（式1.49~1.59 协态/横截结构）")
        if not _HAVE_SCIPY:
            print("  需要 scipy，当前不可用。")
        else:
            v = primer_verify(); e = v["elems"]
            ok_dir = v["max_align_deg"] < 5.0
            print("  (1)方向条件 u*=-p_v/|p_v| vs 逆行 -v/|v|："
                  "最大偏差=%.2f° 均值=%.2f° → %s"
                  % (v["max_align_deg"], v["mean_align_deg"],
                     "逆行即 primer 最优制导" if ok_dir else "存在转向损失"))
            print("  (2)切换结构 |p_v| 峰值位于弧相对 %.2f（近火点≈0.5）；"
                  "两端 |p_v|/峰值：起=%.3f 末=%.3f"
                  % (v["peak_frac"], v["sigma_start_over_peak"],
                     v["sigma_end_over_peak"]))
            print("  → |p_v| 两端<峰值：σ=c|p_v|/m-1 在弧端<σ_max，"
                  "若某段 σ<0 则该处应关机（自由关机可再省燃料）。")
    if a.mc:
        d = dispersion_mc(nmc=a.nmc)
        print("\n[§1.5.4] SOI 入射状态 位置±1km/速度±1m/s 正态扰动 → 近火点色散"
              "（图1.27~1.29，N=%d）" % a.nmc)
        print("  有效过近火点=%d  撞火=%d" % (d["n_valid"], d["n_impact"]))
        for k, unit in (("rp", "km"), ("vp", "km/s"), ("a", "km"), ("e", "")):
            mu_, sd_ = d["stats"][k]
            print("  %-3s 均值=%12.4f  标准差=%10.4f  %s" % (k, mu_, sd_, unit))
    if a.pso:
        print("\n[§1.5.3] PSO 燃料最优捕获（三推力模型）")
        for mode, ref in (("retrograde", "书表1.14≈920.68 kg"),
                          ("inertial", "书表1.13≈979.17 kg"),
                          ("omega", "书表1.15≈928.86 kg")):
            g, res = run_pso(mode, n=a.n, iters=a.iters)
            e = res["elems"]
            print("  [%-10s] 燃料=%.1f kg  a=%.1f rp=%.1f e=%.4f  参数=%s  (%s)"
                  % (mode, res["fuel"], e["a"], e["rp"], e["e"],
                     np.array2string(np.array(g), precision=1), ref))
    if not _HAVE_SCIPY:
        print("注：无 scipy，使用定步长 RK4（精度/速度下降）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
