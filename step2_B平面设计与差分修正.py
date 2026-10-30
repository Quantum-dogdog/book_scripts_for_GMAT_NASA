"""
step2 · 火心段 B 平面设计与差分修正（书 §1.2.3 / §1.3）

算什么：
  A 初步设计（圆锥拼接）：目标倾角 i2、近火点半径 rp2、到达剩余速度 V∞2
    → 标称 B^nom = [BT, BR, τ]
  B 精确模型修正：日心 N 体传播（日 + 金/木/土）→ 实际 B 平面 →
    数值雅可比 K = ∂[BT,BR,τ]/∂v0 → Δv0 = K⁻¹(B^nom − F(v0))，牛顿迭代到残差→0
  C 与 step1 的接口只有一组 dep/arr 字符串：dep 填 step1 报告给出的"实际出 SOI
    时刻"（可带 HH:MM:SS），Lambert 初值就与探测器真实日心状态同源，
    修正量降到 m/s 级。

读数前必知的两条约定：
  · 历表时刻按 UTC≈TT 处理（差 ~69 s），到达端误差由差分修正吸收。
  · N 体下 BT/BR/τ 只在二体极限内严格沿双曲线不变；潮汐摄动让 SOI 入口与
    近火点的 osculating B 相差 ~10 km。残差收敛到 1e-6（归一化）是模型内收敛，
    物理精度按 ~10 km 计。

用法：
  python step2_B平面设计与差分修正.py                      # 默认 2015-12-25 → 2016-09-29
  python step2_B平面设计与差分修正.py --dep 2020-07-25 --arr 2021-02-11
  python step2_B平面设计与差分修正.py --i2 95 --rp2 400    # 目标倾角 / 近火点高度(km)
  python step2_B平面设计与差分修正.py --no-correct         # 只出初步设计

依赖 numpy；有 scipy 就用 solve_ivp，没有则自动降级细步 RK4（评估规则一致）。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
MU_SUN = 1.32712440018e11        # km^3/s^2
MU_MARS = 42828.37
AU = 1.495978707e8
J2000 = datetime(2000, 1, 1, 12, 0, 0)
R_MARS = 3396.2                  # 火星平均半径 km
R_SOI_MARS = 5.7e5               # 火星影响球半径 km

# 参考矢量 N̂：书式(1.14) 明确"选取该坐标系(J2000 火心惯性系)的 Z 轴为参考矢量"。
# 本机全程在 J2000 日心黄道系传播，火心相对状态即该系矢量，故 N̂ = 黄道极 [0,0,1]。
# 校核：表1.3 窗口2 V∞2=(-2.4726,-2.0737,-1.6583)，asin(Vz/|V|)=-27.20°=书 δ∞2，
# 证明书的"赤纬"实为出黄纬、参考轴就是 Z。曾误用火星自转轴 N_MARS，已废弃。
N_REF = np.array([0.0, 0.0, 1.0])

# ---- 数值/算法常数 ----
SCAN_PRE_D = 10.0                # 接近段扫描窗起点（相对 t1，天）
SCAN_POST_D = 2.0                # 接近段扫描窗终点（相对 t1，天）
N_SCAN = 400                     # 最近接近点粗扫点数（dense 路径）
RK4_ARCH_STEP_S = 21600.0        # RK4 降级：存档步长 6 h
RK4_FINE_STEP_S = 600.0          # RK4 降级：段内重积分步长
TERNARY_ITERS = 60               # 最近接近点三分细化迭代
SOI_BISEC_ITERS = 60             # SOI 入口二分迭代
DV_EPS = 1e-3                    # 雅可比差分步长 km/s
DV_MAX_STEP = 0.2                # 单步日心速度限幅 km/s
CORR_TOL = 1e-6                  # 归一化残差容差
CORR_MAX_ITER = 60

# Standish(1992) 日心黄道根数 [1800-2050]：(a,e,I,L,wbar,Om) + 每世纪速率 + GM
# 用于 N 体摄动（金、地、火、木、土）
_PLANETS = {
    "venus": ((0.72333566, 0.00677672, 3.39467605,
               181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890,
               58517.81538729, 0.00268329, -0.27769418),
              324859.0),
    "earth": ((1.00000261, 0.01671123, -0.00001531,
               100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668,
               35999.37244981, 0.32327364, 0.0),
              398600.4),
    "mars": ((1.52371034, 0.09339410, 1.84969142,
              -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131,
              19140.30268499, 0.44441088, -0.29257343),
             MU_MARS),
    "jupiter": ((5.20288700, 0.04838624, 1.30439695,
                 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714,
                 3034.74612775, 0.21252668, 0.20469106),
                126686534.0),
    "saturn": ((9.53667594, 0.05386179, 2.48599187,
                49.95424423, 92.60405805, 113.66344615),
               (-0.00125060, -0.00050991, 0.00193609,
                1222.49362201, -0.41897216, -0.28867794),
               37931187.0),
}

# 日心 N 体传播：探测器受太阳 + 第三体（金/木/土）间接摄动。
# 地球、火星不计入：出发/到达剩余速度 V∞1、V∞2 已含两体逃逸，
# 且探测器起点在地球、终点在火星附近，施其直接引力项会奇异。
# 这正是 §1.3 的"日心段只受太阳引力 + 行星球引力"模型。
_PERTURBERS = [p for p in _PLANETS if p not in ("earth", "mars")]


def unit(v):
    return v / np.linalg.norm(v)


def fmt_dt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def parse_dt(s: str) -> datetime:
    """宽容解析：YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'（与 step1 交棒约定一致）"""
    s = s.strip()
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError("时间格式：YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'")


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
    xv = a * (math.cos(E) - e)
    yv = a * math.sqrt(1 - e * e) * math.sin(E)
    vx = -a * math.sin(E) * Ed
    vy = a * math.sqrt(1 - e * e) * math.cos(E) * Ed
    w = math.radians(wb - Om)
    cw, sw = math.cos(w), math.sin(w)
    cO, sO = math.cos(math.radians(Om)), math.sin(math.radians(Om))
    cI, sI = math.cos(math.radians(I)), math.sin(math.radians(I))
    P = np.array([cw * cO - sw * sO * cI, cw * sO + sw * cO * cI, sw * sI])
    Q = np.array([-sw * cO - cw * sO * cI, -sw * sO + cw * cO * cI, cw * sI])
    return xv * P + yv * Q, vx * P + vy * Q


def _stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (1 - math.cos(s)) / z, (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z)
        return (1 - math.cosh(s)) / z, (math.sinh(s) - s) / s ** 3
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
            lo = z
            continue
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
    return (np.array(r2) - f * np.array(r1)) / g, \
           (gd * np.array(r2) - np.array(r1)) / g


# ---------------------------------------------------------------------------
# 日心 N 体传播（统一轨迹接口）
# ---------------------------------------------------------------------------
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


def _rk4_seg(t0, y0, t1, n_steps):
    """固定步长 RK4 积分 [t0, t1]（n 步）"""
    h = (t1 - t0) / n_steps
    t, y = t0, np.array(y0, float)
    for _ in range(n_steps):
        k1 = _accel(t, y)
        k2 = _accel(t + h / 2, y + h / 2 * k1)
        k3 = _accel(t + h / 2, y + h / 2 * k2)
        k4 = _accel(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


class _Traj:
    """统一轨迹访问：任意 t ∈ [t_start, t_end] 的日心状态。
    scipy 路径：DOP853 + dense_output 插值；
    RK4 降级：6 h 存档 + 段内 600 s 细步重积分。
    两条路径对上层（扫描/细化/评估）完全同接口、同规则。
    """

    def __init__(self, t0, y0, t_end):
        self.t0, self.t_end = t0, t_end
        if _HAVE_SCIPY:
            sol = solve_ivp(_accel, (t0, t_end), np.asarray(y0, float),
                            rtol=1e-11, atol=1e-6, method="DOP853",
                            dense_output=True)
            self._sol = sol.sol
            self._mode = "dense"
        else:
            n = int((t_end - t0) / RK4_ARCH_STEP_S) + 1
            h = (t_end - t0) / n
            ts = np.empty(n + 1)
            ys = np.empty((n + 1, 6))
            t, y = t0, np.asarray(y0, float)
            ts[0], ys[0] = t, y
            for k in range(n):
                k1 = _accel(t, y)
                k2 = _accel(t + h / 2, y + h / 2 * k1)
                k3 = _accel(t + h / 2, y + h / 2 * k2)
                k4 = _accel(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                ts[k + 1], ys[k + 1] = t, y
            self._ts, self._ys = ts, ys
            self._mode = "rk4"

    def state(self, t: float) -> np.ndarray:
        t = min(max(t, self.t0), self.t_end)   # 数值安全钳位
        if self._mode == "dense":
            return self._sol(t)
        i = int(np.searchsorted(self._ts, t)) - 1
        i = min(max(i, 0), len(self._ts) - 2)
        span = self._ts[i + 1] - self._ts[i]
        nseg = max(2, int(span / RK4_FINE_STEP_S))
        return _rk4_seg(self._ts[i], self._ys[i], t, nseg)

    def sample(self, lo, hi):
        """返回 (ts, states) 网格采样（states 形状 (N,6)）"""
        if self._mode == "dense":
            ts = np.linspace(lo, hi, N_SCAN)
            return ts, self._sol(ts).T
        m = (self._ts >= lo - 1e-9) & (self._ts <= hi + 1e-9)
        return self._ts[m], self._ys[m]


def _rel_mars_state(traj, t):
    rm, vm = helio_state("mars", t)
    y = traj.state(t)
    return y[:3] - rm, y[3:] - vm


def _rel_mars_dist(traj, t):
    rm, _ = helio_state("mars", t)
    return float(np.linalg.norm(traj.state(t)[:3] - rm))


def arrival_bplane(t0, r0, v0, t1):
    """
    传播到接近段，在统一评估点上提取 B 平面。
    BT、BR 沿整条接近双曲线为常量（二体极限内），τ 按 τ+t_eval 换算回 t1
    基准，故评估点取"入段 SOI 穿越"或"最近接近点"在模型内等价；
    取 SOI/最近接近而非 r→0 处，是为了避免退化并贴近书的拼接圆锥定义。
    模型内敏感度：N 体潮汐使 SOI 入口与近火点处的 osculating B 相差
    ~10–100 km（Δv_tide ~0.8 m/s × 杠杆臂），即 F 的定义精度量级。
    注：雅可比扰动使 dmin 跨越 SOI 边界时评估规则切换会引入 ~10 km 级
    F 不连续——远初值时雅可比略脏，近收敛时无碍（LM 阻尼 + 回溯兜底）。
    """
    dt = 24 * 3600.0
    lo = max(t1 - SCAN_PRE_D * dt, t0 + 3600.0)
    hi = t1 + SCAN_POST_D * dt
    traj = _Traj(t0, np.concatenate([r0, v0]), hi)

    # ---- 粗扫最近接近点（dense 向量化 / RK4 用存档网格，规则一致）----
    ts, ys = traj.sample(lo, hi)
    mars = np.array([helio_state("mars", t)[0] for t in ts])
    ds = np.linalg.norm(ys[:, :3] - mars, axis=1)
    i0 = int(np.argmin(ds))
    t_ca = float(ts[i0])
    step = float(ts[1] - ts[0]) if len(ts) > 1 else 3600.0

    # ---- 三分细化（边界钳位到扫描窗）----
    a, b = max(lo, t_ca - step), min(hi, t_ca + step)
    for _ in range(TERNARY_ITERS):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if _rel_mars_dist(traj, m1) < _rel_mars_dist(traj, m2):
            b = m2
        else:
            a = m1
    t_ca = 0.5 * (a + b)
    dmin = _rel_mars_dist(traj, t_ca)

    # ---- 评估点：真实掠过取入段 SOI 穿越；否则取最近接近点 ----
    if dmin < R_SOI_MARS:
        aa, bb = lo, t_ca
        if _rel_mars_dist(traj, aa) < R_SOI_MARS:
            t_eval = t_ca   # 评估窗起点已在 SOI 内（早到轨迹），退回最近接近点
        else:
            for _ in range(SOI_BISEC_ITERS):
                mm = 0.5 * (aa + bb)
                if _rel_mars_dist(traj, mm) > R_SOI_MARS:
                    aa = mm
                else:
                    bb = mm
            t_eval = 0.5 * (aa + bb)
    else:
        t_eval = t_ca

    rf, vf = _rel_mars_state(traj, t_eval)
    bp = bplane_from_state(rf, vf)
    tau_from_t1 = bp.tau + (t_eval - t1)
    return BPlane(bp.BT, bp.BR, tau_from_t1, bp.r_p, bp.v_inf,
                  bp.i, bp.Bv, bp.S)


# ---------------------------------------------------------------------------
# 实际 B 平面参数（式 1.21~1.30），全部在火心 J2000 系
# ---------------------------------------------------------------------------
@dataclass
class BPlane:
    BT: float
    BR: float
    tau: float        # 距近火点时间 (s)，>0 表示尚未到近火点
    r_p: float        # 实际近火点半径 km
    v_inf: float
    i: float          # 轨道倾角 deg（相对参考轴 N̂=Z）
    Bv: np.ndarray = None   # B 矢量（日心黄道直角系）
    S: np.ndarray = None    # 渐近线方向 Ŝ


def bplane_from_state(rf, vf, mu=MU_MARS, ref=None):
    """式 1.21~1.30：由火心相对状态求实际 B 平面。
    ref 为参考面法向 N̂，默认取 N_REF = 黄道极 [0,0,1]（书式1.14 的 Z 轴）。
    """
    if ref is None:
        ref = N_REF
    rf = np.asarray(rf, float)
    vf = np.asarray(vf, float)
    r = float(np.linalg.norm(rf))
    v2 = float(vf @ vf)
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

    # 入射渐近线单位矢 Ŝ（书式1.24）：θ 为拱线-渐近线夹角 = acos(1/e)。
    # S = cosθ·(e2/|e2|) + sinθ·(n2×e2/|n2×e2|)
    # 书表3 自证：θ1=36.8827°=acos(1/1.250210)，非 asin(1/e)=53.1173°。
    # 误用 asin 会让 Ŝ 偏离真渐近线：偏差 = 90°−2·acos(1/e)，
    # 火星到达典型 e≈2.4 时约 42°（地球出发 e≈1.25 时约 16°），T̂/R̂ 基全偏。
    theta_b = math.acos(1.0 / e)
    ct, st = math.cos(theta_b), math.sin(theta_b)
    cross = np.cross(n2, e2)
    S = ct * e2 / np.linalg.norm(e2) + st * cross / np.linalg.norm(cross)
    S = S / np.linalg.norm(S)

    Bvec = b * np.cross(S, n2)
    T = np.cross(S, ref)
    nT = float(np.linalg.norm(T))
    if nT < 1e-12:   # Ŝ ∥ N̂：BT/BR 退化，给个正交占位基
        tmp = np.array([1.0, 0.0, 0.0]) if abs(ref[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        T = np.cross(S, tmp)
        nT = float(np.linalg.norm(T))
    T = T / nT
    R = np.cross(S, T)
    BT = float(Bvec @ T)
    BR = float(Bvec @ R)

    # 真近点角 f
    cosf = (1 / e) * (p / r - 1)
    sinf = math.sqrt(p / mu) * (rf @ vf) / (e * r)
    f = math.atan2(sinf, cosf)
    # 双曲偏近点角 H：coshH=(e+cosf)/(1+e cosf)
    cosE = (e + cosf) / (1 + e * cosf)
    cosE = max(1.0, cosE)
    sign = 1.0 if math.sin(f) >= 0 else -1.0
    E = sign * math.acosh(cosE)
    sinhH = math.sinh(E)
    # 双曲平近点角 M = e·sinhH − H（入段 H<0 ⇒ M<0），M = n(t − t_p)
    # ⇒ 距近火点时间 τ = t_p − t = −M/n，入段为正
    tau = -math.sqrt(-a ** 3 / mu) * (e * sinhH - E)
    i = math.degrees(math.acos(max(-1.0, min(1.0,
                         float(n2 @ np.asarray(ref, float))))))
    return BPlane(BT, BR, tau, abs(a) * (e - 1), v_inf, i, Bvec, S)


# ---------------------------------------------------------------------------
# 差分修正：牛顿迭代 Δv0 = K⁻¹ (B^nom − F(v0))（式 1.31~1.33）
# ---------------------------------------------------------------------------
@dataclass
class Correction:
    v0: np.ndarray
    F: BPlane
    Bnom: tuple
    iters: int
    resid: float     # 归一化残差范数（BT/BR ÷1e3 km，τ ÷1e5 s）


def differential_correction(t0, r0, v0, t1, Bnom_vec,
                            dv_eps=DV_EPS, tol=CORR_TOL,
                            max_iter=CORR_MAX_ITER):
    """
    牛顿-莱文伯格迭代求 Δv0 = K⁻¹(B^nom − F(v0))（式1.31~1.33）。
    目标 [BT, BR, τ]，BT/BR 按式 1.21~1.30 在"当前 Ŝ 的 B 平面基"上定义，
    因 Bv ⟂ Ŝ 恒成立，故 BT²+BR²=‖Bv‖²=b²=‖B‖²，等价于同时锁定 ‖B‖ → r_p。
    若改用固定标称基做投影，则 Bv 沿 Ŝ0 的分量不受约束，‖B‖ 可以任意大，
    解不物理（近火点半径失控）——这也是常见 TPBVIP 实现的一个坑。
    数值设计：BT/BR(~1e3 km) 与 τ(~1e5 s) 量级差两个数量级 →
    归一化残差/雅可比 + LM 阻尼 + 回溯线搜索 + 单步限幅。
    传播失败（线搜索远点相对能量≤0 等）返回 None 而非崩溃：
    雅可比该方向退化为单侧差分，线搜索视为失败自动回退。
    收敛判据与返回的 resid 均为归一化范数。
    """
    v0 = np.array(v0, float)
    target = np.array(Bnom_vec, float)
    scale = np.array([1e3, 1e3, 1e5])

    def F(v):
        try:
            return arrival_bplane(t0, r0, v, t1)
        except ValueError:
            return None

    def vec(bp):
        return np.array([bp.BT, bp.BR, bp.tau])

    def resid(bp):
        return (target - vec(bp)) / scale

    f0 = F(v0)
    if f0 is None:
        raise ValueError("初值传播即无 B 平面（相对火星能量≤0），请检查初值")
    r_n = float(np.linalg.norm(resid(f0)))
    lam = 1e-2
    it_used = 0
    for it in range(1, max_iter + 1):
        it_used = it
        if r_n < tol:
            break
        # 归一化雅可比 K = ∂[BT,BR,τ]/∂v0：必须用 vec 差分（正向）。
        # 若用 resid 差分会得到 -J，牛顿方向反向，线搜索全部失败
        # （症状：迭代次数恰为 lam 熔断阶梯 1e-2·4^k > 1e6，Δv0 = 0）。
        K = np.zeros((3, 3))
        for j in range(3):
            dp = v0.copy(); dp[j] += dv_eps
            dm = v0.copy(); dm[j] -= dv_eps
            fp, fm = F(dp), F(dm)
            if fp is not None and fm is not None:
                K[:, j] = ((vec(fp) - vec(fm)) / scale) / (2 * dv_eps)
            elif fp is not None:
                K[:, j] = ((vec(fp) - vec(f0)) / scale) / dv_eps
            elif fm is not None:
                K[:, j] = ((vec(f0) - vec(fm)) / scale) / dv_eps
        rr = (target - vec(f0)) / scale
        A = K.T @ K + lam * np.eye(3)
        try:
            dv = np.linalg.solve(A, K.T @ rr)
        except np.linalg.LinAlgError:
            dv = np.linalg.lstsq(K, rr, rcond=None)[0]
        step = float(np.linalg.norm(dv))
        if step > DV_MAX_STEP:
            dv *= DV_MAX_STEP / step
        # 回溯线搜索
        ok = False
        for _ in range(20):
            vtry = v0 + dv
            ftry = F(vtry)
            if ftry is not None:
                r_try = float(np.linalg.norm(resid(ftry)))
                if r_try < r_n:
                    v0, f0, r_n = vtry, ftry, r_try
                    lam = max(lam * 0.5, 1e-8)
                    ok = True
                    break
            dv *= 0.5
        if not ok:
            lam *= 4.0
            if lam > 1e6:
                break
    return Correction(v0, f0, tuple(target), it_used, r_n)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _jd(dt: datetime) -> float:
    return (dt - J2000).total_seconds() / 86400.0 + 2451545.0


def compute(i2=95.0, rp2_alt=400.0, correct=True, dep=None, arr=None):
    """
    默认窗口1（书表1.2）：2015-12-25 → 2016-09-29，转移 278 天。
    注意：默认 rp2_alt=400 km 为工程常用值；书 §1.2.3 窗口2 例
    （由表1.5 反解）约 1294 km —— 对书请用 --rp2 1294。
    式1.16 不可达（|cos i2| > cos δ∞2）时跳过差分修正。

    接口约定：dep/arr 与 step1 同源。dep 建议填 step1 报告的
    "实际出 SOI 时刻"（可带 HH:MM:SS），Lambert 初值即探测器真实
    日心状态的同源近似。r0 取地球中心而非 SOI 出口点：拼接圆锥的
    标准接口，~0.6% 的位置偏移由差分修正调 v0 吸收（B 平面对 v0
    的敏感度足够），不必在几何上换心。
    """
    if dep is None:
        dep = datetime(2015, 12, 25, 0, 0, 0)
    if arr is None:
        arr = datetime(2016, 9, 29, 0, 0, 0)
    if arr <= dep:
        raise ValueError("到达日期必须晚于出发日期")
    if rp2_alt <= 0:
        raise ValueError("近火点高度必须为为正")

    t0 = (_jd(dep) - 2451545.0) * 86400.0
    t1 = (_jd(arr) - 2451545.0) * 86400.0
    tof = t1 - t0

    # 全程日心黄道 J2000，自洽：
    # Lambert 给出转移弧两端速度 → V∞1(出发剩余)、V∞2(到达剩余)
    re, ve = helio_state("earth", t0)
    rm, vm = helio_state("mars", t1)
    v1, v2 = lambert(re, rm, tof)
    V_inf1 = v1 - ve          # 出发地心剩余速度(黄道)
    V_inf2 = v2 - vm          # 到达火心剩余速度(黄道)
    v1m = float(np.linalg.norm(V_inf1))
    v2m = float(np.linalg.norm(V_inf2))

    r_p2 = R_MARS + rp2_alt
    # 标称 ||B|| 由 r_p2 与 V∞2（式1.17 第一式）
    Bmag = math.sqrt(r_p2 ** 2 + 2 * r_p2 * MU_MARS / v2m ** 2)
    # 标称 θb：cos i2 = cosθb · cosδ∞2（式1.16）
    # δ∞2 = Ŝ 的出黄纬（书式1.13/1.16 的赤纬，参考轴为 Z）：asin(Ŝ·N̂)
    dec2 = math.degrees(math.asin(max(-1.0, min(1.0,
                        float(np.dot(V_inf2 / v2m, N_REF))))))
    cos_th = math.cos(math.radians(i2)) / math.cos(math.radians(dec2))
    feasible = abs(cos_th) <= 1.0   # 式1.16：|cos i2| ≤ |cos δ∞2| 才可达
    cos_th = max(-1.0, min(1.0, cos_th))
    th_deg = math.degrees(math.acos(cos_th))
    BT_nom = Bmag * math.cos(math.radians(th_deg))
    BR_nom = Bmag * math.sin(math.radians(th_deg))

    # 精确模型：以 Lambert 初值 N 体传播 → 实际 B 平面（修正前）
    bp_unc = arrival_bplane(t0, re, v1, t1)

    corr = None
    skipped = False
    if correct:
        if feasible:
            corr = differential_correction(t0, re, v1, t1,
                                           [BT_nom, BR_nom, 0.0])
        else:
            skipped = True   # 不可达：跳过修正，报告注明

    return dict(dep=dep, arr=arr, V_inf1=V_inf1, v1=v1m,
                V_inf2=V_inf2, v2=v2m, dec2=dec2, i2=i2,
                r_p2=r_p2, rp2_alt=rp2_alt, Bmag=Bmag, th_deg=th_deg,
                feasible=feasible, BT_nom=BT_nom, BR_nom=BR_nom,
                bp_unc=bp_unc, corr=corr, skipped=skipped,
                v0_lambert=np.asarray(v1, float), tof_days=tof / 86400.0)


def report(r) -> str:
    L = []
    L.append("=" * 70)
    L.append("火心段 B 平面设计 + 精确模型差分修正")
    L.append("=" * 70)
    L.append("[窗口] 出发 %s UTC → 到达 %s UTC（转移 %.0f 天）"
             % (fmt_dt(r["dep"]), fmt_dt(r["arr"]), r["tof_days"]))
    L.append("       （历表时刻按 UTC≈TT，差 ~69 s 由差分修正吸收）")
    L.append("")
    L.append("[剩余速度（Lambert，日心黄道自洽）]")
    L.append("  V∞1 = [%+.4f, %+.4f, %+.4f]  |V∞1| = %.4f km/s（出发，参考）"
             % (*r["V_inf1"], r["v1"]))
    L.append("  V∞2 = [%+.4f, %+.4f, %+.4f]  |V∞2| = %.4f km/s（到达）"
             % (*r["V_inf2"], r["v2"]))
    L.append("  δ∞2 = %.2f°（相对 N̂=Z 的出黄纬，式1.14）" % r["dec2"])
    L.append("")
    L.append("[A] 初步设计（式1.17）  i2=%.1f°  近火点高度=%.0f km (r_p2=%.1f km)"
             % (r["i2"], r["rp2_alt"], r["r_p2"]))
    L.append("  ||B|| = %.1f km   B 平面钟角 θ = %.3f° (式1.17)"
             % (r["Bmag"], r["th_deg"]))
    if not r["feasible"]:
        L.append("  ⚠ 式1.16 不可达：|cos i2|>cos δ∞2，"
                 "可达范围 [%.2f°, %.2f°]，结果不可用"
                 % (abs(r["dec2"]), 180.0 - abs(r["dec2"])))
    L.append("  B^nom = [BT=%.1f, BR=%.1f km, τ=0 s]"
             % (r["BT_nom"], r["BR_nom"]))
    L.append("")
    bu = r["bp_unc"]
    L.append("[B] 精确模型 · 圆锥拼接初值（未修正）")
    L.append("  实际 B = [BT=%.1f, BR=%.1f km, τ=%.1f s]"
             % (bu.BT, bu.BR, bu.tau))
    L.append("  实际近火点 r_p=%.1f km (高度 %.0f)  i=%.2f°"
             % (bu.r_p, bu.r_p - R_MARS, bu.i))
    L.append("  → 残差 ΔBT=%.1f ΔBR=%.1f km（圆锥拼接满足不了火心段，需修正）"
             % (bu.BT - r["BT_nom"], bu.BR - r["BR_nom"]))
    L.append("")
    if r["corr"] is not None:
        c = r["corr"]
        L.append("[B] 差分修正（式1.31~1.33，牛顿迭代 %d 次）" % c.iters)
        L.append("  Lambert 初值 v0(日心) = [%+.4f, %+.4f, %+.4f] km/s"
                 % (*r["v0_lambert"],))
        L.append("  修正后     v0(日心) = [%+.4f, %+.4f, %+.4f] km/s"
                 % (*c.v0,))
        L.append("  修正量 Δv0 = %.4f km/s"
                 % float(np.linalg.norm(c.v0 - r["v0_lambert"])))
        L.append("  修正后 B = [BT=%.2f, BR=%.2f km, τ=%.3f s]"
                 % (c.F.BT, c.F.BR, c.F.tau))
        L.append("  终端 r_p=%.1f km (目标 %.1f)  i=%.4f° (目标 %.1f°)"
                 % (c.F.r_p, r["r_p2"], c.F.i, r["i2"]))
        L.append("  终端残差 ΔBT=%.2e ΔBR=%.2e km  Δτ=%.2e s"
                 "（归一 %.2e，容差 %.0e）"
                 % (abs(r["BT_nom"] - c.F.BT), abs(r["BR_nom"] - c.F.BR),
                    abs(c.F.tau), c.resid, CORR_TOL))
        if c.resid > 1e-4:
            L.append("  ⚠ 未收敛到容差，结果仅供参考")
        L.append("  注：N 体下 BT/BR/τ 的模型内定义精度 ~10 km（潮汐摄动），"
                 "残差即模型内零点。")
    elif r["skipped"]:
        L.append("[B] 差分修正：已跳过（式1.16 不可达，B^nom 无物理解）")
    else:
        L.append("[B] 差分修正：未启用（--no-correct）")
    L.append("=" * 70)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="火心段 B 平面设计与精确模型差分修正（单文件）")
    ap.add_argument("--i2", type=float, default=95.0,
                    help="目标火心倾角 deg")
    ap.add_argument("--rp2", type=float, default=400.0,
                    help="近火点高度 km（书§1.2.3 窗口2 例约 1294；"
                         "默认 400 为工程常用）")
    ap.add_argument("--dep", default=None,
                    help="出发日期，可带时刻（如 '2020-07-21 17:59:19'，"
                         "= step1 实际出SOI交棒钟；默认 2015-12-25，书表1.2）")
    ap.add_argument("--arr", default=None,
                    help="到达日期，可带时刻（默认 2016-09-29）")
    ap.add_argument("--no-correct", action="store_true",
                    help="只出初步设计")
    a = ap.parse_args(argv)

    dep = arr = None
    try:
        if a.dep:
            dep = parse_dt(a.dep)
        if a.arr:
            arr = parse_dt(a.arr)
    except ValueError as e:
        sys.exit("错误：%s" % e)
    if dep is not None and arr is not None and arr <= dep:
        sys.exit("错误：到达日期必须晚于出发日期")

    try:
        r = compute(i2=a.i2, rp2_alt=a.rp2, correct=not a.no_correct,
                    dep=dep, arr=arr)
    except ValueError as e:
        sys.exit("错误：%s" % e)
    print(report(r))
    if not _HAVE_SCIPY:
        print("注：未检测到 scipy，使用存档+局部细步 RK4"
              "（评估规则与 scipy 路径一致，精度略低）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
