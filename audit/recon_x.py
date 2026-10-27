# -*- coding: utf-8 -*-
"""从 emej_nodes.json 反演冠军染色体 x（t0/T/eta/rp 从 meta+历元直接除出来，
β 用 pykep 自己的 fb_vout 数值反解），再交 emej_export.py --x 复算。
目的：TA 未上传 *_x.json——本脚本证明染色体可从交付件唯一重构。"""
import json, math, sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, "/home/z/my-project/emej-audit")
import pykep as pk
from pykep import epoch

try:
    from pykep.planet import jpl_lp          # pykep 2.x
except Exception:
    from pykep.udpla import jpl_lp as _jpl_lp  # pykep 3.x
    def jpl_lp(name):
        return pk.planet(_jpl_lp(name))

nodes = json.load(open("/home/z/my-project/emej-audit/data/emej_nodes.json"))
seq = [jpl_lp(n) for n in ("earth", "mars", "earth", "jupiter")]
MJD_REF = 51544.0

t0 = nodes["meta"]["t0_mjd2000"]
T = nodes["meta"]["tof_days"]
epo = [t0 + sum(T[:i]) for i in range(4)]

# --- 出发 vinf -> x[1], x[2], x[3] ----------------------------------------
vd = np.array(nodes["departure"]["vinf_km_s"]) * 1e3   # m/s
mag = float(np.linalg.norm(vd))
th = math.atan2(vd[1], vd[0]) % (2 * math.pi)
vv = -vd[2] / mag                                       # z = -mag*vv
x = [0.0] * 14
x[0] = t0
x[1] = th / (2 * math.pi)
x[2] = (vv + 1.0) / 2.0
x[3] = mag
x[4] = (nodes["dsms"][0]["epoch_mjd"] - nodes["departure"]["epoch_mjd"]) / T[0]
x[5], x[9], x[13] = T[0], T[1], T[2]

# --- 飞越：rp 分率 + beta 反解 ---------------------------------------------
for i, f in enumerate(nodes["flybys"]):          # i=0(mars)->x[6..8], i=1(earth)->x[10..12]
    pl = seq[i + 1]
    rp_m = f["rp_km"] * 1e3
    x[7 + 4 * i] = rp_m / pl.radius
    rP, vP = pl.eph(epoch(epo[i + 1]))
    vP = np.array(vP, float)
    v_in = vP + np.array(f["vinf_in_km_s"]) * 1e3
    v_tgt = vP + np.array(f["vinf_out_km_s"]) * 1e3
    mu = pl.mu_self
    def resid(b, _vi=list(map(float, v_in)), _vp=list(map(float, vP)),
               _rp=float(rp_m), _mu=float(mu), _vt=v_tgt):
        vo = np.array(pk.fb_vout(_vi, _vp, _rp, b, _mu), float)
        return float(np.sum((vo - _vt) ** 2))
    grid = np.linspace(-math.pi / 2, math.pi / 2, 2001)
    vals = [resid(b) for b in grid]
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 2, 0)], grid[min(k + 2, len(grid) - 1)]
    res = minimize_scalar(resid, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-13))
    x[6 + 4 * i] = float(res.x)
    x[8 + 4 * i] = (nodes["dsms"][i + 1]["epoch_mjd"] - f["epoch_mjd"]) / T[i + 1]
    print("flyby%d: beta=%.15f  残差=%.3e m²/s²" % (i, res.x, res.fun))

print("重构 x = [%s]" % ", ".join("%.17g" % v for v in x))
json.dump(dict(x=x), open("/home/z/my-project/emej-audit/audit/emej_x_recon.json", "w"),
          indent=1)
print("已写 audit/emej_x_recon.json")
