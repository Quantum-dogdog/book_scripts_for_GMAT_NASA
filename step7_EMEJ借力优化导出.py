#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emej_export.py -- pykep MGA-1DSM(地->火->地->木) 优化 + 导出 6.py 的 nodes.json

直接跑本文件即可（等价于你原来的 emej_sade.py），最后多出一个 emej_nodes.json：

    python 6.py --nodes emej_nodes.json --csv emej.csv

注意 pykep 内部是 SI（m, m/s, m^3/s^2），写出去的 JSON 一律换成 km / km/s / MJD。

常用参数：
    --x "1.2,0.3,..."   直接用已知冠军染色体，跳过优化
    --seed 42           固定 DE 随机种子（默认不固定，每次结果不同）
    --out a.json        输出文件名
    --no-plot           不弹图
"""
import argparse
import json
import math
import os

import numpy as np
import pykep as pk
import pygmo as pg
from pykep import epoch
from pykep.trajopt import mga_1dsm

try:                                        # pykep 2.x
    from pykep.planet import jpl_lp
except Exception:                           # noqa: BLE001  pykep 3.x
    from pykep.udpla import jpl_lp as _jpl_lp

    def jpl_lp(name):
        return pk.planet(_jpl_lp(name))

MJD_REF = 51544.0          # pykep: epoch(0) = 2000-01-01T00:00 -> 真 MJD 51544.0


def _body_name(pl):
    """'mars(jpl_lp)' / 'Mars' -> 'mars'（给 6.py 认名字）"""
    return str(pl.name).split("(")[0].strip().lower()


def _vinf_vec_ms(x):
    """x[1]=u, x[2]=v, x[3]=Vinf[m/s] -> 日心 vinf 矢量 [m/s]，与 pykep 解码逐位一致。"""
    th = 2.0 * math.pi * x[1]
    vv = max(-1.0, min(1.0, 2.0 * x[2] - 1.0))
    ph = math.acos(vv) - math.pi / 2.0
    mag = float(x[3])
    return np.array([mag * math.cos(ph) * math.cos(th),
                     mag * math.cos(ph) * math.sin(th),
                     mag * math.sin(ph)], float)


def export_nodes(seq, x, path="emej_nodes.json", tof_encoding="direct"):
    """把 pykep MGA-1DSM 的冠军染色体 x 解码成桥接器 nodes 字典并落盘。

    全程在 pykep 的 SI 单位下计算，落盘时再换算成 km / km/s。
    """
    if tof_encoding != "direct":
        raise NotImplementedError("本导出按 direct 编码写；alpha/eta 先转 direct")

    mu = seq[0].mu_central_body            # m^3/s^2
    n = len(seq) - 1
    T = np.array(x[5::4], dtype=float)     # 各段飞行时间 [天]
    if len(T) != n:
        raise ValueError("染色体长度与段数不符：len(x)=%d, n_legs=%d" % (len(x), n))

    epo = [float(x[0] + sum(T[:i])) for i in range(n + 1)]   # mjd2000
    rP, vP = [], []
    for i, pl in enumerate(seq):
        r, v = pl.eph(epoch(epo[i]))
        rP.append(np.array(r, float))      # m
        vP.append(np.array(v, float))      # m/s

    def prop(r, v, dt_days):
        rr, vv = pk.propagate_lagrangian([list(r), list(v)],
                                        dt_days * pk.DAY2SEC, mu)
        return np.array(rr, float), np.array(vv, float)

    def lambert(r1, r2, dt_days):
        lp = pk.lambert_problem(r1, r2, dt_days * pk.DAY2SEC, mu,
                                cw=False, multi_revs=0)
        return np.array(lp.v0[0], float), np.array(lp.v1[0], float)

    Vinf0 = _vinf_vec_ms(x)                # m/s
    flies, dsms = [], []
    kms = 1e-3

    # --- leg 0: 出发 -> 第一次借力 ------------------------------------------
    eta = float(x[4])
    r_b, v_b = prop(rP[0], vP[0] + Vinf0, eta * T[0])
    v_beg, v_end = lambert(r_b, rP[1], (1.0 - eta) * T[0])
    dsm = v_beg - v_b
    dsms.append(dict(epoch_mjd=epo[0] + eta * T[0] + MJD_REF,
                     dv_km_s=[float(q * kms) for q in dsm]))
    v_in = v_end                            # 到达 planet1 的航天器速度 [m/s]

    # --- 每个借力 + 其后一段 -------------------------------------------------
    for i in range(1, n):
        rr = float(x[7 + 4 * (i - 1)]) * seq[i].radius     # m
        beta = float(x[6 + 4 * (i - 1)])
        v_out = np.array(pk.fb_vout(list(v_in), list(vP[i]), rr, beta,
                                    seq[i].mu_self), float)
        flies.append(dict(
            body=_body_name(seq[i]),
            epoch_mjd=epo[i] + MJD_REF,
            vinf_in_km_s=[float(q * kms) for q in (v_in - vP[i])],
            vinf_out_km_s=[float(q * kms) for q in (v_out - vP[i])],
            rp_km=rr * kms))

        eta = float(x[8 + 4 * (i - 1)])
        r_b, v_b = prop(rP[i], v_out, eta * T[i])
        v_beg, v_end = lambert(r_b, rP[i + 1], (1.0 - eta) * T[i])
        dsm = v_beg - v_b
        dsms.append(dict(epoch_mjd=epo[i] + eta * T[i] + MJD_REF,
                         dv_km_s=[float(q * kms) for q in dsm]))
        v_in = v_end

    nodes = dict(
        departure=dict(body=_body_name(seq[0]), epoch_mjd=epo[0] + MJD_REF,
                       vinf_km_s=[float(q * kms) for q in Vinf0]),
        flybys=flies,
        dsms=dsms,
        arrival=dict(epoch_mjd=epo[-1] + MJD_REF, target=_body_name(seq[-1])),
        meta=dict(source="pykep mga_1dsm (x decoded, SI->km)",
                  t0_mjd2000=float(x[0]),
                  tof_days=[float(t) for t in T],
                  dsm_mag_m_s=[float(np.linalg.norm(d["dv_km_s"]) * 1e3)
                               for d in dsms],
                  vinf_in_mag=[float(np.linalg.norm(f["vinf_in_km_s"]))
                               for f in flies],
                  vinf_out_mag=[float(np.linalg.norm(f["vinf_out_km_s"]))
                                for f in flies],
                  vinf_arr_m_s=float(np.linalg.norm(v_in - vP[-1]))))

    # 自检 1：无动力飞越必须 |vinf_out| == |vinf_in|
    for f in flies:
        d = abs(np.linalg.norm(f["vinf_out_km_s"]) - np.linalg.norm(f["vinf_in_km_s"]))
        if d > 1e-6:
            print("[warn] %s 飞越 |vinf_out|-|vinf_in| = %.3e km/s（应为 0）"
                  % (f["body"], d))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nodes, fh, indent=1, ensure_ascii=False)
    return nodes, path


def _build_udp():
    seq = [jpl_lp('earth'), jpl_lp('mars'), jpl_lp('earth'), jpl_lp('jupiter')]
    udp = mga_1dsm(
        seq=seq,
        t0=[epoch(9053), epoch(9054)],
        tof=[
            [0.3 * 365.25, 0.4 * 365.25],
            [1.8 * 365.25, 1.9 * 365.25],
            [3.3 * 365.25, 3.4 * 365.25],
        ],
        vinf=[0.1, 5],
        add_vinf_dep=True,
        add_vinf_arr=True,
        multi_objective=False,
    )
    return seq, udp


def _check_against_udp(udp, x, nodes):
    """用 pykep 自己的 _compute_dvs 复核导出值（DV[0] 含出发 vinf）。"""
    try:
        dv, _lam, T, _bl, _be = udp._compute_dvs(x)
    except Exception as exc:                       # noqa: BLE001
        print("[warn] 无法调用 _compute_dvs 复核：%s" % exc)
        return
    ref_dsm = [dv[0] - float(x[3])] + list(dv[1:-1])
    arr = nodes["meta"]["dsm_mag_m_s"]
    err = max(abs(a - b) for a, b in zip(arr, ref_dsm)) if ref_dsm else 0.0
    print("pykep 自带 DSM  m/s:", ["%.4f" % v for v in ref_dsm])
    print("导出    DSM  m/s:", ["%.4f" % v for v in arr])
    print("最大偏差 %.3e m/s；到达 vinf %.4f / %.4f m/s"
          % (err, nodes["meta"]["vinf_arr_m_s"], dv[-1]))


def run(argv=None):
    ap = argparse.ArgumentParser(description="pykep MGA-1DSM -> nodes.json")
    ap.add_argument("--out", default="emej_nodes.json")
    ap.add_argument("--x", default=None, help="冠军染色体，逗号分隔（跳过优化）")
    ap.add_argument("--xfile", default=None, help="从 json/txt 读染色体")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--islands", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    seq, udp = _build_udp()

    if a.x:
        x = [float(q) for q in a.x.replace(" ", "").split(",") if q]
    elif a.xfile:
        with open(a.xfile, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        x = [float(q) for q in (raw["x"] if isinstance(raw, dict) else raw)]
    else:
        kw = dict(gen=a.gen)
        if a.seed is not None:
            kw["seed"] = a.seed
        archi = pg.archipelago(algo=pg.sade(**kw), prob=udp,
                               n=a.islands, pop_size=a.pop)
        print("Running SADE on %d islands ..." % a.islands)
        archi.evolve(10)
        archi.wait()
        fs = [float(np.ravel(v)[0]) for v in archi.get_champions_f()]
        idx = int(np.argmin(fs))
        x = [float(q) for q in archi.get_champions_x()[idx]]
        print("Done!! Solutions found are: ", fs)

    udp.pretty(x)
    if not a.no_plot:
        try:
            udp.plot(x)
        except Exception as exc:                   # noqa: BLE001
            print("[warn] plot 跳过：%s" % exc)

    print("\nX =", repr(list(x)))
    with open(os.path.splitext(a.out)[0] + "_x.json", "w", encoding="utf-8") as fh:
        json.dump(dict(x=list(x)), fh, indent=1)

    nodes, path = export_nodes(seq, x, a.out)
    print("\n--- 导出校验 ---")
    _check_against_udp(udp, x, nodes)
    print("节点历元 MJD :", ["%.6f" % e for e in
                             [nodes["departure"]["epoch_mjd"]]
                             + [f["epoch_mjd"] for f in nodes["flybys"]]
                             + [nodes["arrival"]["epoch_mjd"]]])
    print("飞越 rp km  :", ["%.1f" % f["rp_km"] for f in nodes["flybys"]])
    print("nodes -> %s" % path)
    print(json.dumps(nodes, ensure_ascii=False))


if __name__ == "__main__":
    run()
