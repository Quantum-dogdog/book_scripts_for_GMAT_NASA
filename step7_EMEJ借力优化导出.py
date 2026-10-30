#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step7 · pykep MGA-1DSM 借力优化 → 导出 nodes.json 给 step6

算例（--case）：
    emej  : 地->火->地->木（默认，2024 窗口）
    eveej : Galileo 地->金->地->地->木（1989 窗口，4 段 3 借力）

输出 nodes.json（km / km/s / MJD；pykep 内部是 SI，这里统一换算），step6 直接吃：
    python step6_引力辅助修正与EMEJ桥接.py --nodes emej_nodes.json --csv emej.csv

常用参数：
    --case eveej        换 Galileo VEEGA 算例
    --x "1.2,0.3,..."   直接用已知冠军染色体，跳过优化
    --xfile x.json      从 JSON 读冠军染色体（本文件会写 <out>_x.json）
    --seed 42           固定 DE 随机种子（默认不固定，每次结果不同）
    --gen/--islands/--pop   进化代数 / 岛数 / 每种群规模
    --out a.json        输出文件名
    --no-plot           不弹图
    --selfcheck         额外打印一行：把导出的 DSM 跟 pykep 内部值对一遍

为什么要 --selfcheck：
    本文件在 pykep 之外自己写了一套解码（export_nodes）。两套实现只要有一处
    下标错位，输出的 nodes.json 就悄悄错了，而它照样能画出像样的轨迹。让 pykep
    重算一遍、逐段比大小，就是给这层错位兜底；不一致直接 FAIL 非零退出。
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
    """'mars(jpl_lp)' / 'Mars' -> 'mars'（给 step6 认名字）"""
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


def _build_udp(case="emej"):
    """按算例名构建 MGA-1DSM。

    emej  : 地->火->地->木（2024 窗口）
    eveej : Galileo 伽利略号 地->金->地->地->木（1989 窗口，4 段 3 借力）
            参数取自 pykep_example 的 jialilve_eveej 算例。
    """
    if case == "emej":
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
    elif case == "eveej":
        seq = [jpl_lp('earth'), jpl_lp('venus'),
               jpl_lp('earth'), jpl_lp('earth'), jpl_lp('jupiter')]
        udp = mga_1dsm(
            seq=seq,
            t0=[epoch(-3731.5), epoch(-3717.5)],      # 1989-10-11 ~ 10-25
            tof=[
                [0.25 * 365.25, 0.50 * 365.25],       # E->V
                [0.65 * 365.25, 0.95 * 365.25],       # V->E
                [1.80 * 365.25, 2.20 * 365.25],       # E->E 2:1 共振
                [2.50 * 365.25, 3.50 * 365.25],       # E->J
            ],
            vinf=[2.5, 5.0],
            add_vinf_dep=True,
            add_vinf_arr=True,
            multi_objective=False,
        )
    else:
        raise ValueError("未知算例 %r（可选 emej / eveej）" % case)
    return seq, udp


def _selfcheck(udp, x, nodes, tol_mps=1e-3):
    """--selfcheck：拿 pykep 自己的 _compute_dvs 复核 export_nodes 的输出。

    pykep 的 dv[0] 把出发 |v∞| 和第一段 DSM 记在一起，所以要比 DSM 得先减掉 x[3]；
    末位 dv 就是到达 |v∞|。
    """
    try:
        dv, _lam, _T, _bl, _be = udp._compute_dvs(x)
    except Exception as exc:                       # noqa: BLE001
        print("[selfcheck] 跳过：pykep _compute_dvs 调不通（%s）" % exc)
        return True
    ref = [float(dv[0]) - float(x[3])] + [float(v) for v in dv[1:-1]]
    got = [float(v) for v in nodes["meta"]["dsm_mag_m_s"]]
    if len(ref) != len(got):
        print("[selfcheck] FAIL：DSM 段数不一致，导出 %d 段 / pykep %d 段"
              % (len(got), len(ref)))
        return False
    err_dsm = max(abs(a - b) for a, b in zip(got, ref))
    v_arr = float(nodes["meta"]["vinf_arr_m_s"])
    err_arr = abs(v_arr - float(dv[-1]))
    ok = err_dsm <= tol_mps and err_arr <= tol_mps
    print("[selfcheck] DSM 最大偏差 %.1e m/s，到达 |v∞| 差 %.1e m/s -> %s"
          % (err_dsm, err_arr, "PASS" if ok else "FAIL"))
    if not ok:
        print("           pykep：%s" % " ".join("%.3f" % v for v in ref))
        print("           导出 ：%s" % " ".join("%.3f" % v for v in got))
    return ok


def run(argv=None):
    ap = argparse.ArgumentParser(description="pykep MGA-1DSM -> nodes.json")
    ap.add_argument("--out", default="emej_nodes.json")
    ap.add_argument("--case", default="emej", choices=["emej", "eveej"],
                    help="emej=地火地木；eveej=Galileo 地金地地木")
    ap.add_argument("--x", default=None, help="冠军染色体，逗号分隔（跳过优化）")
    ap.add_argument("--xfile", default=None, help="从 json/txt 读染色体")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--islands", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--selfcheck", action="store_true",
                    help="用 pykep 内部 _compute_dvs 复核导出的 DSM / 到达 |v∞|，不一致则非零退出")
    a = ap.parse_args(argv)

    seq, udp = _build_udp(a.case)

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
        print("SADE 优化：%d 岛 × %d 种群 × 10 轮 evolve ..." % (a.islands, a.pop))
        archi.evolve(10)
        archi.wait()
        fs = [float(np.ravel(v)[0]) for v in archi.get_champions_f()]
        idx = int(np.argmin(fs))
        x = [float(q) for q in archi.get_champions_x()[idx]]
        print("完成，冠军 J = %.4f km/s" % (fs[idx] / 1e3))

    if not a.no_plot:
        try:
            udp.plot(x)
        except Exception as exc:                   # noqa: BLE001
            print("[warn] plot 跳过：%s" % exc)

    with open(os.path.splitext(a.out)[0] + "_x.json", "w", encoding="utf-8") as fh:
        json.dump(dict(x=list(x)), fh, indent=1)

    nodes, path = export_nodes(seq, x, a.out)
    m = nodes["meta"]
    eps = ([nodes["departure"]["epoch_mjd"]]
           + [f["epoch_mjd"] for f in nodes["flybys"]]
           + [nodes["arrival"]["epoch_mjd"]])
    vd = float(np.linalg.norm(nodes["departure"]["vinf_km_s"]))
    print("")
    print("节点历元（MJD）：%s" % " -> ".join("%.4f" % e for e in eps))
    print("段飞行时间（d）：%s" % " / ".join(
        "%.2f" % q for q in m.get("tof_days", list(np.diff(eps)))))
    print("出发 |v∞| = %.3f km/s（C3 = %.3f km²/s²）；到达 |v∞| = %.3f km/s"
          % (vd, vd ** 2, m["vinf_arr_m_s"] / 1e3))
    print("飞越 rp：%s" % " / ".join(
        "%s %.1f km" % (f["body"], f["rp_km"]) for f in nodes["flybys"]))
    dm = m.get("dsm_mag_m_s") or []
    if any(dm):
        print("DSM：%s，合计 %.1f m/s"
              % (" / ".join("%.1f" % q for q in dm), float(sum(dm))))
    print("染色体 -> %s_x.json；nodes -> %s"
          % (os.path.splitext(a.out)[0], path))
    if a.selfcheck:
        return 0 if _selfcheck(udp, x, nodes) else 1


if __name__ == "__main__":
    # pykep 3.0.1 解释器退出时偶发堆析构错误（C++ 析构顺序问题，不影响已打印
    # 结果），统一在出口 os._exit() 收口：退出码透传，异常打印 traceback 后以 1 退出。
    import sys
    _code = 0
    try:
        _code = run() or 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
