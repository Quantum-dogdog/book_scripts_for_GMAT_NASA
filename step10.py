#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmat_outer_target.py -- GMAT 外层打靶通用工具（单文件 / 纯标准库 / Windows + Linux）

==============================================================================
方法论：外层打靶（Outer-Loop Targeting over GMAT's Differential Corrector）
==============================================================================

问题
----
一条多段借力序列已在 GMAT 里跑通：每段有一个 Target/DC 块，用若干 burn 分量
(Vary) 去命中该段的 Achieve 目标（通常是某天体 B 平面的 BdotR/BdotT）。
现在想问：**只调最后一根腿的"目标值"，不动任何 burn 初值，能不能让末端
另一次的观测量命中给定值？** 典型场景：去掉末段深空机动(DSM)，只靠前一段
借力几何把飞船送到目标天体。

做法（本文件自动完成的 6 步）
------------------------------
1. 读 script（只读，绝不写回），解析出所有 Target ... EndTarget 块：
   - 控制量 x = 最后一个 SolveMode = Solve 的 Target 块里的 Achieve 值
     （即"最后一根腿"，这是默认；可用 --control 指定别的字段）
   - 目标量 y* = 其后的 Target 块（常是 RunInitialGuess）里的 Achieve 值
     （可用 --goal 显式给）
   - 截断点 = 目标块的起始行：把 DSM / 悬空打靶 / 不触发的 Periapsis
     停止条件整段丢掉，只保留前面已收敛的段
2. 构造变体：在内存里把控制 Achieve 的数值按行号精确替换，注入一个
   ReportFile，截断尾段，追加「固定历元 Propagate + Report」。
   变体只写到临时目录，原 script 的 md5 前后不变。
3. 驱动 GMAT 控制台跑变体，读 ReportFile，按列名取回 y（不硬编码列号）。
4. 把整个变体运行视为函数 y = f(x)。有限差分求雅可比 J，阻尼牛顿 +
   Broyden 更新 + 步长回溯，迭代到 |f(x) - y*| < tol。
   内层 DC 仍由 GMAT 自己解 burn，所以 Δv 代价自动最小、无需碰初值。
5. 验收（可选 --scan）：加密历元扫描真实传播状态，给出 |r| 最近点和
   burn Δv。区分「瞬时密切 B 平面残差」与「真实最近距离」两个判据，
   前者达标不代表后者是干净到达（可能是撞星/极深掠过）。
6. **只 print 修改建议**：第几行、原值、建议值、可直接粘贴的整行文本；
   以及可选的「删除第 N 行起尾段」建议。不写回原 script。

实战中踩过的坑（已内建为默认行为）
----------------------------------
* 停止条件 Sun.Apoapsis / <Body>.Periapsis 在新轨迹上常常不触发甚至跑飞；
  一律改用固定历元 TDBModJulian 传播。
* GMAT script 里出现非 ASCII 字符会解析失败；变体强制 ASCII + LF 写出。
* GMAT 的 TDBModJulian 与标准 MJD 基准差不一定相同（本例 29999.5 天），
  本工具不硬编码任何换算，历元一律沿用 script 里的单位，用 --epoch 给。
* 逼近大天体时单次积分会从几秒涨到几十秒；每次运行带超时，牛顿步长有阻尼。
* RunInitialGuess 的 Target 只施加 Vary 初值、从不求解，其 Achieve 值
  是"名义目标"，正好拿来当 y*，但不能当作已收敛的结果。

跨平台
------
控制台可执行文件自动探测顺序：--gmat-console > 环境变量 GMAT_CONSOLE >
--gmat-root / GMAT_ROOT > PATH > 各平台常见安装路径。
Linux 设 LD_LIBRARY_PATH(bin:lib:plugins)，Windows 把 bin 加到 PATH；
cwd 固定为 bin（GMAT 需要相对资源）。路径一律用 pathlib，报告/脚本用
绝对路径；GMAT 接受正斜杠，Windows 下自动 as_posix()。
无第三方依赖（不需要 numpy），线性代数自带高斯消元。

用法速查
--------
  # 0) 纯 Python 自检（不调 GMAT）
  python gmat_outer_target.py --self-test

  # 1) 只看解析结果，确认自动发现对不对（不调 GMAT）
  python gmat_outer_target.py --script step6v2.script --inspect

  # 2) 只验证一组给定值（跑 1 次 GMAT），并 print 修改建议
  python gmat_outer_target.py --script step6v2.script --epoch 34731.721007756 \
      --verify-only -3174.805517 12486.774499

  # 3b) 热启动续算（从上次的近似解继续收敛，省掉前面几次探测）
  python gmat_outer_target.py --script step6v2.script --epoch 34731.721007756 \
      --start -3174.5477 12488.0044 --tol 100 --max-iter 2 --fd-step 5

  # 3) 完整外层打靶（默认；每次 probe 跑一次 GMAT）
  python gmat_outer_target.py --script step6v2.script --epoch 34731.721007756 \
      --tol 100 --max-iter 6 --fd-step 50 --scan 34648 34668 0.5

  # Windows 示例（PowerShell）
  python gmat_outer_target.py --script "D:\\gmat\\step6v2.script" `
      --gmat-root "C:\\Program Files\\GMAT\\R2022a" --epoch 34731.721007756 `
      --verify-only -3174.805517 12486.774499

注意：--verify-only / --set 的数值顺序必须与 --inspect 打印的 control 顺序一致。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------- 控制台编码
def _setup_stdio():
    enc = "cp936" if os.name == "nt" else "utf-8"
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding=enc, errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------- 线性代数
def solve_linear(A, b):
    """高斯消元（部分主元），返回 x。A 为 n x n，b 长 n。"""
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError("奇异矩阵，无法求牛顿步长")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f == 0.0:
                continue
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def mat_vec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def lsq_step(J, d, damp=1e-9):
    """求 dx 使 J dx ~ -d；方阵直接解，非方阵用 (J^T J + lam I) 最小二乘。"""
    m, n = len(J), len(J[0])
    if m == n:
        return solve_linear(J, [-v for v in d])
    JTJ = [[sum(J[k][i] * J[k][j] for k in range(m)) for j in range(n)]
           for i in range(n)]
    JTy = [sum(J[k][i] * (-d[k]) for k in range(m)) for i in range(n)]
    for i in range(n):
        JTJ[i][i] += damp * max(1.0, abs(JTJ[i][i]))
    return solve_linear(JTJ, JTy)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def fmt(v):
    """GMAT 可读且保真的数值写法（15 位有效数字，够覆盖 MJD 级历元）。"""
    return "%.15g" % v


# ---------------------------------------------------------------- Script 解析
RE_TARGET = re.compile(r"^\s*Target\s+(\w+)\s*\{(.*)\}\s*;", re.I)
RE_ENDTARGET = re.compile(r"^\s*EndTarget\b", re.I)
RE_ACHIEVE = re.compile(
    r"^\s*Achieve\s+(\w+)\s*\(\s*([A-Za-z0-9_.]+)\s*=\s*"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*,\s*\{([^}]*)\}\s*\)\s*;", re.I)
RE_VARY = re.compile(r"^\s*Vary\s+\w+\s*\(\s*([A-Za-z0-9_.]+)\s*=", re.I)
RE_NUM = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
RE_PROPAGATE = re.compile(r"^\s*Propagate\s+(\w+)\s*\(", re.I)
RE_EPOCH = re.compile(r"TDBModJulian\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
RE_CREATE = re.compile(r"^\s*Create\s+(\w+)\s+(\w+)\s*;", re.I)
RE_BEGIN = re.compile(r"^\s*BeginMissionSequence\s*;", re.I)


class Achieve:
    def __init__(self, dc, field, value, opts, line_no, raw):
        self.dc, self.field, self.value, self.opts = dc, field, value, opts
        self.line_no, self.raw = line_no, raw          # line_no: 0-based

    def __repr__(self):
        return "Achieve(L%d %s = %s)" % (self.line_no + 1, self.field, fmt(self.value))


class TargetBlock:
    def __init__(self, dc, header, start, end, lines):
        self.dc, self.header, self.start, self.end = dc, header, start, end
        m = re.search(r"SolveMode\s*=\s*(\w+)", header, re.I)
        self.mode = m.group(1) if m else "?"
        self.achieves, self.varies = [], []
        for i in range(start + 1, end):
            a = RE_ACHIEVE.match(lines[i])
            if a:
                self.achieves.append(Achieve(a.group(1), a.group(2),
                                             float(a.group(3)), a.group(4), i, lines[i]))
            v = RE_VARY.match(lines[i])
            if v:
                self.varies.append(v.group(1))
        self.lines = lines

    @property
    def solve(self):
        return self.mode.lower() == "solve"

    def __repr__(self):
        return "Target(L%d-L%d %s %s, %d Achieve, %d Vary)" % (
            self.start + 1, self.end + 1, self.dc, self.mode,
            len(self.achieves), len(self.varies))


class ScriptModel:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        # 容错：BOM / 非 UTF-8 / CRLF
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        self.text = text
        self.lines = text.split("\n")
        self.md5 = hashlib.md5(self.path.read_bytes()).hexdigest()

        self.begin = next((i for i, l in enumerate(self.lines) if RE_BEGIN.match(l)), None)
        if self.begin is None:
            raise ValueError("script 里找不到 BeginMissionSequence;")

        self.objects = {}
        for l in self.lines[:self.begin]:
            m = RE_CREATE.match(l)
            if m:
                self.objects.setdefault(m.group(1).lower(), []).append(m.group(2))
        self.spacecraft = (self.objects.get("spacecraft") or ["DefaultSC"])[0]
        self.cs_names = set(self.objects.get("coordinatesystem") or [])
        self.propagators = self.objects.get("propagator") or []

        cnt = {}
        for l in self.lines[self.begin:]:
            m = RE_PROPAGATE.match(l)
            if m:
                cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
        self.propagator = max(cnt, key=cnt.get) if cnt else (self.propagators[0] if self.propagators else "")
        self.prop_counts = cnt

        self.epochs = [float(x) for x in RE_EPOCH.findall(text)]
        self.targets = self._parse_targets()

    def _parse_targets(self):
        out, i, n = [], self.begin + 1, len(self.lines)
        while i < n:
            m = RE_TARGET.match(self.lines[i])
            if m:
                j = i + 1
                while j < n and not RE_ENDTARGET.match(self.lines[j]):
                    j += 1
                out.append(TargetBlock(m.group(1), m.group(2), i, min(j, n - 1), self.lines))
                i = j + 1
            else:
                i += 1
        return out

    # ---- 自动发现 ----
    def control_target(self):
        s = [t for t in self.targets if t.solve and t.achieves]
        if not s:
            raise ValueError("找不到 SolveMode = Solve 且带 Achieve 的 Target 块")
        return s[-1]                       # 最后一根腿

    def goal_target(self):
        ctrl = self.control_target()
        later = [t for t in self.targets if t.start > ctrl.end and t.achieves]
        return later[-1] if later else None

    def auto_cut(self, goal):
        """截断点（0-based 行索引）：目标块起始行；无目标块则到文件末。"""
        return goal.start if goal is not None else len(self.lines)

    def auto_epoch(self):
        return max(self.epochs) if self.epochs else None

    def obs_prefix(self, field):
        """从 Achieve 字段推断坐标系前缀，如 DefaultSC.Juxyz.BdotR -> DefaultSC.Juxyz。"""
        parts = field.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:-1]), parts[-2]
        return None, None

    def xyz_cols(self, fields):
        """给目标字段补 X/Y/Z、VX/VY/VZ 列（仅当坐标系在 script 里声明过）。"""
        cols = []
        for f in fields:
            prefix, short = self.obs_prefix(f)
            if not prefix or (self.cs_names and short not in self.cs_names):
                continue
            for s in (".X", ".Y", ".Z", ".VX", ".VY", ".VZ"):
                c = prefix + s
                if c not in cols:
                    cols.append(c)
        return cols

    def burn_cols(self, ctrl):
        cols = []
        for f in ctrl.varies:
            if re.search(r"\.Element\d+$", f) and f not in cols:
                cols.append(f)
        return cols


# ---------------------------------------------------------------- 变体构造
REPORT_HEADER = """Create ReportFile rf;
GMAT rf.Filename = '{report}';
GMAT rf.WriteHeaders = false;

"""


class VariantBuilder:
    """只在临时目录写变体；原 script 全程只读。"""

    def __init__(self, model, workdir, cut_at, propagator=None, extra_cols=None,
                 no_burn_cols=False):
        self.m = model
        self.workdir = Path(workdir)
        self.cut_at = cut_at
        self.propagator = propagator or model.propagator
        self.extra_cols = extra_cols or []
        self.no_burn_cols = no_burn_cols

    def build(self, controls, x, epochs, tag, cols=None):
        """controls: [Achieve]; x: 同长度数值; epochs: [float]; 返回 (script_path, report_path, cols)"""
        m = self.m
        lines = list(m.lines)

        # 1) 按行号精确替换 Achieve 数值（保留 Tolerance 等全部其余文本）
        for a, v in zip(controls, x):
            if a.line_no >= self.cut_at:
                print("[WARN] 控制行 L%d 在截断点之后，未替换" % (a.line_no + 1))
                continue
            old = lines[a.line_no]
            mm = re.search(r"(=\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", old)
            if not mm:
                print("[WARN] L%d 未找到数值，跳过" % (a.line_no + 1))
                continue
            lines[a.line_no] = old[:mm.start(2)] + fmt(v) + old[mm.end(2):]

        # 2) 报告列：历元 + 目标字段 + 位置/速度 + burn 分量 + 用户追加
        if cols is None:
            goal_fields = [a.field for a in self._goal_achieves] if hasattr(self, "_goal_achieves") else []
            cols = [m.spacecraft + ".TDBModJulian"] + goal_fields
            cols += m.xyz_cols(goal_fields)
            if not self.no_burn_cols:
                cols += m.burn_cols(self._ctrl_target) if hasattr(self, "_ctrl_target") else []
            for c in self.extra_cols:
                if c not in cols:
                    cols.append(c)
        self.last_cols = cols

        # 3) 注入 ReportFile 到 BeginMissionSequence 之前
        rep = str((self.workdir / ("report_%s.txt" % tag)).as_posix())
        inj = REPORT_HEADER.format(report=rep).rstrip("\n").split("\n")
        head = lines[:m.begin] + [""] + inj + lines[m.begin:self.cut_at]

        # 4) 追加固定历元传播 + 报告
        tail = []
        for e in epochs:
            tail.append("Propagate %s(%s) {%s.TDBModJulian = %.9f};"
                        % (self.propagator, m.spacecraft, m.spacecraft, e))
            tail.append("Report rf %s;" % " ".join(cols))
        text = "\n".join(head + tail) + "\n"

        # 5) 强制纯 ASCII + LF（GMAT 对非 ASCII 注释敏感）
        if any(ord(c) > 127 for c in text):
            text = "".join(c if ord(c) < 128 else "?" for c in text)
        sp = self.workdir / ("variant_%s.script" % tag)
        # 用 open() 而非 Path.write_text(newline=...)：后者需要 Python 3.10+，
        # 老版本 Windows 上会 TypeError。显式 newline='\n' 保证任何平台都是 LF。
        with open(sp, "w", encoding="ascii", errors="replace", newline="\n") as f:
            f.write(text)
        return sp, Path(rep), cols


# ---------------------------------------------------------------- GMAT 驱动
class RunResult:
    def __init__(self, rc, out, rows, log, script, report):
        self.rc, self.out, self.rows = rc, out, rows
        self.log, self.script, self.report = log, script, report

    @property
    def converged(self):
        return len(re.findall(r"The Targeter converged", self.out))

    @property
    def ok(self):
        return self.rc == 0 and bool(self.rows) and "Mission run completed" in self.out

    def errors(self, n=8):
        pat = re.compile(r"ERROR|Exception|failed|disallowed|not an allowed|violat|TIMEOUT", re.I)
        return [l.strip()[:200] for l in self.out.splitlines() if pat.search(l)][:n]


class GmatRunner:
    COMMON_ROOTS = {
        "nt": [r"C:\Program Files\GMAT\R2022a", r"C:\Program Files (x86)\GMAT\R2022a",
               r"C:\GMAT\R2022a", r"D:\GMAT\R2022a", r"C:\Program Files\GMAT",
               str(Path.home() / "GMAT" / "R2022a")],
        "posix": ["/opt/gmat/GMAT/R2022a", "/opt/GMAT/R2022a",
                  "/usr/local/GMAT/R2022a", str(Path.home() / "GMAT" / "R2022a")],
    }

    def __init__(self, console=None, root=None, workdir=None, timeout=900, is_windows=None):
        self.is_windows = (os.name == "nt") if is_windows is None else is_windows
        self.timeout = timeout
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="gmat_outer_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.console = self._find(console, root)

    def _find(self, explicit, root):
        names = (["GmatConsole-R2022a.exe", "GmatConsole.exe"] if self.is_windows
                 else ["GmatConsole-R2022a", "GmatConsole"])
        cands = []
        if explicit:
            cands.append(Path(explicit))
        if os.environ.get("GMAT_CONSOLE"):
            cands.append(Path(os.environ["GMAT_CONSOLE"]))
        roots = []
        if root:
            roots.append(Path(root))
        if os.environ.get("GMAT_ROOT"):
            roots.append(Path(os.environ["GMAT_ROOT"]))
        roots += [Path(p) for p in self.COMMON_ROOTS["nt" if self.is_windows else "posix"]]
        for r in roots:
            for n in names:
                cands.append(r / "bin" / n)
                cands.append(r / n)
        for n in names:
            w = shutil.which(n)
            if w:
                cands.append(Path(w))
        for c in cands:
            try:
                if c.is_file():
                    return c.resolve()
            except OSError:
                pass
        raise FileNotFoundError(
            "找不到 GMAT 控制台。用 --gmat-console 指定完整路径，或 --gmat-root 指定安装根目录，"
            "或设环境变量 GMAT_CONSOLE / GMAT_ROOT。已尝试: %s" % ", ".join(map(str, cands[:8])))

    def env(self):
        e = dict(os.environ)
        bin_ = str(self.console.parent)
        root = str(self.console.parent.parent)
        sep = ";" if self.is_windows else ":"
        if self.is_windows:
            e["PATH"] = bin_ + sep + root + sep + e.get("PATH", "")
        else:
            parts = [bin_, root + "/lib", root + "/plugins"]
            if e.get("LD_LIBRARY_PATH"):
                parts.append(e["LD_LIBRARY_PATH"])
            e["LD_LIBRARY_PATH"] = ":".join(parts)
        e.setdefault("RUN_MODE", "EXIT_AFTER_RUN")
        return e

    def run(self, script, report, tag, timeout=None):
        report = Path(report)
        try:
            report.unlink()
        except OSError:
            pass
        log = self.workdir / ("console_%s.log" % tag)
        cmd = [str(self.console), "-r", str(Path(script).resolve()),
               "-l", str(log.resolve())]
        try:
            p = subprocess.run(cmd, cwd=str(self.console.parent), env=self.env(),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout or self.timeout)
            out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
        except subprocess.TimeoutExpired as ex:
            out, rc = ((ex.stdout or b"").decode("utf-8", "replace")
                       + "\n[TIMEOUT]"), -9
        except OSError as ex:
            out, rc = "[LAUNCH FAILED] %s" % ex, -1
        stdlog = self.workdir / ("stdout_%s.log" % tag)
        try:
            stdlog.write_text(out, encoding="utf-8", errors="replace")
        except OSError:
            pass
        rows = []
        try:
            for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
                t = line.split()
                if not t:
                    continue
                try:
                    nums = [float(v) for v in t]
                except ValueError:
                    continue
                rows.append(nums)
        except OSError:
            pass
        return RunResult(rc, out, rows, log, script, report)


# ---------------------------------------------------------------- 外层打靶
class OuterTargeter:
    def __init__(self, model, builder, runner, controls, goal_fields, goal_values,
                 epoch, tol, max_iter, fd_step, max_step, verbose=True):
        self.m, self.b, self.r = model, builder, runner
        self.controls, self.goal_fields = controls, goal_fields
        self.goal = list(goal_values)
        self.epoch, self.tol, self.max_iter = epoch, tol, max_iter
        self.fd_step, self.max_step = fd_step, max_step
        self.verbose = verbose
        self.n = len(controls)
        self.calls = 0

    def _p(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def probe(self, x, tag=None, epochs=None):
        self.calls += 1
        tag = tag or ("p%03d" % self.calls)
        ep = epochs or [self.epoch]
        sp, rp, cols = self.b.build(self.controls, x, ep, tag)
        res = self.r.run(sp, rp, tag)
        if not res.rows:
            self._p("  [%s] 无报告行 rc=%s %s" % (tag, res.rc, " | ".join(res.errors(2))))
            return None, res, cols
        row = res.rows[-1]
        if len(row) < len(cols):
            self._p("  [%s] 报告列数 %d < 期望 %d" % (tag, len(row), len(cols)))
            return None, res, cols
        y = [row[cols.index(f)] for f in self.goal_fields]
        return y, res, cols

    def finite_diff(self, x, h):
        f0, _, _ = self.probe(x, "fd_c")
        if f0 is None:
            return None, None
        J = [[0.0] * self.n for _ in range(len(f0))]
        for k in range(self.n):
            xp = list(x)
            xp[k] += h
            f1, _, _ = self.probe(xp, "fd_%d" % k)
            if f1 is None:
                return None, None
            for i in range(len(f0)):
                J[i][k] = (f1[i] - f0[i]) / h
        return f0, J

    def solve(self, x0):
        x = list(x0)
        self._p("=== 基线 ===")
        f0, res, cols = self.probe(x, "base")
        if f0 is None:
            self._p("基线运行失败，终止。")
            return x, None, res
        d = [f0[i] - self.goal[i] for i in range(len(self.goal))]
        self._p("  x0=%s" % [fmt(v) for v in x])
        self._p("  y =%s  目标=%s  残差=%s  |d|=%.6g  收敛=%d"
                % ([fmt(v) for v in f0], [fmt(v) for v in self.goal],
                   [fmt(v) for v in d], norm(d), res.converged))
        if norm(d) < self.tol:
            self._p("  基线已达标，无需迭代。")
            return x, f0, res

        self._p("=== 有限差分雅可比 (h=%g) ===" % self.fd_step)
        f0, J = self.finite_diff(x, self.fd_step)
        if J is None:
            self._p("有限差分失败，终止。")
            return x, None, res
        for row in J:
            self._p("  " + "  ".join("%+.6g" % v for v in row))
        Jn = [r[:] for r in J]
        last_res = res

        for it in range(self.max_iter):
            f, r_, _ = self.probe(x, "nw%d" % it)
            if f is None:
                self._p("iter %d 探测失败，步长减半重试" % it)
                return x, None, r_
            d = [f[i] - self.goal[i] for i in range(len(self.goal))]
            self._p("iter %d: x=%s" % (it, [fmt(v) for v in x]))
            self._p("         y=%s  |d|=%.6g  收敛=%d"
                    % ([fmt(v) for v in f], norm(d), r_.converged))
            last_res = r_
            if norm(d) < self.tol:
                self._p("*** 收敛：残差 %.6g < tol %g" % (norm(d), self.tol))
                return x, f, r_
            if it > 0:
                dx = [x[k] - x_prev[k] for k in range(self.n)]
                df = [f[i] - f_prev[i] for i in range(len(f))]
                den = sum(v * v for v in dx)
                if den > 1e-30:
                    Jdx = mat_vec(Jn, dx)
                    for i in range(len(f)):
                        for k in range(self.n):
                            Jn[i][k] += (df[i] - Jdx[i]) * dx[k] / den
            try:
                step = lsq_step(Jn, d)
            except ValueError as ex:
                self._p("  %s，终止" % ex)
                return x, f, r_
            ms = self.max_step or max(1.0, 0.2 * max(abs(v) for v in x0))
            sc = min(1.0, ms / max(abs(v) for v in step)) if step else 1.0
            x_prev, f_prev = list(x), list(f)
            # 步长回溯：残差必须下降
            accepted = False
            for _ in range(4):
                xn = [x[k] + sc * step[k] for k in range(self.n)]
                fn, rn, _ = self.probe(xn, "ls%d" % it)
                if fn is not None:
                    dn = [fn[i] - self.goal[i] for i in range(len(self.goal))]
                    if norm(dn) < norm(d) or sc < 0.05:
                        x, accepted = xn, True
                        break
                sc *= 0.5
            if not accepted:
                self._p("  线搜索失败，停止")
                return x, f, r_
            self._p("        step=%s scale=%g" % (["%+.4g" % v for v in step], sc))
        self._p("达到最大迭代次数 %d，将对最终值做一次独立验证" % self.max_iter)
        return x, None, last_res


# ---------------------------------------------------------------- 验收
def verify_scan(targeter, x, e0, e1, step):
    ep = []
    e = e0
    while e <= e1 + 1e-9:
        ep.append(e)
        e += step
    y, res, cols = targeter.probe(x, "accept", epochs=ep)
    print("\n=== 验收：加密历元扫描 (%.2f ~ %.2f step %g, %d 点) ==="
          % (e0, e1, step, len(ep)))
    print("EXIT=%s  打靶收敛=%d  有效报告行=%d/%d"
          % (res.rc, res.converged, len(res.rows), len(ep)))
    if not res.rows:
        for l in res.errors():
            print("  ", l)
        return None
    # 找 XYZ 三列（取第一组），算 |r|
    xyz = None
    for i, c in enumerate(cols):
        if c.endswith(".X") and i + 2 < len(cols) and cols[i + 1].endswith(".Y") and cols[i + 2].endswith(".Z"):
            xyz = i
            break
    if xyz is None:
        print("  报告列里没有 XYZ，跳过最近距离判据。列：", cols)
        return None
    recs = []
    for row in res.rows:
        try:
            r = math.sqrt(row[xyz] ** 2 + row[xyz + 1] ** 2 + row[xyz + 2] ** 2)
            recs.append((row[0], r, row))
        except (IndexError, ValueError):
            continue
    best = min(recs, key=lambda q: q[1])
    print("  真实传播 |r|min = %.6e km @ %.6f (GMAT TDBModJulian)" % (best[1], best[0]))
    print("  注意：这是采样点上的最小值；真实连续极值可能落在两点之间，" 
          "步长 %g 天时建议用更小的 --scan 步长复核。" % step)
    print("  最近点邻域：")
    for mjd, r, _ in recs:
        if abs(mjd - best[0]) <= max(step * 4, 1.0):
            print("    %12.4f  %12.4e km" % (mjd, r))
    # burn 分量模（若报告里有）
    bidx = [i for i, c in enumerate(cols) if re.search(r"\.Element\d+$", c)]
    if bidx:
        b = [best[2][i] for i in bidx if i < len(best[2])]
        print("  burn 分量 = %s  |dv| = %.3f m/s"
              % ([round(v, 6) for v in b], math.sqrt(sum(v * v for v in b)) * 1000))
    if y is not None:
        print("  说明：扫描区间内各历元的 B 平面是瞬时密切量，随历元剧烈变化，")
        print("        不能当作命中判据；命中判据请用单个评估历元 (--epoch) 的 probe 结果。")
    return best


# ---------------------------------------------------------------- 修改建议
def print_plan(model, controls, x_final, goal, cut_at, cut_lines, epoch, res, y):
    print("\n" + "=" * 78)
    print("修改建议（本工具不会写入原文件；下面只是给你手工粘贴用）")
    print("=" * 78)
    print("原文件 : %s" % model.path)
    print("原 md5 : %s" % model.md5)
    now = hashlib.md5(model.path.read_bytes()).hexdigest()
    print("当前 md5: %s   %s" % (now, "未改动 (OK)" if now == model.md5 else "已被外部改动"))
    print("评估历元: %s (沿用 script 的 TDBModJulian 单位)" % fmt(epoch))
    print()
    for i, (a, v) in enumerate(zip(controls, x_final), 1):
        old = a.raw.strip()
        mm = re.search(r"(=\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", old)
        new = old[:mm.start(2)] + fmt(v) + old[mm.end(2):] if mm else old
        print("[%d] 第 %d 行  %s" % (i, a.line_no + 1, old))
        print("    改为      %s" % new)
        print("    即 %s : %s -> %s" % (a.field, fmt(a.value), fmt(v)))
        print()
    if cut_lines:
        print("[可选] 去掉末段 DSM / 悬空打靶：删除第 %d 行起至第 %d 行（共 %d 行）"
              % (cut_at + 1, cut_lines, cut_lines - cut_at))
        print("       这段在变体里是被截断的；若你只想改打靶值而不动序列，可跳过此步。")
        print("       注意：被删段里的 Periapsis/Apoapsis 停止条件在新轨迹上可能不触发，")
        print("       需要在原序列末尾改用固定历元传播 + Report 才能复现上面的验收结果。")
        print()
    if y is not None:
        print("验证（GMAT 实跑）：")
        print("  末端观测 y = %s" % [fmt(v) for v in y])
        print("  目标     y* = %s" % [fmt(v) for v in goal])
        print("  残差     |d| = %.6g km" % norm([y[i] - goal[i] for i in range(len(goal))]))
        if res is not None:
            print("  EXIT=%s  打靶收敛=%d  Mission run completed=%s"
                  % (res.rc, res.converged, "Mission run completed" in res.out))
    print("=" * 78)


# ---------------------------------------------------------------- 自检
def self_test():
    ok = []
    # 1) 线性代数
    x = solve_linear([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
    assert abs(x[0] - 0.8) < 1e-12 and abs(x[1] - 1.4) < 1e-12
    ok.append("高斯消元 2x2")
    x = lsq_step([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0, 3.0])
    assert len(x) == 2
    ok.append("非方阵最小二乘")
    # 2) 不同命名 / 不同天体的假序列解析（CRLF + BOM）
    fake = (
        "\ufeffCreate Spacecraft Sat;\r\n"
        "Create Propagator helio;\r\n"
        "Create CoordinateSystem Marsxyz;\r\n"
        "Create CoordinateSystem Targetxyz;\r\n"
        "Create ImpulsiveBurn TCM1;\r\n"
        "Create DifferentialCorrector MyDC;\r\n"
        "BeginMissionSequence;\r\n"
        "Target MyDC {SolveMode = Solve, ExitMode = SaveAndContinue};\r\n"
        "   Vary MyDC(TCM1.Element1 = 0.1, {Perturbation = 0.001});\r\n"
        "   Vary MyDC(TCM1.Element2 = 0.2, {Perturbation = 0.001});\r\n"
        "   Maneuver TCM1(Sat);\r\n"
        "   Propagate helio(Sat) {Sat.TDBModJulian = 10000.5};\r\n"
        "   Achieve MyDC(Sat.Marsxyz.BdotR = 111.0, {Tolerance = 0.1});\r\n"
        "   Achieve MyDC(Sat.Marsxyz.BdotT = 222.0, {Tolerance = 0.1});\r\n"
        "EndTarget;\r\n"
        "Propagate helio(Sat) {Sat.ElapsedDays = 2};\r\n"
        "Target MyDC {SolveMode = Solve, ExitMode = SaveAndContinue};\r\n"
        "   Vary MyDC(TCM1.Element3 = 0.3, {Perturbation = 0.001});\r\n"
        "   Achieve MyDC(Sat.Targetxyz.BdotR = -333.0, {Tolerance = 0.1});\r\n"
        "   Achieve MyDC(Sat.Targetxyz.BdotT = 444.0, {Tolerance = 0.1});\r\n"
        "EndTarget;\r\n"
        "Target MyDC {SolveMode = RunInitialGuess};\r\n"
        "   Vary MyDC(TCM1.Element1 = 0.9, {Perturbation = 0.001});\r\n"
        "   Achieve MyDC(Sat.Targetxyz.BdotR = 500.0, {Tolerance = 10});\r\n"
        "   Achieve MyDC(Sat.Targetxyz.BdotT = 600.0, {Tolerance = 10});\r\n"
        "EndTarget;\r\n"
    )
    tmp = Path(tempfile.mkdtemp(prefix="gmat_ot_selftest_"))
    p = tmp / "fake.script"
    p.write_bytes(fake.encode("utf-8"))
    m = ScriptModel(p)
    assert m.spacecraft == "Sat", m.spacecraft
    assert m.propagator == "helio", m.propagator
    assert len(m.targets) == 3, m.targets
    ctrl = m.control_target()
    assert ctrl.start == 16 and len(ctrl.achieves) == 2, ctrl
    assert [a.value for a in ctrl.achieves] == [-333.0, 444.0]
    goal = m.goal_target()
    assert goal is not None and [a.value for a in goal.achieves] == [500.0, 600.0]
    assert m.auto_cut(goal) == goal.start
    assert m.auto_epoch() == 10000.5
    gf = [a.field for a in goal.achieves]
    xyz = m.xyz_cols(gf)
    assert "Sat.Targetxyz.X" in xyz and "Sat.Targetxyz.VZ" in xyz, xyz
    assert m.burn_cols(ctrl) == ["TCM1.Element3"]
    ok.append("假序列解析（非 DefaultSC / 非 solprop / CRLF + BOM）")
    # 3) 变体构造：原文件 md5 不变、纯 ASCII、LF、按行号替换
    before = hashlib.md5(p.read_bytes()).hexdigest()
    wd = tmp / "work"
    wd.mkdir()
    b = VariantBuilder(m, wd, m.auto_cut(goal))
    b._goal_achieves = goal.achieves
    b._ctrl_target = ctrl
    sp, rp, cols = b.build(ctrl.achieves, [-300.0, 400.0], [10000.5, 10001.5], "t")
    assert hashlib.md5(p.read_bytes()).hexdigest() == before
    t = sp.read_bytes()
    assert all(c < 128 for c in t), "变体含非 ASCII"
    assert b"\r\n" not in t
    assert "Sat.Targetxyz.BdotR = -300" in sp.read_text()
    assert "Propagate helio(Sat)" in sp.read_text()
    assert len(cols) >= 3
    ok.append("变体构造（只读原件 / ASCII / LF / 精确替换 / 自动列）")
    # 4) 跨平台环境分派（不实际启动 GMAT）
    dummy = wd / ("GmatConsole-R2022a.exe" if os.name == "nt" else "GmatConsole-R2022a")
    dummy.write_text("")
    for win in (True, False):
        r = GmatRunner(console=dummy, workdir=wd / ("env_%s" % win), is_windows=win)
        e = r.env()
        assert (e["PATH"].count(str(r.console.parent)) >= 1) if win else ("LD_LIBRARY_PATH" in e)
        assert e.get("RUN_MODE") == "EXIT_AFTER_RUN"
    ok.append("Windows / Linux 环境分派")
    shutil.rmtree(tmp, ignore_errors=True)
    print("=== SELF-TEST ===")
    for s in ok:
        print("  PASS  " + s)
    print("全部通过（%d 项，未调用 GMAT）" % len(ok))
    return 0


# ---------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="GMAT 外层打靶通用工具：读 script / 驱动 GMAT / 只 print 修改建议",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", help="输入 GMAT script（只读）")
    ap.add_argument("--gmat-console", help="GmatConsole 完整路径")
    ap.add_argument("--gmat-root", help="GMAT 安装根目录（其下有 bin/）")
    ap.add_argument("--workdir", help="临时变体/日志目录，默认自动建")
    ap.add_argument("--timeout", type=float, default=900, help="单次 GMAT 超时秒数")
    ap.add_argument("--epoch", type=float, help="评估历元（script 里 TDBModJulian 的单位）")
    ap.add_argument("--epoch-from-last", type=float, default=0.0,
                    help="在 script 最大 TDBModJulian 上加这个天数作为评估历元")
    ap.add_argument("--control", action="append", help="控制字段名，可重复；默认自动取最后一根腿")
    ap.add_argument("--goal", action="append", metavar="FIELD=VALUE",
                    help="目标观测量，可重复；默认自动取控制块之后 Target 的 Achieve")
    ap.add_argument("--cut-at", default="auto",
                    help="截断点：auto / none / 1-based 行号 / script 中的一段文字")
    ap.add_argument("--propagator", help="追加传播用的 Propagator 名，默认出现次数最多的")
    ap.add_argument("--obs-col", action="append", help="额外 Report 列，可重复")
    ap.add_argument("--no-burn-cols", action="store_true", help="不把 burn 分量加进 Report")
    ap.add_argument("--tol", type=float, default=100.0, help="收敛残差阈值")
    ap.add_argument("--max-iter", type=int, default=6, help="外层牛顿最大迭代")
    ap.add_argument("--fd-step", type=float, default=50.0, help="有限差分步长")
    ap.add_argument("--max-step", type=float, help="单次变量最大变化量（默认按初值 20%% 自动）")
    ap.add_argument("--set", nargs="+", type=float, help="给定控制值，跳过求解直接验证")
    ap.add_argument("--verify-only", nargs="+", type=float, help="同 --set")
    ap.add_argument("--start", nargs="+", type=float,
                    help="外层牛顿的起点（顺序同 --inspect 的 control），默认取 script 原值；用于热启动续算")
    ap.add_argument("--scan", nargs=3, type=float, metavar=("E0", "E1", "STEP"),
                    help="验收：加密历元扫描范围")
    ap.add_argument("--inspect", action="store_true", help="只解析并打印结构，不调 GMAT")
    ap.add_argument("--self-test", action="store_true", help="纯 Python 自检，不调 GMAT")
    ap.add_argument("--quiet", action="store_true")
    return ap


def resolve_cut(model, goal, spec):
    if spec == "auto":
        return model.auto_cut(goal)
    if spec == "none":
        return len(model.lines)
    if re.fullmatch(r"\d+", spec):
        return min(max(int(spec) - 1, model.begin + 1), len(model.lines))
    for i, l in enumerate(model.lines):
        if spec in l:
            return i
    print("[WARN] --cut-at 未匹配到任何行，改用 auto")
    return model.auto_cut(goal)


def main(argv=None):
    _setup_stdio()
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.script:
        ap.error("需要 --script（或用 --self-test）")

    m = ScriptModel(a.script)
    ctrl_t = m.control_target()
    goal_t = m.goal_target()
    cut_at = resolve_cut(m, goal_t, a.cut_at)

    # 控制量
    if a.control:
        pool = [ac for t in m.targets for ac in t.achieves]
        controls = []
        for f in a.control:
            hit = [ac for ac in pool if ac.field == f and ac.line_no < cut_at]
            if not hit:
                ap.error("--control 字段在截断点之前找不到: %s" % f)
            controls.append(hit[-1])
    else:
        controls = ctrl_t.achieves
    x0 = [ac.value for ac in controls]

    # 目标量
    if a.goal:
        gf, gv = [], []
        for s in a.goal:
            if "=" not in s:
                ap.error("--goal 需写成 FIELD=VALUE，收到 %s" % s)
            f, v = s.split("=", 1)
            gf.append(f.strip())
            gv.append(float(v))
    elif goal_t is not None:
        gf = [ac.field for ac in goal_t.achieves]
        gv = [ac.value for ac in goal_t.achieves]
    else:
        ap.error("script 里没有可作为目标的后续 Target 块，请用 --goal FIELD=VALUE 指定")

    # 历元
    if a.epoch is not None:
        epoch = a.epoch
    else:
        base = m.auto_epoch()
        if base is None:
            ap.error("script 里没有 TDBModJulian，请用 --epoch 指定评估历元")
        epoch = base + a.epoch_from_last
        print("[WARN] 未给 --epoch，暂用 script 最大 TDBModJulian %s + %s = %s"
              % (fmt(base), fmt(a.epoch_from_last), fmt(epoch)))
        print("       末端到达历元通常不在这个位置，务必用 --epoch 给正确值（沿用 script 的单位）。")

    print("=" * 78)
    print("GMAT 外层打靶 / 自动发现结果")
    print("=" * 78)
    print("script       : %s (md5 %s)" % (m.path, m.md5))
    print("Spacecraft   : %s    Propagator: %s %s"
          % (m.spacecraft, m.propagator, m.prop_counts))
    print("Target 块    : %d 个" % len(m.targets))
    for t in m.targets:
        print("   %s  %s" % ("CTRL" if t is ctrl_t else ("GOAL" if t is goal_t else "    "), t))
    print("控制量 x (%d) : %s" % (len(controls), ", ".join("%s=%s" % (c.field, fmt(c.value)) for c in controls)))
    print("目标   y*(%d) : %s" % (len(gf), ", ".join("%s=%s" % (f, fmt(v)) for f, v in zip(gf, gv))))
    print("截断点       : 第 %d 行 (保留 1..%d，丢弃 %d..%d)"
          % (cut_at + 1, cut_at, cut_at + 1, len(m.lines)))
    print("评估历元     : %s" % fmt(epoch))
    pre_cols = ([m.spacecraft + ".TDBModJulian"] + gf + m.xyz_cols(gf)
                + ([] if a.no_burn_cols else m.burn_cols(ctrl_t))
                + [c for c in (a.obs_col or []) if c not in gf])
    print("报告列(推断) : %s" % " ".join(pre_cols))
    if goal_t is None:
        print("[提示] script 里没有后续 Target 块，目标由 --goal 指定；默认不截断，")
        print("       会保留原序列全部传播。若末尾已有目标段传播，请用 --cut-at 指定截断点。")

    if a.inspect:
        print("\n(--inspect 模式，未调用 GMAT)")
        return 0

    wd = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="gmat_outer_"))
    wd.mkdir(parents=True, exist_ok=True)
    b = VariantBuilder(m, wd, cut_at, propagator=a.propagator,
                       extra_cols=a.obs_col or [], no_burn_cols=a.no_burn_cols)
    b._goal_achieves = [type("A", (), {"field": f}) for f in gf]
    b._ctrl_target = ctrl_t

    runner = GmatRunner(console=a.gmat_console, root=a.gmat_root,
                        workdir=wd, timeout=a.timeout)
    print("GMAT 控制台  : %s" % runner.console)
    print("临时目录     : %s" % wd)
    print("=" * 78)

    ot = OuterTargeter(m, b, runner, controls, gf, gv, epoch,
                       a.tol, a.max_iter, a.fd_step, a.max_step, verbose=not a.quiet)

    given = a.verify_only or a.set
    if given:
        # 单次验证
        print("\n=== 单次验证（--verify-only）===")
        y, res, cols = ot.probe(given, "verify")
        if y is None:
            print("运行失败：")
            for l in res.errors():
                print("  ", l)
            return 2
        d = [y[i] - gv[i] for i in range(len(gv))]
        print("x     = %s" % [fmt(v) for v in given])
        print("y     = %s" % [fmt(v) for v in y])
        print("y*    = %s" % [fmt(v) for v in gv])
        print("残差  = %s  |d| = %.6g km" % ([fmt(v) for v in d], norm(d)))
        print("EXIT=%s  打靶收敛=%d  Mission run completed=%s"
              % (res.rc, res.converged, "Mission run completed" in res.out))
        if a.scan:
            verify_scan(ot, given, *a.scan)
        print_plan(m, controls, given, gv, cut_at, len(m.lines), epoch, res, y)
        return 0

    # 完整求解
    t0 = time.time()
    x_start = list(a.start) if a.start else x0
    if a.start:
        if len(x_start) != len(controls):
            ap.error("--start 需要 %d 个值，收到 %d" % (len(controls), len(x_start)))
        print("\n热启动：x_start = %s" % [fmt(v) for v in x_start])
    x, y, res = ot.solve(x_start)
    print("\n求解完成：%d 次 GMAT 调用，用时 %.1f s" % (ot.calls, time.time() - t0))
    print("FINAL x = %s" % [fmt(v) for v in x])
    if y is None:
        # 用最终值再验一次，拿到 y 与 res
        y, res, cols = ot.probe(x, "final")
    if a.scan:
        verify_scan(ot, x, *a.scan)
    print_plan(m, controls, x, gv, cut_at, len(m.lines), epoch, res, y)
    return 0 if y is not None and norm([y[i] - gv[i] for i in range(len(gv))]) < a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
