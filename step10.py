
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

跨平台 / 后端
-------------
两种驱动方式，--backend 选择（默认 auto，先找控制台，找不到自动回退）：
  console : GmatConsole -r <script> -l <log>。探测顺序 --gmat-console >
           环境变量 GMAT_CONSOLE > --gmat-root / GMAT_ROOT > PATH > 常见安装路径。
  gmatpy  : Windows 发行版不带 GmatConsole 时走这条。用 GMAT 自带 bin/gmatpy
           （R2020a 起的 Python API）：Setup -> LoadScript -> RunScript，
           每个变体起一个干净子进程，等价于每次新开一个控制台。
           --gmat-root 指到安装根目录；bin/gmatpy/_pyXY 需与解释器版本对应
           （R2022a 官方测试 Python 3.9）。
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

  # Windows 没有 GmatConsole：走自带 Python API（默认 auto 也会自动回退）
  python gmat_outer_target.py --script "D:\\gmat\\step6v2.script" `
      --gmat-root "E:\\GMAT" --backend gmatpy --epoch 34731.721007756 `
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
    'Gaussian elimination with partial pivoting. A is n x n, b has length n.'
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError('Singular matrix; cannot compute the Newton step')
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
    'Solve J dx ~ -d: direct solve for square J, damped least squares (J^T J + lam I) otherwise.'
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
    'GMAT-readable lossless number format (15 significant digits, enough for MJD-scale epochs).'
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
            raise ValueError('BeginMissionSequence; not found in script')

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
            msg = ['No Target block with SolveMode = Solve and an Achieve found.',
                   '  %d lines, BeginMissionSequence at L%d, %d Target block(s) parsed:'
                   % (len(self.lines), self.begin + 1, len(self.targets))]
            for t in self.targets:
                msg.append("    L%d-L%d  SolveMode=%s  Achieve=%d  Vary=%d"
                           % (t.start + 1, t.end + 1, t.mode,
                              len(t.achieves), len(t.varies)))
            kw = [sum(1 for l in self.lines if re.search(p, l, re.I))
                  for p in (r"\bTarget\b", r"\bAchieve\b", r"SolveMode", r"EndTarget")]
            msg.append('  Keyword counts: Target=%d  Achieve=%d  SolveMode=%d  EndTarget=%d'
                       % tuple(kw))
            if not self.targets:
                msg.append('  No Target block parsed: this script may have no '
                           'DifferentialCorrector (e.g. only object definitions / plain propagate / back-propagation).')
                msg.append('  step10 is an outer-loop targeting tool; it requires a '
                           'script with Target{...Achieve...}. Use a different --script.')
            elif kw[1] > 0 and all(not t.achieves for t in self.targets):
                msg.append('  Achieve lines exist but were not matched inside a block: check '
                           'that Achieve DC(field = number, {...}); uses a plain numeric value with complete brackets, commas and semicolon.')
            else:
                msg.append('  Target blocks exist, but none has both SolveMode = Solve '
                           'and an Achieve inside.')
            raise ValueError("\n".join(msg))
        return s[-1]                       # 最后一根腿

    def goal_target(self):
        ctrl = self.control_target()
        later = [t for t in self.targets if t.start > ctrl.end and t.achieves]
        return later[-1] if later else None

    def auto_cut(self, goal):
        'Cut point (0-based line index): start line of the goal block; end of file if none.'
        return goal.start if goal is not None else len(self.lines)

    def auto_epoch(self):
        return max(self.epochs) if self.epochs else None

    def obs_prefix(self, field):
        'Infer the coordinate-system prefix from an Achieve field, e.g. DefaultSC.Juxyz.BdotR -> DefaultSC.Juxyz.'
        parts = field.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:-1]), parts[-2]
        return None, None

    def xyz_cols(self, fields):
        'Add X/Y/Z and VX/VY/VZ columns for goal fields (only if the CS is declared in the script).'
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
    'Write variants only to the temp dir; the original script is always read-only.'

    def __init__(self, model, workdir, cut_at, propagator=None, extra_cols=None,
                 no_burn_cols=False):
        self.m = model
        self.workdir = Path(workdir)
        self.cut_at = cut_at
        self.propagator = propagator or model.propagator
        self.extra_cols = extra_cols or []
        self.no_burn_cols = no_burn_cols

    def build(self, controls, x, epochs, tag, cols=None):
        'controls: [Achieve]; x: same-length numbers; epochs: [float]; returns (script_path, report_path, cols)'
        m = self.m
        lines = list(m.lines)

        # 1) 按行号精确替换 Achieve 数值（保留 Tolerance 等全部其余文本）
        for a, v in zip(controls, x):
            if a.line_no >= self.cut_at:
                print('[WARN] control line L%d is after the cut point, not replaced' % (a.line_no + 1))
                continue
            old = lines[a.line_no]
            mm = re.search(r"(=\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", old)
            if not mm:
                print('[WARN] L%d: no numeric value found, skipped' % (a.line_no + 1))
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
        # 控制台把 "Mission run completed" 打到 stdout；gmatpy 后端靠
        # UseLogFile 把同一份 GMAT 日志并回 out，口径一致。
        # GMATPY_RUNSCRIPT_OK 是 RunScript() 真实返回值的兜底（日志没写成时）。
        completed = ("Mission run completed" in self.out
                     or "GMATPY_RUNSCRIPT_OK" in self.out)
        return self.rc == 0 and bool(self.rows) and completed

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
        self.label = 'console %s' % self.console

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
            'GmatConsole not found. Use --gmat-console for the full path, or --gmat-root for the install root, '
            'or set GMAT_CONSOLE / GMAT_ROOT. Tried: %s' % ", ".join(map(str, cands[:8])))

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


# ---------------------------------------------------------------- gmatpy 后端
# Windows 的 GMAT 发行版不带 GmatConsole，但自带 bin/gmatpy（R2020a 起的 Python
# API）。线索来自 GMAT 自带 api/load_gmat.py 与 api/API_README.txt：
#   Setup(api_startup_file.txt) -> LoadScript(script) -> RunScript()
# 每个变体起一个干净的子进程，等价于「每次新开一个控制台」，避免同一进程反复
# LoadScript 造成对象累积。Report 仍由 script 内的 ReportFile 写出，读法不变。
GMATPY_BOOTSTRAP = r'''
import os, re, sys
bin_path = os.path.abspath(sys.argv[1])
startup  = os.path.abspath(sys.argv[2])
script   = os.path.abspath(sys.argv[3])
log_path = os.path.abspath(sys.argv[4])
gmat_dir = os.path.dirname(bin_path)
if os.name == "nt":
    for p in (bin_path, os.path.join(gmat_dir, "plugins")):
        try:
            os.add_dll_directory(p)
        except (AttributeError, OSError):
            pass
    os.environ["PATH"] = (bin_path + os.pathsep + os.path.join(gmat_dir, "plugins")
                          + os.pathsep + os.environ.get("PATH", ""))
sys.path.insert(0, bin_path)
try:
    import gmatpy as gmat
except ImportError:
    # gmatpy 按 sys.version_info 找 _pyXY 目录；解释器比发行版新时没有对应目录。
    # Linux 上 .so 常能跨小版本加载，这里退到最新的一个（Windows 的 .pyd
    # 绑定 python3X.dll，通常仍会失败，报错会原样打出来）。
    pkg = os.path.join(bin_path, "gmatpy")
    vers = sorted([d for d in os.listdir(pkg) if re.fullmatch(r"_py\d+", d)],
                  key=lambda d: int(d[3:]), reverse=True)
    if not vers:
        raise
    sys.path.insert(0, os.path.join(pkg, vers[0]))
    import gmat_py as gmat
    gmat.FileManager.Instance().SetBinDirectory(
        "gmat_startup_file.txt", bin_path + os.sep)
gmat.Setup(startup)
try:
    # API 不把运行消息打到 stdout，让 GMAT 自己写日志；调用方把它并回 out，
    # 这样 "Mission run completed" / "The Targeter converged" 与控制台口径一致。
    gmat.UseLogFile(log_path)
except Exception:
    pass
gmat.LoadScript(script)
ok = gmat.RunScript()
print("GMATPY_RUNSCRIPT_OK" if ok else "GMATPY_RUNSCRIPT_FAILED")
'''


class GmatPyRunner:
    'Same interface as GmatRunner, but drives GMAT via its bundled Python API; no GmatConsole needed.'

    def __init__(self, root=None, workdir=None, timeout=900, is_windows=None):
        self.is_windows = (os.name == "nt") if is_windows is None else is_windows
        self.timeout = timeout
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="gmat_outer_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.bin = self._find_bin(root)
        self.startup = self._make_startup()
        self.label = "gmatpy (Python API, %s)" % self.bin

    def _find_bin(self, root):
        cands = []
        if root:
            cands.append(Path(root))
        if os.environ.get("GMAT_ROOT"):
            cands.append(Path(os.environ["GMAT_ROOT"]))
        cands += [Path(p) for p in
                  GmatRunner.COMMON_ROOTS["nt" if self.is_windows else "posix"]]
        if self.is_windows:
            for d in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                cands.append(Path(d + ":\\GMAT"))
                cands.append(Path(d + ":\\GMAT\\R2022a"))
        for c in cands:
            for b in (c / "bin", c):
                try:
                    if ((b / "gmatpy" / "__init__.py").is_file()
                            and (b / "gmat_startup_file.txt").is_file()):
                        return b.resolve()
                except OSError:
                    pass
        raise FileNotFoundError(
            'bin/gmatpy not found (Windows builds ship no GmatConsole; use the Python API backend).'
            'Use --gmat-root for the install root (containing bin/gmatpy), or set GMAT_ROOT.')

    def _make_startup(self):
        """照官方 api/BuildApiStartupFile.py：把 gmat_startup_file.txt 里的 ..
        换成安装根目录的绝对路径。写到工作目录，不碰安装目录（免管理员权限）。"""
        src = self.bin / "gmat_startup_file.txt"
        text = src.read_text(encoding="utf-8", errors="replace")
        text = text.replace("..", str(self.bin.parent))
        out = self.workdir / "api_startup_file.txt"
        out.write_text(text, encoding="utf-8", errors="replace")
        return out.resolve()

    def env(self):
        e = dict(os.environ)
        bin_ = str(self.bin)
        root = str(self.bin.parent)
        if self.is_windows:
            e["PATH"] = bin_ + ";" + root + "\\plugins" + ";" + e.get("PATH", "")
        else:
            parts = [bin_, root + "/lib", root + "/plugins"]
            if e.get("LD_LIBRARY_PATH"):
                parts.append(e["LD_LIBRARY_PATH"])
            e["LD_LIBRARY_PATH"] = ":".join(parts)
        return e

    def run(self, script, report, tag, timeout=None):
        report = Path(report)
        try:
            report.unlink()
        except OSError:
            pass
        log = self.workdir / ("gmatpy_%s.log" % tag)
        cmd = [sys.executable, "-c", GMATPY_BOOTSTRAP,
               str(self.bin), str(self.startup), str(Path(script).resolve()),
               str(log.resolve())]
        try:
            p = subprocess.run(cmd, cwd=str(self.bin), env=self.env(),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout or self.timeout)
            out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
        except subprocess.TimeoutExpired as ex:
            o = ex.stdout or ""
            if isinstance(o, bytes):
                o = o.decode("utf-8", "replace")
            out, rc = o + "\n[TIMEOUT]", -9
        except OSError as ex:
            out, rc = "[LAUNCH FAILED] %s" % ex, -1
        try:
            msg = log.read_text(encoding="utf-8", errors="replace")
            if msg:
                out += "\n[GMAT LOG]\n" + msg
        except OSError:
            pass
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
            self._p('  [%s] no report rows rc=%s %s' % (tag, res.rc, " | ".join(res.errors(2))))
            return None, res, cols
        row = res.rows[-1]
        if len(row) < len(cols):
            self._p('  [%s] report has %d cols < expected %d' % (tag, len(row), len(cols)))
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
        self._p('=== Baseline ===')
        f0, res, cols = self.probe(x, "base")
        if f0 is None:
            self._p('Baseline run failed; aborting.')
            return x, None, res
        d = [f0[i] - self.goal[i] for i in range(len(self.goal))]
        self._p("  x0=%s" % [fmt(v) for v in x])
        self._p('  y =%s  goal=%s  residual=%s  |d|=%.6g  converged=%d'
                % ([fmt(v) for v in f0], [fmt(v) for v in self.goal],
                   [fmt(v) for v in d], norm(d), res.converged))
        if norm(d) < self.tol:
            self._p('  Baseline already within tolerance; no iteration needed.')
            return x, f0, res

        self._p('=== Finite-difference Jacobian (h=%g) ===' % self.fd_step)
        f0, J = self.finite_diff(x, self.fd_step)
        if J is None:
            self._p('Finite difference failed; aborting.')
            return x, None, res
        for row in J:
            self._p("  " + "  ".join("%+.6g" % v for v in row))
        Jn = [r[:] for r in J]
        last_res = res

        for it in range(self.max_iter):
            f, r_, _ = self.probe(x, "nw%d" % it)
            if f is None:
                self._p('iter %d probe failed; halving step and retrying' % it)
                return x, None, r_
            d = [f[i] - self.goal[i] for i in range(len(self.goal))]
            self._p("iter %d: x=%s" % (it, [fmt(v) for v in x]))
            self._p('         y=%s  |d|=%.6g  converged=%d'
                    % ([fmt(v) for v in f], norm(d), r_.converged))
            last_res = r_
            if norm(d) < self.tol:
                self._p('*** Converged: residual %.6g < tol %g' % (norm(d), self.tol))
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
                self._p('  %s; stopping' % ex)
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
                self._p('  Line search failed; stopping')
                return x, f, r_
            self._p("        step=%s scale=%g" % (["%+.4g" % v for v in step], sc))
        self._p('Max iterations %d reached; running one independent verification of the final value' % self.max_iter)
        return x, None, last_res


# ---------------------------------------------------------------- 验收
def verify_scan(targeter, x, e0, e1, step):
    ep = []
    e = e0
    while e <= e1 + 1e-9:
        ep.append(e)
        e += step
    y, res, cols = targeter.probe(x, "accept", epochs=ep)
    print('\n=== Verification: fine epoch scan (%.2f ~ %.2f step %g, %d points) ==='
          % (e0, e1, step, len(ep)))
    print('EXIT=%s  targeter converged=%d  valid report rows=%d/%d'
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
        print('  No XYZ columns in report; skipping closest-approach check. Columns:', cols)
        return None
    recs = []
    for row in res.rows:
        try:
            r = math.sqrt(row[xyz] ** 2 + row[xyz + 1] ** 2 + row[xyz + 2] ** 2)
            recs.append((row[0], r, row))
        except (IndexError, ValueError):
            continue
    best = min(recs, key=lambda q: q[1])
    print('  Propagated |r|min = %.6e km @ %.6f (GMAT TDBModJulian)' % (best[1], best[0]))
    print('  Note: this is the minimum over sampled points; the true continuous extremum may fall between two samples, ' 
          'so with a %g-day step, re-check using a smaller --scan step.' % step)
    print('  Neighborhood of closest point:')
    for mjd, r, _ in recs:
        if abs(mjd - best[0]) <= max(step * 4, 1.0):
            print("    %12.4f  %12.4e km" % (mjd, r))
    # burn 分量模（若报告里有）
    bidx = [i for i, c in enumerate(cols) if re.search(r"\.Element\d+$", c)]
    if bidx:
        b = [best[2][i] for i in bidx if i < len(best[2])]
        print('  burn components = %s  |dv| = %.3f m/s'
              % ([round(v, 6) for v in b], math.sqrt(sum(v * v for v in b)) * 1000))
    if y is not None:
        print('  Note: B-plane values across the scan are instantaneous osculating quantities and vary strongly with epoch; ')
        print('        they are not a hit criterion. Use the probe result at the single evaluation epoch (--epoch).')
    return best


# ---------------------------------------------------------------- 修改建议
def print_plan(model, controls, x_final, goal, cut_at, cut_lines, epoch, res, y):
    print("\n" + "=" * 78)
    print('Suggested edits (this tool never writes the original file; copy/paste these by hand)')
    print("=" * 78)
    print('Original  : %s' % model.path)
    print('Original md5 : %s' % model.md5)
    now = hashlib.md5(model.path.read_bytes()).hexdigest()
    print('Current md5  : %s   %s' % (now, 'unchanged (OK)' if now == model.md5 else 'CHANGED EXTERNALLY'))
    print('Eval epoch : %s (TDBModJulian units, as in the script)' % fmt(epoch))
    print()
    for i, (a, v) in enumerate(zip(controls, x_final), 1):
        old = a.raw.strip()
        mm = re.search(r"(=\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", old)
        new = old[:mm.start(2)] + fmt(v) + old[mm.end(2):] if mm else old
        print('[%d] line %d  %s' % (i, a.line_no + 1, old))
        print('    change to %s' % new)
        print('    i.e. %s : %s -> %s' % (a.field, fmt(a.value), fmt(v)))
        print()
    if cut_lines:
        print('[optional] Drop the trailing DSM / dangling targeter: delete lines %d..%d (%d lines)'
              % (cut_at + 1, cut_lines, cut_lines - cut_at))
        print('       This tail was truncated in the variant; skip this step if you only want to change the target values.')
        print('       Note: Periapsis/Apoapsis stop conditions in the deleted tail may never trigger on the new trajectory;')
        print('       replace them with a fixed-epoch Propagate + Report to reproduce the scan results above.')
        print()
    if y is not None:
        print('Verification (actual GMAT run):')
        print('  terminal y  = %s' % [fmt(v) for v in y])
        print('  target  y* = %s' % [fmt(v) for v in goal])
        print('  residual |d| = %.6g km' % norm([y[i] - goal[i] for i in range(len(goal))]))
        if res is not None:
            print('  EXIT=%s  targeter converged=%d  run OK=%s'
                  % (res.rc, res.converged, res.ok))
    print("=" * 78)


# ---------------------------------------------------------------- 自检
def self_test():
    ok = []
    # 1) 线性代数
    x = solve_linear([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
    assert abs(x[0] - 0.8) < 1e-12 and abs(x[1] - 1.4) < 1e-12
    ok.append('Gaussian elimination 2x2')
    x = lsq_step([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0, 3.0])
    assert len(x) == 2
    ok.append('Non-square least squares')
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
    ok.append('Mock sequence parsing (non-DefaultSC / non-solprop / CRLF + BOM)')
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
    assert all(c < 128 for c in t), 'variant must be ASCII-only'
    assert b"\r\n" not in t
    assert "Sat.Targetxyz.BdotR = -300" in sp.read_text()
    assert "Propagate helio(Sat)" in sp.read_text()
    assert len(cols) >= 3
    ok.append('Variant build (read-only original / ASCII / LF / exact replace / auto cols)')
    # 4) 跨平台环境分派（不实际启动 GMAT）
    dummy = wd / ("GmatConsole-R2022a.exe" if os.name == "nt" else "GmatConsole-R2022a")
    dummy.write_text("")
    for win in (True, False):
        r = GmatRunner(console=dummy, workdir=wd / ("env_%s" % win), is_windows=win)
        e = r.env()
        assert (e["PATH"].count(str(r.console.parent)) >= 1) if win else ("LD_LIBRARY_PATH" in e)
        assert e.get("RUN_MODE") == "EXIT_AFTER_RUN"
    ok.append('Windows / Linux environment dispatch')
    shutil.rmtree(tmp, ignore_errors=True)
    print("=== SELF-TEST ===")
    for s in ok:
        print("  PASS  " + s)
    print('All %d checks passed (GMAT not called)' % len(ok))
    return 0


# ---------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description='GMAT outer-loop targeting tool: read script / drive GMAT / print suggested edits only',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", help='Input GMAT script (read-only)')
    ap.add_argument("--gmat-console", help='Full path to GmatConsole')
    ap.add_argument("--gmat-root", help='GMAT install root (contains bin/)')
    ap.add_argument("--backend", choices=("auto", "console", "gmatpy"), default="auto",
                    help='auto=console first, fall back to gmatpy; console=GmatConsole only; '
                         'gmatpy=bundled Python API only (for Windows without a console)')
    ap.add_argument("--workdir", help='Work dir for variants/logs; auto-created by default')
    ap.add_argument("--timeout", type=float, default=900, help='Timeout in seconds for one GMAT run')
    ap.add_argument("--epoch", type=float, help='Evaluation epoch (TDBModJulian units as used in the script)')
    ap.add_argument("--epoch-from-last", type=float, default=0.0,
                    help='Add this many days to the script max TDBModJulian to get the evaluation epoch')
    ap.add_argument("--control", action="append", help='Control field name, repeatable; default = last leg (auto)')
    ap.add_argument("--goal", action="append", metavar="FIELD=VALUE",
                    help='Goal observable, repeatable; default = Achieve of the Target block after the control block')
    ap.add_argument("--cut-at", default="auto",
                    help='Cut point: auto / none / 1-based line number / a text fragment from the script')
    ap.add_argument("--propagator", help='Propagator name for the appended propagate; default = most frequent one')
    ap.add_argument("--obs-col", action="append", help='Extra Report column, repeatable')
    ap.add_argument("--no-burn-cols", action="store_true", help='Do not add burn components to the Report')
    ap.add_argument("--tol", type=float, default=100.0, help='Residual tolerance for convergence')
    ap.add_argument("--max-iter", type=int, default=6, help='Max outer Newton iterations')
    ap.add_argument("--fd-step", type=float, default=50.0, help='Finite-difference step')
    ap.add_argument("--max-step", type=float, help='Max per-variable step (default: auto, 20%% of initial value)')
    ap.add_argument("--set", nargs="+", type=float, help='Given control values; skip solving and verify once')
    ap.add_argument("--verify-only", nargs="+", type=float, help='Alias of --set')
    ap.add_argument("--start", nargs="+", type=float,
                    help='Outer Newton start point (same order as --inspect controls); defaults to script values; for warm-start continuation')
    ap.add_argument("--scan", nargs=3, type=float, metavar=("E0", "E1", "STEP"),
                    help='Verification: fine epoch scan range')
    ap.add_argument("--inspect", action="store_true", help='Parse and print structure only; do not call GMAT')
    ap.add_argument("--self-test", action="store_true", help='Pure-Python self test; do not call GMAT')
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
    print('[WARN] --cut-at matched no line, falling back to auto')
    return model.auto_cut(goal)


def main(argv=None):
    _setup_stdio()
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.script:
        ap.error('--script is required (or use --self-test)')

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
                ap.error('--control field not found before the cut point: %s' % f)
            controls.append(hit[-1])
    else:
        controls = ctrl_t.achieves
    x0 = [ac.value for ac in controls]

    # 目标量
    if a.goal:
        gf, gv = [], []
        for s in a.goal:
            if "=" not in s:
                ap.error('--goal must be FIELD=VALUE, got %s' % s)
            f, v = s.split("=", 1)
            gf.append(f.strip())
            gv.append(float(v))
    elif goal_t is not None:
        gf = [ac.field for ac in goal_t.achieves]
        gv = [ac.value for ac in goal_t.achieves]
    else:
        ap.error('No later Target block usable as goal in script; specify it with --goal FIELD=VALUE')

    # 历元
    if a.epoch is not None:
        epoch = a.epoch
    else:
        base = m.auto_epoch()
        if base is None:
            ap.error('No TDBModJulian in script; specify the evaluation epoch with --epoch')
        epoch = base + a.epoch_from_last
        print('[WARN] --epoch not given; using max TDBModJulian %s + %s = %s'
              % (fmt(base), fmt(a.epoch_from_last), fmt(epoch)))
        print('       The terminal arrival epoch is usually elsewhere; pass the correct value with --epoch (same units as the script).')

    print("=" * 78)
    print('GMAT outer-loop targeting / auto-discovery')
    print("=" * 78)
    print("script       : %s (md5 %s)" % (m.path, m.md5))
    print("Spacecraft   : %s    Propagator: %s %s"
          % (m.spacecraft, m.propagator, m.prop_counts))
    print('Target blocks : %d' % len(m.targets))
    for t in m.targets:
        print("   %s  %s" % ("CTRL" if t is ctrl_t else ("GOAL" if t is goal_t else "    "), t))
    print('Controls x (%d) : %s' % (len(controls), ", ".join("%s=%s" % (c.field, fmt(c.value)) for c in controls)))
    print('Goals   y*(%d) : %s' % (len(gf), ", ".join("%s=%s" % (f, fmt(v)) for f, v in zip(gf, gv))))
    print('Cut point    : line %d (keep 1..%d, drop %d..%d)'
          % (cut_at + 1, cut_at, cut_at + 1, len(m.lines)))
    print('Eval epoch   : %s' % fmt(epoch))
    pre_cols = ([m.spacecraft + ".TDBModJulian"] + gf + m.xyz_cols(gf)
                + ([] if a.no_burn_cols else m.burn_cols(ctrl_t))
                + [c for c in (a.obs_col or []) if c not in gf])
    print('Report cols  : %s' % " ".join(pre_cols))
    if goal_t is None:
        print('[NOTE] No later Target block; goals come from --goal and no truncation is applied by default, ')
        print('       so the full original sequence is kept. If the tail already propagates to the goal, set --cut-at explicitly.')

    if a.inspect:
        print('\n(--inspect mode; GMAT was not called)')
        return 0

    wd = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="gmat_outer_"))
    wd.mkdir(parents=True, exist_ok=True)
    b = VariantBuilder(m, wd, cut_at, propagator=a.propagator,
                       extra_cols=a.obs_col or [], no_burn_cols=a.no_burn_cols)
    b._goal_achieves = [type("A", (), {"field": f}) for f in gf]
    b._ctrl_target = ctrl_t

    if a.backend == "gmatpy":
        runner = GmatPyRunner(root=a.gmat_root, workdir=wd, timeout=a.timeout)
    elif a.backend == "console":
        runner = GmatRunner(console=a.gmat_console, root=a.gmat_root,
                            workdir=wd, timeout=a.timeout)
    else:
        try:
            runner = GmatRunner(console=a.gmat_console, root=a.gmat_root,
                                workdir=wd, timeout=a.timeout)
        except FileNotFoundError:
            print('[INFO] GmatConsole not found; falling back to gmatpy (Python API) backend.')
            runner = GmatPyRunner(root=a.gmat_root, workdir=wd, timeout=a.timeout)
    print('GMAT backend : %s' % runner.label)
    print('Work dir     : %s' % wd)
    print("=" * 78)

    ot = OuterTargeter(m, b, runner, controls, gf, gv, epoch,
                       a.tol, a.max_iter, a.fd_step, a.max_step, verbose=not a.quiet)

    given = a.verify_only or a.set
    if given:
        # 单次验证
        print('\n=== Single verification (--verify-only) ===')
        y, res, cols = ot.probe(given, "verify")
        if y is None:
            print('Run failed:')
            for l in res.errors():
                print("  ", l)
            return 2
        d = [y[i] - gv[i] for i in range(len(gv))]
        print("x     = %s" % [fmt(v) for v in given])
        print("y     = %s" % [fmt(v) for v in y])
        print("y*    = %s" % [fmt(v) for v in gv])
        print('residual = %s  |d| = %.6g km' % ([fmt(v) for v in d], norm(d)))
        print('EXIT=%s  targeter converged=%d  run OK=%s'
              % (res.rc, res.converged, res.ok))
        if a.scan:
            verify_scan(ot, given, *a.scan)
        print_plan(m, controls, given, gv, cut_at, len(m.lines), epoch, res, y)
        return 0

    # 完整求解
    t0 = time.time()
    x_start = list(a.start) if a.start else x0
    if a.start:
        if len(x_start) != len(controls):
            ap.error('--start needs %d values, got %d' % (len(controls), len(x_start)))
        print('\nWarm start: x_start = %s' % [fmt(v) for v in x_start])
    x, y, res = ot.solve(x_start)
    print('\nSolve done: %d GMAT calls, %.1f s' % (ot.calls, time.time() - t0))
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
