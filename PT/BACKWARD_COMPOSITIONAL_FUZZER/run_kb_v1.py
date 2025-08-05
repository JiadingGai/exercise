#!/usr/bin/env python3
"""
run_level.py – compile‑and‑run every KernelBench task in one level
• 35‑minute per‑task timeout
• continues past failures (unless early‑stop flags reached)
• coloured summary table
• early stop via --max-tasks / --max-fails
"""

from pathlib import Path
import argparse, importlib.util, uuid, os, gc, sys, datetime as dt, traceback
import signal, re, textwrap, subprocess
from textwrap import shorten

GLYPH = dict(PASS="\033[92m✓\033[0m", FAIL="\033[91m✗\033[0m", TIME="⚠")
SEP = "─" * 78
TIMEOUT = 35 * 60  # seconds

p = argparse.ArgumentParser()
p.add_argument("--level",     default="level2", help="level1 / level2 / level3")
p.add_argument("--kb-root",   default="KernelBench/KernelBench")
p.add_argument("--max-tasks", type=int, default=None)
p.add_argument("--max-fails", type=int, default=None)
args = p.parse_args()

root = Path.cwd()
level_dir = root / args.kb_root / args.level
if not level_dir.is_dir():
    sys.exit(f"❌ cannot find {level_dir}")

out_root = root / "outputs"; out_root.mkdir(exist_ok=True)

os.environ.update({
    "PJRT_DEVICE": "NEURON",
    "MASTER_ADDR": "localhost",
    "TORCH_XLA_DEBUG": "1",
    "XLA_FLAGS": f"--xla_dump_to={out_root}/xla-dump --xla_dump_hlo_as_text "
                 f"--xla_dump_hlo_pass_re=.*",
    "HF_HOME": str(out_root / "hf-home"),
})

import torch, torch.nn as nn
import torch_xla.core.xla_model as xm
device = xm.xla_device()

# kill residual Neuron compiler processes
def kill_neuron_processes():
    for proc in ["neuron-cc", "neuron_parallel_compile", "tensorizer", "neuron_compile_worker"]:
        subprocess.run(f"pkill -f {proc}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_BANNER = "An Internal Compiler Error has occurred"
_BRACKET = re.compile(r"\[([A-Z]{1,4}\d{3})\]\s*(.*)")
_STAR_ROW = re.compile(r"\*{5,}")
_FALLBACK_MAP = {
    r"SIM09\d": "BIRSIM mismatch",
    r"Internal Compiler Error": "ICE · internal compiler error",
    r"Assertion failure": "Assertion failure",
}
_FALLBACK_PAT = re.compile("|".join(_FALLBACK_MAP.keys()), re.I)

def _short(msg: str, width: int = 210) -> str:
    return textwrap.shorten(msg.strip(), width=width, placeholder="…")

def _skip(line: str) -> bool:
    return (not line.strip()
            or _STAR_ROW.fullmatch(line.rsplit(":", 1)[-1].strip()))

def extract_neuron_signal(log_file: Path, tail_file: Path) -> str | None:
    if not log_file.exists(): return None
    tail = log_file.read_bytes()[-16384:].decode(errors="replace")
    tail_file.write_text(tail)
    lines = tail.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _BANNER in ln) + 1
    except StopIteration:
        start = 0
    first_info = None
    for ln in lines[start:]:
        if _skip(ln): continue
        m = _BRACKET.search(ln)
        if m: return _short(f"{m.group(1)} · {m.group(2)}")
        if first_info is None: first_info = _short(ln.strip())
    for ln in reversed(lines):
        m = _BRACKET.search(ln)
        if m: return _short(f"{m.group(1)} · {m.group(2)}")
    for ln in lines:
        if _FALLBACK_PAT.search(ln):
            key = next(k for k in _FALLBACK_MAP if re.search(k, ln, re.I))
            return _short(_FALLBACK_MAP[key])
    if first_info: return first_info
    for ln in reversed(lines):
        if ln.strip(): return _short(ln)
    return None

def _raise_timeout(signum, frame): raise TimeoutError(f"exceeded {TIMEOUT//60}-min compile window")
signal.signal(signal.SIGALRM, _raise_timeout)

records = []
passed = failed = timed = 0

for idx, py_file in enumerate(sorted(level_dir.glob("*.py")), 1):
    if args.max_tasks and idx > args.max_tasks: break
    if args.max_fails and (failed + timed) >= args.max_fails: break

    task = f"{args.level}/{py_file.stem}"
    run_id = f"{py_file.stem}_{uuid.uuid4().hex[:8]}"
    run_dir = out_root / run_id
    (run_dir / "compiler-work").mkdir(parents=True)
    (run_dir / "compiler-cache").mkdir(parents=True)

    print(f"\n{SEP}\n▶  [{idx}] Compiling {task}\n{SEP}")
    os.environ["NEURON_CC_FLAGS"] = (
        f"--compile_workdir={run_dir}/compiler-work "
        f"--cache_dir={run_dir}/compiler-cache "
        f"--tensorizer-options=' --print-nki --dump-after=All --dump-nki ' "
        f"--pipeline verify"
    )

    spec = importlib.util.spec_from_file_location("kb_mod", py_file)
    kb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)

    status, note = "PASS", ""
    try:
        signal.alarm(TIMEOUT)
        model  = kb.Model(*kb.get_init_inputs()).to(device).eval()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in kb.get_inputs()]
        with torch.no_grad():
            _ = model(*inputs)
            xm.mark_step()
        passed += 1
    except TimeoutError as e:
        status, note = "TIME", str(e)
        timed += 1
    except Exception as e:
        status, note = "FAIL", str(e).splitlines()[0] or "RuntimeError"
        failed += 1
        (run_dir / "error.txt").write_text(traceback.format_exc())
        try:
            work_root = run_dir / "compiler-work"
            log_path  = max(work_root.rglob("log-neuron-cc.txt"), key=os.path.getmtime)
            tail_path = run_dir / "compiler_error_tail.txt"
            sig = extract_neuron_signal(log_path, tail_path)
            if sig: note = sig
        except Exception:
            pass
    finally:
        signal.alarm(0)
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {GLYPH[status]} {task:45} → {run_dir}")
        print(SEP)
        records.append((task, status, note, run_dir))
        for v in ("kb", "model", "inputs", "_"):
            if v in locals(): del locals()[v]
        gc.collect()
        kill_neuron_processes()  # cleanup step

# Summary table
print("\nSummary\n" + SEP)
header = f"{'Problem':45}  {'Status':6}  Note"
print(header + "\n" + "-" * len(header))
for task, status, note, run_dir in records:
    shown_note = note if status == "TIME" else shorten(note, width=30, placeholder="…")
    print(f"{task:45}  {GLYPH[status]}  {shown_note}")
print(SEP)
total = passed + failed + timed
print(f"🏁 {args.level}: {passed}/{total} passed, {failed} failed, {timed} timeout\n")

