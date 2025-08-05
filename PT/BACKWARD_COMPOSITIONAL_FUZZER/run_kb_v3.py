#!/usr/bin/env python3
"""
run_kernels.py  –  compile‑and‑run KernelBench (by level) or KernelBook samples on Trainium

Features
• Per‑task timeout (default: 35 minutes)
• Continues past failures (unless early‑stop flags reached)
• Colored summary table
• Early stop via --max-tasks / --max-fails
• Extracts short compiler headline from log-neuron-cc.txt
• Best‑effort cleanup of Neuron compiler processes after each task
• KernelBook support: loads `python_code`, reconstructs a module, runs it

Examples
• KernelBench level2:
    python run_kernels.py --source kernelbench --level level2 --kb-root KernelBench/KernelBench
• KernelBook train split, first 50 samples whose repo contains "Akhil":
    python run_kernels.py --source kernelbook --kbk-limit 50 --kbk-repo-like Akhil
"""

from pathlib import Path
import argparse, importlib.util, uuid, os, gc, sys, datetime as dt, traceback
import signal, re, textwrap, tempfile, subprocess, psutil

# optional: only needed for KernelBook
try:
    from datasets import load_dataset  # pip install datasets
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

# ─────────────────────────────────────────────────────────────────────────────
#  visuals / constants
# ─────────────────────────────────────────────────────────────────────────────
GLYPH = dict(PASS="\033[92m✓\033[0m", FAIL="\033[91m✗\033[0m", TIME="⚠")
SEP   = "─" * 78
TIMEOUT = 35 * 60  # seconds

# names we’ll try to clean up between runs (best‑effort)
NEURON_PROC_PATTERNS = [
    "neuron-cc", "neuronx-cc", "neuronx-rt", "neuronx-dis", "neuronx-profiler",
    "nkic", "nkicodegen", "nki-compiler", "tpc", "tpcd"
]

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--source", choices=["kernelbench", "kernelbook"], default="kernelbench",
               help="Run local KernelBench (by level) or HuggingFace KernelBook")
# KernelBench options
p.add_argument("--level",     default="level2", help="level1 / level2 / level3")
p.add_argument("--kb-root",   default="KernelBench/KernelBench")
# KernelBook options
p.add_argument("--kbk-split", default="train", help="KernelBook split: train/validation/test")
p.add_argument("--kbk-limit", type=int, default=None, help="Stop after N KernelBook samples")
p.add_argument("--kbk-repo-like",  default=None, help="Substring filter on record['repo']")
p.add_argument("--kbk-class-like", default=None, help="Substring filter on detected module class")
# early stop / runtime controls
p.add_argument("--max-tasks", type=int, default=None, help="process at most N problems then stop")
p.add_argument("--max-fails", type=int, default=None, help="abort after M failures/timeouts")
p.add_argument("--timeout-min", type=int, default=35, help="per-task timeout minutes")
args = p.parse_args()

TIMEOUT = int(args.timeout_min) * 60

root = Path.cwd()
out_root = root / "outputs"; out_root.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  env‑vars (must precede torch_xla import)
# ─────────────────────────────────────────────────────────────────────────────
os.environ.update({
    "PJRT_DEVICE": "NEURON",
    "MASTER_ADDR": "localhost",
    "TORCH_XLA_DEBUG": "1",
    "XLA_FLAGS": f"--xla_dump_to={out_root}/xla-dump --xla_dump_hlo_as_text "
                 f"--xla_dump_hlo_pass_re=.*",
    "HF_HOME": str(out_root / "hf-home"),
})

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
device = xm.xla_device()

# ─────────────────────────────────────────────────────────────────────────────
#  compiler headline extraction (bracket‑first)
# ─────────────────────────────────────────────────────────────────────────────
_BANNER    = "An Internal Compiler Error has occurred"
_BRACKET   = re.compile(r"\[([A-Z]{1,4}\d{3})\]\s*(.*)")
_STAR_ROW  = re.compile(r"\*{5,}")

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
    if not log_file.exists():
        return None
    tail = log_file.read_bytes()[-16384:].decode(errors="replace")
    tail_file.write_text(tail)
    lines = tail.splitlines()

    try:
        start = next(i for i, ln in enumerate(lines) if _BANNER in ln) + 1
    except StopIteration:
        start = 0

    first_info = None
    for ln in lines[start:]:
        if _skip(ln):
            continue
        m = _BRACKET.search(ln)
        if m:
            code, msg = m.groups()
            return _short(f"{code} · {msg}")
        if first_info is None:
            first_info = _short(ln.strip())

    for ln in reversed(lines):
        m = _BRACKET.search(ln)
        if m:
            code, msg = m.groups()
            return _short(f"{code} · {msg}")

    for ln in lines:
        if _FALLBACK_PAT.search(ln):
            key = next(k for k in _FALLBACK_MAP if re.search(k, ln, re.I))
            return _short(_FALLBACK_MAP[key])

    if first_info:
        return first_info
    for ln in reversed(lines):
        if ln.strip():
            return _short(ln)
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  timeout machinery
# ─────────────────────────────────────────────────────────────────────────────
def _raise_timeout(signum, frame):
    raise TimeoutError(f"exceeded {TIMEOUT//60}-min compile window")
signal.signal(signal.SIGALRM, _raise_timeout)

# ─────────────────────────────────────────────────────────────────────────────
#  process cleanup: kill lingering Neuron compiler processes (best‑effort)
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_neuron_processes() -> list[str]:
    """
    Try to terminate known compiler/rt processes to avoid zombie CPU hogs.
    Returns a list of 'killed' PIDs (as strings) for logging.
    """
    killed = []
    try:
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            name = (proc.info.get("name") or "").lower()
            cmd  = " ".join(proc.info.get("cmdline") or []).lower()
            if any(pat in name or pat in cmd for pat in NEURON_PROC_PATTERNS):
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed.append(str(proc.pid))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception:
        # as a last resort, a quick pkill sweep (ignored if not present)
        for pat in NEURON_PROC_PATTERNS:
            try:
                subprocess.run(["pkill", "-f", pat], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    return killed

# ─────────────────────────────────────────────────────────────────────────────
#  helpers for dynamic import and KernelBook reconstruction
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_RE = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*nn\.Module\s*\)\s*:", re.M)

def _detect_first_module_class(py_src: str) -> str | None:
    m = _CLASS_RE.search(py_src)
    return m.group(1) if m else None

def _ensure_model_alias(py_src: str, class_name: str) -> str:
    """
    Append a 'Model = <class_name>' alias if not already present.
    """
    if re.search(r"^\s*class\s+Model\s*\(", py_src, re.M):
        return py_src  # already has Model class
    return py_src + f"\n\n# Auto-alias for runner\nModel = {class_name}\n"

def _has_funcs(py_src: str, name: str) -> bool:
    pat = rf"^\s*def\s+{re.escape(name)}\s*\("
    return re.search(pat, py_src, re.M) is not None

def _write_module_and_import(py_src: str, run_dir: Path):
    """
    Writes py_src to a file in run_dir and imports it as a module object.
    """
    mod_path = run_dir / "kb_mod.py"
    mod_path.write_text(py_src)
    spec = importlib.util.spec_from_file_location("kb_mod", mod_path)
    kb   = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
    return kb

# ─────────────────────────────────────────────────────────────────────────────
#  core execution for a loaded module (KernelBench or reconstructed KernelBook)
# ─────────────────────────────────────────────────────────────────────────────
def run_module_once(kb, device, run_dir: Path) -> None:
    """
    Executes kb.Model(*get_init_inputs()) and a forward pass with get_inputs().
    Assumes environment variables already set.
    """
    model  = kb.Model(*kb.get_init_inputs()).to(device).eval()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
              for x in kb.get_inputs()]
    with torch.no_grad():
        _ = model(*inputs)
        xm.mark_step()
    # return locals that caller may want to cleanup explicitly
    return dict(model=model, inputs=inputs, _: _)

# ─────────────────────────────────────────────────────────────────────────────
#  main loops
# ─────────────────────────────────────────────────────────────────────────────
records = []  # (task, status, note, run_dir)
passed = failed = timed = 0

def process_one_task(task_label: str, kb_loader_callable, run_dir: Path):
    """
    kb_loader_callable must return a module object ready to run or raise.
    """
    global passed, failed, timed

    os.environ["NEURON_CC_FLAGS"] = (
        f"--compile_workdir={run_dir}/compiler-work "
        f"--cache_dir={run_dir}/compiler-cache "
        f"--tensorizer-options=' --print-nki --dump-after=All --dump-nki ' "
        f"--pipeline verify"
    )
    (run_dir / "compiler-work").mkdir(parents=True, exist_ok=True)
    (run_dir / "compiler-cache").mkdir(parents=True, exist_ok=True)

    status, note = "PASS", ""
    try:
        signal.alarm(TIMEOUT)
        kb = kb_loader_callable()
        locals_created = run_module_once(kb, device, run_dir)
        passed += 1
    except TimeoutError as e:
        status, note = "TIME", str(e)
        timed += 1
    except Exception as e:
        status, note = "FAIL", str(e).splitlines()[0] or "RuntimeError"
        failed += 1
        (run_dir / "error.txt").write_text(traceback.format_exc())

        # refine note with compiler headline if present
        try:
            work_root = run_dir / "compiler-work"
            log_path  = max(work_root.rglob("log-neuron-cc.txt"),
                            key=os.path.getmtime)
            tail_path = run_dir / "compiler_error_tail.txt"
            sig = extract_neuron_signal(log_path, tail_path)
            if sig:
                note = sig
        except Exception:
            pass
    finally:
        signal.alarm(0)
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {GLYPH[status]} {task_label:45} → {run_dir}")
        print(SEP)
        records.append((task_label, status, note, run_dir))

        # cleanup locals that may or may not exist
        for var_name in ("kb", "model", "inputs", "_", "locals_created"):
            if var_name in locals():
                try:
                    del locals()[var_name]
                except Exception:
                    pass
        gc.collect()

        # kill lingering compiler processes (best‑effort)
        killed = cleanup_neuron_processes()
        if killed:
            print(f"🧹 Cleaned Neuron procs: {', '.join(killed)}")

def run_kernelbench():
    level_dir = root / args.kb_root / args.level
    if not level_dir.is_dir():
        sys.exit(f"❌ cannot find {level_dir}")

    for idx, py_file in enumerate(sorted(level_dir.glob("*.py")), 1):
        if args.max_tasks and idx > args.max_tasks:
            print(f"\n🛑 Reached --max-tasks={args.max_tasks}. Stopping early.\n")
            break
        if args.max_fails and (failed + timed) >= args.max_fails:
            print(f"\n🛑 Reached --max-fails={args.max_fails}. Stopping early.\n")
            break

        task   = f"{args.level}/{py_file.stem}"
        run_id = f"{py_file.stem}_{uuid.uuid4().hex[:8]}"
        run_dir = out_root / run_id

        print(f"\n{SEP}\n▶  [{idx}] Compiling {task}\n{SEP}")

        def _loader():
            spec = importlib.util.spec_from_file_location("kb_mod", py_file)
            kb   = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
            return kb

        process_one_task(task, _loader, run_dir)

def run_kernelbook():
    if not _HAS_DATASETS:
        sys.exit("❌ 'datasets' package not installed. pip install datasets")

    ds = load_dataset("GPUMODE/KernelBook", split=args.kbk_split)
    seen = 0
    for i, ex in enumerate(ds):
        if args.kbk_limit and seen >= args.kbk_limit:
            break
        raw = ex.get("python_code") or ex.get("code") or ex.get("text") or ""
        code = textwrap.dedent(str(raw)).replace("\r\n", "\n")
        if not code.strip():
            continue

        repo = (ex.get("repo") or ex.get("github_repo") or "").strip()
        class_hint = (ex.get("class_name") or ex.get("model_name") or "").strip()

        if args.kbk_repo_like and args.kbk_repo_like.lower() not in repo.lower():
            continue

        # ensure we have a module class and glue funcs
        cls = _detect_first_module_class(code) or class_hint or ""
        if not cls:
            # cannot determine class; skip
            continue
        if not _has_funcs(code, "get_inputs") or not _has_funcs(code, "get_init_inputs"):
            # we require both; skip rather than guessing inputs
            continue

        patched = _ensure_model_alias(code, cls)

        task = f"KBK/{i}_{cls or 'UnknownModule'}"
        if args.kbk_class_like and args.kbk_class_like.lower() not in task.lower():
            continue

        run_id = f"kernelbook_{i}_{uuid.uuid4().hex[:6]}"
        run_dir = out_root / run_id

        print(f"\n{SEP}\n▶  [{i}] Compiling {task}\n{SEP}")

        def _loader():
            return _write_module_and_import(patched, run_dir)

        process_one_task(task, _loader, run_dir)
        seen += 1

# ─────────────────────────────────────────────────────────────────────────────
#  entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if args.source == "kernelbench":
        run_kernelbench()
    else:
        run_kernelbook()

    # summary
    print("\nSummary\n" + SEP)
    from textwrap import shorten
    header = f"{'Problem':45}  {'Status':6}  Note"
    print(header + "\n" + "-" * len(header))
    for task, status, note, run_dir in records:
        shown_note = note if status == "TIME" else shorten(note, width=30, placeholder="…")
        print(f"{task:45}  {GLYPH[status]}  {shown_note}")
    print(SEP)

    total = passed + failed + timed
    print(f"🏁 Completed: {passed}/{total} passed, {failed} failed, {timed} timeout\n")

