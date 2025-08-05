#!/usr/bin/env python3
"""
run_level.py – compile-and-run kernels from:
  • KernelBench (by level: level1/level2/level3)
  • KernelBook (HF dataset: GPUMODE/KernelBook)

Trainium/Torch-XLA:
  • 35-min per-task timeout
  • continues past failures
  • coloured summary table
  • early stop via --max-tasks / --max-fails
  • cleans up Neuron compiler processes after each task

Examples
--------
# KernelBench level2
python run_level.py --source kernelbench --level level2

# KernelBook train split, limit to 50, filter class names containing "Sum"
python run_level.py --source kernelbook --kbk-split train --kbk-limit 50 --kbk-class-like Sum

Prereqs for KernelBook:
  pip install datasets
"""

from pathlib import Path
import argparse, importlib.util, uuid, os, gc, sys, datetime as dt, traceback
import signal, re, textwrap, subprocess, tempfile, shutil

# ─────────────────────────────────────────────────────────────────────────────
# visuals / constants
# ─────────────────────────────────────────────────────────────────────────────
GLYPH = dict(PASS="\033[92m✓\033[0m", FAIL="\033[91m✗\033[0m", TIME="⚠")
SEP   = "─" * 78
TIMEOUT = 35 * 60  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--source", choices=["kernelbench", "kernelbook"], default="kernelbench",
               help="Where to load tasks from")
p.add_argument("--level",     default="level2", help="(kernelbench) level1 / level2 / level3")
p.add_argument("--kb-root",   default="KernelBench/KernelBench",
               help="(kernelbench) path to KernelBench root (contains level*/)")
p.add_argument("--max-tasks", type=int, default=None, help="process at most N problems then stop")
p.add_argument("--max-fails", type=int, default=None, help="abort after M failures/timeouts")

# KernelBook options
p.add_argument("--kbk-split", default="train", help="(kernelbook) HF split: train/validation/test")
p.add_argument("--kbk-limit", type=int, default=None, help="(kernelbook) limit number of samples")
p.add_argument("--kbk-class-like", default=None, help="(kernelbook) only run samples whose detected class name contains this substring (case-insensitive)")
p.add_argument("--kbk-repo-like", default=None, help="(kernelbook) only run samples whose repo contains this substring (case-insensitive)")

args = p.parse_args()

root = Path.cwd()
out_root = root / "outputs"; out_root.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# env‑vars (must precede torch_xla import)
# ─────────────────────────────────────────────────────────────────────────────
os.environ.update({
    "PJRT_DEVICE": "NEURON",
    "MASTER_ADDR": "localhost",
    "TORCH_XLA_DEBUG": "1",
    "XLA_FLAGS": f"--xla_dump_to={out_root}/xla-dump --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*",
    "HF_HOME": str(out_root / "hf-home"),
})

import torch, torch.nn as nn
import torch_xla.core.xla_model as xm
device = xm.xla_device()

# ─────────────────────────────────────────────────────────────────────────────
# cleanup: kill residual Neuron compiler processes
# ─────────────────────────────────────────────────────────────────────────────
def kill_neuron_processes():
    procs = [
        "neuron-cc",
        "neuron_parallel_compile",
        "tensorizer",
        "neuron_compile_worker",
        "neuronx-cc",
    ]
    for proc in procs:
        subprocess.run(f"pkill -f {proc}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ─────────────────────────────────────────────────────────────────────────────
# helper: compiler headline extraction (bracket‑first)
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
    return textwrap.shorten(str(msg).strip(), width=width, placeholder="…")

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
        if _skip(ln): continue
        m = _BRACKET.search(ln)
        if m:
            return _short(f"{m.group(1)} · {m.group(2)}")
        if first_info is None:
            first_info = _short(ln.strip())
    for ln in reversed(lines):
        m = _BRACKET.search(ln)
        if m:
            return _short(f"{m.group(1)} · {m.group(2)}")
    for ln in lines:
        if _FALLBACK_PAT.search(ln):
            key = next(k for k in _FALLBACK_MAP if re.search(k, ln, re.I))
            return _short(_FALLBACK_MAP[key])
    if first_info: return first_info
    for ln in reversed(lines):
        if ln.strip(): return _short(ln)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# timeout machinery
# ─────────────────────────────────────────────────────────────────────────────
def _raise_timeout(signum, frame):
    raise TimeoutError(f"exceeded {TIMEOUT//60}-min compile window")
signal.signal(signal.SIGALRM, _raise_timeout)

records = []  # (task, status, note, run_dir)
passed = failed = timed = 0

# ─────────────────────────────────────────────────────────────────────────────
# KernelBook helpers
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_RE = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*\(\s*nn\.Module\s*\)\s*:', re.M)

def _detect_class_name(code: str) -> str | None:
    m = None
    for m in _CLASS_RE.finditer(code):
        pass  # keep the last class deriving from nn.Module
    return m.group(1) if m else None

def _slice_to_minimal_module(code: str, preferred_class: str | None = None) -> str | None:
    """
    Heuristics:
      • Keep ONLY the last nn.Module class (or the one named preferred_class if provided),
      • Keep trailing get_inputs()/get_init_inputs() definitions,
      • Append `Model = <ClassName>`.
    This avoids CUDA/Triton prologues present earlier in many KernelBook entries.
    """
    # If preferred class provided, try to find its declaration explicitly.
    if preferred_class:
        pat = re.compile(rf'^\s*class\s+({re.escape(preferred_class)})\s*\(\s*nn\.Module\s*\)\s*:', re.M)
        m = pat.search(code)
        if m:
            start = m.start()
        else:
            # fallback to last nn.Module class
            cls = _detect_class_name(code)
            if not cls: return None
            m2 = list(_CLASS_RE.finditer(code))[-1]
            start = m2.start()
            preferred_class = cls
    else:
        cls = _detect_class_name(code)
        if not cls: return None
        preferred_class = cls
        m2 = list(_CLASS_RE.finditer(code))[-1]
        start = m2.start()

    # Take from class start to end
    tail = code[start:]

    # Keep only one class block + following helpers; defensively remove earlier extra classes
    # If there is another "class ..." later, we still keep the segment up to that point.
    nxt = re.search(r'^\s*class\s+[A-Za-z_]\w*\s*\(', tail, re.M)
    if nxt and nxt.start() > 0:
        tail = tail[:nxt.start()]  # keep first class block in tail

    # Ensure imports for torch/nn exist
    header = "import torch\nimport torch.nn as nn\n"
    out = header + "\n" + tail

    # Ensure get_inputs() exists; if not, create a trivial one to avoid crashes.
    if "def get_inputs(" not in out:
        out += "\n\ndef get_inputs():\n    return []\n"
    if "def get_init_inputs(" not in out:
        out += "\n\ndef get_init_inputs():\n    return [[], {}]\n"

    # Alias to `Model` expected by the runner
    out += f"\n\nModel = {preferred_class}\n"
    return out

def _iter_kernelbook(temp_root: Path):
    """
    Yields tuples: (task_name, module_path)
    Requires `datasets` to be installed.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        sys.exit("❌ KernelBook support requires `pip install datasets`")

    ds = load_dataset("GPUMODE/KernelBook", split=args.kbk_split)
    count = 0
    for i, ex in enumerate(ds):
        if args.kbk_limit and count >= args.kbk_limit:
            break

        # Common field guesses
        code = ex.get("code") or ex.get("text") or ""
        if not code.strip():
            continue
        repo = (ex.get("repo") or ex.get("github_repo") or "").strip()
        class_hint = (ex.get("class_name") or ex.get("model_name") or "").strip()

        # optional filters
        if args.kbk_repo_like and args.kbk_repo_like.lower() not in repo.lower():
            continue

        minimal = _slice_to_minimal_module(code, preferred_class=class_hint or None)
        if not minimal:
            # try again without hint, using last nn.Module subclass
            minimal = _slice_to_minimal_module(code, preferred_class=None)
            if not minimal:
                continue

        # detect final class name to annotate task name
        final_cls = _detect_class_name(minimal) or class_hint or "UnknownModule"
        if args.kbk_class_like and args.kbk_class_like.lower() not in final_cls.lower():
            continue

        run_id = f"kbk_{final_cls}_{uuid.uuid4().hex[:8]}"
        mod_path = temp_root / f"{run_id}.py"
        mod_path.write_text(minimal)
        task_name = f"kernelbook/{final_cls}"
        yield task_name, mod_path
        count += 1

# ─────────────────────────────────────────────────────────────────────────────
# Task stream (KernelBench | KernelBook)
# ─────────────────────────────────────────────────────────────────────────────
def task_iterator():
    if args.source == "kernelbench":
        level_dir = root / args.kb_root / args.level
        if not level_dir.is_dir():
            sys.exit(f"❌ cannot find {level_dir}")
        for py_file in sorted(level_dir.glob("*.py")):
            yield f"{args.level}/{py_file.stem}", py_file
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="kernelbook_"))
        try:
            yield from _iter_kernelbook(temp_root)
        finally:
            # Keep temp files under outputs for post-mortem? Uncomment to preserve.
            # shutil.copytree(temp_root, out_root / ("kernelbook_tmp_" + uuid.uuid4().hex[:6]))
            shutil.rmtree(temp_root, ignore_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
for idx, (task, py_file) in enumerate(task_iterator(), 1):
    if args.max_tasks and idx > args.max_tasks:
        print(f"\n🛑 Reached --max-tasks={args.max_tasks}. Stopping early.\n")
        break
    if args.max_fails and (failed + timed) >= args.max_fails:
        print(f"\n🛑 Reached --max-fails={args.max_fails}. Stopping early.\n")
        break

    run_id = f"{Path(py_file).stem}_{uuid.uuid4().hex[:8]}"
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
    kb   = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)

    status, note = "PASS", ""
    try:
        signal.alarm(TIMEOUT)

        model  = kb.Model(*kb.get_init_inputs()).to(device).eval()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                  for x in kb.get_inputs()]
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
        # refine note with compiler headline if possible
        try:
            work_root = run_dir / "compiler-work"
            log_path  = max(work_root.rglob("log-neuron-cc.txt"), key=os.path.getmtime)
            tail_path = run_dir / "compiler_error_tail.txt"
            sig = extract_neuron_signal(log_path, tail_path)
            if sig:
                note = sig
        except Exception:
            pass
    finally:
        signal.alarm(0)
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {GLYPH.get(status, status)} {task:45} → {run_dir}")
        print(SEP)
        records.append((task, status, note, run_dir))
        for v in ("kb", "model", "inputs", "_"):
            if v in locals(): del locals()[v]
        gc.collect()
        kill_neuron_processes()  # cleanup step

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\nSummary\n" + SEP)
header = f"{'Problem':45}  {'Status':6}  Note"
print(header + "\n" + "-" * len(header))
from textwrap import shorten as _shorten
for task, status, note, run_dir in records:
    shown_note = note if status == "TIME" else _shorten(note, width=30, placeholder="…")
    print(f"{task:45}  {GLYPH.get(status, status)}  {shown_note}")
print(SEP)
total = passed + failed + timed
print(f"🏁 {args.source}: {passed}/{total} passed, {failed} failed, {timed} timeout\n")

