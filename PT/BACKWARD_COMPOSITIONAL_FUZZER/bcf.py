# backward_fuzzer_softmatch_no_cycles_forbid.py
import random
from typing import Tuple, List, Dict, NamedTuple, Optional, Set

# ───────────────────────── 1) Tiny tensor "type" ─────────────────────────
class TType(NamedTuple):
    shape: Tuple[int, ...]
    dtype: str = "float32"
    @property
    def rank(self) -> int:
        return len(self.shape)

# ───────────────────────── 2) Unary ops (shape-preserving) ────────────────
UNARY_FMTS = [
    ("relu",    lambda x: f"torch.relu({x})"),
    ("sigmoid", lambda x: f"torch.sigmoid({x})"),
    ("tanh",    lambda x: f"torch.tanh({x})"),
    ("softmax", lambda x: f"torch.softmax({x}, dim=-1)"),
]
def can_use_softmax(tt: TType) -> bool:
    return tt.rank >= 1

# ───────────────────────── 3) Fresh names ─────────────────────────────────
def fresh(counter: List[int]) -> str:
    counter[0] += 1
    return f"t{counter[0]}"

# ───────────────────────── 4) Soft-matching helpers ───────────────────────
def broadcastable_expand(from_shape: Tuple[int, ...], to_shape: Tuple[int, ...]) -> bool:
    """True if from_shape can be expanded to to_shape (same rank, dims equal or 1)."""
    if len(from_shape) != len(to_shape):
        return False
    for f, t in zip(from_shape, to_shape):
        if f == t or f == 1:
            continue
        return False
    return True

def maybe_unary_wrap(
    name: str,
    ty: TType,
    produced: Dict[str, tuple[str, TType, List[str]]],
    counter: List[int]
) -> str:
    """Optionally wrap `name` in a random unary op (shape-preserving)."""
    pool = UNARY_FMTS if can_use_softmax(ty) else UNARY_FMTS[:-1]
    if not pool or random.random() >= 0.5:
        return name
    op, fmt = random.choice(pool)
    w = fresh(counter)
    produced[w] = (fmt(name), ty, [name])
    return w

def try_reuse_rankN(
    required_ty: TType,
    produced: Dict[str, tuple[str, TType, List[str]]],
    counter: List[int],
    allow_expand: bool = True,
    forbid: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Reuse an existing node with same rank:
      - exact shape → (maybe unary) reuse
      - broadcastable → insert .expand(target_shape) (then maybe unary)
    Skips any names in `forbid` (all ancestors on the current path) to avoid cycles.
    """
    if forbid is None:
        forbid = set()
    for cname, (_, cty, _) in produced.items():
        if cname in forbid:
            continue
        if cty.rank != required_ty.rank:
            continue
        if cty.shape == required_ty.shape and random.random() < 0.5:
            return maybe_unary_wrap(cname, required_ty, produced, counter)
        if allow_expand and broadcastable_expand(cty.shape, required_ty.shape):
            e = fresh(counter)
            import pdb; pdb.set_trace()
            produced[e] = (f"{cname}.expand{required_ty.shape}", required_ty, [cname])
            return maybe_unary_wrap(e, required_ty, produced, counter)
    return None

def try_reuse_B_for_matmul(
    required_shape_B: Tuple[int, int],
    produced: Dict[str, tuple[str, TType, List[str]]],
    counter: List[int],
    forbid: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Reuse a rank-2 candidate for B (k,j):
      - exact (k,j) → use directly
      - transposed (j,k) → wrap with .transpose(-2,-1)
      - broadcastable → wrap with .expand((k,j))
    Skips any names in `forbid` (ancestors) to avoid cycles.
    """
    if forbid is None:
        forbid = set()
    k, j = required_shape_B
    req_ty = TType((k, j))
    for cname, (_, cty, _) in produced.items():
        if cname in forbid:
            continue
        if cty.rank != 2:
            continue
        if cty.shape == (k, j):
            return cname
        if cty.shape == (j, k):
            tname = fresh(counter)
            produced[tname] = (f"{cname}.transpose(-2, -1)", req_ty, [cname])
            return tname
        if broadcastable_expand(cty.shape, (k, j)):
            ename = fresh(counter)
            produced[ename] = (f"{cname}.expand{(k, j)}", req_ty, [cname])
            return ename
    return None

# ───────────────────────── 5) Backward builder (goal → operands) ──────────
def backward_build(
    *,
    out_shape: Tuple[int, ...],
    depth: int = 12,
    min_chain: int = 3,
    seed: Optional[int] = None,
) -> Dict[str, tuple[str, TType, List[str]]]:
    """
    Build a DAG mapping:
        name -> (rhs_string, TType, deps)
    Start from 't_out' of shape `out_shape`, expand backwards for up to `depth` steps.
    Use `min_chain` as minimal op-depth budget along the main path.
    Cycle-safe: each subgoal carries a `forbid` set (all ancestors on that path).
    """
    if seed is not None:
        random.seed(seed)

    produced: Dict[str, tuple[str, TType, List[str]]] = {}
    counter = [-1]  # next fresh is t0

    # unresolved holds (name, type, budget, forbid_set)
    unresolved: List[tuple[str, TType, int, Set[str]]] = [
        ("t_out", TType(out_shape), min_chain, {"t_out"})
    ]
    steps_left = depth

    def make_leaf(name: str, ty: TType):
        produced[name] = (f"torch.randn({ty.shape}, dtype=torch.float32)", ty, [])

    while unresolved and steps_left > 0:
        name, ty, budget, forbid = unresolved.pop()

        # If already defined (via reuse/wrappers earlier), continue
        if name in produced:
            steps_left -= 1
            continue

        # Decide op family
        if ty.rank == 4:
            choice = random.choices(["unary", "einsum4"], weights=[0.7, 0.3], k=1)[0]
        elif ty.rank == 3:
            choice = random.choices(["unary", "matmul"], weights=[0.75, 0.25], k=1)[0]
        elif ty.rank in (1, 2):
            choice = "unary"
        else:
            make_leaf(name, ty)
            steps_left -= 1
            continue

        # Stop if budget exhausted
        if budget <= 0:
            make_leaf(name, ty)
            steps_left -= 1
            continue

        if choice == "unary":
            pool = UNARY_FMTS if can_use_softmax(ty) else UNARY_FMTS[:-1]
            op, fmt = random.choice(pool)

            # Try to reuse some compatible node (skipping all ancestors in `forbid`)
            reused = try_reuse_rankN(ty, produced, counter, allow_expand=True, forbid=forbid)
            if reused is not None:
                produced[name] = (fmt(reused), ty, [reused])
            else:
                # Create a fresh predecessor subgoal
                src = fresh(counter)
                produced[name] = (fmt(src), ty, [src])
                # New subgoal inherits ancestors + current node as forbidden
                unresolved.append((src, ty, budget - 1, set(forbid) | {name}))

        elif choice == "matmul" and ty.rank == 3:
            # (b,i,j) = (b,i,k) @ (k,j)
            b, i, j = ty.shape
            k = random.randint(8, 128)
            A_ty = TType((b, i, k))
            B_ty = TType((k, j))

            # Reuse B (rank-2) with transpose/expand if possible; skip ancestors
            B_name = try_reuse_B_for_matmul(B_ty.shape, produced, counter, forbid=forbid)
            if B_name is None:
                # Create fresh B as a leaf (we typically don't grow B)
                B_name = fresh(counter)
                produced[B_name] = (f"torch.randn({B_ty.shape}, dtype=torch.float32)", B_ty, [])

            # Reuse A (rank-3) with expand/unary if possible; skip ancestors
            A_name = try_reuse_rankN(A_ty, produced, counter, allow_expand=True, forbid=forbid)
            if A_name is None:
                A_name = fresh(counter)
                # Grow A branch; inherit forbid + current node
                unresolved.append((A_name, A_ty, budget - 1, set(forbid) | {name}))

            produced[name] = (f"torch.matmul({A_name}, {B_name})", ty, [A_name, B_name])

        elif choice == "einsum4" and ty.rank == 4:
            # 4D einsum: "bijk,kl->bijl"
            # out (b,i,j,l) = A(b,i,j,k) x B(k,l)
            b, i, j, l = ty.shape
            k = random.randint(8, 128)
            A_ty = TType((b, i, j, k))
            B_ty = TType((k, l))

            # Reuse B (rank-2) as above
            B_name = try_reuse_B_for_matmul(B_ty.shape, produced, counter, forbid=forbid)
            if B_name is None:
                B_name = fresh(counter)
                produced[B_name] = (f"torch.randn({B_ty.shape}, dtype=torch.float32)", B_ty, [])

            # Reuse A (rank-4) via expand/unary; skip ancestors
            A_name = try_reuse_rankN(A_ty, produced, counter, allow_expand=True, forbid=forbid)
            if A_name is None:
                A_name = fresh(counter)
                unresolved.append((A_name, A_ty, budget - 1, set(forbid) | {name}))

            produced[name] = (f'torch.einsum("bijk,kl->bijl", {A_name}, {B_name})', ty, [A_name, B_name])

        else:
            make_leaf(name, ty)

        steps_left -= 1

    # Materialize any remaining subgoals as leaves
    for n, ty, _, _ in unresolved:
        if n not in produced:
            make_leaf(n, ty)

    return produced

# ───────────────────────── 6) Topological order (Kahn) ────────────────────
def topo_sort(graph: Dict[str, tuple[str, TType, List[str]]]) -> List[str]:
    from collections import defaultdict, deque
    indeg = defaultdict(int)
    children = defaultdict(list)
    for node, (_, _, deps) in graph.items():
        for d in deps:
            indeg[node] += 1
            children[d].append(node)
    q = deque([n for n in graph if indeg[n] == 0])
    ordered: List[str] = []
    while q:
        n = q.popleft()
        ordered.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(ordered) != len(graph):
        raise RuntimeError("cycle detected")
    return ordered

# ───────────────────────── 7) Emit a runnable module ──────────────────────
def make_module(
    *,
    depth: int = 12,
    min_chain: int = 3,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        random.seed(seed)

    # Randomize between 3D and 4D targets to exercise both paths
    if random.random() < 0.5:
        out_shape = (
            random.randint(8, 64),
            random.randint(8, 64),
            random.randint(8, 64),
        )  # 3D → matmul/unary
    else:
        out_shape = (
            random.randint(8, 48),
            random.randint(8, 48),
            random.randint(8, 48),
            random.randint(8, 48),
        )  # 4D → einsum4/unary

    g = backward_build(out_shape=out_shape, depth=depth, min_chain=min_chain, seed=seed)
    order = topo_sort(g)

    # Emit forward() with no inputs; leaves are randn, internals are ops
    body_lines: List[str] = []
    for n in order:
        if n == "t_out":
            continue
        rhs, _, _ = g[n]
        body_lines.append(f"{n} = {rhs}")
    body_lines.append("t_out = " + g["t_out"][0])
    body_lines.append("return t_out")

    indented = "\n".join(" " * 8 + ln for ln in body_lines)
    return (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self):\n"
        f"{indented}\n\n"
        "def get_inputs():\n"
        "    return []\n"
    )

# ───────────────────────── 8) CLI demo ────────────────────────────────────
if __name__ == "__main__":
    for i in range(10):
        print(make_module(depth=16, min_chain=4, seed=i))

