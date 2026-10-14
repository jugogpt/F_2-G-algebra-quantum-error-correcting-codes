"""sweep driver: fix a symmetry set, enumerate every code it admits, screen by distance, classify the logical group, write a csv.

pipeline per candidate (f1,f2):  dedupe by module -> k filter -> sparsest generator -> check-weight filter
  -> distance screen against the merit threshold -> exact distance -> logical group from the design symmetries -> csv row.
merit is k*d^2/n; baselines are the gross code (144,12,12) and the two-gross code (288,12,18).
"""
from __future__ import annotations
import csv, hashlib, itertools, math, os, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement
from bb_theta_inverse import _mons
from bb_logical import code_dimension, logical_basis
from bb_crt import CRTBasis
from bb_script_b import construct_code_theta
from bb_distance import code_distance
from bb_lowweight import module_basis, sparsest_generator_best, low_weight_words_isd
from bb_kdist import _prepare, _rank_bits, convolve
from bb_logical_action import induced_logical_action
from bb_logical_group import classify_logical_group
from bb_theta import identity4

BASELINES = {"gross": (144, 12, 12), "two_gross": (288, 12, 18)}
COLUMNS = ["L", "f1", "f2", "n", "k", "check_weight", "status", "d", "dX", "dZ", "d_ge", "exact", "merit", "beats_gross",
           "beats_two_gross", "design_syms", "group_order", "group_status", "clusters", "full_clifford", "seconds", "module_key"]


def merit(n, k, d):
    return k * d * d / n


def _pair_int(f1, f2, mons, idx):
    D = len(mons)
    v = 0
    for m in f1.mons:
        v |= 1 << idx[m]
    for m in f2.mons:
        v |= 1 << (D + idx[m])
    return v


def _int_pair(v, L, mons):
    D = len(mons)
    return (RingElement(L, [mons[i] for i in range(D) if (v >> i) & 1]),
            RingElement(L, [mons[i] for i in range(D) if (v >> (D + i)) & 1]))


def canonical_key(f1, f2, rd):
    """hash of the fully reduced echelon basis of the module, identical for all generator pairs of one code."""
    basis = module_basis(f1, f2, rd)
    leads = sorted(basis)
    vec = {h: basis[h] for h in leads}
    for h in leads:
        for g in leads:
            if g > h and (vec[g] >> h) & 1:
                vec[g] ^= vec[h]
    blob = ",".join(f"{h}:{vec[h]:x}" for h in leads)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def block_elements_by_k(block_basis, orbit, rd, max_dim=16):
    """{k_O: [mask,...]} over all 2^d elements of an orbit block, by gray code; None when d is too large."""
    d = len(block_basis)
    if d > max_dim:
        return None
    comps, img = _prepare(block_basis, orbit, rd)
    rows = [[0] * len(img[0][ci]) for ci in range(len(comps))] if d else [[0] * 2 for _ in comps]
    dims = [c.dim for c in comps]
    kval = lambda: sum(2 * (dims[ci] - _rank_bits(rows[ci])) for ci in range(len(comps)))
    out = {kval(): [0]}
    gray = 0
    for m in range(1, 1 << d):
        i = (m & -m).bit_length() - 1
        for ci in range(len(comps)):
            r, im = rows[ci], img[i][ci]
            for j in range(len(r)):
                r[j] ^= im[j]
        gray ^= 1 << i
        out.setdefault(kval(), []).append(gray)
    return out


def enumerate_blocks(rd, res, k_targets, max_candidates, seed):
    """candidates from the orbit blocks with exact control of k: choose per-block k_O summing to a target, then sample members."""
    rng = random.Random(seed)
    per_block = []
    for (bb, orb) in zip(res["blocks"], res["orbits"]):
        if not bb:
            continue
        per_block.append(block_elements_by_k(bb, orb, rd))
    if any(p is None for p in per_block):
        return None
    live_idx = [i for i, bb in enumerate(res["blocks"]) if bb]
    live = list(zip(live_idx, per_block))
    mons = _mons(rd.L)
    out, seen = [], set()
    ks_options = [sorted(p) for _, p in live]
    splits = [c for c in itertools.product(*ks_options) if sum(c) in k_targets]
    rng.shuffle(splits)
    for split in splits:
        space = math.prod(len(live[j][1][split[j]]) for j in range(len(live)))
        if space <= max_candidates:
            choices = itertools.product(*[live[j][1][split[j]] for j in range(len(live))])
        else:
            choices = (tuple(rng.choice(live[j][1][split[j]]) for j in range(len(live))) for _ in range(max_candidates * 2))
        for masks in choices:
            f1, f2 = RingElement(rd.L, []), RingElement(rd.L, [])
            for (bi, p), mask in zip(live, masks):
                for t, (a, b) in enumerate(res["blocks"][bi]):
                    if (mask >> t) & 1:
                        f1, f2 = f1 + a, f2 + b
            key = (frozenset(f1.mons), frozenset(f2.mons))
            if f1.mons and f2.mons and key not in seen:
                seen.add(key)
                out.append((f1, f2))
            if len(out) >= max_candidates:
                return out
    return out


def enumerate_sparse(rd, res, max_weight, iters, seed, p=3, min_poly_weight=2):
    """low-weight members of the solution space by information-set sampling; both polynomials nonzero."""
    mons = _mons(rd.L)
    idx = {m: i for i, m in enumerate(mons)}
    basis = {}
    for a, b in res["basis"]:
        v = _pair_int(a, b, mons, idx)
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                break
    seen = set()
    for v in low_weight_words_isd(basis, 2 * len(mons), max_weight, iters=iters, p=p, seed=seed):
        f1, f2 = _int_pair(v, rd.L, mons)
        if len(f1.mons) >= min_poly_weight and len(f2.mons) >= min_poly_weight and v not in seen:
            seen.add(v)
            yield f1, f2


def enumerate_exhaustive(rd, weights, seed=0):
    """all pairs with wt(f1),wt(f2) in weights, f1 containing the identity monomial (translation normal form)."""
    mons = _mons(rd.L)
    zero = tuple(0 for _ in rd.L)
    others = [m for m in mons if m != zero]
    for w1 in weights:
        for rest in itertools.combinations(others, w1 - 1):
            f1 = RingElement(rd.L, (zero,) + rest)
            for w2 in weights:
                for sup in itertools.combinations(mons, w2):
                    yield f1, RingElement(rd.L, sup)


_RD = {}


def fast_dimension(f1, f2, rd):
    """k = 2D - rank(Hx) - rank(Hz) with rows as python ints; much cheaper than the crt route for bulk filtering."""
    mons = _mons(rd.L)
    idx = {m: i for i, m in enumerate(mons)}
    D = len(mons)
    L = rd.L

    def rows_of(pairs):
        out = []
        for g in mons:
            v = 0
            for off, f in pairs:
                for m in f:
                    v |= 1 << (off + idx[tuple((m[i] + g[i]) % L[i] for i in range(len(L)))])
            out.append(v)
        return out

    def rank(rows):
        basis = {}
        r = 0
        for v in rows:
            while v:
                h = v.bit_length() - 1
                b = basis.get(h)
                if b is None:
                    basis[h] = v
                    r += 1
                    break
                v ^= b
        return r

    A, B = list(f1.mons), list(f2.mons)
    Af = [tuple((-x) % L[i] for i, x in enumerate(m)) for m in A]
    Bf = [tuple((-x) % L[i] for i, x in enumerate(m)) for m in B]
    rx = rank(rows_of([(0, A), (D, B)]))
    rz = rank(rows_of([(0, Bf), (D, Af)]))
    return 2 * D - rx - rz


def _rd(L):
    if L not in _RD:
        _RD[L] = ring_setup(L)
    return _RD[L]


def evaluate_code(task):
    """one candidate through the whole pipeline; returns a csv row dict (never raises)."""
    L, m1, m2, design, cfg = task
    t0 = time.time()
    rd = _rd(L)
    f1, f2 = RingElement(L, m1), RingElement(L, m2)
    row = {"L": str(L)}
    try:
        k = code_dimension(f1, f2, rd)
        n = 2 * rd.DL()
        row.update(n=n, k=k)
        if cfg["minimize_weight"]:
            g = sparsest_generator_best(f1, f2, rd, isd_iters=cfg["isd_iters"])
            if g.get("f1") is not None:
                f1, f2 = g["f1"], g["f2"]
        cw = len(f1.mons) + len(f2.mons)
        row.update(f1=str(f1), f2=str(f2), check_weight=cw, module_key=canonical_key(f1, f2, rd))
        if cw > cfg["max_check_weight"]:
            row.update(status="over_weight_budget")
            return _finish(row, t0)
        crt = CRTBasis(rd)
        thr = cfg["merit_threshold"]
        dmin = None if thr is None else max(2, math.ceil(math.sqrt(thr * n / k)))
        if cfg["min_distance"]:
            dmin = max(dmin or 0, cfg["min_distance"])
        scr = code_distance(f1, f2, rd, crt=crt, backend=cfg["backend"], time_limit=cfg["screen_time_limit"],
                            reject_below=dmin) if dmin else None
        if scr is not None and scr.get("below_target"):
            row.update(status="below_target_d", d=scr["d"], d_ge="", exact=scr["exact"])
            return _finish(row, t0)
        if scr is not None and not scr["exact"]:
            row.update(status="screen_timeout")
            return _finish(row, t0)
        dd = code_distance(f1, f2, rd, crt=crt, backend=cfg["backend"], time_limit=cfg["time_limit"])
        row.update(status="evaluated", d=dd["d"], dX=dd["dX"], dZ=dd["dZ"], exact=dd["exact"])
        if dd["d"]:
            m = merit(n, k, dd["d"])
            row.update(merit=round(m, 3), beats_gross=m >= merit(*BASELINES["gross"]),
                       beats_two_gross=m >= merit(*BASELINES["two_gross"]))
        if cfg["discover"]:
            from bb_script_a import find_symmetries
            found = find_symmetries(f1, f2, rd, include_depth1=cfg["discover_depth1"])
            design = [(r["psi"], r["theta"]) for r in found]
        LX, LZ = logical_basis(f1, f2, rd, crt)
        mats = [induced_logical_action(psi, Th if Th is not None else identity4(L), f1, f2, LX, LZ, rd) for psi, Th in design]
        if cfg["include_translations"]:
            from bb_automorphisms import identity_matrix
            from bb_theta import zero4
            for i in range(len(L)):
                e = tuple(1 if j == i else 0 for j in range(len(L)))
                T = zero4(L)
                for r_ in range(4):
                    T[r_][r_] = RingElement(L, [e])
                mats.append(induced_logical_action(identity_matrix(len(L)), T, f1, f2, LX, LZ, rd))
        nfold = 0
        if cfg["discover"] and cfg["fold_s"]:
            from bb_fold_s import find_fold_s, sigma_function
            for h in find_fold_s(f1, f2, rd):
                mats.append(induced_logical_action(None, None, f1, f2, LX, LZ, rd, sigma_fn=sigma_function(h[1], h[2], h[3], L)))
                nfold += 1
        rep = classify_logical_group(mats, k, max_points=cfg.get("max_points", 3000))
        row.update(design_syms=len(design) + nfold, group_order=rep["order"], group_status=rep["order_status"],
                   clusters="+".join(str(c["m"]) for c in rep["clusters"]) or "none",
                   full_clifford=rep["acts_as_full_clifford"])
    except Exception as e:
        row.update(status=f"error:{type(e).__name__}:{e}"[:120])
    return _finish(row, t0)


def _finish(row, t0):
    row["seconds"] = round(time.time() - t0, 1)
    return row


DEFAULTS = dict(mode="sparse", k_targets=None, max_candidates=200, max_check_weight=8, sparse_max_weight=8, isd_iters=200,
                minimize_weight=False, weights=(3, 4), min_distance=None, discover=False, discover_depth1=True, fold_s=True, include_translations=True, isd_p=3, min_poly_weight=2, merit_threshold=None, backend="highs", screen_time_limit=60, time_limit=300,
                workers=1, time_budget=None, seed=0, max_points=3000)


def run_sweep(L, gens, csv_path, **kw):
    """enumerate codes admitted by gens=[(psi,Theta,W),...] on R_L, evaluate each, append rows to csv_path (resumable)."""
    cfg = {**DEFAULTS, **kw}
    rd = _rd(tuple(L))
    L = tuple(L)
    res = None if (cfg["mode"] == "exhaustive" or cfg.get("candidates") is not None) else construct_code_theta(rd, gens)
    design = [(psi, Th) for psi, Th, _ in gens]
    if cfg.get("candidates") is not None:
        cands = cfg["candidates"]
    elif cfg["mode"] == "blocks":
        cands = enumerate_blocks(rd, res, set(cfg["k_targets"] or []), cfg["max_candidates"], cfg["seed"]) \
            if cfg["k_targets"] else None
        if cands is None:
            raise ValueError("blocks mode needs k_targets and every block of dimension <= 16; use mode='sparse'")
    elif cfg["mode"] == "exhaustive":
        cands = enumerate_exhaustive(rd, cfg["weights"])
    else:
        cands = enumerate_sparse(rd, res, cfg["sparse_max_weight"], cfg["isd_iters"], cfg["seed"], cfg["isd_p"],
                                 cfg["min_poly_weight"])
    done = set()
    exists = os.path.exists(csv_path)
    if exists:
        with open(csv_path) as fh:
            done = {r["module_key"] for r in csv.DictReader(fh) if r.get("module_key")}
    t_start = time.time()
    stats = {"enumerated": 0, "skipped_dup": 0, "written": 0}
    seen_keys = set(done)
    tasks = []
    for f1, f2 in cands:
        if cfg["time_budget"] and time.time() - t_start > cfg["time_budget"]:
            break
        if len(tasks) >= cfg["max_candidates"]:
            break
        stats["enumerated"] += 1
        if cfg["k_targets"]:
            k = code_dimension(f1, f2, rd)
            if k not in cfg["k_targets"]:
                continue
        key = canonical_key(f1, f2, rd)
        if key in seen_keys:
            stats["skipped_dup"] += 1
            continue
        seen_keys.add(key)
        tasks.append((L, sorted(f1.mons), sorted(f2.mons), design, cfg))
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        if cfg["workers"] > 1:
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                for fut in as_completed([ex.submit(evaluate_code, t) for t in tasks]):
                    w.writerow(fut.result()); fh.flush(); stats["written"] += 1
        else:
            for t in tasks:
                w.writerow(evaluate_code(t)); fh.flush(); stats["written"] += 1
    return stats


def summarize(csv_path, top=10):
    """best evaluated rows by merit."""
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "evaluated" and r.get("merit")]
    rows.sort(key=lambda r: -float(r["merit"]))
    return rows[:top]


if __name__ == "__main__":
    import tempfile
    from bb_theta import identity4 as I4
    L = (4, 3)
    one, zero = RingElement(L, [(0, 0)]), RingElement(L, [])
    gens = [([[3, 0], [0, 1]], I4(L), [[one, zero], [zero, one]])]
    path = os.path.join(tempfile.gettempdir(), "bb_sweep_test.csv")
    if os.path.exists(path):
        os.remove(path)
    print("blocks mode k=4:", run_sweep(L, gens, path, mode="blocks", k_targets={4}, max_candidates=10, time_limit=30))
    print("resume (expect duplicates):", run_sweep(L, gens, path, mode="blocks", k_targets={4}, max_candidates=10, time_limit=30))
    print("sparse mode:", run_sweep(L, gens, path, mode="sparse", max_candidates=10, sparse_max_weight=8, isd_p=3,
                                    k_targets={4, 6, 8}, time_limit=30))
    for r in summarize(path, 6):
        print({k: r[k] for k in ("k", "check_weight", "d", "merit", "group_order", "clusters", "exact")})
