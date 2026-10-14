"""Exact minimum-distance computation for kB codes via a per-logical ILP, generalizing Pryadko's AB-reduction script.

Two changes from the univariate GB script this is built from:

1. The AB gcd-reduction is dropped. That trick pulls a common factor f out of f1=f*p, f2=f*q via
   gcd(a,b,x^ell-1) in the PID F2[x]/(x^ell-1); it only makes sense when R_L is a principal ideal
   ring, i.e. when at most one l_i is even (Theorem 2.4). For two or more even l_i, R_L has no
   single-generator ideals, so there is no ring-theoretic analogue of that gcd. Dropping it costs
   nothing but speed: Hx, Hz are built directly from f1, f2 with rho_matrix (reusing
   bb_script_b._mult_matrix), which is always correct, just not AB-minimized.

2. The cyclic symmetry-breaking constraint v[0] + v[D] >= 1 generalizes verbatim, with D = |G| in
   place of ell. Every monomial g in G is a symmetry of every code on R_L (Theta = g*I4 is always
   physical, since bar(g) = g^{-1} for a group element, so g*bar(g) = 1 trivially satisfies the
   symplectic condition). G acts regularly on each block's D positions, so the same shifting
   argument as the univariate case applies: any nonzero codeword can be rotated so that block-L or
   block-R position 0 is set, with no loss of generality on the minimum weight.
"""
from __future__ import annotations
import csv, itertools, random, time
import numpy as np
from bb_core import RingData, ring_setup
from bb_ring_elt import RingElement
from bb_logical import code_dimension, logical_basis
from bb_crt import CRTBasis
from bb_script_b import _mult_matrix, construct_code_theta
from bb_theta_inverse import _mons, _vec, _elt

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except ImportError:
    HAVE_GUROBI = False

from scipy.optimize import milp, LinearConstraint, Bounds


# GF(2) matrices: Hx, Hz, and the flattened logical operators, built from the toolkit's own rho map.

def rho_matrix(f: RingElement, rd: RingData, monomials=None) -> np.ndarray:
    """the D x D F2 matrix of multiplication by f, reusing bb_script_b._mult_matrix."""
    monomials = monomials or _mons(rd.L)
    return np.array(_mult_matrix(f, rd, monomials), dtype=np.uint8)


def build_checks(f1: RingElement, f2: RingElement, rd: RingData):
    """Hx = [rho(f1)|rho(f2)], Hz = [rho(f1)^T|rho(f2)^T]^swapped = [rho(f2)^T|rho(f1)^T], each D x 2D."""
    monomials = _mons(rd.L)
    A = rho_matrix(f1, rd, monomials)
    B = rho_matrix(f2, rd, monomials)
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])
    assert not np.any((Hx @ Hz.T) % 2), "CSS commutativity violated: Hx Hz^T != 0"
    return Hx, Hz


def flatten_pair(P: RingElement, Q: RingElement, rd: RingData, monomials=None) -> np.ndarray:
    """(P,Q) -> the length-2D support vector matching Hx's column blocks."""
    monomials = monomials or _mons(rd.L)
    return np.concatenate([_vec(P, monomials), _vec(Q, monomials)]).astype(np.uint8)


# GF(2) linear algebra.

def gf2_rank(M: np.ndarray) -> int:
    m = M.copy() % 2
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = next((ro for ro in range(r, rows) if m[ro, c] % 2), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        for ro in range(rows):
            if ro != r and m[ro, c] % 2:
                m[ro] = (m[ro] + m[r]) % 2
        r += 1
    return r


def check_low_weight(constraint_matrix: np.ndarray, logical_matrix: np.ndarray, max_weight: int = 2):
    """exhaustive search for a low-weight vector in ker(constraint_matrix) that is nonzero against some row of logical_matrix."""
    C = constraint_matrix % 2
    Lm = logical_matrix % 2
    nn = C.shape[1]
    for w in range(1, max_weight + 1):
        for combo in itertools.combinations(range(nn), w):
            syn = np.zeros(C.shape[0], dtype=np.uint8)
            for j in combo:
                syn ^= C[:, j]
            if np.any(syn):
                continue
            lsyn = np.zeros(Lm.shape[0], dtype=np.uint8)
            for j in combo:
                lsyn ^= Lm[:, j]
            if np.any(lsyn):
                v = np.zeros(nn, dtype=np.uint8)
                v[list(combo)] = 1
                return w, v
    return None, None


# the per-logical ILP: minimize wt(v) s.t. constraint_matrix.v = 0 (mod 2) and logical_vector.v = 1 (mod 2).
# the GF(2) equations are rewritten as integer equations sum = 2*slack (even) or 2*slack+1 (odd), exactly
# as in the original script; only the backend differs.

def _solve_single_logical_highs(constraint_matrix, logical_vector, D, upper_bound=None, time_limit=30):
    C = constraint_matrix % 2
    lv = logical_vector % 2
    n_constraints, nn = C.shape
    C_nz = [np.flatnonzero(C[i]) for i in range(n_constraints)]
    L_nz = np.flatnonzero(lv)
    if len(L_nz) == 0:
        return None, None, "infeasible"

    n_t = n_constraints
    n_vars = nn + n_t + 1
    u_idx = nn + n_t

    c = np.zeros(n_vars)
    c[:nn] = 1.0

    rows, cols, vals, lb_list, ub_list = [], [], [], [], []
    for i, nz in enumerate(C_nz):
        row = len(lb_list)
        for j in nz:
            rows.append(row); cols.append(j); vals.append(1.0)
        rows.append(row); cols.append(nn + i); vals.append(-2.0)
        lb_list.append(0.0); ub_list.append(0.0)
    row = len(lb_list)
    for j in L_nz:
        rows.append(row); cols.append(j); vals.append(1.0)
    rows.append(row); cols.append(u_idx); vals.append(-2.0)
    lb_list.append(1.0); ub_list.append(1.0)

    if D is not None and nn == 2 * D:
        row = len(lb_list)
        rows.append(row); cols.append(0); vals.append(1.0)
        rows.append(row); cols.append(D); vals.append(1.0)
        lb_list.append(1.0); ub_list.append(np.inf)

    if upper_bound is not None:
        row = len(lb_list)
        for j in range(nn):
            rows.append(row); cols.append(j); vals.append(1.0)
        lb_list.append(0.0); ub_list.append(float(upper_bound))

    from scipy.sparse import coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(len(lb_list), n_vars))
    con = LinearConstraint(A, lb_list, ub_list)

    lo = np.zeros(n_vars); hi = np.ones(n_vars)
    hi[nn:nn + n_t] = [len(nz) // 2 for nz in C_nz] or [0]
    hi[u_idx] = len(L_nz) // 2
    bounds = Bounds(lo, hi)
    integrality = np.ones(n_vars)

    res = milp(c, constraints=[con], integrality=integrality, bounds=bounds,
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, None, ("infeasible" if res.status == 2 else "timeout")
    v = np.round(res.x[:nn]).astype(np.uint8)
    return int(round(res.fun)), v, ("optimal" if res.status == 0 else "timeout")


def _solve_single_logical_gurobi(constraint_matrix, logical_vector, D, time_limit=300, threads=0,
                                  verbose=False, upper_bound=None, warm_start=None):
    """mirrors the pasted script's Gurobi formulation; used only when gurobipy is installed and licensed."""
    C = constraint_matrix % 2
    lv = logical_vector % 2
    n_constraints, nn = C.shape
    C_nz = [np.flatnonzero(C[i]).tolist() for i in range(n_constraints)]
    L_nz = np.flatnonzero(lv).tolist()
    if not L_nz:
        return None, None, "infeasible"

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", threads)
    model.setParam("MIPFocus", 2)
    model.setParam("Cuts", 2)
    model.setParam("Presolve", 2)
    model.setParam("Symmetry", 2)

    v = model.addVars(nn, vtype=GRB.BINARY, name="v")
    if warm_start is not None:
        for i in range(nn):
            v[i].Start = int(warm_start[i])

    if D is not None and nn == 2 * D:
        model.addConstr(v[0] + v[D] >= 1, name="cyc_sym_break")

    max_row_wt = max((len(nz) for nz in C_nz), default=0)
    t = model.addVars(n_constraints, vtype=GRB.INTEGER, lb=0, ub=max_row_wt // 2, name="t")
    max_lw = len(L_nz)
    u = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_lw // 2, name="u")

    for i, nz in enumerate(C_nz):
        if nz:
            model.addConstr(gp.quicksum(v[j] for j in nz) == 2 * t[i], name=f"par_{i}")
    model.addConstr(gp.quicksum(v[j] for j in L_nz) == 2 * u + 1, name="log_odd")

    weight_expr = gp.quicksum(v[i] for i in range(nn))
    if upper_bound is not None:
        model.addConstr(weight_expr <= upper_bound, name="ub")
    model.setObjective(weight_expr, GRB.MINIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return None, None, ("infeasible" if model.Status == GRB.INFEASIBLE else "timeout")
    sol = np.array([int(round(v[i].X)) for i in range(nn)], dtype=np.uint8)
    return int(round(model.ObjVal)), sol, ("optimal" if model.Status == GRB.OPTIMAL else "timeout")


def min_weight_over_logicals(constraint_matrix, logical_matrix, D, backend="highs",
                              time_limit=30, precheck_weight=2, verbose=False, reject_below=None):
    """min weight over all logicals; returns (weight, vector, exact) with exact false if any ilp timed out unresolved.

    with reject_below set, stops at the first logical of weight < reject_below; if none exists the result is (None, None, exact),
    meaning every logical has weight >= reject_below (proven when exact).
    """
    w0, v0 = check_low_weight(constraint_matrix, logical_matrix, max_weight=precheck_weight)
    best_w, best_v = (w0, v0) if w0 is not None else (None, None)
    exact = True
    if reject_below is not None and best_w is not None and best_w < reject_below:
        return best_w, best_v, True

    solve_one = _solve_single_logical_gurobi if (backend == "gurobi" and HAVE_GUROBI) else _solve_single_logical_highs
    for j in range(logical_matrix.shape[0]):
        lv = logical_matrix[j]
        if not np.any(lv):
            continue
        ub = None if best_w is None else best_w - 1
        if best_w is None and reject_below is not None:
            ub = reject_below - 1
        if ub is not None and ub < 1:
            break
        w, v, st = solve_one(constraint_matrix, lv, D, upper_bound=ub, time_limit=time_limit)
        if st == "timeout":
            exact = False
        if w is not None and (best_w is None or w < best_w):
            best_w, best_v = w, v
        if verbose:
            print(f"    logical {j}: status {st}, weight {w}, running best {best_w}")
        if reject_below is not None and w is not None and w < reject_below:
            return best_w, best_v, exact
    return best_w, best_v, exact


def _bar(P: RingElement, rd: RingData) -> RingElement:
    """antipode g -> g^-1; logical_basis returns representatives whose bar lies in the kernel of the opposite check."""
    return RingElement(rd.L, [tuple((-m[i]) % rd.L[i] for i in range(len(rd.L))) for m in P.mons])


def logical_matrices(f1, f2, rd, crt):
    """binary matrices of true X and Z logical representatives, in the column convention of build_checks."""
    LX, LZ = logical_basis(f1, f2, rd, crt)
    monomials = _mons(rd.L)
    mk = lambda LL: np.array([flatten_pair(_bar(P, rd), _bar(Q, rd), rd, monomials) for P, Q in LL], dtype=np.uint8)
    return mk(LX), mk(LZ)


def code_distance(f1: RingElement, f2: RingElement, rd: RingData, crt: CRTBasis = None,
                   backend="highs", time_limit=30, precheck_weight=2, verbose=False, reject_below=None):
    """d = min(dX, dZ), each found by a per-logical ILP against the other side's logical basis.

    with reject_below set this is a screen: d is None and d_ge=reject_below when no logical lighter than the threshold exists.
    """
    if crt is None:
        crt = CRTBasis(rd)
    D = rd.DL()
    k = code_dimension(f1, f2, rd)
    if k == 0:
        return {"n": 2 * D, "k": 0, "dX": None, "dZ": None, "d": None, "exact": True}

    Hx, Hz = build_checks(f1, f2, rd)
    LX_mat, LZ_mat = logical_matrices(f1, f2, rd, crt)

    if verbose:
        print(f"  n={2*D}, k={k}: solving for dX (errors in ker Hz, nontrivial on some Z logical)")
    dX, _, exX = min_weight_over_logicals(Hz, LZ_mat, D, backend=backend, time_limit=time_limit,
                                      precheck_weight=precheck_weight, verbose=verbose, reject_below=reject_below)
    if reject_below is not None and dX is not None and dX < reject_below:
        return {"n": 2 * D, "k": k, "dX": dX, "dZ": None, "d": dX, "exact": True, "below_target": True}
    if verbose:
        print(f"  dX = {dX}; solving for dZ (errors in ker Hx, nontrivial on some X logical)")
    dZ, _, exZ = min_weight_over_logicals(Hx, LX_mat, D, backend=backend, time_limit=time_limit,
                                      precheck_weight=precheck_weight, verbose=verbose, reject_below=reject_below)
    if reject_below is not None:
        if dZ is not None:
            return {"n": 2 * D, "k": k, "dX": None, "dZ": dZ, "d": dZ, "exact": True, "below_target": True}
        return {"n": 2 * D, "k": k, "dX": None, "dZ": None, "d": None, "d_ge": reject_below, "exact": exX and exZ}
    d = None if (dX is None or dZ is None) else min(dX, dZ)
    return {"n": 2 * D, "k": k, "dX": dX, "dZ": dZ, "d": d, "exact": exX and exZ}


# enumerate codes for a prescribed symmetry set and scan them for distance, writing results to csv.

def scan_codes_to_csv(rd: RingData, gens: list, csv_path: str, target_k=None, max_codes: int = 200,
                       backend="highs", time_limit=30, seed: int = 0, verbose: bool = False):
    """codes = construct_code_theta(rd, gens)'s solution space, filtered to target_k if given, each sent through code_distance."""
    random.seed(seed)
    R = construct_code_theta(rd, gens)
    V = R["basis"]
    dim = len(V)
    crt = CRTBasis(rd)

    if dim <= 20:
        combos = list(itertools.product([0, 1], repeat=dim))[1:]
    else:
        combos = [tuple(random.randint(0, 1) for _ in range(dim)) for _ in range(max_codes * 4)]

    seen = set()
    rows = []
    for coef in combos:
        if len(rows) >= max_codes:
            break
        f1 = RingElement(rd.L, [])
        f2 = RingElement(rd.L, [])
        for c, (a, b) in zip(coef, V):
            if c:
                f1 = f1 + a
                f2 = f2 + b
        key = (frozenset(f1.mons), frozenset(f2.mons))
        if key in seen or (not f1.mons and not f2.mons):
            continue
        seen.add(key)

        if not f1.mons or not f2.mons:
            continue  # a bicycle code needs both generators nonzero; skip degenerate f1=0 or f2=0 picks
        k = code_dimension(f1, f2, rd)
        if k == 0 or (target_k is not None and k != target_k):
            continue

        t0 = time.time()
        res = code_distance(f1, f2, rd, crt=crt, backend=backend, time_limit=time_limit, verbose=verbose)
        res["f1"] = str(f1)
        res["f2"] = str(f2)
        res["L"] = str(rd.L)
        res["seconds"] = round(time.time() - t0, 1)
        rows.append(res)

    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["L", "f1", "f2", "n", "k", "dX", "dZ", "d", "exact", "seconds"])
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    from bb_automorphisms import full_fold

    # correctness check on a tiny ring (n=8): HiGHS ILP result must match exhaustive brute force.
    rd22 = ring_setup((2, 2))
    f1 = RingElement.from_string((2, 2), "x + y")
    f2 = RingElement.from_string((2, 2), "1 + x*y")
    print("R_{2,2}, f1=x+y, f2=1+xy:")
    res = code_distance(f1, f2, rd22, backend="highs", time_limit=30, verbose=True)
    print(" ILP:", res)

    Hx, Hz = build_checks(f1, f2, rd22)
    crt22 = CRTBasis(rd22)
    LX_mat, LZ_mat = logical_matrices(f1, f2, rd22, crt22)
    dX_bf, _ = check_low_weight(Hz, LZ_mat, max_weight=8)
    dZ_bf, _ = check_low_weight(Hx, LX_mat, max_weight=8)
    print(" brute force: dX =", dX_bf, " dZ =", dZ_bf, " d =", min(dX_bf, dZ_bf))
    assert res["dX"] == dX_bf and res["dZ"] == dZ_bf, "ILP and brute force disagree"
    print("  pass\n")

    # a second, less symmetric tiny code on the same ring, same cross-check.
    f1b = RingElement.from_string((2, 2), "1 + x")
    f2b = RingElement.from_string((2, 2), "1 + y")
    res_b = code_distance(f1b, f2b, rd22, backend="highs", verbose=False)
    Hxb, Hzb = build_checks(f1b, f2b, rd22)
    LXb_mat, LZb_mat = logical_matrices(f1b, f2b, rd22, crt22)
    dXb_bf, _ = check_low_weight(Hzb, LZb_mat, max_weight=8)
    dZb_bf, _ = check_low_weight(Hxb, LXb_mat, max_weight=8)
    print("R_{2,2}, f1=1+x, f2=1+y:  ILP =", res_b, " brute force dX,dZ =", (dXb_bf, dZb_bf))
    assert res_b["dX"] == dXb_bf and res_b["dZ"] == dZb_bf
    print("  pass\n")

    # the Example 1 forward code on R_{4,3} (n=24): ILP only, brute force is infeasible at this size.
    rd43 = ring_setup((4, 3))
    g1 = RingElement.from_string((4, 3), "1 + x*y")
    g2 = RingElement.from_string((4, 3), "y^2 + x*y + x^2 + x^3")
    t0 = time.time()
    res43 = code_distance(g1, g2, rd43, backend="highs", time_limit=30)
    print(f"R_{{4,3}}, Example-1 code: [[{res43['n']},{res43['k']},{res43['d']}]]  ({time.time()-t0:.1f}s)")
    print("  (dX={}, dZ={})".format(res43["dX"], res43["dZ"]))

    # scan_codes_to_csv end-to-end: enumerate the iota_x-fixed codes on R_{4,3} (Example 5a's family) and tabulate.
    one = RingElement(rd43.L, [(0, 0)])
    zero = RingElement(rd43.L, [])
    W = [[one, zero], [zero, one]]
    from bb_theta import identity4
    ix = [[3, 0], [0, 1]]
    gens = [(ix, identity4(rd43.L), W)]
    rows = scan_codes_to_csv(rd43, gens, "/tmp/distance_scan.csv", target_k=4, max_codes=5,
                              backend="highs", time_limit=20)
    print(f"\nscan_codes_to_csv: {len(rows)} codes with k=4 written to /tmp/distance_scan.csv")
    for r in rows:
        print(" ", r)
