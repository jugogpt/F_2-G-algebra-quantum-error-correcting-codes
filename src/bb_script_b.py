"""Script B: inverse construction of codes for a given symmetry, by scalar target (Tier 1) or fixed Theta (Tier 2)."""

from __future__ import annotations
from itertools import product as iproduct
import galois
from bb_core import RingData
from bb_ring_elt import RingElement
from bb_local_ring import check_cyclic_matching
from bb_logical import code_dimension

GF2 = galois.GF(2)


def _permutation_matrix(psi_matrix, rd: RingData, monomials):
    """P_psi: the DL x DL F2 matrix of psi's action on the monomial basis."""
    DL = rd.DL()
    M = GF2.Zeros((DL, DL))
    idx = {m: i for i, m in enumerate(monomials)}
    L = rd.L
    k = rd.k
    for j, m in enumerate(monomials):
        newexp = tuple(sum(psi_matrix[i][t] * m[t] for t in range(k)) % L[i] for i in range(k))
        M[idx[newexp], j] = GF2(1)
    return M


def _mult_matrix(w: RingElement, rd: RingData, monomials):
    """Rho_w: the DL x DL F2 matrix of multiplication by w."""
    DL = rd.DL()
    M = GF2.Zeros((DL, DL))
    idx = {m: i for i, m in enumerate(monomials)}
    for j, m in enumerate(monomials):
        e = RingElement(rd.L, [m])
        prod = e * w
        for pm in prod.mons:
            M[idx[pm], j] = GF2(1)
    return M


def target_kernel(psi_matrix, w: RingElement, rd: RingData, monomials):
    """F2-basis of {f in R_L : psi(f) = w*f}."""
    P = _permutation_matrix(psi_matrix, rd, monomials)
    Rw = _mult_matrix(w, rd, monomials)
    diff = P - Rw
    return diff.null_space()


def _subspace_intersection(basis1, basis2):
    """intersect two subspaces via the combined null-space trick."""
    if len(basis1) == 0 or len(basis2) == 0:
        return GF2.Zeros((0, basis1.shape[1] if len(basis1) else basis2.shape[1]))
    n = basis1.shape[1]
    d1, d2 = basis1.shape[0], basis2.shape[0]
    N = GF2.Zeros((n, d1 + d2))
    N[:, :d1] = basis1.T
    N[:, d1:] = basis2.T
    ns = N.null_space()
    if len(ns) == 0:
        return GF2.Zeros((0, n))
    rows = []
    for combo in ns:
        a = combo[:d1]
        v = a @ basis1
        rows.append(v)
    stacked = GF2(rows)
    R = stacked.row_reduce()
    return R[[i for i in range(R.shape[0]) if any(int(x) != 0 for x in R[i])], :]


def joint_target_space(rd: RingData, targets: list):
    """intersection, over all (psi,w) targets, of their target kernels."""
    L = rd.L
    monomials = [tuple(m) for m in iproduct(*[range(li) for li in L])]
    V = None
    for psi_matrix, w in targets:
        K = target_kernel(psi_matrix, w, rd, monomials)
        V = K if V is None else _subspace_intersection(V, K)
        if V.shape[0] == 0:
            break
    return V, monomials


def _vec_to_ring_element(vec, L, monomials):
    mons = [monomials[i] for i in range(len(monomials)) if int(vec[i]) != 0]
    return RingElement(L, mons)


def construct_code(rd: RingData, targets: list, max_pairs: int = 30, verbose: bool = False):
    """find codes (f1,f2) satisfying every (psi,w) target exactly, verified against check_cyclic_matching."""
    V, monomials = joint_target_space(rd, targets)
    L = rd.L
    dimV = V.shape[0]
    if verbose:
        print(f"joint solution space dimension: {dimV}")
    if dimV == 0:
        return []

    basis_elems = [_vec_to_ring_element(V[i], L, monomials) for i in range(dimV)]
    candidates = list(basis_elems)
    for i in range(dimV):
        for j in range(i + 1, dimV):
            candidates.append(_vec_to_ring_element(V[i] + V[j], L, monomials))
    candidates = candidates[:max_pairs] if len(candidates) > max_pairs else candidates

    results = []
    seen_pairs = set()
    for f1 in candidates:
        for f2 in candidates:
            if f1 == f2:
                continue
            key = (frozenset(f1.mons), frozenset(f2.mons))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            k = code_dimension(f1, f2, rd)
            if k == 0:
                continue

            all_ok = True
            for psi_matrix, w in targets:
                ok, _ = check_cyclic_matching(f1, f2, psi_matrix, rd)
                if not ok:
                    all_ok = False
                    break
            if all_ok:
                results.append({"f1": f1, "f2": f2, "k": k})
    return results


def construct_code_bruteforce(rd: RingData, targets: list, max_weight: int = 2,
                               max_candidates: int = 4000, max_results: int = 10,
                               include_depth1: bool = True, verbose: bool = False):
    """limited fallback for entangling Theta targets: enumerate low-weight candidates and re-verify via find_symmetries."""
    from bb_script_a import find_symmetries
    from itertools import combinations

    L = rd.L
    monomials = [tuple(m) for m in iproduct(*[range(li) for li in L])]
    zero_mon = tuple(0 for _ in L)

    pool = []
    for w in range(1, max_weight + 1):
        for combo in combinations(monomials, w):
            pool.append(RingElement(L, combo))
    if verbose:
        print(f"brute-force candidate pool size: {len(pool)}")

    tried = 0
    results = []
    for f1 in pool:
        for f2 in pool:
            if f1 == f2:
                continue
            tried += 1
            if tried > max_candidates:
                if verbose:
                    print(f"stopped after {max_candidates} candidates")
                return results
            k = code_dimension(f1, f2, rd)
            if k == 0:
                continue
            psi_list = [t[0] for t in targets]
            found = find_symmetries(f1, f2, rd, psi_candidates=psi_list,
                                     include_depth1=include_depth1, verbose=False)
            if _covers_all_targets(found, targets):
                results.append({"f1": f1, "f2": f2, "k": k, "symmetries": found})
                if len(results) >= max_results:
                    return results
    return results


def _covers_all_targets(found_symmetries, targets):
    for psi_matrix, theta_spec in targets:
        match = False
        for r in found_symmetries:
            if r["psi"] != psi_matrix:
                continue
            if theta_spec is None:
                match = True
                break
            name, meta = theta_spec
            if r["theta_name"] == name and r["theta_meta"] == meta:
                match = True
                break
        if not match:
            return False
    return True


if __name__ == "__main__":
    from bb_core import ring_setup
    from bb_automorphisms import full_fold, identity_matrix

    # construct a code invariant under the exact bar involution on L=(3,3), then self-verify.
    rd = ring_setup((3, 3))
    one = RingElement(rd.L, [tuple(0 for _ in rd.L)])
    A_fold = full_fold(rd.L)

    results = construct_code(rd, [(A_fold, one)], verbose=True)
    print(f"codes found invariant under the exact bar involution (w=1): {len(results)}")
    for r in results[:5]:
        print(f"  f1={r['f1']}, f2={r['f2']}, k={r['k']}")
    assert len(results) > 0, "expected at least one nontrivial fold-invariant code"

    f1, f2 = results[0]["f1"], results[0]["f2"]
    ok, _ = check_cyclic_matching(f1, f2, A_fold, rd)
    print(f"independent re-verification of first result: {ok}")
    assert ok is True
    print("  pass\n")

    # with the identity as the only target, every valid nonzero pair should work.
    results_id = construct_code(rd, [(identity_matrix(2), one)], max_pairs=20, verbose=True)
    print(f"codes found with only the trivial (identity, w=1) target: {len(results_id)}")
    assert len(results_id) > 0
    print("  pass\n")

    # brute-force fallback smoke test: rediscover a code with the qubit-swap symmetry on L=(2,2).
    rd22 = ring_setup((2, 2))
    swap = [[0, 1], [1, 0]]
    results_bf = construct_code_bruteforce(rd22, [(swap, None)], max_weight=2,
                                            max_candidates=500, max_results=3, verbose=True)
    print(f"brute-force Tier 2: codes found with the swap symmetry: {len(results_bf)}")
    for r in results_bf:
        print(f"  f1={r['f1']}, f2={r['f2']}, k={r['k']}")
    assert len(results_bf) > 0
    print("  pass")


# script B, Tier 2 (replaces the brute-force fallback): fixed (psi, Theta, W) generators.
def construct_code_theta(rd, gens, report_clifford: bool = False, max_report_codes: int = 20,
                          target_k=None, discover_extra: bool = False, discover_depth1: bool = False,
                          clifford_exact_cap: int = 2_000_000, seed: int = 0, verbose: bool = False):
    """given gens = [(psi_matrix, Theta, W), ...], return an F2-basis of all codes with sigma_j(M)=M for every generator.

    With report_clifford=True, also samples up to max_report_codes member codes from that solution
    space (filtered to target_k if given) and attaches "sample_reports": one
    {f1, f2, k, clifford} dict per sampled code. "clifford" is symmetry_clifford_report's verdict
    for that specific code against the DESIGN gens' (psi, Theta) pairs by default; set
    discover_extra=True to instead re-run find_symmetries per sampled code (far more expensive,
    but can surface extra symmetries the code happens to admit beyond what it was designed for --
    this is exactly how to check whether different codes in the same physical-symmetry family end
    up with different realized logical actions, see the module's __main__ for a worked case).
    """
    from bb_theta_inverse import joint_solution_basis
    from bb_orbits import component_perm, bar_perm, orbits_of, split_solution_space
    from bb_crt import CRTBasis
    V = joint_solution_basis(rd.L, gens)
    perms = [component_perm(psi, rd) for psi, _, _ in gens] + [bar_perm(rd)]
    orbs = orbits_of(perms, rd)
    crt = CRTBasis(rd)
    blocks = [split_solution_space(V, rd, crt, O) for O in orbs]
    result = {"basis": V, "orbits": orbs, "blocks": blocks, "dim": len(V)}

    if not report_clifford:
        return result

    import random, itertools
    from bb_logical import code_dimension
    from bb_clifford_order import symmetry_clifford_report

    rng = random.Random(seed)
    dim = len(V)
    if dim == 0:
        result["sample_reports"] = []
        return result
    if dim <= 16:
        combos = list(itertools.product([0, 1], repeat=dim))[1:]
        rng.shuffle(combos)
    else:
        combos = [tuple(rng.randint(0, 1) for _ in range(dim)) for _ in range(max_report_codes * 8)]

    design_symmetries = [(psi, Theta) for psi, Theta, _ in gens]
    seen, reports = set(), []
    for coef in combos:
        if len(reports) >= max_report_codes:
            break
        f1, f2 = RingElement(rd.L, []), RingElement(rd.L, [])
        for c, (a, b) in zip(coef, V):
            if c:
                f1 = f1 + a
                f2 = f2 + b
        if not f1.mons or not f2.mons:
            continue
        key = (frozenset(f1.mons), frozenset(f2.mons))
        if key in seen:
            continue
        seen.add(key)

        k = code_dimension(f1, f2, rd)
        if k == 0 or (target_k is not None and k != target_k):
            continue

        if discover_extra:
            from bb_script_a import find_symmetries
            _, clifford = find_symmetries(f1, f2, rd, include_depth1=discover_depth1, report_clifford=True,
                                           clifford_exact_cap=clifford_exact_cap, verbose=False)
        else:
            clifford = symmetry_clifford_report(f1, f2, rd, design_symmetries, crt=crt,
                                                 exact_cap=clifford_exact_cap, verbose=False)
        reports.append({"f1": f1, "f2": f2, "k": k, "clifford": clifford})
        if verbose:
            print(f"  sampled code k={k}: {clifford}")

    result["sample_reports"] = reports
    return result


if __name__ == "__main__":
    from bb_core import ring_setup
    from bb_automorphisms import multiplier
    from bb_theta import identity4

    # does report_clifford=False still return exactly the old shape? (backward compatibility)
    rd43 = ring_setup((4, 3))
    one43 = RingElement(rd43.L, [(0, 0)])
    zero43 = RingElement(rd43.L, [])
    W = [[one43, zero43], [zero43, one43]]
    ix = [[3, 0], [0, 1]]
    gens_ix = [(ix, identity4(rd43.L), W)]

    plain = construct_code_theta(rd43, gens_ix)
    assert set(plain.keys()) == {"basis", "orbits", "blocks", "dim"}, "default shape must be unchanged"
    print("construct_code_theta default return keys:", sorted(plain.keys()))
    print("  pass\n")

    # the integrated question: within ONE physical-symmetry family (all members fixed exactly by
    # iota_x, same psi and Theta for every member by construction), does the single designed
    # symmetry induce the SAME logical gate on every member, or does it vary? discover_extra=False
    # (the default): report against exactly the design symmetry {iota_x}, no catalog scan, fast.
    rep = construct_code_theta(rd43, gens_ix, report_clifford=True, max_report_codes=8,
                                target_k=4, verbose=True)
    print(f"\n{len(rep['sample_reports'])} sampled k=4 codes, all fixed by the SAME (iota_x, Theta=I); "
          f"logical order of iota_x alone on each:")
    orders_seen = set()
    for r in rep["sample_reports"]:
        c = r["clifford"]
        print(f"  f1={r['f1']}, f2={r['f2']}: order={c.get('order')}, verdict={c['verdict']}")
        orders_seen.add(c.get("order"))
    print(f"\ndistinct realized orders of the SAME physical symmetry, across the family: {orders_seen}")
    assert len(rep["sample_reports"]) > 0, "expected at least one k=4 sample"
    print("  pass\n")

    # a second, cheap look: do different family members admit DIFFERENT extra scalar-tier
    # symmetries beyond the one they were designed for? (scalar tier only: no depth-1 catalog scan)
    rep2 = construct_code_theta(rd43, gens_ix, report_clifford=True, max_report_codes=4,
                                 target_k=4, discover_extra=True, discover_depth1=False, verbose=False)
    print(f"{len(rep2['sample_reports'])} sampled k=4 codes, scalar-tier symmetries actually admitted by each:")
    for r in rep2["sample_reports"]:
        print(f"  f1={r['f1']}, f2={r['f2']}: num_generators={r['clifford']['num_generators']}, "
              f"order={r['clifford'].get('order')}")
    print("  pass")
