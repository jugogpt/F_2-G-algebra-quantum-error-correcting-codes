"""Does a code's symmetry set generate the full local Clifford group (Sp(2k,2), acting mod Pauli and phase)?

Two tiers, because |Sp(2k,2)| grows too fast for exact enumeration past k=3 or so (see the module
test for the actual numbers: k=3 is 1.45M, k=4 is 47 billion, k=12, the gross code's k, is already
larger than the number of atoms in the observable universe).

Tier A (exact, k small): literally close the generators under multiplication and count. Correct and
complete whenever the closure finishes under the size cap; this is the only tier that can prove
equality to the full group on its own.

Tier B (necessary condition, any k): Sp(2k,2) acts transitively on the 2^(2k)-1 nonzero vectors of
its natural module, so if the generated group does NOT act transitively (its orbit on a probe vector
is smaller than 2^(2k)-1), it is certainly a proper subgroup. This orbit computation is bounded by
the DEGREE (2^(2k)-1), not by the group order, so it stays cheap even when the group itself is
far too large to enumerate. It can only ever reject or stay silent: passing it is necessary but not
sufficient for generating the whole group. Proving sufficiency at large k needs a real computational
group theory tool (randomized Schreier-Sims, as used in GAP/Magma); see the module docstring tail
for what that would take and why it is not attempted here.
"""
from __future__ import annotations
import random
import numpy as np
import galois

GF2 = galois.GF(2)


def sp_order(k: int) -> int:
    """|Sp(2k,2)| = 2^(k^2) * prod_{i=1}^{k} (4^i - 1)."""
    o = 2 ** (k * k)
    for i in range(1, k + 1):
        o *= (4 ** i - 1)
    return o


def _to_rows(M, n) -> tuple:
    """an n x n GF(2) matrix -> a tuple of n ints, row i's bit j = M[i,j]."""
    arr = np.array(M, dtype=np.uint8)
    rows = []
    for i in range(n):
        r = 0
        for j in range(n):
            if arr[i, j]:
                r |= 1 << j
        rows.append(r)
    return tuple(rows)


def _identity_rows(n) -> tuple:
    return tuple(1 << i for i in range(n))


def _apply(v: int, rows: tuple, n: int) -> int:
    """v @ M, with v and the rows of M given as bit-packed ints."""
    out = 0
    vv = v
    i = 0
    while vv:
        if vv & 1:
            out ^= rows[i]
        vv >>= 1
        i += 1
    return out


def _mat_mul(rowsA: tuple, rowsB: tuple, n: int) -> tuple:
    """A @ B, given the row-bit representations of A and B."""
    return tuple(_apply(rowsA[i], rowsB, n) for i in range(n))


# tier A: exact closure, for the k where |Sp(2k,2)| is actually enumerable.

def group_closure_order(matrices: list, n: int, max_size: int = 2_000_000):
    """exact |<matrices>|, by BFS closure under multiplication, capped at max_size.

    Returns (order, complete): complete=True means the closure finished (order is exact);
    complete=False means the cap was hit first (order is only a lower bound)."""
    gens = [_to_rows(M, n) for M in matrices]
    ident = _identity_rows(n)
    seen = {ident}
    frontier = [ident]
    while frontier:
        new_frontier = []
        for M in frontier:
            for g in gens:
                P = _mat_mul(M, g, n)
                if P not in seen:
                    seen.add(P)
                    new_frontier.append(P)
                    if len(seen) > max_size:
                        return len(seen), False
        frontier = new_frontier
    return len(seen), True


# tier B: the cheap necessary condition, transitivity on the natural module.

def is_transitive_on_vectors(matrices: list, n: int, trials: int = 3, seed: int = 0):
    """does <matrices> act transitively on the 2^n - 1 nonzero vectors of F2^n?

    Runs the orbit computation from a few random nonzero start vectors (the orbit partitions
    the nonzero vectors, so any one orbit's size already upper-bounds transitivity; a few
    restarts just protect against an unlucky start inside a small orbit when there are several).
    Cost is bounded by min(orbit size, 2^n - 1), not by the group order.
    """
    rng = random.Random(seed)
    gens = [_to_rows(M, n) for M in matrices]
    full = (1 << n) - 1
    best_orbit = 0
    for _ in range(trials):
        v0 = rng.randint(1, full)
        orbit = {v0}
        frontier = [v0]
        while frontier:
            new_frontier = []
            for v in frontier:
                for g in gens:
                    w = _apply(v, g, n)
                    if w not in orbit:
                        orbit.add(w)
                        new_frontier.append(w)
            frontier = new_frontier
            if len(orbit) == full + 1:  # includes 0; nonzero orbit can be at most `full`
                break
        best_orbit = max(best_orbit, len(orbit))
        if best_orbit > full:
            break
    transitive = (best_orbit - (1 if 0 in {0} else 0)) >= full  # orbit of a nonzero start never contains 0
    return (best_orbit >= full), best_orbit, full


# the user-facing test.

def generates_full_clifford(matrices: list, k: int, exact_cap: int = 2_000_000,
                             small_order_cap: int = 10_000, verbose: bool = False):
    """classify a generator set's closure against Sp(2k,2).

    Three tiers, cheapest first:
      Tier A-small (always attempted when small_order_cap > 0): a closure attempt capped very low.
        Catches the common case of a FEW generators with a genuinely small generated order even
        when k is large and the full group is far too big to enumerate -- e.g. a single generator
        of order 2 is found in a couple of steps regardless of k. Set small_order_cap=0 to skip
        this and go straight to Tier B, for maximum throughput on a large sweep.
      Tier A-full (only attempted when |Sp(2k,2)| <= exact_cap): the complete closure, for the k
        where exact proof is actually tractable (k <= 3 or so; see the module docstring).
      Tier B (necessary condition, any k): transitivity on the natural module.

    Returns a dict with:
      verdict: "full" (proven), "not_full" (proven proper subgroup, exactly or via Tier B), or
               "unknown" (passed the necessary condition but k is too large to prove sufficiency)
      order: exact order whenever a closure attempt (Tier A-small or Tier A-full) completed, else None
      target_order: |Sp(2k,2)|
      transitive: whether Tier B's necessary condition passed (True whenever verdict != "not_full")
    """
    n = 2 * k
    target = sp_order(k)

    if small_order_cap > 0:
        order, complete = group_closure_order(matrices, n, max_size=small_order_cap)
        if complete:
            verdict = "full" if order == target else "not_full"
            if verbose:
                print(f"  Tier A-small (exact, cheap): order {order:,} of {target:,} -> {verdict}")
            return {"verdict": verdict, "order": order, "target_order": target, "transitive": order == target}

    if target <= exact_cap:
        order, complete = group_closure_order(matrices, n, max_size=exact_cap)
        if complete:
            verdict = "full" if order == target else "not_full"
            if verbose:
                print(f"  Tier A-full (exact): order {order:,} of {target:,} -> {verdict}")
            return {"verdict": verdict, "order": order, "target_order": target, "transitive": order == target}

    transitive, orbit, full = is_transitive_on_vectors(matrices, n)
    if verbose:
        print(f"  Tier B (necessary condition): orbit {orbit:,} of {full:,} nonzero vectors -> "
              f"{'passes' if transitive else 'FAILS: definitely not the full group'}")
    if not transitive:
        return {"verdict": "not_full", "order": None, "target_order": target, "transitive": False}
    return {"verdict": "unknown", "order": None, "target_order": target, "transitive": True}


if __name__ == "__main__":
    import itertools

    # ---- Tier A validation: k=1 and k=2 against orders we already trust independently. ----
    H1 = GF2([[0, 1], [1, 0]]); S1 = GF2([[1, 0], [1, 1]])
    r1 = generates_full_clifford([H1, S1], k=1, verbose=True)
    print("k=1, <H,S>:", r1)
    assert r1 == {"verdict": "full", "order": 6, "target_order": 6, "transitive": True}
    print("  pass\n")

    CNOT = GF2([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]])
    CZ   = GF2([[1,0,0,0],[0,1,0,0],[0,1,1,0],[1,0,0,1]])
    H1_ = GF2([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    H2_ = GF2([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    S1_ = GF2([[1,0,0,0],[0,1,0,0],[1,0,1,0],[0,0,0,1]])
    S2_ = GF2([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,1]])
    r2 = generates_full_clifford([CNOT, CZ, H1_, H2_, S1_, S2_], k=2, verbose=True)
    print("k=2, <CNOT,CZ,H1,H2,S1,S2>:", r2)
    assert r2["verdict"] == "full" and r2["order"] == 720
    print("  pass\n")

    # negative control: a generator set that provably does NOT generate the full group at k=2
    # (just CNOT and CZ, with no local gates -- these preserve the X/Z split, can't be everything).
    r2_bad = generates_full_clifford([CNOT, CZ], k=2, verbose=True)
    print("k=2, <CNOT,CZ> only:", r2_bad)
    assert r2_bad["verdict"] == "not_full"
    print("  pass\n")

    # Tier A at the practical ceiling, k=3 (|Sp(6,2)| = 1,451,520, right at the cap):
    # build a standard "H+S on every qubit, CNOT in a ring" generating set and confirm it's full.
    def embed(mat2x2_blocks, k, qubit_map):
        """place a local 2x2-per-qubit block pattern into a 2k x 2k matrix (X,Z interleaved per qubit at positions i, k+i)."""
        M = np.eye(2 * k, dtype=np.uint8)
        for (qi, qj), block in qubit_map.items():
            idx = [qi, k + qi] if qj is None else [qi, k + qi, qj, k + qj]
            for a, ia in enumerate(idx):
                for b, ib in enumerate(idx):
                    M[ia, ib] = block[a, b]
        return GF2(M)

    def local_H(k, q):
        M = np.eye(2 * k, dtype=np.uint8)
        M[q, q] = 0; M[k + q, k + q] = 0; M[q, k + q] = 1; M[k + q, q] = 1
        return GF2(M)

    def local_S(k, q):
        M = np.eye(2 * k, dtype=np.uint8)
        M[k + q, q] = 1
        return GF2(M)

    def cnot(k, ctrl, tgt):
        M = np.eye(2 * k, dtype=np.uint8)
        M[tgt, ctrl] = 1          # X_ctrl -> X_ctrl X_tgt
        M[k + ctrl, k + tgt] = 1  # Z_tgt  -> Z_ctrl Z_tgt
        return GF2(M)

    k3 = 3
    gens3 = [local_H(k3, q) for q in range(k3)] + [local_S(k3, q) for q in range(k3)] + \
            [cnot(k3, q, (q + 1) % k3) for q in range(k3)]
    t0 = __import__("time").time()
    r3 = generates_full_clifford(gens3, k=k3, verbose=True)
    print(f"k=3, standard H+S+ring-CNOT generators: {r3}  ({__import__('time').time()-t0:.1f}s)")
    assert r3["verdict"] == "full" and r3["order"] == sp_order(3)
    print("  pass\n")

    # Tier B at a k where Tier A is hopeless (k=6, |Sp(12,2)| ~ 2e23): same generator recipe,
    # same claim (it's the standard universal Clifford generating set), now only Tier B runs.
    k6 = 6
    gens6 = [local_H(k6, q) for q in range(k6)] + [local_S(k6, q) for q in range(k6)] + \
            [cnot(k6, q, (q + 1) % k6) for q in range(k6)]
    t0 = __import__("time").time()
    r6 = generates_full_clifford(gens6, k=k6, verbose=True)
    print(f"k=6, same recipe: {r6}  ({__import__('time').time()-t0:.1f}s)")
    assert r6["verdict"] == "unknown" and r6["transitive"] is True
    print("  (as expected: Tier A can't run at this size, Tier B can't prove sufficiency, "
          "but it correctly fails to reject a generating set we know is complete)\n")

    # negative control at k=6: drop the CNOTs, keep only local H,S -- this is a direct sum of k
    # copies of Sp(2,2), which can never mix qubits, so it must fail transitivity.
    gens6_bad = [local_H(k6, q) for q in range(k6)] + [local_S(k6, q) for q in range(k6)]
    r6_bad = generates_full_clifford(gens6_bad, k=k6, verbose=True)
    print(f"k=6, local gates only (no CNOT): {r6_bad}")
    assert r6_bad["verdict"] == "not_full"
    print("  pass")


# wired to the toolkit: given a code and its admitted symmetries, report the Clifford-generation verdict.

def symmetry_clifford_report(f1, f2, rd, symmetries: list, crt=None, exact_cap: int = 2_000_000,
                              small_order_cap: int = 10_000, verbose: bool = False):
    """symmetries: a list of (psi_matrix, Theta) pairs already verified against the code (e.g. from
    find_symmetries, or the (psi,Theta) half of a construct_code_theta gens list). Theta=None is
    read as the identity (the scalar-twist tier)."""
    from bb_logical import code_dimension, logical_basis
    from bb_logical_action import induced_logical_action
    from bb_crt import CRTBasis
    from bb_theta import identity4

    k = code_dimension(f1, f2, rd)
    if k == 0:
        return {"verdict": "no_logical_qubits", "k": 0}
    if crt is None:
        crt = CRTBasis(rd)
    LX, LZ = logical_basis(f1, f2, rd, crt)

    matrices = []
    for psi, Theta in symmetries:
        Theta = Theta if Theta is not None else identity4(rd.L)
        matrices.append(induced_logical_action(psi, Theta, f1, f2, LX, LZ, rd))

    result = generates_full_clifford(matrices, k, exact_cap=exact_cap, small_order_cap=small_order_cap, verbose=verbose)
    result["k"] = k
    result["num_generators"] = len(symmetries)
    return result


if __name__ == "__main__":
    print("\n--- grounding on real codes already built in this project ---\n")
    from bb_core import ring_setup
    from bb_ring_elt import RingElement as RE
    from bb_automorphisms import identity_matrix, multiplier, full_fold, automorphism_group, preserves_local_ideal_structure
    from bb_theta import form_C_theta
    import numpy as _np

    # Example 1's code on R_{4,3}: find_symmetries found only {id, iota_x}.
    rd43 = ring_setup((4, 3))
    f1 = RE.from_string((4, 3), "1 + x*y")
    f2 = RE.from_string((4, 3), "y^2 + x*y + x^2 + x^3")
    ix = [[3, 0], [0, 1]]
    rep = symmetry_clifford_report(f1, f2, rd43, [(identity_matrix(2), None), (ix, None)], verbose=True)
    print("Example 1 code [[24,2,.]], symmetries {id, iota_x}:", rep)
    print("  (iota_x acted as the identity logical gate in Example 6, so this *should* fail to be full)\n")

    # the gross code on R_{3,4,6}: Example 3b found {id with Theta=I, full-fold with Theta=SWAP.H1.H2}.
    rd346 = ring_setup((3, 4, 6))
    g1 = RE.from_string((3, 4, 6), "y^3 + z + z^2")
    g2 = RE.from_string((3, 4, 6), "z^3 + x*y + x^2*y^2")
    SWAP = _np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    H1 = _np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    H2 = _np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    swap_h1_h2 = (SWAP @ H1 @ H2) % 2
    fold = full_fold((3, 4, 6))
    Theta_fold = form_C_theta((3, 4, 6), (0, 0, 0), swap_h1_h2)
    rep_gross = symmetry_clifford_report(g1, g2, rd346, [(identity_matrix(3), None), (fold, Theta_fold)], verbose=True)
    print("Gross code [[144,12,.]], symmetries {id, full-fold+SWAP.H1.H2}:", rep_gross)
    print("  (only 2 generators on k=12 logical qubits -- correctly far too few to be the full group)")
