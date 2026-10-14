"""
Gamma = Aut(G), G = Z_{l1} x ... x Z_{lk}, represented as k x k integer
matrices A=(a_ij) with psi(x_j) = prod_i x_i^{a_ij} (eq. 31). Well-definedness
(eq. 32) restricts a_ij to the gcd(l_i,l_j) values that are multiples of
l_i/gcd(l_i,l_j) mod l_i; composition is matrix multiplication (verified
against bb_ring_elt.apply_automorphism's convention in the module docstring
below), and invertibility is checked directly by testing that the induced
map is a bijection of the finite group G (fine at the problem sizes this
prototype targets -- see planning discussion for the scaling caveat).

Also: Example 2's named bivariate/general catalog (multipliers, partial
folds, the full fold/antipode, shears, variable permutations), used both as
convenient constructors and as a cross-check on automorphism_group's count
(|Aut(Z3 x Z3)| = |GL(2,F3)| = 48, |Aut(Z2 x Z2)| = |S3| = 6).

And Proposition 4.13: a necessary (not sufficient) filter for a symmetry
sigma=(psi,Theta) to exist -- psi must map the unordered ideal pair
{I_c, Ibar_c} to itself (as an unordered pair) at every component c, where
I = <f1,f2> and Ibar = <f1bar,f2bar> (f1bar/f2bar computed via the full-fold
automorphism, reusing apply_automorphism rather than a separate bar-involution
routine, per the "apply globally then take local jet" design decision).
"""

from __future__ import annotations
from itertools import product as iproduct
from math import gcd
from bb_core import RingData
from bb_ring_elt import RingElement, local_jet


# ---------------------------------------------------------------------------
# Example 2's named catalog
# ---------------------------------------------------------------------------

def identity_matrix(k: int):
    return [[1 if i == j else 0 for j in range(k)] for i in range(k)]


def multiplier(L: tuple, js: tuple):
    """psi: x_i -> x_i^{j_i}, j_i a unit mod l_i (Example 2, item 1)."""
    k = len(L)
    A = [[0] * k for _ in range(k)]
    for i in range(k):
        A[i][i] = js[i] % L[i]
    return A


def partial_fold(L: tuple, i: int):
    """psi: x_i -> x_i^{-1}, all other x_j fixed (Example 2, item 2)."""
    js = tuple(-1 if t == i else 1 for t in range(len(L)))
    return multiplier(L, js)


def full_fold(L: tuple):
    """The antipode / bar involution: x_i -> x_i^{-1} for every i (Example 2, item 3)."""
    return multiplier(L, tuple(-1 for _ in L))


def shear(L: tuple, i: int, j: int, sfac: int):
    """psi: x_i -> x_i * x_j^{sfac * l_i/d}, all other x_t fixed, d=gcd(l_i,l_j)
    (Example 2, item 4; off-diagonal entry a_ij)."""
    k = len(L)
    A = identity_matrix(k)
    d = gcd(L[i], L[j])
    A[i][j] = (sfac * (L[i] // d)) % L[i]
    return A


def swap_variables(L: tuple, i: int, j: int):
    """psi: x_i <-> x_j, requires l_i == l_j (Example 2, item 5)."""
    assert L[i] == L[j]
    k = len(L)
    A = identity_matrix(k)
    A[i][i], A[j][j] = 0, 0
    A[i][j], A[j][i] = 1, 1
    return A


def matmul_mod(A, B, L):
    """(A . B) with row i reduced mod l_i -- the composition rule of eq. (31)/(32)."""
    k = len(L)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % L[i] for j in range(k)] for i in range(k)]


# ---------------------------------------------------------------------------
# Full enumeration of Gamma = Aut(G)
# ---------------------------------------------------------------------------

def _candidate_entries(li: int, lj: int):
    """The gcd(li,lj) valid values of a_ij (eq. 32)."""
    g = gcd(li, lj)
    step = li // g
    return [(t * step) % li for t in range(g)]


def _is_bijection(A, L) -> bool:
    """Direct bijectivity check of v -> (A v mod L) on G = prod Z_{l_i}."""
    k = len(L)
    seen = set()
    for v in iproduct(*[range(li) for li in L]):
        img = tuple(sum(A[i][t] * v[t] for t in range(k)) % L[i] for i in range(k))
        if img in seen:
            return False
        seen.add(img)
    return True


def automorphism_group(rd: RingData, max_raw_candidates: int = 500_000):
    """Enumerate Gamma = Aut(G) as a list of k x k matrices. Builds every
    eq.-(32)-valid matrix, then filters for bijectivity. Raises if the raw
    (pre-filter) candidate count would exceed max_raw_candidates, since this
    prototype enumerates rather than constructs Aut(G) structurally."""
    L = rd.L
    k = rd.k
    per_entry = {(i, j): _candidate_entries(L[i], L[j]) for i in range(k) for j in range(k)}
    raw_count = 1
    for v in per_entry.values():
        raw_count *= len(v)
    if raw_count > max_raw_candidates:
        raise ValueError(f"Candidate space too large ({raw_count}); restrict via a named "
                          f"subgroup or raise max_raw_candidates explicitly.")

    entries_order = [(i, j) for i in range(k) for j in range(k)]
    gamma = []
    for combo in iproduct(*[per_entry[e] for e in entries_order]):
        A = [[0] * k for _ in range(k)]
        for (i, j), val in zip(entries_order, combo):
            A[i][j] = val
        if _is_bijection(A, L):
            gamma.append(A)
    return gamma


# ---------------------------------------------------------------------------
# Proposition 4.13: local ideal preservation
# ---------------------------------------------------------------------------

def local_ideal_basis(gens: list, rd: RingData):
    """Row-reduced kappa_c-basis of the ideal generated (inside J_c) by the
    given jets, via u^a . g for a in rd.A and each generator g (Algorithm 1
    Step 3's construction)."""
    from bb_local_ring import local_mul, _zero
    GF = rd.GF
    rows = []
    for g in gens:
        for a in rd.A:
            e = _zero(rd)
            e[a] = GF(1)
            rows.append(local_mul(e, g, rd))
    return _row_reduce_ideal(rows, rd)


def _row_reduce_ideal(rows: list, rd: RingData):
    """RREF of a list of J_c elements (viewed as vectors over rd.A), zero
    rows dropped; returns the RREF matrix rows as a canonical subspace
    representative (two ideals are equal iff their RREFs, in this fixed
    basis order, are identical)."""
    GF = rd.GF
    A = rd.A
    n = len(A)
    if not rows:
        return GF.Zeros((0, n))
    M = GF.Zeros((len(rows), n))
    for r, g in enumerate(rows):
        for c, a in enumerate(A):
            M[r, c] = g[a]
    R = M.row_reduce()
    nonzero = [i for i in range(R.shape[0]) if any(int(x) != 0 for x in R[i])]
    return R[nonzero, :] if nonzero else GF.Zeros((0, n))


def ideal_equal(basis1, basis2) -> bool:
    """Two RREF-canonicalized ideal bases (from local_ideal_basis) are equal
    iff they're literally the same matrix."""
    if basis1.shape != basis2.shape:
        return False
    return bool((basis1 == basis2).all())


def preserves_local_ideal_structure(f1: RingElement, f2: RingElement, Amat, rd: RingData) -> bool:
    """Proposition 4.13: necessary condition for psi=Amat to admit ANY Theta'
    with sigma(M)=M. Checks {I_c, Ibar_c} = {psi(I)_c, psi(Ibar)_c} as an
    unordered pair, at every component."""
    fold = full_fold(rd.L)
    f1bar = f1.apply_automorphism(fold)
    f2bar = f2.apply_automorphism(fold)
    psi_f1 = f1.apply_automorphism(Amat)
    psi_f2 = f2.apply_automorphism(Amat)
    psi_f1bar = f1bar.apply_automorphism(Amat)
    psi_f2bar = f2bar.apply_automorphism(Amat)

    for c in rd.components:
        I_c = local_ideal_basis([local_jet(f1, c, rd), local_jet(f2, c, rd)], rd)
        Ibar_c = local_ideal_basis([local_jet(f1bar, c, rd), local_jet(f2bar, c, rd)], rd)
        psiI_c = local_ideal_basis([local_jet(psi_f1, c, rd), local_jet(psi_f2, c, rd)], rd)
        psiIbar_c = local_ideal_basis([local_jet(psi_f1bar, c, rd), local_jet(psi_f2bar, c, rd)], rd)

        matches = (ideal_equal(I_c, psiI_c) and ideal_equal(Ibar_c, psiIbar_c)) or \
                  (ideal_equal(I_c, psiIbar_c) and ideal_equal(Ibar_c, psiI_c))
        if not matches:
            return False
    return True


if __name__ == "__main__":
    from bb_core import ring_setup
    from bb_local_ring import check_cyclic_matching

    # --- Validation 1: |Aut(Z3 x Z3)| = |GL(2,F3)| = 48 ---
    rd33 = ring_setup((3, 3))
    gamma33 = automorphism_group(rd33)
    print(f"|Aut(Z3 x Z3)| = {len(gamma33)}  (expect 48)")
    assert len(gamma33) == 48
    print("  PASS\n")

    # --- Validation 2: |Aut(Z2 x Z2)| = |S3| = 6 ---
    rd22 = ring_setup((2, 2))
    gamma22 = automorphism_group(rd22)
    print(f"|Aut(Z2 x Z2)| = {len(gamma22)}  (expect 6)")
    assert len(gamma22) == 6
    print("  PASS\n")

    # --- Validation 3: named catalog entries actually land inside Gamma ---
    A_mult = multiplier((3, 3), (1, 2))
    A_fold = full_fold((3, 3))
    A_swap = swap_variables((3, 3), 0, 1)
    for name, A in [("multiplier(1,2)", A_mult), ("full_fold", A_fold), ("swap", A_swap)]:
        ok = any(A == G for G in gamma33)
        print(f"  {name} in Gamma(3,3)? {ok}")
        assert ok
    print("  PASS\n")

    # --- Validation 4 (consistency): every psi that PASSES the full scalar
    # matching test (Tier 1) must also pass Prop 4.13 (necessary condition) ---
    f1 = RingElement.from_string((3, 3), "x + y")
    f2 = RingElement.from_string((3, 3), "1 + x*y")
    A_id = identity_matrix(2)

    for name, A in [("identity", A_id), ("full_fold", A_fold)]:
        matches, _ = check_cyclic_matching(f1, f2, A, rd33)
        p413 = preserves_local_ideal_structure(f1, f2, A, rd33)
        print(f"  {name}: matching={matches}, Prop4.13={p413}")
        assert matches is True and p413 is True, "Prop 4.13 must not reject a genuine symmetry"
    print("  PASS (Prop 4.13 never rejects a real symmetry)\n")

    # --- Informative (not a pass/fail against the draft): psi=(1,2), which
    # the draft shows FAILS full matching at the {c3,c4} orbit -- report
    # what the cheaper Prop 4.13 filter says about it.
    A_12 = multiplier((3, 3), (1, 2))
    p413_12 = preserves_local_ideal_structure(f1, f2, A_12, rd33)
    print(f"psi=(1,2) [known to fail full matching]: Prop4.13 says preserves ideal structure = {p413_12}")
