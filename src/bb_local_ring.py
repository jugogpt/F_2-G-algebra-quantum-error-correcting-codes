"""
Arithmetic inside a single local component J_c = kappa_c[u_i : i in E]/(u_i^{2^{s_i}}),
elements represented as dicts {a: coeff} with a ranging over rd.A and coeff a
GF(2^N) scalar (kappa_c sits inside GF(2^N) as a subfield, so ordinary GF(2^N)
arithmetic already computes the right answer -- no separate subfield handling
needed, per the design note in bb_ring_elt.py).

Then: the scalar matching-equation solver of Theorem 4.3 / Algorithm 3.12 --
given psi (as a k x k automorphism matrix) and f1, f2, decide whether
psi(f1, f2) = w . (f1, f2) for some unit w in R_L, i.e. whether sigma =
Phi_psi (scalar twist only, Theta = w . I) is a symmetry of the CYCLIC
submodule R_L.(f1,f2). This is exactly what Algorithm 3.12 / Sec 4.2.1's
worked examples do, and is validated below against the full ell=m=3
Forward Detection example of Sec 6.2 (including its negative result).
"""

from __future__ import annotations
from itertools import product as iproduct
from bb_core import RingData, Component
from bb_ring_elt import RingElement, local_jet


def _zero(rd: RingData) -> dict:
    return {a: rd.GF(0) for a in rd.A}


def local_add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def local_mul(a: dict, b: dict, rd: RingData) -> dict:
    """Multiply two elements of J_c using the monomial rule u^p . u^q = u^{p+q}
    if p+q is still in-bounds (no coordinate overflow), else 0."""
    bounds = [2 ** rd.s[i] for i in rd.E]
    result = _zero(rd)
    for pa, ca in a.items():
        if int(ca) == 0:
            continue
        for pb, cb in b.items():
            if int(cb) == 0:
                continue
            p = tuple(pa[i] + pb[i] for i in range(len(pa)))
            if all(p[i] < bounds[i] for i in range(len(p))):
                result[p] = result[p] + ca * cb
    return result


def local_scale(a: dict, scalar, rd: RingData) -> dict:
    return {k: v * scalar for k, v in a.items()}


def local_is_unit(a: dict, rd: RingData) -> bool:
    """A finite local ring element is a unit iff its residue (constant term)
    is nonzero -- everything else is in the maximal ideal, hence non-invertible."""
    zero_key = tuple(0 for _ in rd.E)
    return int(a[zero_key]) != 0


def local_inverse(a: dict, rd: RingData) -> dict:
    """Invert a unit a = c0*(1 - n), n nilpotent (n has zero constant term),
    via the truncated Neumann series a^{-1} = c0^{-1} * sum_t n^t (finite
    since n is nilpotent of index <= dim_kappa_c(J_c) = 2^S)."""
    assert local_is_unit(a, rd), "local_inverse called on a non-unit"
    zero_key = tuple(0 for _ in rd.E)
    c0 = a[zero_key]
    c0inv = c0 ** -1
    # n = 1 - a/c0  (as a dict; "1" is the multiplicative identity element)
    a_over_c0 = local_scale(a, c0inv, rd)
    one = _zero(rd)
    one[zero_key] = rd.GF(1)
    n = {k: one[k] - a_over_c0[k] for k in one}  # char 2: subtraction == addition

    total = dict(one)  # t=0 term
    term = dict(one)
    for _ in range(2 ** rd.S):
        term = local_mul(term, n, rd)
        if all(int(v) == 0 for v in term.values()):
            break
        total = local_add(total, term)
    return local_scale(total, c0inv, rd)


def local_exists_r(target1: dict, target2: dict, base1: dict, base2: dict, rd: RingData) -> bool:
    """Plain submodule-membership existence test: does there exist r in J_c
    (NO unit requirement -- this is Theorem 4.4's condition, not Theorem 4.3's)
    with r*base1=target1 and r*base2=target2? Used for general 2-generator
    matching (entangling Theta), as opposed to local_solve_omega's unit
    search which is specific to the single-generator Theorem 4.3 case."""
    if local_is_unit(base1, rd):
        r = local_mul(target1, local_inverse(base1, rd), rd)
        return local_mul(r, base2, rd) == target2
    if local_is_unit(base2, rd):
        r = local_mul(target2, local_inverse(base2, rd), rd)
        return local_mul(r, base1, rd) == target1

    # General case: just check consistency of the linear system (no search
    # for a unit witness -- any solution, including zero or a zero divisor,
    # certifies membership per Theorem 4.4).
    GF = rd.GF
    A = rd.A
    n = len(A)
    M = GF.Zeros((2 * n, n))
    rhs = GF.Zeros((2 * n, 1))
    for k, a in enumerate(A):
        e = _zero(rd)
        e[a] = GF(1)
        col1 = local_mul(e, base1, rd)
        col2 = local_mul(e, base2, rd)
        for i, aa in enumerate(A):
            M[i, k] = col1[aa]
            M[n + i, k] = col2[aa]
    for i, aa in enumerate(A):
        rhs[i, 0] = target1[aa]
        rhs[n + i, 0] = target2[aa]
    aug = GF.Zeros((2 * n, n + 1))
    aug[:, :n] = M
    aug[:, n] = rhs[:, 0]
    aug_rref = aug.row_reduce()
    for row in aug_rref:
        if all(int(x) == 0 for x in row[:n]) and int(row[n]) != 0:
            return False
    return True


def local_solve_omega(target1: dict, target2: dict, base1: dict, base2: dict, rd: RingData):
    """Solve, for omega in J_c, the pair of equations
        omega * base1 = target1,   omega * base2 = target2
    and return a unit solution if one exists, else None.

    Fast path (matches the paper's description "field division here" / the
    unit-inversion shortcut used throughout Sec 4.2.1): if base1 (or base2)
    is itself a unit, omega is *forced* = target1 * base1^{-1}, and we simply
    certify it against the second equation -- this covers every worked
    example in the draft (all of Sec 6.2, and the field case in general).

    General path (nilpotent case, "Jc has nilpotents... solved by linear
    algebra, with omega certified by a nonzero constant term"): set up the
    linear system in omega's |A| unknown coefficients directly from the
    multiplication table, solve, and if the particular solution isn't a
    unit, search the kernel (affine solution space) for one that is.
    """
    if local_is_unit(base1, rd):
        omega = local_mul(target1, local_inverse(base1, rd), rd)
        return omega if local_mul(omega, base2, rd) == target2 else None
    if local_is_unit(base2, rd):
        omega = local_mul(target2, local_inverse(base2, rd), rd)
        return omega if local_mul(omega, base1, rd) == target1 else None

    return _general_solve_omega(target1, target2, base1, base2, rd)


def _general_solve_omega(target1, target2, base1, base2, rd: RingData):
    GF = rd.GF
    A = rd.A
    n = len(A)
    idx = {a: i for i, a in enumerate(A)}

    # Build the (2n) x n matrix M where column k = [mul(e_k,base1) ; mul(e_k,base2)],
    # e_k the indicator basis vector of J_c at position A[k].
    M = GF.Zeros((2 * n, n))
    rhs = GF.Zeros((2 * n, 1))
    for k, a in enumerate(A):
        e = _zero(rd)
        e[a] = GF(1)
        col1 = local_mul(e, base1, rd)
        col2 = local_mul(e, base2, rd)
        for i, aa in enumerate(A):
            M[i, k] = col1[aa]
            M[n + i, k] = col2[aa]
    for i, aa in enumerate(A):
        rhs[i, 0] = target1[aa]
        rhs[n + i, 0] = target2[aa]

    # Augmented row reduction to test consistency and extract a particular
    # solution + kernel basis.
    aug = GF.Zeros((2 * n, n + 1))
    aug[:, :n] = M
    aug[:, n] = rhs[:, 0]
    aug_rref = aug.row_reduce()

    # Inconsistent iff some row is [0,...,0 | nonzero].
    for row in aug_rref:
        if all(int(x) == 0 for x in row[:n]) and int(row[n]) != 0:
            return None

    pivots = []
    particular = [GF(0)] * n
    r = 0
    for col in range(n):
        if r < aug_rref.shape[0] and int(aug_rref[r, col]) == 1 and \
           all(int(aug_rref[r, c]) == 0 for c in range(col)):
            pivots.append(col)
            particular[col] = aug_rref[r, n]
            r += 1
    free_cols = [c for c in range(n) if c not in pivots]

    # kernel basis: one vector per free column (standard back-substitution)
    kernel_basis = []
    for fc in free_cols:
        vec = [GF(0)] * n
        vec[fc] = GF(1)
        for ridx, pc in enumerate(pivots):
            vec[pc] = -aug_rref[ridx, fc] if ridx < aug_rref.shape[0] else GF(0)
        kernel_basis.append(vec)

    def to_dict(vec):
        return {A[i]: vec[i] for i in range(n)}

    zero_key = tuple(0 for _ in rd.E)
    zero_pos = idx[zero_key]

    if int(particular[zero_pos]) != 0:
        return to_dict(particular)

    # Search the affine solution space {particular + kernel span} for a unit,
    # i.e. a point with nonzero constant-term coordinate. Exhaustive over
    # kappa_c-combinations of the kernel basis; fine for the small |A|, small
    # kappa_c cases this prototype targets (documented limitation for larger
    # search spaces -- see planning discussion).
    if not kernel_basis:
        return None
    field_elems = list(GF.elements)
    for coeffs in iproduct(field_elems, repeat=len(kernel_basis)):
        cand = list(particular)
        for c, kv in zip(coeffs, kernel_basis):
            if int(c) == 0:
                continue
            for i in range(n):
                cand[i] = cand[i] + c * kv[i]
        if int(cand[zero_pos]) != 0:
            return to_dict(cand)
    return None


# ---------------------------------------------------------------------------
# Theorem 4.3 / Algorithm 3.12: scalar (cyclic, Theta = w.I) matching check
# ---------------------------------------------------------------------------

def check_cyclic_matching(f1: RingElement, f2: RingElement, Amat, rd: RingData, verbose=False):
    """Decide whether psi (given by matrix Amat, eq. 31) admits a unit
    w in R_L with psi(f1,f2) = w.(f1,f2), i.e. whether Phi_psi is a symmetry
    of the cyclic submodule R_L.(f1,f2) (Theorem 4.3, applied with generator
    g=(f1,f2) as in Sec 4.2.1's Detection algorithm).

    Returns (True, omegas_dict) on success, (False, reason) on failure --
    reason names the first component at which no unit solution exists,
    mirroring the diagnostic in Sec 6.2.6.
    """
    psi_f1 = f1.apply_automorphism(Amat)
    psi_f2 = f2.apply_automorphism(Amat)

    omegas = {}
    for c in rd.components:
        f1c = local_jet(f1, c, rd)
        f2c = local_jet(f2, c, rd)
        t1c = local_jet(psi_f1, c, rd)
        t2c = local_jet(psi_f2, c, rd)

        zero_key = tuple(0 for _ in rd.E)
        both_zero = all(int(v) == 0 for v in f1c.values()) and all(int(v) == 0 for v in f2c.values())
        if both_zero:
            # c in K0: Mc = 0, no unit constraint (Theorem 4.3) -- but the
            # transported data must also vanish exactly, else psi doesn't
            # preserve the (zero) local module.
            if not (all(int(v) == 0 for v in t1c.values()) and all(int(v) == 0 for v in t2c.values())):
                if verbose:
                    print(f"  component {c.index}: K0 mismatch (target nonzero)")
                return False, f"component {c.index}: target data nonzero where source is exactly zero"
            omegas[c.index] = None  # any unit works
            continue

        omega = local_solve_omega(t1c, t2c, f1c, f2c, rd)
        if omega is None:
            if verbose:
                print(f"  component {c.index}: no unit omega solves the matching equations")
            return False, f"component {c.index}: no unit solution"
        omegas[c.index] = omega

    return True, omegas


if __name__ == "__main__":
    from bb_core import ring_setup

    # --- Validation: §6.2 Forward Detection, L=(3,3), f1=x+y, f2=1+xy, psi=(1,2) ---
    # Expect: NOT a symmetry, failing on the length-2 orbit {c3,c4} (their
    # own conclusion: "no solution at c3 and c4").
    rd = ring_setup((3, 3))
    f1 = RingElement.from_string((3, 3), "x + y")
    f2 = RingElement.from_string((3, 3), "1 + x*y")

    # psi: x -> x, y -> y^2   i.e.  A = [[1,0],[0,2]]
    A = [[1, 0], [0, 2]]
    ok, info = check_cyclic_matching(f1, f2, A, rd, verbose=True)
    print(f"psi=(1,2) on (x+y, 1+xy): symmetry? {ok}  ({info})")
    assert ok is False
    print("  PASS (correctly rejected, matching the draft's own negative result)\n")

    # --- Sanity check: the trivial automorphism psi=identity must ALWAYS pass ---
    A_id = [[1, 0], [0, 1]]
    ok2, info2 = check_cyclic_matching(f1, f2, A_id, rd)
    print(f"identity automorphism: symmetry? {ok2}")
    assert ok2 is True
    print("  PASS\n")

    # --- Sanity check: the full fold (bar involution), ell=(3,3) ---
    # x -> x^{-1}=x^2, y -> y^{-1}=y^2.  For f1=x+y (symmetric under swap of
    # roots with inverses only if f1 itself is "fold-invariant" up to a unit);
    # test it runs without error and report the result either way.
    A_fold = [[2, 0], [0, 2]]
    ok3, info3 = check_cyclic_matching(f1, f2, A_fold, rd, verbose=True)
    print(f"full fold on (x+y, 1+xy): symmetry? {ok3}  ({info3})")
