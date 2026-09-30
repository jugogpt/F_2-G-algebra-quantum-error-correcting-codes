"""Arithmetic inside a single local component J_c, and the scalar matching-equation solver."""

from __future__ import annotations
from itertools import product as iproduct
from bb_core import RingData, Component
from bb_ring_elt import RingElement, local_jet


def _zero(rd: RingData) -> dict:
    return {a: rd.GF(0) for a in rd.A}


def local_add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def local_mul(a: dict, b: dict, rd: RingData) -> dict:
    """multiply two elements of J_c by the monomial rule u^p . u^q = u^{p+q}, or 0 on overflow."""
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
    """an element of a finite local ring is a unit iff its constant term is nonzero."""
    zero_key = tuple(0 for _ in rd.E)
    return int(a[zero_key]) != 0


def local_inverse(a: dict, rd: RingData) -> dict:
    """invert a unit via the truncated Neumann series over its nilpotent part."""
    assert local_is_unit(a, rd), "local_inverse called on a non-unit"
    zero_key = tuple(0 for _ in rd.E)
    c0 = a[zero_key]
    c0inv = c0 ** -1
    a_over_c0 = local_scale(a, c0inv, rd)
    one = _zero(rd)
    one[zero_key] = rd.GF(1)
    n = {k: one[k] - a_over_c0[k] for k in one}

    total = dict(one)
    term = dict(one)
    for _ in range(2 ** rd.S):
        term = local_mul(term, n, rd)
        if all(int(v) == 0 for v in term.values()):
            break
        total = local_add(total, term)
    return local_scale(total, c0inv, rd)


def local_exists_r(target1: dict, target2: dict, base1: dict, base2: dict, rd: RingData) -> bool:
    """submodule-membership test: does r*base1=target1, r*base2=target2 have a solution with no unit requirement?"""
    if local_is_unit(base1, rd):
        r = local_mul(target1, local_inverse(base1, rd), rd)
        return local_mul(r, base2, rd) == target2
    if local_is_unit(base2, rd):
        r = local_mul(target2, local_inverse(base2, rd), rd)
        return local_mul(r, base1, rd) == target1

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
    """solve omega*base1=target1, omega*base2=target2 for a unit omega in J_c, or return None."""
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

    # column k of M is [mul(e_k,base1) ; mul(e_k,base2)] for the indicator basis vector e_k.
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

    # search the affine solution space for a point with nonzero constant term.
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


# theorem 4.3 / algorithm 3.12: scalar (cyclic, Theta = w.I) matching check.

def check_cyclic_matching(f1: RingElement, f2: RingElement, Amat, rd: RingData, verbose=False):
    """decide whether psi (matrix Amat) admits a unit w with psi(f1,f2)=w.(f1,f2), returning (ok, omegas or reason)."""
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
            # c is purely logical: no unit constraint, but the transported data must also vanish exactly.
            if not (all(int(v) == 0 for v in t1c.values()) and all(int(v) == 0 for v in t2c.values())):
                if verbose:
                    print(f"  component {c.index}: mismatch, target nonzero where source is exactly zero")
                return False, f"component {c.index}: target data nonzero where source is exactly zero"
            omegas[c.index] = None
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

    # L=(3,3), f1=x+y, f2=1+xy, psi=(1,2): expect rejection on the length-2 orbit {c3,c4}.
    rd = ring_setup((3, 3))
    f1 = RingElement.from_string((3, 3), "x + y")
    f2 = RingElement.from_string((3, 3), "1 + x*y")

    A = [[1, 0], [0, 2]]
    ok, info = check_cyclic_matching(f1, f2, A, rd, verbose=True)
    print(f"psi=(1,2) on (x+y, 1+xy): symmetry? {ok}  ({info})")
    assert ok is False
    print("  pass\n")

    # sanity check: the identity automorphism must always pass.
    A_id = [[1, 0], [0, 1]]
    ok2, info2 = check_cyclic_matching(f1, f2, A_id, rd)
    print(f"identity automorphism: symmetry? {ok2}")
    assert ok2 is True
    print("  pass\n")

    # sanity check: the full fold on L=(3,3), run without asserting either outcome.
    A_fold = [[2, 0], [0, 2]]
    ok3, info3 = check_cyclic_matching(f1, f2, A_fold, rd, verbose=True)
    print(f"full fold on (x+y, 1+xy): symmetry? {ok3}  ({info3})")
