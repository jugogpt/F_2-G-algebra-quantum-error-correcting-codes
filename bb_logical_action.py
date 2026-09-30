"""Given a verified symmetry (psi, Theta), compute the 2k x 2k F2 matrix it induces on the logical space L = LX (+) LZ."""

from __future__ import annotations
from itertools import product as iproduct
import galois
from bb_core import RingData
from bb_ring_elt import RingElement
from bb_theta import mat4_mul


def _mat4_vec_mul(Theta, gvec, L):
    out = []
    for i in range(4):
        acc = RingElement(L, [])
        for j in range(4):
            acc = acc + (Theta[i][j] * gvec[j])
        out.append(acc)
    return out


def _apply_sigma(gvec, psi_matrix, Theta, L):
    Tv = _mat4_vec_mul(Theta, gvec, L)
    return [comp.apply_automorphism(psi_matrix) for comp in Tv]


def _flatten4(gvec, rd: RingData, monomials):
    """flatten 4 RingElements into a length-4*DL F2 bit vector."""
    GF2 = galois.GF(2)
    DL = rd.DL()
    out = GF2.Zeros(4 * DL)
    for slot in range(4):
        mons = gvec[slot].mons
        for i, m in enumerate(monomials):
            if m in mons:
                out[slot * DL + i] = GF2(1)
    return out


def _shift(f: RingElement, delta, L):
    """x^delta * f, a monomial shift used only to build M's spanning set."""
    k = len(L)
    new_mons = set()
    for m in f.mons:
        nm = tuple((m[i] + delta[i]) % L[i] for i in range(k))
        if nm in new_mons:
            new_mons.discard(nm)
        else:
            new_mons.add(nm)
    return RingElement(L, new_mons)


def _m_basis(f1, f2, rd: RingData, monomials):
    """an F2-basis of M = MX (+) MZ, as flattened 4*DL vectors."""
    from bb_automorphisms import full_fold
    L = rd.L
    fold = full_fold(L)
    f2bar = f2.apply_automorphism(fold)
    f1bar = f1.apply_automorphism(fold)
    zero = RingElement(L, [])
    G = list(iproduct(*[range(li) for li in L]))

    rows = []
    for delta in G:
        gX = [_shift(f1, delta, L), _shift(f2, delta, L), zero, zero]
        rows.append(_flatten4(gX, rd, monomials))
    for delta in G:
        gZ = [zero, zero, _shift(f2bar, delta, L), _shift(f1bar, delta, L)]
        rows.append(_flatten4(gZ, rd, monomials))
    GF2 = galois.GF(2)
    M = GF2(rows)
    R = M.row_reduce()
    return [row for row in R if any(int(x) != 0 for x in row)]


def induced_logical_action(psi_matrix, Theta, f1: RingElement, f2: RingElement,
                            LX: list, LZ: list, rd: RingData):
    """the 2k x 2k F2 matrix that sigma=Phi_psi o Theta induces on L = LX (+) LZ."""
    L = rd.L
    GF2 = galois.GF(2)
    DL = rd.DL()
    monomials = [tuple(m) for m in iproduct(*[range(li) for li in L])]
    zero = RingElement(L, [])

    m_basis = _m_basis(f1, f2, rd, monomials)
    lxlz_vecs = [[P, Q, zero, zero] for (P, Q) in LX] + [[zero, zero, P, Q] for (P, Q) in LZ]
    lxlz_flat = [_flatten4(v, rd, monomials) for v in lxlz_vecs]

    combined = GF2([list(r) for r in m_basis] + [list(r) for r in lxlz_flat])
    rank_combined = sum(1 for row in combined.row_reduce() if any(int(x) != 0 for x in row))
    expected_rank = len(m_basis) + len(lxlz_flat)
    if rank_combined != expected_rank:
        raise ValueError(f"M-basis and LX/LZ-basis are not independent as expected "
                          f"(rank {rank_combined} != {expected_rank}).")

    n_m = len(m_basis)
    n_l = len(lxlz_flat)
    k2 = n_l

    out_cols = []
    for v in lxlz_vecs:
        sv = _apply_sigma(v, psi_matrix, Theta, L)
        target = _flatten4(sv, rd, monomials)

        A = GF2.Zeros((4 * DL, n_m + n_l))
        for i in range(n_m):
            A[:, i] = GF2(list(m_basis[i]))
        for i in range(n_l):
            A[:, n_m + i] = GF2(list(lxlz_flat[i]))
        aug = GF2.Zeros((4 * DL, n_m + n_l + 1))
        aug[:, :n_m + n_l] = A
        aug[:, n_m + n_l] = target
        R = aug.row_reduce()
        for row in R:
            if all(int(x) == 0 for x in row[:n_m + n_l]) and int(row[n_m + n_l]) != 0:
                raise ValueError("sigma(v) not in span(M, LX, LZ): inconsistent symmetry or logical data.")
        coeffs = [0] * (n_m + n_l)
        r = 0
        for col in range(n_m + n_l):
            if r < R.shape[0] and int(R[r, col]) == 1 and all(int(R[r, c]) == 0 for c in range(col)):
                coeffs[col] = int(R[r, n_m + n_l])
                r += 1
        out_cols.append(coeffs[n_m:])

    Mat = GF2.Zeros((k2, k2))
    for j, col in enumerate(out_cols):
        for i in range(k2):
            Mat[i, j] = GF2(col[i])
    return Mat


if __name__ == "__main__":
    from bb_core import ring_setup
    from bb_ring_elt import RingElement as RE
    from bb_logical import logical_basis, code_dimension
    from bb_crt import CRTBasis
    from bb_theta import identity4
    from bb_automorphisms import identity_matrix

    # the identity symmetry must induce the identity matrix.
    rd = ring_setup((2, 2))
    f1 = RE.from_string((2, 2), "x + y")
    f2 = RE.from_string((2, 2), "1 + x*y")
    crt = CRTBasis(rd)
    LX, LZ = logical_basis(f1, f2, rd, crt)
    k = code_dimension(f1, f2, rd)
    print(f"k={k}, dim LX={len(LX)}, dim LZ={len(LZ)}")

    Theta_id = identity4((2, 2))
    action = induced_logical_action(identity_matrix(2), Theta_id, f1, f2, LX, LZ, rd)
    print("induced action of (identity psi, identity Theta):")
    print(action)

    GF2 = galois.GF(2)
    expected = GF2.Identity(2 * k)
    ok = bool((action == expected).all())
    print(f"equals the {2*k}x{2*k} identity? {ok}")
    assert ok
    print("  pass")
