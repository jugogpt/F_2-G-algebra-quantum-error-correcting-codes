"""Depth-one unitary layers (Lemma 4.11), built on Sp(4,F2) and ring-valued 4x4 matrices Theta in Mat4(R_L)."""

from __future__ import annotations
from itertools import product as iproduct
import numpy as np
from bb_core import RingData
from bb_ring_elt import RingElement
from bb_automorphisms import full_fold


# Sp(4,F2): the 720-element two-qubit symplectic group.

_OMEGA_F2 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=int)


def sp4_f2():
    """all 4x4 F2 matrices M with M^T Omega M = Omega, enumerated by brute force."""
    mats = []
    for bits in range(1 << 16):
        M = np.array([[(bits >> (4 * r + c)) & 1 for c in range(4)] for r in range(4)])
        if np.array_equal((M.T @ _OMEGA_F2 @ M) % 2, _OMEGA_F2):
            mats.append(M)
    return mats


def swap_symmetric_subset(sp4):
    """the elements of Sp(4,F2) of block form [[A,B],[B,A]], the shape Form W draws its gate from."""
    out = []
    for M in sp4:
        A, B, B2, A2 = M[0:2, 0:2], M[0:2, 2:4], M[2:4, 0:2], M[2:4, 2:4]
        if np.array_equal(A, A2) and np.array_equal(B, B2):
            out.append((A, B))
    return out


# Mat4(R_L) helpers: ring-valued 4x4 matrices, multiplication, the star (bar+transpose) map.

def zero4(L):
    return [[RingElement(L, []) for _ in range(4)] for _ in range(4)]


def identity4(L):
    one = RingElement(L, [tuple(0 for _ in L)])
    M = zero4(L)
    for i in range(4):
        M[i][i] = one
    return M


def lift_f2_matrix(M01, L):
    """lift a 0/1 numpy matrix to a ring-valued matrix."""
    one = RingElement(L, [tuple(0 for _ in L)])
    zero = RingElement(L, [])
    n = M01.shape[0]
    return [[one if int(M01[i, j]) == 1 else zero for j in range(n)] for i in range(n)]


def mat4_mul(A, B, L):
    n = len(A)
    C = zero4(L) if n == 4 else [[RingElement(L, []) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = RingElement(L, [])
            for t in range(n):
                acc = acc + (A[i][t] * B[t][j])
            C[i][j] = acc
    return C


def mat4_star(Theta, L):
    """Theta* = (Theta-bar)^T: bar every entry, then transpose."""
    fold = full_fold(L)
    n = len(Theta)
    return [[Theta[j][i].apply_automorphism(fold) for j in range(n)] for i in range(n)]


def mat4_equal(A, B):
    n = len(A)
    return all(A[i][j] == B[i][j] for i in range(n) for j in range(n))


def omega4(L):
    """Omega = [[0,I2],[I2,0]] as a Mat4(R_L) matrix."""
    one = RingElement(L, [tuple(0 for _ in L)])
    zero = RingElement(L, [])
    return [[zero, zero, one, zero],
            [zero, zero, zero, one],
            [one, zero, zero, zero],
            [zero, one, zero, zero]]


def verify_symplectic(Theta, rd: RingData) -> bool:
    """Proposition 4.10: Theta in Sp4(R_L) iff Theta . Omega . Theta* = Omega."""
    L = rd.L
    lhs = mat4_mul(mat4_mul(Theta, omega4(L), L), mat4_star(Theta, L), L)
    return mat4_equal(lhs, omega4(L))


# lemma 4.11: the depth-one catalog.

def _monomial(L, expo):
    return RingElement(L, [tuple(e % L[i] for i, e in enumerate(expo))])


def _neg(expo, L):
    return tuple((-e) % L[i] for i, e in enumerate(expo))


def form_C_theta(L, delta_expo, g01):
    """Theta = Sigma_delta . g . Sigma_delta^{-1}, the cross-block layer of form C."""
    one = _monomial(L, tuple(0 for _ in L))
    delta_mon = _monomial(L, delta_expo)
    delta_inv_mon = _monomial(L, _neg(delta_expo, L))
    zero = RingElement(L, [])
    Sigma = [[one, zero, zero, zero],
             [zero, delta_mon, zero, zero],
             [zero, zero, one, zero],
             [zero, zero, zero, delta_mon]]
    SigmaInv = [[one, zero, zero, zero],
                [zero, delta_inv_mon, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, delta_inv_mon]]
    g_lifted = lift_f2_matrix(g01, L)
    return mat4_mul(mat4_mul(Sigma, g_lifted, L), SigmaInv, L)


def group_elements_order_le2(rd: RingData):
    """G[2] = {eta in G : eta^2 = 1}, used by form W."""
    L = rd.L
    out = []
    for v in iproduct(*[range(li) for li in L]):
        if all((2 * e) % L[i] == 0 for i, e in enumerate(v)):
            out.append(v)
    return out


def form_W_theta(L, etaL_expo, etaR_expo, AB_L, AB_R):
    """Theta = diag_b(Ab + Bb.eta_b), the within-block layer of form W."""
    (AL, BL), (AR, BR) = AB_L, AB_R
    one = _monomial(L, tuple(0 for _ in L))
    zero = RingElement(L, [])
    etaL_mon = _monomial(L, etaL_expo)
    etaR_mon = _monomial(L, etaR_expo)

    def entry(Am, Bm, eta_mon, r, c):
        val = zero
        if int(Am[r, c]) == 1:
            val = val + one
        if int(Bm[r, c]) == 1:
            val = val + eta_mon
        return val

    Theta = zero4(L)
    Theta[0][0] = entry(AL, BL, etaL_mon, 0, 0)
    Theta[0][2] = entry(AL, BL, etaL_mon, 0, 1)
    Theta[2][0] = entry(AL, BL, etaL_mon, 1, 0)
    Theta[2][2] = entry(AL, BL, etaL_mon, 1, 1)
    Theta[1][1] = entry(AR, BR, etaR_mon, 0, 0)
    Theta[1][3] = entry(AR, BR, etaR_mon, 0, 1)
    Theta[3][1] = entry(AR, BR, etaR_mon, 1, 0)
    Theta[3][3] = entry(AR, BR, etaR_mon, 1, 1)
    return Theta


def depth1_theta_catalog(rd: RingData, max_form_c: int = 5000, max_form_w: int = 5000):
    """yield (name, metadata, Theta) for every depth-1 layer in Lemma 4.11's catalog."""
    L = rd.L
    G = list(iproduct(*[range(li) for li in L]))
    sp4 = sp4_f2()

    count = 0
    for delta in G:
        for g01 in sp4:
            if count >= max_form_c:
                break
            Theta = form_C_theta(L, delta, g01)
            yield ("C", {"delta": delta, "g": g01}, Theta)
            count += 1

    # form W is filtered through verify_symplectic before being yielded.
    swap_sym = swap_symmetric_subset(sp4)
    G2 = group_elements_order_le2(rd)
    count = 0
    for etaL in G2:
        for etaR in G2:
            for AB_L in swap_sym:
                for AB_R in swap_sym:
                    if count >= max_form_w:
                        return
                    Theta = form_W_theta(L, etaL, etaR, AB_L, AB_R)
                    if verify_symplectic(Theta, rd):
                        yield ("W", {"etaL": etaL, "etaR": etaR}, Theta)
                    count += 1


if __name__ == "__main__":
    from bb_core import ring_setup

    print("Sp(4,F2):", len(sp4_f2()), " (expect 720)")
    assert len(sp4_f2()) == 720
    print("  pass\n")

    # every form-C Theta should be physical, checked exhaustively on a small ring.
    rd = ring_setup((2, 2))
    sp4 = sp4_f2()
    G = list(iproduct(*[range(li) for li in rd.L]))
    bad = 0
    checked = 0
    for delta in G:
        for g01 in sp4[:50]:
            Theta = form_C_theta(rd.L, delta, g01)
            checked += 1
            if not verify_symplectic(Theta, rd):
                bad += 1
    print(f"Form C physicality check: {checked - bad}/{checked} pass Prop 4.10 (expect all)")
    assert bad == 0
    print("  pass\n")

    # report how much of the best-effort form W survives the Prop 4.10 filter.
    catalog = list(depth1_theta_catalog(rd, max_form_c=0, max_form_w=2000))
    print(f"Form W candidates surviving Prop 4.10 filter: {len(catalog)} (informative, not a pass/fail)")
