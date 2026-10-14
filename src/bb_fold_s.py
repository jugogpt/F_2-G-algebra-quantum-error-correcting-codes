"""s-type fold gates: sigma(x|z) = (x | z + P x) with P the permutation matrix of an involutive pairing pi of the physical qubits.

this realizes s on fixed points of pi and cz between paired qubits (breuckmann-burton fold-transversal s). it is not psi-semilinear for any
single psi, so it lies outside the class Phi_psi o Theta; pi = (translation by delta) o psi, optionally composed with the block swap.
sigma preserves M iff its z-side image of gX, namely (0,0,pi(f1),pi(f2)) (or swapped), lies in M.
"""
from __future__ import annotations
from itertools import product as iproduct
from bb_ring_elt import RingElement
from bb_automorphisms import automorphism_group
from bb_symcheck import SymChecker


def _mono(L, e):
    return RingElement(L, [tuple(x % L[i] for i, x in enumerate(e))])


def _compose_is_identity(psi, L):
    k = len(L)
    return all(sum(psi[i][t] * psi[t][j] for t in range(k)) % L[i] == (1 if i == j else 0) % L[i] for i in range(k) for j in range(k))


def _apply_exp(psi, e, L):
    k = len(L)
    return tuple(sum(psi[i][t] * e[t] for t in range(k)) % L[i] for i in range(k))


def fold_s_variants(rd):
    """(name, psi, delta, swap) for every involutive pairing pi(g)=psi(g)+delta, optionally with the block swap."""
    L = rd.L
    G = list(iproduct(*[range(l) for l in L]))
    out = []
    for psi in automorphism_group(rd):
        if not _compose_is_identity(psi, L):
            continue
        for delta in G:
            pd = _apply_exp(psi, delta, L)
            if all((pd[i] + delta[i]) % L[i] == 0 for i in range(len(L))):
                for swap in (False, True):
                    out.append((f"fold-S psi={tuple(map(tuple, psi))} delta={delta} swap={swap}", psi, delta, swap))
    return out


def sigma_function(psi, delta, swap, L):
    """v (list of four RingElements: XL, XR, ZL, ZR) -> sigma(v)."""
    m = _mono(L, delta)
    pi = lambda a: m * a.apply_automorphism(psi)
    def sigma(v):
        a, b = (v[1], v[0]) if swap else (v[0], v[1])
        return [v[0], v[1], v[2] + pi(a), v[3] + pi(b)]
    return sigma


def preserves(f1, f2, variant, rd, checker=None):
    from bb_automorphisms import full_fold
    _, psi, delta, swap = variant
    L = rd.L
    chk = checker or SymChecker(f1, f2, rd)
    sig = sigma_function(psi, delta, swap, L)
    zero = RingElement(L, [])
    fold = full_fold(L)
    gX = [f1, f2, zero, zero]
    gZ = [zero, zero, f2.apply_automorphism(fold), f1.apply_automorphism(fold)]
    for g in (gX, gZ):
        if not chk.contains(chk._int([chk._arr(c) for c in sig(g)])):
            return False
    return True


def find_fold_s(f1, f2, rd, variants=None):
    chk = SymChecker(f1, f2, rd)
    return [v for v in (variants or fold_s_variants(rd)) if preserves(f1, f2, v, rd, chk)]
