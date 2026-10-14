"""fast, uncapped symmetry detection: sigma = psi o Theta preserves M iff sigma(gX), sigma(gZ) lie in M, tested on bit vectors.

replaces the capped ring-matrix path in bb_script_a.find_symmetries for the depth-one catalog (all shifts delta, all 720 g, all form W).
a layer Theta is a sparse list of terms (i, j, e) meaning slot j is shifted by monomial e into slot i.
"""
from __future__ import annotations
import numpy as np
from itertools import product as iproduct
from bb_core import RingData
from bb_ring_elt import RingElement
from bb_theta import (sp4_f2, swap_symmetric_subset, group_elements_order_le2, form_W_theta, verify_symplectic,
                      identity4)
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure, full_fold
from bb_theta_inverse import _mons


class SymChecker:
    """bit-level membership engine for one fixed pair (f1,f2)."""

    def __init__(self, f1: RingElement, f2: RingElement, rd: RingData):
        self.rd, self.L = rd, rd.L
        self.mons = _mons(rd.L)
        self.D = len(self.mons)
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self._shift = {}
        fold = full_fold(rd.L)
        zero = np.zeros(self.D, dtype=np.uint8)
        self.gX = [self._arr(f1), self._arr(f2), zero, zero]
        self.gZ = [zero, zero, self._arr(f2.apply_automorphism(fold)), self._arr(f1.apply_automorphism(fold))]
        self.basis = {}
        for g in self.gX, self.gZ:
            for e in self.mons:
                self._insert(self._int(self._shift_vec(g, e)))

    def _arr(self, f):
        a = np.zeros(self.D, dtype=np.uint8)
        for m in f.mons:
            a[self.idx[m]] = 1
        return a

    def shift_table(self, e):
        t = self._shift.get(e)
        if t is None:
            t = np.array([self.idx[tuple((m[i] + e[i]) % self.L[i] for i in range(len(self.L)))] for m in self.mons])
            self._shift[e] = t
        return t

    def _shift_vec(self, vec4, e):
        t = self.shift_table(e)
        out = []
        for a in vec4:
            b = np.empty_like(a)
            b[t] = a
            out.append(b)
        return out

    def _int(self, vec4):
        return int.from_bytes(np.packbits(np.concatenate(vec4)).tobytes(), "big")

    def _insert(self, v):
        while v:
            h = v.bit_length() - 1
            if h in self.basis:
                v ^= self.basis[h]
            else:
                self.basis[h] = v
                return

    def contains(self, v):
        while v:
            h = v.bit_length() - 1
            b = self.basis.get(h)
            if b is None:
                return False
            v ^= b
        return True

    def psi_perm(self, Amat):
        k = len(self.L)
        return np.array([self.idx[tuple(sum(Amat[i][t] * m[t] for t in range(k)) % self.L[i] for i in range(k))]
                         for m in self.mons])

    def apply(self, terms, perm, vec4):
        out = [np.zeros(self.D, dtype=np.uint8) for _ in range(4)]
        for i, j, e in terms:
            src = vec4[j]
            if not src.any():
                continue
            t = self.shift_table(e)
            tmp = np.empty_like(src)
            tmp[t] = src
            out[i] ^= tmp
        res = []
        for a in out:
            b = np.empty_like(a)
            b[perm] = a
            res.append(b)
        return res

    def preserves(self, terms, perm) -> bool:
        return (self.contains(self._int(self.apply(terms, perm, self.gX)))
                and self.contains(self._int(self.apply(terms, perm, self.gZ))))


def _neg(e, L):
    return tuple((-x) % L[i] for i, x in enumerate(e))


def form_c_terms(L, delta, g01):
    """terms of Sigma_delta g Sigma_delta^-1 with Sigma_delta = diag(1,x^delta,1,x^delta)."""
    s = [tuple(0 for _ in L), tuple(delta), tuple(0 for _ in L), tuple(delta)]
    out = []
    for i in range(4):
        for j in range(4):
            if int(g01[i, j]):
                out.append((i, j, tuple((s[i][t] - s[j][t]) % L[t] for t in range(len(L)))))
    return tuple(sorted(out))


def theta_to_terms(Theta):
    return tuple(sorted((i, j, m) for i in range(4) for j in range(4) for m in Theta[i][j].mons))


def terms_to_theta(terms, L):
    from bb_theta import zero4
    T = zero4(L)
    for i, j, e in terms:
        T[i][j] = T[i][j] + RingElement(L, [e])
    return T


_CATALOG_CACHE = {}


def full_depth1_catalog(rd: RingData):
    """deduplicated (name, meta, terms) for every form C layer (all delta, all of Sp(4,F2)) and every physical form W layer."""
    key = rd.L
    if key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]
    L = rd.L
    seen, out = set(), []
    sp4 = sp4_f2()
    for delta in iproduct(*[range(l) for l in L]):
        for g in sp4:
            t = form_c_terms(L, delta, g)
            if t not in seen:
                seen.add(t)
                out.append(("C", {"delta": delta, "g": g}, t))
    sw = swap_symmetric_subset(sp4)
    G2 = group_elements_order_le2(rd)
    for eL in G2:
        for eR in G2:
            for ABL in sw:
                for ABR in sw:
                    Th = form_W_theta(L, eL, eR, ABL, ABR)
                    t = theta_to_terms(Th)
                    if t in seen or not verify_symplectic(Th, rd):
                        continue
                    seen.add(t)
                    out.append(("W", {"etaL": eL, "etaR": eR}, t))
    _CATALOG_CACHE[key] = out
    return out


def find_symmetries_fast(f1, f2, rd, psi_candidates=None, include_depth1=True, survivors=None, verbose=False):
    """same result format as find_symmetries, with no catalog cap; scalar tier uses identity terms."""
    L = rd.L
    chk = SymChecker(f1, f2, rd)
    if psi_candidates is None:
        psi_candidates = automorphism_group(rd)
    if survivors is None:
        survivors = [p for p in psi_candidates if preserves_local_ideal_structure(f1, f2, p, rd)]
    if verbose:
        print(f"prop 4.13: {len(survivors)}/{len(psi_candidates)} psi survive")
    id_terms = tuple((i, i, tuple(0 for _ in L)) for i in range(4))
    cat = full_depth1_catalog(rd) if include_depth1 else []
    if verbose:
        print(f"depth-one catalog: {len(cat)} layers")
    id4 = identity4(L)
    results = []
    for psi in survivors:
        perm = chk.psi_perm(psi)
        if chk.preserves(id_terms, perm):
            results.append({"psi": psi, "theta": id4, "theta_name": "scalar", "theta_meta": None, "tier": 1})
        for name, meta, terms in cat:
            if chk.preserves(terms, perm):
                results.append({"psi": psi, "theta": terms_to_theta(terms, L), "theta_name": name,
                                "theta_meta": meta, "tier": 2})
    return results
