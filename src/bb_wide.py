"""wider symmetry class: qubit permutations (g in block B) -> (psi_B(g) + a_B in block B'), with independent psi_L, psi_R in Aut(G),
independent translations, optional block exchange, optionally composed with H on every qubit (x and z checks exchanged).

translations common to both blocks are symmetries of every code, so a_L = 0 is fixed and they are added separately.
a candidate is first filtered on the image of one check row by syndrome hashing, then verified exactly on all check rows.
the logical action is read off from commutation with the true logical representatives.
"""
from __future__ import annotations
from itertools import product as iproduct
import numpy as np
from bb_distance import build_checks, logical_matrices, _mons
from bb_crt import CRTBasis
from bb_automorphisms import automorphism_group


def _gf2_inv(A):
    n = A.shape[0]
    m = np.hstack([A % 2, np.eye(n, dtype=np.uint8)]).astype(np.uint8)
    for c in range(n):
        p = next(i for i in range(c, n) if m[i, c])
        m[[c, p]] = m[[p, c]]
        for i in range(n):
            if i != c and m[i, c]:
                m[i] ^= m[c]
    return m[:, n:]


def logical_action_of_permutation(qp, swaps, LX, LZ, Winv):
    """2k x 2k matrix in the basis (x_1..x_k | z_1..z_k), acting on column vectors, of qubit permutation qp (composed with H^n if swaps)."""
    k = LX.shape[0]
    perm = lambda rows: np.array([np.bincount(qp[np.nonzero(r)[0]], minlength=len(r)) % 2 for r in rows], dtype=np.uint8)
    M = np.zeros((2 * k, 2 * k), dtype=np.uint8)
    W = (LX.astype(int) @ LZ.T.astype(int)) % 2
    if not swaps:
        PX, PZ = perm(LX), perm(LZ)
        cx = (PX.astype(int) @ LZ.T.astype(int) @ Winv) % 2
        cz = (PZ.astype(int) @ LX.T.astype(int) @ Winv.T) % 2
        for i in range(k):
            M[:k, i] = cx[i]
            M[k:, k + i] = cz[i]
    else:
        PX, PZ = perm(LX), perm(LZ)
        # image of an x logical is a z operator on the permuted support, and vice versa
        dz = (PX.astype(int) @ LX.T.astype(int) @ Winv.T) % 2
        dx = (PZ.astype(int) @ LZ.T.astype(int) @ Winv) % 2
        for i in range(k):
            M[k:, i] = dz[i]
            M[:k, k + i] = dx[i]
    return M



def _idx_maps(rd):
    """index maps of every psi in Aut(G) on the monomial list, and the translation index maps."""
    mons = _mons(rd.L)
    pos = {m: i for i, m in enumerate(mons)}
    L = rd.L
    psis = []
    for psi in automorphism_group(rd):
        psis.append(np.array([pos[tuple(sum(psi[i][t] * m[t] for t in range(len(L))) % L[i] for i in range(len(L)))] for m in mons]))
    trans = [np.array([pos[tuple((m[i] + b[i]) % L[i] for i in range(len(L)))] for m in mons]) for b in mons]
    return mons, psis, trans


def wide_permutations(f1, f2, rd, Hx=None, Hz=None):
    """all qubit permutations of the wide class that preserve the code; returns (list of (qubit_perm, swaps_xz), counts)."""
    if Hx is None:
        Hx, Hz = build_checks(f1, f2, rd)
    D = Hx.shape[0]
    mons, psis, trans = _idx_maps(rd)
    n = 2 * D
    s1 = np.nonzero(Hx[0, :D])[0]
    s2 = np.nonzero(Hx[0, D:])[0]
    out = []
    pack = lambda col: int("".join(map(str, col.astype(int))), 2)
    for typ, K in (("I", Hz), ("H", Hx)):
        cols = [pack(K[:, c]) for c in range(n)]
        for xch in (False, True):
            offL, offR = (D, 0) if xch else (0, D)
            tab = {}
            for iL, pL in enumerate(psis):
                key = 0
                for h in s1:
                    key ^= cols[offL + pL[h]]
                tab.setdefault(key, []).append(iL)
            for iR, pR in enumerate(psis):
                for ib, tb in enumerate(trans):
                    key = 0
                    for h in s2:
                        key ^= cols[offR + tb[pR[h]]]
                    for iL in tab.get(key, []):
                        pL = psis[iL]
                        qp = np.empty(n, dtype=int)
                        qp[:D] = offL + pL
                        qp[D:] = offR + tb[pR]
                        P = np.zeros((n, n), dtype=np.uint8)
                        P[qp, np.arange(n)] = 1
                        src = Hx if typ == "I" else Hx
                        img = (src @ P.T) % 2
                        if not ((K @ img.T) % 2).any():
                            out.append((qp, typ == "H"))
    return out


def wide_logical_actions(f1, f2, rd, crt=None):
    """(distinct logical matrices, number of wide permutations found, number of swapping ones)."""
    crt = crt or CRTBasis(rd)
    LXm, LZm = logical_matrices(f1, f2, rd, crt)
    W = (LXm.astype(int) @ LZm.T.astype(int)) % 2
    Winv = _gf2_inv(W.astype(np.uint8)).astype(int)
    perms = wide_permutations(f1, f2, rd)
    seen, mats = set(), []
    for qp, sw in perms:
        M = logical_action_of_permutation(qp, sw, LXm, LZm, Winv)
        if M.tobytes() not in seen:
            seen.add(M.tobytes())
            mats.append(M)
    return mats, len(perms), sum(1 for _, s in perms if s)


# ---------------------------------------------------------------------------------------------------------------
# full wide class: sigma = Phi(psi_L, psi_R, b, exchange) o Theta, with Theta from the depth-one catalog.
# slots are (xL, xR, zL, zR); psi_L acts on slots 0,2, translation-by-b after psi_R acts on slots 1,3, exchange swaps 0<->1, 2<->3.

def _perp_columns(chk):
    """for each slot and index, the syndrome (int) of that coordinate against the parity checks of M."""
    import galois
    GF2 = galois.GF(2)
    D, N = chk.D, 4 * chk.D
    rows = []
    for v in chk.basis.values():
        bits = np.unpackbits(np.frombuffer(v.to_bytes((N + 7) // 8, "big"), dtype=np.uint8))[:N]
        rows.append(bits)
    B = GF2(np.array(rows, dtype=np.uint8))
    K = B.null_space()
    Kn = np.array(K, dtype=np.uint8)
    cols = [int("".join(map(str, Kn[:, c])), 2) if Kn.shape[0] else 0 for c in range(N)]
    return [cols[s * D:(s + 1) * D] for s in range(4)]


def apply_wide(chk, terms, permL, permR, tb, xch, vec4):
    out = [np.zeros(chk.D, dtype=np.uint8) for _ in range(4)]
    for i, j, e in terms:
        src = vec4[j]
        if not src.any():
            continue
        t = chk.shift_table(e)
        tmp = np.empty_like(src)
        tmp[t] = src
        out[i] ^= tmp
    pr = tb[permR]
    res = [None] * 4
    for s, perm in ((0, permL), (2, permL), (1, pr), (3, pr)):
        b = np.empty_like(out[s])
        b[perm] = out[s]
        res[s] = b
    if xch:
        res = [res[1], res[0], res[3], res[2]]
    return res


def _preserves_all(chk, terms, permL, permR, tb, xch):
    """sigma is not R-semilinear when psi_L != psi_R, so images of the two generators are not enough: test every basis vector of M."""
    N, D = 4 * chk.D, chk.D
    for v in chk.basis.values():
        b = np.unpackbits(np.frombuffer(v.to_bytes((N + 7) // 8, "big"), dtype=np.uint8))[:N]
        vec = [b[s * D:(s + 1) * D].copy() for s in range(4)]
        if not chk.contains(chk._int(apply_wide(chk, terms, permL, permR, tb, xch, vec))):
            return False
    return True


def wide_symmetries(f1, f2, rd, catalog=None, psis=None, verbose=False, only_nonsemilinear=False):
    """all sigma = Phi(psi_L,psi_R,b,xch) o Theta (b with a_L=0) preserving M. psis: allowed Aut(G) elements (default all)."""
    from bb_symcheck import SymChecker, full_depth1_catalog
    chk = SymChecker(f1, f2, rd)
    L = rd.L
    mons, allpsi, trans = _idx_maps(rd)
    autos = automorphism_group(rd)
    if psis is not None:
        keep = [i for i, p in enumerate(autos) if any(p == q for q in psis)]
        allpsi = [allpsi[i] for i in keep]
        autos = [autos[i] for i in keep]
    cat = catalog if catalog is not None else full_depth1_catalog(rd)
    idt = tuple((i, i, tuple(0 for _ in L)) for i in range(4))
    cat = [("scalar", None, idt)] + list(cat)
    Kc = _perp_columns(chk)
    found = []
    for name, meta, terms in cat:
        for gi, g in enumerate((chk.gX, chk.gZ)):
            pass
        # image of gX under Theta alone, as four bit arrays
        wX = apply_wide(chk, terms, np.arange(chk.D), np.arange(chk.D), np.arange(chk.D), False, chk.gX)
        for xch in (False, True):
            sL, sR = ((1, 3), (0, 2)) if xch else ((0, 2), (1, 3))
            tab = {}
            for iL, pL in enumerate(allpsi):
                key = 0
                for h in np.nonzero(wX[0])[0]:
                    key ^= Kc[sL[0]][pL[h]]
                for h in np.nonzero(wX[2])[0]:
                    key ^= Kc[sL[1]][pL[h]]
                tab.setdefault(key, []).append(iL)
            sup1, sup3 = np.nonzero(wX[1])[0], np.nonzero(wX[3])[0]
            for iR, pR in enumerate(allpsi):
                for ib, tb in enumerate(trans):
                    key = 0
                    for h in sup1:
                        key ^= Kc[sR[0]][tb[pR[h]]]
                    for h in sup3:
                        key ^= Kc[sR[1]][tb[pR[h]]]
                    for iL in tab.get(key, ()):
                        if chk.contains(chk._int(apply_wide(chk, terms, allpsi[iL], allpsi[iR], tb, xch, chk.gZ))) and \
                           chk.contains(chk._int(apply_wide(chk, terms, allpsi[iL], allpsi[iR], tb, xch, chk.gX))):
                            semi = iL == iR and ib == 0
                            if not semi and not _preserves_all(chk, terms, allpsi[iL], allpsi[iR], tb, xch):
                                continue
                            found.append({"theta": name, "terms": terms, "iL": iL, "iR": iR, "ib": ib, "xch": xch,
                                          "psiL": autos[iL], "psiR": autos[iR], "semilinear": semi})
    return found


class LogicalSolver:
    """coordinates of a vector modulo M in the basis (LX..., LZ...), by tagged echelon reduction on bit-integers."""

    def __init__(self, chk, f1, f2, rd, crt=None):
        from bb_logical import logical_basis
        crt = crt or CRTBasis(rd)
        LX, LZ = logical_basis(f1, f2, rd, crt)
        z = [np.zeros(chk.D, dtype=np.uint8)] * 2
        vecs = [[chk._arr(P), chk._arr(Q), z[0], z[1]] for P, Q in LX] + [[z[0], z[1], chk._arr(P), chk._arr(Q)] for P, Q in LZ]
        self.k2 = len(vecs)
        self.vecs = vecs
        self.basis = {}
        for v in chk.basis.values():
            self._ins(v, 0)
        for i, v in enumerate(vecs):
            left = self._ins(chk._int(v), 1 << i)
            assert left, "logical vector dependent on M"
        self.chk = chk

    def _ins(self, v, tag):
        while v:
            h = v.bit_length() - 1
            b = self.basis.get(h)
            if b is None:
                self.basis[h] = (v, tag)
                return True
            v ^= b[0]
            tag ^= b[1]
        return False

    def coords(self, v):
        tag = 0
        while v:
            h = v.bit_length() - 1
            b = self.basis[h]
            v ^= b[0]
            tag ^= b[1]
        return tag

    def matrix_of(self, image_of):
        """image_of(i) -> bit-integer image of logical basis vector i; returns 2k x 2k matrix, columns are images."""
        M = np.zeros((self.k2, self.k2), dtype=np.uint8)
        for i in range(self.k2):
            t = self.coords(image_of(i))
            for j in range(self.k2):
                M[j, i] = (t >> j) & 1
        return M


def _store_vecs(self_vecs):
    return self_vecs


def logical_action(solver, terms, permL, permR, tb, xch):
    chk = solver.chk
    return solver.matrix_of(lambda i: chk._int(apply_wide(chk, terms, permL, permR, tb, xch, solver.vecs[i])))
