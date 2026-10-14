"""logical group of a code from every symmetry we can certify: wide class (incl. depth-one catalog), translations, fold-S."""
import numpy as np
from bb_wide import LogicalSolver, logical_action, _idx_maps, wide_symmetries
from bb_symcheck import SymChecker
from bb_ring_elt import RingElement
from bb_fold_s import find_fold_s, sigma_function
from bb_logical_group import classify_logical_group


def full_group_mats(f1, f2, rd, use_wide=True, use_fold=True, catalog=None, psis=None):
    L = rd.L
    mons, allpsi, trans = _idx_maps(rd)
    chk = SymChecker(f1, f2, rd)
    S = LogicalSolver(chk, f1, f2, rd)
    k = S.k2 // 2
    mats, seen = [], set()

    def add(M):
        key = M.tobytes()
        if key not in seen:
            seen.add(key)
            mats.append(M)

    info = {"k": k, "wide": 0, "nonsemilinear": 0, "fold": 0}
    if use_wide:
        fs = wide_symmetries(f1, f2, rd, catalog=catalog, psis=psis)
        info["wide"] = len(fs)
        info["nonsemilinear"] = sum(1 for x in fs if not x["semilinear"])
        for x in fs:
            add(logical_action(S, x["terms"], allpsi[x["iL"]], allpsi[x["iR"]], trans[x["ib"]], x["xch"]))
    idp = np.arange(chk.D)
    tmats = []
    for v in range(len(L)):
        e = tuple(1 if j == v else 0 for j in range(len(L)))
        terms = tuple((i, i, e) for i in range(4))
        Tm = logical_action(S, terms, idp, idp, idp, False)
        add(Tm)
        tmats.append(Tm)
    if use_fold:
        for h in find_fold_s(f1, f2, rd):
            sg = sigma_function(h[1], h[2], h[3], L)

            def img(i, sg=sg):
                v = [RingElement(L, [mons[j] for j in np.nonzero(a)[0]]) for a in S.vecs[i]]
                return chk._int([chk._arr(c) for c in sg(v)])

            add(S.matrix_of(img))
            info["fold"] += 1
    info['trans'] = tmats
    return mats, k, info




def nilradical_certificate(mats, trans, L):
    """N V for the nilradical N=(x_i^{m_i}+1), m_i the odd part of l_i: a proper nonzero invariant subspace proves the group is not Sp(2k,2)."""
    import numpy as np
    def rank_basis(cols):
        basis, r = {}, 0
        for v in cols:
            while v:
                h = v.bit_length() - 1
                if h in basis:
                    v ^= basis[h]
                else:
                    basis[h] = v
                    r += 1
                    break
        return r, basis
    def colints(M):
        M = np.asarray(M, dtype=np.uint8) % 2
        return [sum(int(M[i, j]) << i for i in range(M.shape[0])) for j in range(M.shape[1])]
    def apply(M, v):
        M = np.asarray(M, dtype=np.uint8)
        out = 0
        for j in range(M.shape[0]):
            if v >> j & 1:
                out ^= sum(int(M[i, j]) << i for i in range(M.shape[0]))
        return out
    n = trans[0].shape[0]
    I = np.eye(n, dtype=np.uint8)
    gens = []
    for T, l in zip(trans, L):
        m = l
        while m % 2 == 0:
            m //= 2
        P = np.eye(n, dtype=np.int64)
        for _ in range(m):
            P = (P @ np.asarray(T, dtype=np.int64)) % 2
        gens += colints((P.astype(np.uint8) ^ I) % 2)
    r, basis = rank_basis(gens)
    if r == 0:
        return {"dim_V": n, "dim_NV": 0, "invariant": None}
    W = list(basis.values())
    inv = all(rank_basis(W + [apply(M, w) for w in W])[0] == r for M in mats)
    return {"dim_V": n, "dim_NV": r, "invariant": bool(inv), "proper_certificate": bool(inv and 0 < r < n)}


def narrow_group_mats(f1, f2, rd, use_fold=True):
    """semilinear class (psi, Theta) via find_symmetries, plus translations and fold-S; much cheaper than the wide class."""
    import numpy as np
    from bb_script_a import find_symmetries
    from bb_crt import CRTBasis
    from bb_logical import logical_basis, code_dimension
    from bb_logical_action import induced_logical_action
    from bb_automorphisms import identity_matrix
    from bb_theta import zero4, identity4
    L = rd.L
    crt = CRTBasis(rd)
    LX, LZ = logical_basis(f1, f2, rd, crt)
    k = code_dimension(f1, f2, rd)
    found = find_symmetries(f1, f2, rd, include_depth1=True)
    mats = [induced_logical_action(r["psi"], r["theta"], f1, f2, LX, LZ, rd) for r in found]
    tm = []
    for i in range(len(L)):
        e = tuple(1 if j == i else 0 for j in range(len(L)))
        T = zero4(L)
        for r_ in range(4):
            T[r_][r_] = RingElement(L, [e])
        M = induced_logical_action(identity_matrix(len(L)), T, f1, f2, LX, LZ, rd)
        mats.append(M)
        tm.append(np.array(M))
    nfold = 0
    if use_fold:
        from bb_fold_s import find_fold_s, sigma_function
        for h in find_fold_s(f1, f2, rd):
            mats.append(induced_logical_action(None, None, f1, f2, LX, LZ, rd, sigma_fn=sigma_function(h[1], h[2], h[3], L)))
            nfold += 1
    return [np.array(M) for M in mats], k, {"k": k, "semilinear": len(found), "fold": nfold, "trans": tm}
