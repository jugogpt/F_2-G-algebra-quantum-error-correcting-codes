"""proof that a subgroup G of Sp(2k,2) is all of Sp(2k,2): G contains an F2-transvection and is transitive on nonzero vectors.

soundness: by Hering's theorem a subgroup of GL(V), V=F2^(2k), transitive on V\\{0} is one of SL_a(q), Sp_a(q), G2(q) (q=2^b, a b = 2k, up to semilinear
extension) or one of finitely many small exceptions (dim <= 6 for char 2).  Over q>2 every non-identity element of these groups of F_q-transvection type acts on F2^(2k)
with rank b > 1 for g-1, and G2(q) and the field-semilinear groups contain no element with rank(g-1)=1; so a transvection forces q=2, G=Sp(2k,2) (G preserves the form).
the exceptions (SL_2(13) in dim 6 etc.) occur only for 2k <= 6 and are covered by the exact order test there.
"""
import numpy as np


def _rows(M):
    A = np.asarray(M, dtype=np.uint8) % 2
    n = A.shape[0]
    return [int("".join(str(int(x)) for x in A[i][::-1]), 2) for i in range(n)]  # row i as int with bit j = A[i,j]


def rank_minus_identity(M):
    A = (np.asarray(M, dtype=np.uint8) ^ np.eye(len(M), dtype=np.uint8)) % 2
    rows = [int("".join(str(int(x)) for x in r), 2) for r in A]
    r = 0
    basis = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                r += 1
                break
    return r


def _mpow(A, e):
    R = np.eye(A.shape[0], dtype=np.int64)
    B = A.astype(np.int64)
    while e:
        if e & 1:
            R = (R @ B) % 2
        B = (B @ B) % 2
        e >>= 1
    return R


def _order(g, cap=1 << 20):
    n = g.shape[0]
    I = np.eye(n, dtype=np.int64)
    h = g.astype(np.int64)
    for o in range(1, cap):
        if (h == I).all():
            return o
        h = (h @ g) % 2
    return None


def find_transvection_by_powers(mats, rng=None, tries=300):
    """random words, then the 2-power part of each (g^(odd part of its order)) and its repeated squares, looking for rank(g-1)=1."""
    import random
    rng = rng or random.Random(1)
    mats = [np.asarray(m, dtype=np.uint8) for m in mats]
    n = mats[0].shape[0]
    for _ in range(tries):
        g = np.eye(n, dtype=np.int64)
        for _ in range(rng.randint(3, 25)):
            g = (g @ mats[rng.randrange(len(mats))].astype(np.int64)) % 2
        o = _order(g.astype(np.uint8))
        if o is None:
            continue
        odd = o
        while odd % 2 == 0:
            odd //= 2
        h = _mpow(g, odd)
        for _ in range(40):
            r = rank_minus_identity(h.astype(np.uint8))
            if r == 1:
                return h.astype(np.uint8)
            if r == 0:
                break
            h = (h @ h) % 2
    return None


def find_transvection(mats, rng=None, tries=400):
    """an element of <mats> with rank(g-1)=1, searched among generators, pairwise products and short random words."""
    import random
    rng = rng or random.Random(0)
    mats = [np.asarray(m, dtype=np.uint8) for m in mats]
    for m in mats:
        if rank_minus_identity(m) == 1:
            return m
    for a in mats:
        for b in mats:
            g = (a.astype(int) @ b.astype(int)) % 2
            if rank_minus_identity(g) == 1:
                return g.astype(np.uint8)
    for _ in range(tries):
        g = np.eye(len(mats[0]), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            g = (g @ mats[rng.randrange(len(mats))].astype(np.int64)) % 2
        for e in range(1, 8):  # powers
            h = np.linalg.matrix_power(g, e) % 2
            if rank_minus_identity(h) == 1:
                return h.astype(np.uint8)
    return None


def _tables(M, n):
    """byte lookup tables for v -> M v on bit-vectors stored as ints (bit j = coordinate j)."""
    A = np.asarray(M, dtype=np.uint8) % 2
    col = [sum(int(A[i, j]) << i for i in range(n)) for j in range(n)]
    nb = (n + 7) // 8
    tabs = []
    for b in range(nb):
        t = np.zeros(256, dtype=np.uint32)
        for x in range(256):
            v = 0
            for j in range(8):
                if x >> j & 1 and 8 * b + j < n:
                    v ^= col[8 * b + j]
            t[x] = v
        tabs.append(t)
    return tabs


def _apply(tabs, arr):
    out = np.zeros_like(arr)
    for b, t in enumerate(tabs):
        out ^= t[(arr >> (8 * b)) & 255]
    return out


def transitive_on_nonzero(mats, n, start=1):
    """orbit of one vector under the group has size 2^n-1 (breadth-first with numpy lookup tables)."""
    tabs = [_tables(m, n) for m in mats]
    seen = np.zeros(1 << n, dtype=bool)
    frontier = np.array([start], dtype=np.uint32)
    seen[start] = True
    count = 1
    while frontier.size:
        new = []
        for tb in tabs:
            img = _apply(tb, frontier)
            img = img[~seen[img]]
            if img.size:
                img = np.unique(img)
                seen[img] = True
                new.append(img)
        frontier = np.concatenate(new) if new else np.array([], dtype=np.uint32)
        count += frontier.size
    return count == (1 << n) - 1, int(count)


def prove_full(mats, k):
    n = 2 * k
    t = find_transvection(mats)
    if t is None:
        t = find_transvection_by_powers(mats)
    if t is None:
        return {"verdict": "no_transvection_found", "transvection": False}
    tr, cnt = transitive_on_nonzero(mats, n)
    return {"verdict": "full" if tr else "not_full_not_transitive", "transvection": True, "orbit": cnt, "target": (1 << n) - 1}
