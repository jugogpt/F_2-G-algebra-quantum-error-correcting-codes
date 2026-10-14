"""equivalence of bivariate pairs under psi in Aut(G), independent translations of f1 and f2, and f1 <-> f2.

these are qubit relabelings (and a block swap), so equivalent pairs give codes with identical [[n,k,d]].
"""
from bb_automorphisms import automorphism_group


def _act(psi, m, L):
    return tuple(sum(psi[i][t] * m[t] for t in range(len(L))) % L[i] for i in range(len(L)))


def _norm(ms, L):
    best = None
    for s in ms:
        t = tuple(sorted(tuple((m[i] - s[i]) % L[i] for i in range(len(L))) for m in ms))
        if best is None or t < best:
            best = t
    return best


_AUT = {}


def _auts(rd):
    if rd.L not in _AUT:
        _AUT[rd.L] = automorphism_group(rd)
    return _AUT[rd.L]


def equivalence_keys(f1, f2, rd):
    """set of canonical keys of all images of (f1,f2) under the group above; two pairs are equivalent iff the sets meet."""
    L = rd.L
    keys = set()
    for psi in _auts(rd):
        a = _norm([_act(psi, m, L) for m in f1.mons], L)
        b = _norm([_act(psi, m, L) for m in f2.mons], L)
        keys.add((a, b))
        keys.add((b, a))
    return keys


def equivalent(p, q, rd):
    return bool(equivalence_keys(*p, rd) & equivalence_keys(*q, rd))
