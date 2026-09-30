"""Script B, Tier 2 without brute force: the linear constraint that sigma(M)=M imposes on (f1,f2) for fixed (psi, Theta, W)."""
from __future__ import annotations
import itertools, galois, numpy as np
from bb_ring_elt import RingElement
from bb_automorphisms import full_fold

GF2 = galois.GF(2)

def _mons(L): return [tuple(m) for m in itertools.product(*[range(l) for l in L])]

def _elt(L, vec, mons): return RingElement(L, [mons[i] for i in range(len(mons)) if int(vec[i])])

def _vec(f, mons):
    idx = {m: i for i, m in enumerate(mons)}
    v = np.zeros(len(mons), dtype=np.uint8)
    for m in f.mons: v[idx[m]] = 1
    return v

def _apply_theta(Theta, gvec, L):
    out = []
    for i in range(4):
        acc = RingElement(L, [])
        for j in range(4): acc = acc + Theta[i][j] * gvec[j]
        out.append(acc)
    return out

def constraint_matrix(L, psi, Theta, W):
    """the binary matrix C with C @ [f1;f2] = 0 iff sigma(M)=M under the twist matrix W."""
    mons = _mons(L); n = len(mons); fold = full_fold(L)
    zero = RingElement(L, [])
    cols = []
    for slot in range(2):
        for j in range(n):
            e = RingElement(L, [mons[j]])
            f1, f2 = (e, zero) if slot == 0 else (zero, e)
            f1b, f2b = f1.apply_automorphism(fold), f2.apply_automorphism(fold)
            gX = [f1, f2, zero, zero]; gZ = [zero, zero, f2b, f1b]
            col = []
            for src, (wa, wb) in ((gX, W[0]), (gZ, W[1])):
                s = [c.apply_automorphism(psi) for c in _apply_theta(Theta, src, L)]
                tgt = [wa * gX[i] + wb * gZ[i] for i in range(4)]
                col += [_vec(s[i] + tgt[i], mons) for i in range(4)]
            cols.append(np.concatenate(col))
    return GF2(np.array(cols, dtype=np.uint8).T)

def solve_codes(L, psi, Theta, W):
    """an F2-basis of {(f1,f2) : sigma(M)=M with twist W}."""
    mons = _mons(L); n = len(mons)
    N = constraint_matrix(L, psi, Theta, W).null_space()
    return [(_elt(L, v[:n], mons), _elt(L, v[n:], mons)) for v in N]


# several generators at once: the joint solution space of a symmetry set.
def joint_constraint_matrix(L, gens):
    """gens: list of (psi, Theta, W); stack their constraint matrices."""
    return GF2(np.vstack([np.array(constraint_matrix(L, psi, Th, W), dtype=np.uint8)
                          for psi, Th, W in gens]))

def joint_solution_basis(L, gens):
    mons = _mons(L); n = len(mons)
    N = joint_constraint_matrix(L, gens).null_space()
    return [(_elt(L, v[:n], mons), _elt(L, v[n:], mons)) for v in N]
