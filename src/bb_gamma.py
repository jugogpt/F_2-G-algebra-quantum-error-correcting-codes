"""inverse design for a symmetry set gamma = {automorphism gates, fold gates, automorphism+fold gates}.

each gate imposes linear conditions on (f1,f2) once its ring-element twist is fixed:
 - automorphism gate Phi_psi (Theta = identity): psi(f_i) = u f_i, hence psi(fbar_i) = ubar fbar_i,
 - fold gate x|z -> x|z+P_pi x with pi(g)=m psi_f(g) involutive and swap flag: (pi(a), pi(b)) = s (fbar_2, fbar_1) with (a,b)=(f1,f2) or (f2,f1);
   psi_f = id is a plain fold gate, psi_f != id an automorphism+fold gate.
a gate preserves M iff these hold (images of g_X, g_Z lie in M; the semilinear structure covers the remaining elements of M).
"""
import itertools, random
import numpy as np
import galois
from bb_ring_elt import RingElement
from bb_automorphisms import automorphism_group, full_fold
from bb_theta import identity4
from bb_theta_inverse import constraint_matrix, _mons, _vec, _elt
from bb_fold_s import fold_s_variants

GF2 = galois.GF(2)


def mono(L, e):
    return RingElement(L, [tuple(x % L[i] for i, x in enumerate(e))])


def auto_gate_matrix(L, psi, u_exp):
    u = mono(L, u_exp)
    ub = u.apply_automorphism(full_fold(L))
    zero = RingElement(L, [])
    return np.array(constraint_matrix(L, psi, identity4(L), ((u, zero), (zero, ub))), dtype=np.uint8)


def fold_gate_matrix(L, psi, delta, swap, s_exp):
    mons = _mons(L)
    m = mono(L, delta)
    s = mono(L, s_exp)
    fold = full_fold(L)
    zero = RingElement(L, [])
    cols = []
    for slot in range(2):
        for j in range(len(mons)):
            e = RingElement(L, [mons[j]])
            f1, f2 = (e, zero) if slot == 0 else (zero, e)
            a, b = (f2, f1) if swap else (f1, f2)
            pi = lambda x: m * x.apply_automorphism(psi)
            r1 = pi(a) + s * f2.apply_automorphism(fold)
            r2 = pi(b) + s * f1.apply_automorphism(fold)
            cols.append(np.concatenate([_vec(r1, mons), _vec(r2, mons)]))
    return np.array(cols, dtype=np.uint8).T


def solution_basis(L, mats):
    n = len(_mons(L))
    A = GF2(np.vstack(mats))
    N = A.null_space()
    return [(_elt(L, v[:n], _mons(L)), _elt(L, v[n:], _mons(L))) for v in N]


def is_id(psi):
    k = len(psi)
    return all(psi[i][j] == (1 if i == j else 0) for i in range(k) for j in range(k))


def sample_gamma(rd, rng, n_auto=2, n_fold=2, n_autofold=2):
    """random gate parameters; returns list of ('auto',psi,u) / ('fold',psi,delta,swap,s)."""
    L = rd.L
    G = list(itertools.product(*[range(l) for l in L]))
    autos = [p for p in automorphism_group(rd) if not is_id(p)]
    folds = fold_s_variants(rd)
    pure = [v for v in folds if is_id(v[1])]
    mixed = [v for v in folds if not is_id(v[1])]
    gens = []
    for _ in range(n_auto):
        gens.append(("auto", rng.choice(autos), rng.choice(G)))
    for _ in range(n_fold):
        v = rng.choice(pure)
        gens.append(("fold", v[1], v[2], v[3], rng.choice(G)))
    for _ in range(n_autofold):
        v = rng.choice(mixed)
        gens.append(("fold", v[1], v[2], v[3], rng.choice(G)))
    return gens


def gamma_basis(L, gens):
    mats = []
    for g in gens:
        mats.append(auto_gate_matrix(L, g[1], g[2]) if g[0] == "auto" else fold_gate_matrix(L, g[1], g[2], g[3], g[4]))
    return solution_basis(L, mats)
