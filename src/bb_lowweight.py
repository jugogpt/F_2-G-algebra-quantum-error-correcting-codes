"""multivariate sparsest generator pair: find u with R(u f1,u f2)=R(f1,f2) minimizing wt(u f1)+wt(u f2).

admissible u are exactly those with nonzero residue u(alpha_c) at every component c where (f1,f2)_c is nonzero (nakayama).
the search has three modes: exhaustive for small |G|, enumeration plus descent as an upper bound, and an exact ilp.
"""
from __future__ import annotations
import itertools, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from bb_core import RingData, ring_setup
from bb_ring_elt import RingElement, local_jet
from bb_theta_inverse import _mons


class LowWeightContext:
    """translates and component residues for a fixed pair."""

    def __init__(self, f1: RingElement, f2: RingElement, rd: RingData):
        self.rd, self.f1, self.f2 = rd, f1, f2
        self.mons = _mons(rd.L)
        self.D = len(self.mons)
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.t1 = [self._translate(f1, g) for g in self.mons]
        self.t2 = [self._translate(f2, g) for g in self.mons]
        nz = lambda jet: any(int(v) != 0 for v in jet.values())
        self.active = [c for c in rd.components
                       if nz(local_jet(f1, c, rd)) or nz(local_jet(f2, c, rd))]
        self.res = [[int(RingElement(rd.L, [g]).evaluate(c.rep, rd.GF)) for c in self.active] for g in self.mons]

    def _translate(self, f, g):
        L, bits = self.rd.L, 0
        for m in f.mons:
            bits |= 1 << self.idx[tuple((m[i] + g[i]) % L[i] for i in range(len(L)))]
        return bits

    def residues(self, support):
        out = [0] * len(self.active)
        for j in support:
            r = self.res[j]
            for t in range(len(out)):
                out[t] ^= r[t]
        return out

    def admissible(self, support) -> bool:
        return all(self.residues(support))

    def weights(self, support):
        a = b = 0
        for j in support:
            a ^= self.t1[j]
            b ^= self.t2[j]
        return a.bit_count() + b.bit_count()

    def product_pair(self, support):
        a = b = 0
        for j in support:
            a ^= self.t1[j]
            b ^= self.t2[j]
        mk = lambda bits: RingElement(self.rd.L, [self.mons[i] for i in range(self.D) if (bits >> i) & 1])
        return mk(a), mk(b)


def module_basis(f1: RingElement, f2: RingElement, rd: RingData):
    """binary basis of R*(f1,f2) as int bitmasks keyed by leading bit."""
    mons = _mons(rd.L)
    D = len(mons)
    idx = {m: i for i, m in enumerate(mons)}
    rows = []
    for g in mons:
        v = 0
        for m in f1.mons:
            v |= 1 << idx[tuple((m[i] + g[i]) % rd.L[i] for i in range(len(rd.L)))]
        for m in f2.mons:
            v |= 1 << (D + idx[tuple((m[i] + g[i]) % rd.L[i] for i in range(len(rd.L)))])
        rows.append(v)
    basis = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                break
    return basis


def same_module(p, q, rd: RingData) -> bool:
    """true iff the two pairs generate equal binary row spaces."""
    bp, bq = module_basis(*p, rd), module_basis(*q, rd)
    return len(bp) == len(bq) and _contains(bp, bq)


def _contains(big, small) -> bool:
    for v in small.values():
        while v:
            h = v.bit_length() - 1
            if h in big:
                v ^= big[h]
            else:
                return False
    return True


def sparsest_generator(f1: RingElement, f2: RingElement, rd: RingData, max_u_weight: int = 4,
                       exhaustive_cap: int = 20, restarts: int = 200, seed: int = 0):
    """exhaustive for |G|<=exhaustive_cap, else enumeration to max_u_weight plus descent (upper bound only)."""
    ctx = LowWeightContext(f1, f2, rd)
    D = ctx.D
    best = (ctx.weights([0]), (0,))
    base_w = best[0]

    def consider(sup):
        nonlocal best
        if not sup or not ctx.admissible(sup):
            return
        w = ctx.weights(sup)
        if w < best[0]:
            best = (w, tuple(sup))

    if D <= exhaustive_cap:
        mode = "exhaustive"
        for mask in range(1, 1 << D):
            if mask & 1:
                consider([j for j in range(D) if (mask >> j) & 1])
    else:
        mode = f"enum<= {max_u_weight} + descent"
        for w in range(1, max_u_weight + 1):
            for rest in itertools.combinations(range(1, D), w - 1):
                consider((0,) + rest)
        rng = random.Random(seed)
        for _ in range(restarts):
            sup = set(rng.sample(range(D), rng.randint(1, max(2, D // 4))))
            if not ctx.admissible(sorted(sup)):
                continue
            cur = ctx.weights(sorted(sup))
            improved = True
            while improved:
                improved = False
                for j in rng.sample(range(D), D):
                    t = sup ^ {j}
                    if not t or not ctx.admissible(sorted(t)):
                        continue
                    w = ctx.weights(sorted(t))
                    if w < cur:
                        sup, cur, improved = t, w, True
            if cur < best[0]:
                best = (cur, tuple(sorted(sup)))
    w, sup = best
    g1, g2 = ctx.product_pair(sup)
    return {"weight": w, "original_weight": base_w, "u_support": [ctx.mons[j] for j in sup],
            "f1": g1, "f2": g2, "mode": mode}


def verify_admissibility_criterion(f1, f2, rd, trials=300, seed=0):
    """compare the residue test with row-space equality on random u."""
    ctx = LowWeightContext(f1, f2, rd)
    rng = random.Random(seed)
    base = (f1, f2)
    agree = bad = adm = 0
    for _ in range(trials):
        sup = sorted(rng.sample(range(ctx.D), rng.randint(1, max(1, ctx.D // 2))))
        pred = ctx.admissible(sup)
        p = ctx.product_pair(sup)
        truth = same_module(base, p, rd)
        adm += truth
        if pred == truth:
            agree += 1
        else:
            bad += 1
    return {"trials": trials, "agree": agree, "disagree": bad, "admissible_true": adm,
            "active_components": len(ctx.active), "all_components": len(rd.components)}


def low_weight_words_isd(basis: dict, n: int, max_weight: int, iters: int = 300, p: int = 2, seed: int = 0):
    """prange-style isd: random pivot order, systematic form, combinations of <=p rows; yields words of weight <= max_weight."""
    rng = random.Random(seed)
    rows0 = list(basis.values())
    seen = set()
    for _ in range(iters):
        order = list(range(n))
        rng.shuffle(order)
        rows = rows0[:]
        piv_rows = []
        for c in order:
            for i, r in enumerate(rows):
                if (r >> c) & 1:
                    rows.pop(i)
                    piv_rows = [q ^ r if (q >> c) & 1 else q for q in piv_rows]
                    rows = [q ^ r if (q >> c) & 1 else q for q in rows]
                    piv_rows.append(r)
                    break
            if not rows:
                break
        for w in range(1, p + 1):
            for combo in itertools.combinations(piv_rows, w):
                v = 0
                for r in combo:
                    v ^= r
                if v and v.bit_count() <= max_weight and v not in seen:
                    seen.add(v)
                    yield v


def sparsest_generator_isd(f1: RingElement, f2: RingElement, rd: RingData, max_weight: int | None = None,
                           iters: int = 300, p: int = 2, seed: int = 0):
    """upper bound by isd: low-weight words of M that still generate M (same binary dimension), best first."""
    ctx = LowWeightContext(f1, f2, rd)
    D = ctx.D
    basis = module_basis(f1, f2, rd)
    target = len(basis)
    max_weight = max_weight or (len(f1.mons) + len(f2.mons))
    best = None
    for v in low_weight_words_isd(basis, 2 * D, max_weight, iters, p, seed):
        a = RingElement(rd.L, [ctx.mons[i] for i in range(D) if (v >> i) & 1])
        b = RingElement(rd.L, [ctx.mons[i] for i in range(D) if (v >> (D + i)) & 1])
        if best is not None and v.bit_count() >= best[0]:
            continue
        if len(module_basis(a, b, rd)) == target:
            best = (v.bit_count(), a, b)
            max_weight = best[0] - 1
    if best is None:
        return {"weight": None}
    return {"weight": best[0], "f1": best[1], "f2": best[2], "mode": f"isd p={p} iters={iters}"}


def sparsest_generator_ilp(f1: RingElement, f2: RingElement, rd: RingData, time_limit: float = 120,
                           upper_bound: int | None = None):
    """exact sparsest generator by ilp over binary u with u_0=1; effective only for small rings (no certificate at |G|=72)."""
    ctx = LowWeightContext(f1, f2, rd)
    D = ctx.D
    T = np.zeros((2 * D, D), dtype=np.int64)
    for j in range(D):
        for i in range(D):
            T[i, j] = (ctx.t1[j] >> i) & 1
            T[D + i, j] = (ctx.t2[j] >> i) & 1
    forms = []
    for t in range(len(ctx.active)):
        nb = max(ctx.res[j][t].bit_length() for j in range(D))
        forms.append([[ (ctx.res[j][t] >> b) & 1 for j in range(D)] for b in range(nb)])
    nf = sum(len(f) for f in forms)
    nv = D + 2 * D + 2 * D + nf + nf
    iu, iv, it, iw, iy = 0, D, 3 * D, 5 * D, 5 * D + nf
    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        r = np.zeros(nv)
        for k, c in coefs:
            r[k] += c
        rows.append(r); lo.append(l); hi.append(h)

    for i in range(2 * D):
        add([(iu + j, T[i, j]) for j in range(D) if T[i, j]] + [(iv + i, -1), (it + i, -2)], 0, 0)
    k = 0
    for t, fs in enumerate(forms):
        ys = []
        for fb in fs:
            add([(iu + j, 1) for j in range(D) if fb[j]] + [(iw + k, -2), (iy + k, -1)], 0, 0)
            ys.append(iy + k)
            k += 1
        add([(y, 1) for y in ys], 1, len(ys))
    if upper_bound is not None:
        add([(iv + i, 1) for i in range(2 * D)], 0, upper_bound)
    lb, ub = np.zeros(nv), np.ones(nv)
    lb[iu] = 1
    ub[it:it + 2 * D] = D
    ub[iw:iw + nf] = D
    res = milp(np.concatenate([np.zeros(D), np.ones(2 * D), np.zeros(nv - 3 * D)]),
               constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.x is None:
        return {"weight": None, "optimal": False, "status": res.message}
    u = [j for j in range(D) if round(res.x[iu + j])]
    g1, g2 = ctx.product_pair(u)
    return {"weight": len(g1.mons) + len(g2.mons), "optimal": res.status == 0, "u_support": [ctx.mons[j] for j in u],
            "f1": g1, "f2": g2, "status": res.message}


def _pgcd(a: int, b: int) -> int:
    while b:
        while a and a.bit_length() >= b.bit_length():
            a ^= b << (a.bit_length() - b.bit_length())
        a, b = b, a
    return a


def _pmulmod(a: int, b: int, l: int) -> int:
    out = 0
    for i in range(l):
        if (a >> i) & 1:
            sh = ((b << i) | (b >> (l - i))) & ((1 << l) - 1) if i else b
            out ^= sh
    return out


def univariate_gcd_check(p: int, q: int, l: int):
    """compare the gcd criterion gcd(u p,u q,x^l-1)=gcd(p,q,x^l-1) with the residue test for every u."""
    rd = ring_setup((l,))
    f1 = RingElement(rd.L, [(i,) for i in range(l) if (p >> i) & 1])
    f2 = RingElement(rd.L, [(i,) for i in range(l) if (q >> i) & 1])
    ctx = LowWeightContext(f1, f2, rd)
    xl = (1 << l) | 1
    g0 = _pgcd(_pgcd(p, q), xl)
    bad = 0
    for u in range(1, 1 << l):
        sup = [i for i in range(l) if (u >> i) & 1]
        ref = _pgcd(_pgcd(_pmulmod(u, p, l), _pmulmod(u, q, l)), xl) == g0
        bad += ref != ctx.admissible(sup)
    return {"l": l, "u_checked": (1 << l) - 1, "disagree": bad}


def sparsest_generator_best(f1: RingElement, f2: RingElement, rd: RingData, isd_iters: int = 200, seed: int = 0):
    """lowest-weight pair over exhaustive/descent and isd; weight is an upper bound unless |G|<=20 (then exact)."""
    a = sparsest_generator(f1, f2, rd, seed=seed)
    if a["mode"] == "exhaustive":
        return a
    b = sparsest_generator_isd(f1, f2, rd, iters=isd_iters, seed=seed)
    return b | {"original_weight": a["original_weight"]} if b["weight"] is not None and b["weight"] < a["weight"] else a


if __name__ == "__main__":
    for L, s1, s2 in [((6,), "1 + x + x^3", "1 + x^2 + x^3"), ((4, 3), "1 + x + y", "1 + x*y + y^2"),
                      ((2, 2, 3), "1 + x + y*z", "1 + x*y + z")]:
        rd = ring_setup(L)
        f1, f2 = RingElement.from_string(L, s1), RingElement.from_string(L, s2)
        print(L, verify_admissibility_criterion(f1, f2, rd))
        r = sparsest_generator(f1, f2, rd)
        print("  sparsest:", r["original_weight"], "->", r["weight"], r["mode"],
              "same module:", same_module((f1, f2), (r["f1"], r["f2"]), rd))
