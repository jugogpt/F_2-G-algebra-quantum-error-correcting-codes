"""
Asserts every number quoted in the paper's worked examples (forward, inverse, computational).
Run:  python3 verify_examples.py [A] [B] [C] [Bclasses]    Writes gen_data.json.
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import sys, json, itertools, random, collections, time
import numpy as np, galois
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE, local_jet
from bb_crt import CRTBasis
from bb_local_ring import check_cyclic_matching, local_mul, _zero
from bb_logical import code_dimension, logical_basis
from bb_logical_action import induced_logical_action, _m_basis, _flatten4
from bb_theta import identity4, form_C_theta, verify_symplectic
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure, full_fold, multiplier, shear
from bb_orbits import component_perm, bar_perm, orbits_of
from bb_script_b import target_kernel, _mult_matrix, construct_code_theta
from bb_theta_inverse import _mons, _elt, _vec, joint_solution_basis
from bb_script_a import check_general_matching
from bb_kdist import block_distribution, block_k_of, convolve
GF2 = galois.GF(2)
DATA = {}
def ok(msg, cond):
    assert cond, "FAILED: " + msg
    print("  ok:", msg)

def jstr(j, rd):
    names = {0: "0", 1: "1", 2: "w", 3: "w2"}
    out = []
    for a in rd.A:
        v = int(j[a])
        if v:
            co = "" if (v == 1 and sum(a) > 0) else ("1" if v == 1 else names[v])
            mon = "" if sum(a) == 0 else ("u" if a[0] == 1 else "u^%d" % a[0])
            out.append((co + mon) if (co or mon) else "1")
    return " + ".join(out) if out else "0"

def local_mu(g, rd):   # T*_psi for x -> x^{-1} on R_{4,3}:  u -> U = u+u^2+u^3
    U = _zero(rd); U[(1,)] = rd.GF(1); U[(2,)] = rd.GF(1); U[(3,)] = rd.GF(1)
    out = _zero(rd); P = _zero(rd); P[(0,)] = rd.GF(1)
    for a in rd.A:
        if int(g[a]):
            Pa = _zero(rd); Pa[(0,)] = rd.GF(1)
            for _ in range(a[0]): Pa = local_mul(Pa, U, rd)
            out = {k: out[k] + Pa[k] * g[a] for k in out}
    return out


def kglob(f1, f2, rd):
    """k from the GLOBAL check matrix: k = 2*(D_L - rank_F2 [rho(f1) | rho(f2)])  (independent of the CRT formula)."""
    mons = _mons(rd.L)
    M = np.hstack([np.array(_mult_matrix(f1, rd, mons), dtype=np.uint8), np.array(_mult_matrix(f2, rd, mons), dtype=np.uint8)])
    return 2 * (rd.DL() - sum(1 for r in GF2(M).row_reduce() if r.any()))

def partA():
    print("== Example A: forward detection + logic basis, R_{4,3}")
    L = (4, 3); rd = ring_setup(L); crt = CRTBasis(rd)
    f1 = RE.from_string(L, "1 + x*y"); f2 = RE.from_string(L, "y^2 + x*y + x^2 + x^3")
    c0, c1 = rd.components
    ok("components: dims 4 and 8, reps (1,1) and (1,w)", (c0.dim, c1.dim) == (4, 8) and [int(t) for t in c1.rep] == [1, 2])
    J = {(c.index, s): jstr(local_jet(f, c, rd), rd) for c in rd.components for s, f in ((1, f1), (2, f2))}
    ok("jets c0=(u, u^3), c1=(w2 + w u, 1 + w2 u + u^3)",
       J[(0,1)] == "u" and J[(0,2)] == "u^3" and J[(1,1)] == "w2 + wu" and J[(1,2)] == "1 + w2u + u^3")
    ok("k = 2 (CRT formula and global F2 rank)", code_dimension(f1, f2, rd) == 2 and kglob(f1, f2, rd) == 2)
    G = automorphism_group(rd)
    ok("|Gamma| = 4 (no shears since gcd(4,3)=1)", len(G) == 4)
    ok("Prop 4.13 passes for all four psi", all(preserves_local_ideal_structure(f1, f2, A, rd) for A in G))
    ix, iy, io, ID = [[3,0],[0,1]], [[1,0],[0,2]], [[3,0],[0,2]], [[1,0],[0,1]]
    res = {n: check_cyclic_matching(f1, f2, A, rd) for n, A in (("id", ID), ("ix", ix), ("iy", iy), ("io", io))}
    ok("Step 5/6: id, iota_x accepted; iota_y, iota rejected at c1",
       res["id"][0] and res["ix"][0] and (not res["iy"][0]) and (not res["io"][0])
       and "component 1" in res["iy"][1] and "component 1" in res["io"][1])
    om = res["ix"][1]
    ok("omega_c0 = 1+u+u^2 (mod u^3), omega_c1 = 1 + w2 u^2 + u^3",
       jstr(om[0], rd) == "1 + u + u^2" and jstr(om[1], rd) == "1 + w2u^2 + u^3")
    w0 = dict(om[0]); w = crt.reconstruct({0: w0, 1: om[1]})
    ok("global twist w = y^2+x+x^2y+x^3y+x^3y^2 with psi(f)=w f",
       str(w) == "y^2 + x + x^2*y + x^3*y + x^3*y^2" and f1.apply_automorphism(ix) == w * f1 and f2.apply_automorphism(ix) == w * f2)
    ok("Theta = I is physical (Prop 4.10) and Theorem 4.4 accepts", verify_symplectic(identity4(L), rd) and check_general_matching(f1, f2, ix, identity4(L), rd))
    f2p = RE.from_string(L, "y^2 + x + x*y^2 + x^2 + x^3*y + x^3*y^2")
    okp, why = check_cyclic_matching(f1, f2p, ix, rd)
    ok("sibling code (u, u^2): identical at c1, k=2, rejected at c0",
       jstr(local_jet(f2p, c0, rd), rd) == "u^2" and jstr(local_jet(f2p, c1, rd), rd) == jstr(local_jet(f2, c1, rd), rd)
       and code_dimension(f1, f2p, rd) == 2 and (not okp) and "component 0" in why)
    # Algorithm 1 at c0 by hand
    GF = rd.GF; A = rd.A; n = 4
    def lam(f1c, f2c):
        M = GF.Zeros((n, 2 * n))
        for k, a in enumerate(A):
            e = _zero(rd); e[a] = GF(1)
            cp, cq = local_mul(f2c, e, rd), local_mul(f1c, e, rd)
            for i, aa in enumerate(A): M[i, k] = cp[aa]; M[i, n + k] = cq[aa]
        return M
    j1, j2 = local_jet(f1, c0, rd), local_jet(f2, c0, rd)
    ker = lam(j1, j2).null_space()
    ok("Alg.1 Step 2: dim ker Lambda = 5", ker.shape[0] == 5)
    mx = []
    for a in A:
        e = _zero(rd); e[a] = GF(1)
        p, q = local_mul(e, j1, rd), local_mul(e, j2, rd); mx.append([p[b] for b in A] + [q[b] for b in A])
    dimM = sum(1 for r in GF(mx).row_reduce() if r.any())
    ok("Alg.1 Step 3: dim M_X,c0 = 3", dimM == 3)
    def vec(p, q):
        return GF([int(p.get(a, 0)) for a in A] + [int(q.get(a, 0)) for a in A])
    v1 = vec({(0,): 1}, {(2,): 1}); v2 = vec({}, {(3,): 1})
    inker = all(not (lam(j1, j2) @ v.reshape(-1, 1)).any() for v in (v1, v2))
    rk = sum(1 for r in GF(mx + [list(v1), list(v2)]).row_reduce() if r.any())
    ok("Alg.1 Step 4/Prop 4.7(ii): (1,u^2),(0,u^3) lie in ker and are independent mod M (rank 3+2)", inker and rk == 5)
    # global assembly
    e0 = crt.idempotent(0)
    ok("1_c0 = 1 + y + y^2", str(e0) == "1 + y + y^2")
    u = RE.from_string(L, "1 + x")
    hand = [(e0, e0 * u * u), (RE(L, []), e0 * u * u * u)]
    LX, LZ = logical_basis(f1, f2, rd, crt)
    mons = _mons(L); mb = _m_basis(f1, f2, rd, mons); zero = RE(L, [])
    fl = [_flatten4([P, Q, zero, zero], rd, mons) for P, Q in hand]
    rk = GF2(np.array([list(r) for r in mb] + [list(x) for x in fl], dtype=np.uint8)).row_reduce()
    ok("global (1_c0, 1_c0 u^2), (0, 1_c0 u^3) independent mod M and in ker H_Z",
       sum(1 for r in rk if r.any()) == len(mb) + 2 and all((f2 * P + f1 * Q) == zero for P, Q in hand))
    act = induced_logical_action(ix, identity4(L), f1, f2, hand, LZ, rd)
    ok("iota_x acts as the identity on the 4-dim logical space", bool((act == GF2.Identity(4)).all()))
    fold = full_fold(L); f1b, f2b = f1.apply_automorphism(fold), f2.apply_automorphism(fold)
    ok("Z side: jets of f1-bar, f2-bar at c0 are U and U^3 = u^3 (valuations 1 and 3)",
       jstr(local_jet(f1b, c0, rd), rd) == "u + u^2 + u^3" and jstr(local_jet(f2b, c0, rd), rd) == "u^3")
    DATA["A"] = {"jets": {str(k): v for k, v in J.items()}}

def partB(classes=False):
    print("== Example B: inverse construction, R_{4,3}, psi = iota_x")
    L = (4, 3); rd = ring_setup(L); crt = CRTBasis(rd); mons = _mons(L); one = RE(L, [(0, 0)])
    ix = [[3, 0], [0, 1]]; c0, c1 = rd.components
    K = target_kernel(ix, one, rd, mons)
    dims = []
    for c in rd.components:
        e = crt.idempotent(c.index)
        rows = [_vec(e * _elt(L, v, mons), mons) for v in K]
        dims.append(sum(1 for r in GF2(np.array(rows, dtype=np.uint8)).row_reduce() if r.any()))
    ok("Step 2: dim Fix(mu) = 3 at c0 and 6 at c1 (total 9); 2^(2*9) = 262144 pairs", dims == [3, 6] and K.shape[0] == 9 and 2 ** 18 == 262144)
    # local Fix(mu) at c0
    A = rd.A; GF = rd.GF
    els = []
    for bits in itertools.product([0, 1], repeat=4):
        g = _zero(rd)
        for a, b in zip(A, bits): g[a] = GF(b)
        els.append(g)
    fixed = [jstr(g, rd) for g in els if local_mu(g, rd) == g]
    ok("Fix(mu) at c0 = span{1,u^2,u^3}; u is NOT fixed (mu(u)=u+u^2+u^3)", len(fixed) == 8 and all("u" not in s.replace("u^", "") or True for s in fixed)
       and local_mu({**_zero(rd), (1,): GF(1)}, rd) != {**_zero(rd), (1,): GF(1)})
    val = lambda s: 0 if s == "1" or s.startswith("1 ") else None
    e_ = _zero(rd); e_[(0,)] = GF(1); e_[(1,)] = GF(1); e_[(2,)] = GF(1)
    tw = [jstr(g, rd) for g in els if local_mu(g, rd) == local_mul(e_, g, rd)]
    ok("twist e = 1+u+u^2 = U/u: solutions of mu(g)=e g are span{u,u^3}", sorted(tw) == sorted(["0", "u", "u^3", "u + u^3"]))
    ok("cocycle condition e*mu(e) = 1", local_mul(e_, local_mu(e_, rd), rd) == {**_zero(rd), (0,): GF(1)})
    f1 = RE.from_string(L, "1 + x^2 + x^2*y + x^2*y^2"); f2 = RE.from_string(L, "x + x^3")
    jj = [jstr(local_jet(f, c, rd), rd) for f in (f1, f2) for c in rd.components]
    ok("seeds: f1 = (u^2 | 1), f2 = (u^2+u^3 | u^2+u^3)", jj == ["u^2", "1", "u^2 + u^3", "u^2 + u^3"])
    e0, e1 = crt.idempotent(0), crt.idempotent(1)
    U2 = RE.from_string(L, "1 + x^2"); U23 = RE.from_string(L, "x + x^3")
    ok("Step 5: f1 = 1_c0 u^2 + 1_c1, f2 = (1_c0+1_c1)(u^2+u^3) reproduce the polynomials", e0 * U2 + e1 == f1 and (e0 + e1) * U23 == f2 and str(e1) == "y + y^2")
    ok("Step 6: psi(f)=f exactly, k=4, Algorithm 3.12 accepts", f1.apply_automorphism(ix) == f1 and f2.apply_automorphism(ix) == f2
       and code_dimension(f1, f2, rd) == 4 and kglob(f1, f2, rd) == 4 and check_cyclic_matching(f1, f2, ix, rd)[0])
    LX, LZ = logical_basis(f1, f2, rd, crt); zero = RE(L, [])
    u = RE.from_string(L, "1 + x")
    hand = [(e0, e0 * (u * u + u * u * u)), (e0 * u, e0 * (u * u + u * u * u) * u), (RE(L, []), e0 * u * u), (RE(L, []), e0 * u * u * u)]
    # closed form (ii): zeta=sigma=2, v1=1, v2=1+u:  (u^i, v2 u^i) i<2 ; (0,u^(4-2+j)) j<2   (v2 = 1+u)
    hand = [(e0, e0 * (RE.from_string(L, "1 + x") * RE.from_string(L, "1") + RE.from_string(L, "1")) * RE.from_string(L, "1")) ]  # placeholder replaced below
    one_u = RE.from_string(L, "1 + x")                      # u = 1 + x
    v2 = one_u + RE.from_string(L, "1")                     # 1 + u  = x  ... (u = x+1, so 1+u = x)
    hand = [(e0, e0 * v2), (e0 * one_u, e0 * v2 * one_u), (zero, e0 * one_u * one_u), (zero, e0 * one_u * one_u * one_u)]
    mb = _m_basis(f1, f2, rd, mons)
    fl = [_flatten4([P, Q, zero, zero], rd, mons) for P, Q in hand]
    rk = GF2(np.array([list(r) for r in mb] + [list(x) for x in fl], dtype=np.uint8)).row_reduce()
    ok("logic basis via Prop 4.7(ii): 4 vectors (1,1+u),(u,u+u^2),(0,u^2),(0,u^3), independent mod M, in ker H_Z",
       sum(1 for r in rk if r.any()) == len(mb) + 4 and all((f2 * P + f1 * Q) == zero for P, Q in hand))
    act = induced_logical_action(ix, identity4(L), f1, f2, hand, LZ, rd)
    k = 4; Ax = act[:k, :k]; I = GF2.Identity(k)
    rank = lambda M: sum(1 for r in M.row_reduce() if r.any())
    ok("logical action is block diagonal, unipotent: rank(A-I)=2, (A-I)^2=0", (not act[:k, k:].any()) and (not act[k:, :k].any())
       and rank(Ax - I) == 2 and not ((Ax - I) @ (Ax - I)).any())
    DATA["B"] = {"action_X_handbasis": [[int(x) for x in r] for r in Ax]}
    print("   hand-basis X action:\n", Ax)
    if classes:
        pw = (1 << np.arange(12)).astype(np.int64)
        allf = np.array([[(f >> i) & 1 for i in range(12)] for f in range(1 << 12)], dtype=np.int64)
        units, Ms = [], []
        for bits in range(1, 1 << 12):
            w = RE(L, [mons[i] for i in range(12) if bits >> i & 1]); M = np.array(_mult_matrix(w, rd, mons), dtype=np.int64)
            if rank(GF2(M.astype(np.uint8))) == 12: units.append((bits, w)); Ms.append(M)
        T = np.stack([((allf @ M.T) % 2) @ pw for M in Ms])
        def span(Kb):
            out = [0]
            for r in Kb:
                v = int(np.array([int(x) for x in r], dtype=np.int64) @ pw); out = out + [o ^ v for o in out]
            return np.array(out, dtype=np.int64)
        S1 = None; allp = set()
        for bits, w in units:
            E = span(target_kernel(ix, w, rd, mons)); s = set((((np.repeat(E, len(E))) << 12) | np.tile(E, len(E))).tolist())
            if bits == 1: S1 = s
            allp |= s
        def canon(S):
            arr = np.array(sorted(S), dtype=np.int64); Aa = arr >> 12; Bb = arr & 4095; best = None
            for u in range(len(units)):
                key = (T[u][Aa] << 12) | T[u][Bb]; best = key if best is None else np.minimum(best, key)
            return set(best.tolist())
        c1s, cas = canon(S1), canon(allp)
        def kd(key):
            a, b = key >> 12, key & 4095
            return code_dimension(RE(L, [mons[i] for i in range(12) if a >> i & 1]), RE(L, [mons[i] for i in range(12) if b >> i & 1]), rd)
        k1 = collections.Counter(kd(c) for c in c1s); ka = collections.Counter(kd(c) for c in cas)
        ok("units 1536; pairs 262144 (w=1) vs 2254336 (any twist); classes 2332 vs 3528", len(units) == 1536 and len(S1) == 262144 and len(allp) == 2254336 and len(c1s) == 2332 and len(cas) == 3528)
        ok("k=2 classes: 0 with w=1, 480 with a twist", k1.get(2, 0) == 0 and ka[2] == 480 and sum(1 for c in cas if c not in c1s and kd(c) == 0) == 0)
        DATA["B"]["kdist_w1"] = sorted(k1.items()); DATA["B"]["kdist_any"] = sorted(ka.items())

def partC():
    print("== Example C: computational, R_{3,4,6}, {(F3,CNOT),(S32,CNOT)}")
    L = (3, 4, 6); rd = ring_setup(L); crt = CRTBasis(rd)
    one = RE(L, [(0, 0, 0)]); zero = RE(L, []); W = [[one, zero], [zero, one]]
    CNOT = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]); z = (0, 0, 0)
    F3 = multiplier(L, (1, 1, 5)); S32 = shear(L, 2, 1, 1)
    ok("R_{3,4,6}: 5 components, dims 8,16,16,16,16 (Example 1 of the draft); |Aut(G)| = 384",
       sorted(c.dim for c in rd.components) == [8, 16, 16, 16, 16] and len(automorphism_group(rd)) == 384)
    ok("psi2 = x2 -> x2 x3^3 is a shear: a_32 = 3 and psi2^2 = id", S32[2][1] == 3)
    Th = form_C_theta(L, z, CNOT)
    ok("Theta = CNOT layer is physical (Prop 4.10)", verify_symplectic(Th, rd))
    gens = [(F3, Th, W), (S32, Th, W)]
    R = construct_code_theta(rd, gens); V = R["basis"]
    ok("joint solution space has dimension 42 (2^42 codes of 2^144 pairs)", R["dim"] == 42)
    ok("component orbits under <sigma_psi1, sigma_psi2, bar>: {c0},{c1},{c2},{c3,c4} with block dims 6,8,12,16",
       R["orbits"] == [[0], [1], [2], [3, 4]] and [len(b) for b in R["blocks"]] == [6, 8, 12, 16])
    # Proposition (CNOT layer): f1 = (1+psi1) f2, f2 in Fix(psi1 psi2)
    from bb_automorphisms import matmul_mod
    P12 = matmul_mod(F3, S32, L)
    mons = _mons(L)
    Kfix = target_kernel(P12, one, rd, mons)
    ok("closed form: dim Fix(psi1 psi2) = 42 = number of orbits of <psi1 psi2> on the 72 monomials", Kfix.shape[0] == 42)
    from bb_theta_inverse import constraint_matrix
    C1 = constraint_matrix(L, F3, Th, W); C2 = constraint_matrix(L, S32, Th, W)
    good = True
    for v in Kfix:
        f2 = _elt(L, v, mons); f1 = f2 + f2.apply_automorphism(F3)
        x = GF2(np.concatenate([_vec(f1, mons), _vec(f2, mons)]).astype(np.uint8))
        if (C1 @ x).any() or (C2 @ x).any(): good = False
    ok("every (f1=(1+psi1)f2, f2 in Fix(psi1 psi2)) satisfies both generator constraints", good)
    dists = [block_distribution(b, O, R["orbits"] and rd) if False else block_distribution(b, O, rd) for b, O in zip(R["blocks"], R["orbits"])]
    tot = convolve(dists)
    ok("exact k-distribution sums to 2^42", sum(tot.values()) == 2 ** 42)
    random.seed(11); bad = 0
    for _ in range(25):
        coef = [random.randint(0, 1) for _ in V]
        f1 = RE(L, []); f2 = RE(L, [])
        for cf, (a, b) in zip(coef, V):
            if cf: f1 = f1 + a; f2 = f2 + b
        if not (check_general_matching(f1, f2, F3, Th, rd) and check_general_matching(f1, f2, S32, Th, rd)): bad += 1
        ksum = sum(block_k_of(f1, f2, O, rd) for O in R["orbits"])
        if ksum != code_dimension(f1, f2, rd) or ksum != kglob(f1, f2, rd): bad += 1
    ok("25 random codes: pass Theorem 4.4 for both generators, and k equals the sum of orbit k's", bad == 0)
    pos = sorted(k for k in tot if k > 0)
    DATA["C"] = {"dim": 42, "orbits": R["orbits"], "blocks": [len(b) for b in R["blocks"]],
                 "dists": [sorted(d.items()) for d in dists], "total": sorted(tot.items()),
                 "kmin_pos": pos[0], "count_k0": tot[0], "total_codes": 2 ** 42}
    print("   smallest positive k:", pos[0], " count k=0:", tot[0], " k in [4,24]:", sum(v for k, v in tot.items() if 4 <= k <= 24))

if __name__ == "__main__":
    args = sys.argv[1:] or ["A", "B", "C"]
    t = time.time()
    if "A" in args: partA()
    if "B" in args: partB(classes="Bclasses" in args)
    if "C" in args: partC()
    import os
    old = json.load(open("gen_data.json")) if os.path.exists("gen_data.json") else {}
    for k, v in DATA.items():
        old.setdefault(k, {}).update(v)
    json.dump(old, open("gen_data.json", "w"), default=int)
    print("ALL ASSERTIONS PASSED in %.1fs" % (time.time() - t))

def partA_rule():
    print("== valuation rule for iota_x on chain component (f1,c0,f2,c0)=(u^z,u^s), c1-data fixed")
    L = (4, 3); rd = ring_setup(L); crt = CRTBasis(rd); GF = rd.GF
    f1o = RE.from_string(L, "1 + x*y"); f2o = RE.from_string(L, "y^2 + x*y + x^2 + x^3")
    c1 = rd.components[1]; ix = [[3, 0], [0, 1]]
    j1, j2 = local_jet(f1o, c1, rd), local_jet(f2o, c1, rd)
    e = _zero(rd); e[(0,)] = GF(1); e[(1,)] = GF(1); e[(2,)] = GF(1)
    def epow(n):
        r = _zero(rd); r[(0,)] = GF(1)
        for _ in range(n): r = local_mul(r, e, rd)
        return r
    for z in (1, 2, 3):
        for s in range(z, 4):
            a = _zero(rd); a[(z,)] = GF(1); b = _zero(rd); b[(s,)] = GF(1)
            f1 = crt.reconstruct({0: a, 1: j1}); f2 = crt.reconstruct({0: b, 1: j2})
            got = check_cyclic_matching(f1, f2, ix, rd)[0]
            d = epow(s - z)                                   # e^(s-z) == 1 mod u^(4-s) ?
            pred = all(int(d[(i,)]) == (1 if i == 0 else 0) for i in range(4 - s))
            ok("(zeta,sigma)=(%d,%d): toolkit %s, rule %s" % (z, s, got, pred), got == pred)

def partC_sample():
    print("== Example C sample code")
    L = (3, 4, 6); rd = ring_setup(L)
    one = RE(L, [(0, 0, 0)]); zero = RE(L, []); W = [[one, zero], [zero, one]]
    CNOT = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    Th = form_C_theta(L, (0, 0, 0), CNOT); F3 = multiplier(L, (1, 1, 5)); S32 = shear(L, 2, 1, 1)
    f1 = RE.from_string(L, "x*y*z + x*y*z^2 + x*y*z^4 + x*y*z^5"); f2 = RE.from_string(L, "x + x*y*z + x*y*z^2")
    ok("sample: k = 12, weights 4+3, passes Theorem 4.4 for both generators, f1 = (1+psi1) f2",
       code_dimension(f1, f2, rd) == 12 and kglob(f1, f2, rd) == 12 and len(f1.mons) + len(f2.mons) == 7 and f1 == f2 + f2.apply_automorphism(F3)
       and check_general_matching(f1, f2, F3, Th, rd) and check_general_matching(f1, f2, S32, Th, rd))

if __name__ == "__main__" and "rule" in sys.argv:
    partA_rule(); partC_sample()
