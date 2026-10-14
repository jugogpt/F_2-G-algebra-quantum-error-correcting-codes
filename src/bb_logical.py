"""
Logical operators: Corollary 2's dimension count, then Algorithm 1's local
kernel/quotient basis at each component (using galois's built-in
.null_space()/row-space machinery rather than hand-rolled RREF, since it's
available and exactly what Sec 5.6.1 needs), assembled globally (Sec 5.8)
via bb_crt's CRT reconstruction, with the Z-side obtained by the
f1,f2 -> f2bar,f1bar substitution of Sec 5.9 (using the same "apply globally
then take jet" equivalence used everywhere else in this codebase, which
automatically handles the c -> c* relabeling Sec 5.9 describes).
"""

from __future__ import annotations
from bb_core import RingData
from bb_ring_elt import RingElement, local_jet
from bb_local_ring import local_mul, _zero
from bb_automorphisms import local_ideal_basis, full_fold
from bb_crt import CRTBasis, kappa_generator


def code_dimension(f1: RingElement, f2: RingElement, rd: RingData) -> int:
    """Corollary 2 / eq. (54): k = 2 * sum_c dim_F2(J_c / I_c)."""
    total = 0
    for c in rd.components:
        Ic = local_ideal_basis([local_jet(f1, c, rd), local_jet(f2, c, rd)], rd)
        dim_Jc = c.dim  # F2-dimension of J_c
        dim_Ic = Ic.shape[0] * c.Dc  # Ic.shape[0] is the kappa_c-rank; each row is Dc F2-dims
        total += dim_Jc - dim_Ic
    return 2 * total


def _rank(rows, rd):
    """kappa_c-rank of a list of equal-length GF(2^N)-vectors (as galois arrays)."""
    if not rows:
        return 0
    stacked = rd.GF(rows)
    R = stacked.row_reduce()
    return sum(1 for row in R if any(int(x) != 0 for x in row))


def local_kernel_quotient_basis(f1c: dict, f2c: dict, rd: RingData, Dc: int):
    """Algorithm 1 (Sec 5.6): an F2-basis of ker(Lambda_c)/MX,c at one
    component, where Lambda_c(p,q) = f2c*p + f1c*q. Returns a list of (p,q)
    dicts (each an element of J_c^2, F2-basis after the kappa_c -> F2
    restriction of Step 5). Dc is the component's residue-field degree,
    needed to pick how many power-basis copies Step 5 produces."""
    GF = rd.GF
    A = rd.A
    n = len(A)

    # Step 1: matrix of Lambda_c ("form the matrix via the multiplication
    # table"): n rows (output coords in A), 2n columns (p_a then q_a unknowns).
    M = GF.Zeros((n, 2 * n))
    for k, a in enumerate(A):
        e = _zero(rd)
        e[a] = GF(1)
        col_p = local_mul(f2c, e, rd)
        col_q = local_mul(f1c, e, rd)
        for i, aa in enumerate(A):
            M[i, k] = col_p[aa]
            M[i, n + k] = col_q[aa]

    # Step 2: kernel of Lambda_c (galois's null_space in place of hand-rolled RREF)
    ker_basis = M.null_space()  # rows: length-2n vectors over kappa_c (in GF(2^N))

    # Step 3: basis of MX,c = {(u^a f1c, u^a f2c) : a in A}
    mx_rows = []
    for a in A:
        e = _zero(rd)
        e[a] = GF(1)
        pf1 = local_mul(e, f1c, rd)
        pf2 = local_mul(e, f2c, rd)
        mx_rows.append([pf1[aa] for aa in A] + [pf2[aa] for aa in A])
    mx_stacked = GF(mx_rows) if mx_rows else GF.Zeros((0, 2 * n))
    mx_reduced = mx_stacked.row_reduce()
    mx_basis = [row for row in mx_reduced if any(int(x) != 0 for x in row)]

    # Step 4: quotient basis -- greedily accept ker_c vectors that increase
    # rank beyond the running (MX,c + accepted) set (mirrors Sec 5.6.1's
    # "reduce x against E; if nonzero, keep it").
    running = [GF(row) for row in mx_basis]
    quotient_reps = []
    for v in ker_basis:
        trial = running + [v]
        if _rank(trial, rd) > _rank(running, rd):
            running.append(v)
            quotient_reps.append(v)

    # Step 5: restrict scalars to F2 -- expand each kappa_c-basis rep into
    # Dc many F2-basis vectors via the power basis {1,beta,...,beta^{Dc-1}}.
    beta = kappa_generator(Dc, rd) if Dc > 1 else GF(1)
    out = []
    for v in quotient_reps:
        for t in range(Dc):
            scaled = v * (beta ** t)
            p = {A[i]: scaled[i] for i in range(n)}
            q = {A[i]: scaled[n + i] for i in range(n)}
            out.append((p, q))
    return out


def logical_basis(f1: RingElement, f2: RingElement, rd: RingData, crt: CRTBasis = None):
    """Full construction of L_X and L_Z (Sec 5.8/5.9): local quotient basis
    at every component, assembled into global (P,Q) pairs via CRT
    reconstruction. Returns (LX_pairs, LZ_pairs), each a list of
    (RingElement, RingElement)."""
    if crt is None:
        crt = CRTBasis(rd)
    L = rd.L
    fold = full_fold(L)
    f2bar = f2.apply_automorphism(fold)
    f1bar = f1.apply_automorphism(fold)

    def build(genA: RingElement, genB: RingElement):
        pairs = []
        for c in rd.components:
            gA_c = local_jet(genA, c, rd)
            gB_c = local_jet(genB, c, rd)
            local_reps = local_kernel_quotient_basis(gA_c, gB_c, rd, c.Dc)
            for (p, q) in local_reps:
                P = crt.reconstruct({c.index: p})
                Q = crt.reconstruct({c.index: q})
                pairs.append((P, Q))
        return pairs

    LX = build(f1, f2)
    LZ = build(f2bar, f1bar)
    return LX, LZ


if __name__ == "__main__":
    from bb_core import ring_setup

    # --- Validation: §5.10's complete worked example ---
    # L=(2,2), f1=x+y, f2=1+xy, single component c0, expect dim LX = 4,
    # matching 2*dim(Jc0/Ic0) = 2*2 = 4, and the SAME SPAN as the paper's
    # stated basis {(u+v,0),(uv,0),(1+u,1),(u,u)} (basis choice needn't be
    # identical, only the subspace).
    rd = ring_setup((2, 2))
    f1 = RingElement.from_string((2, 2), "x + y")
    f2 = RingElement.from_string((2, 2), "1 + x*y")

    k = code_dimension(f1, f2, rd)
    print(f"code_dimension: k = {k}  (expect 4, matching §5.10's 'k=4=2dim(Jc0/Ic0)')")
    assert k == 4
    print("  PASS\n")

    crt = CRTBasis(rd)
    LX, LZ = logical_basis(f1, f2, rd, crt)
    print(f"dim LX = {len(LX)}  (expect 4)")
    print(f"dim LZ = {len(LZ)}  (expect 4, by symmetry of Prop 5.3)")
    assert len(LX) == 4 and len(LZ) == 4
    print("  PASS\n")

    # Compare row-space of my computed LX (as (p,q) jets at c0) against the
    # paper's explicitly stated basis.
    c0 = rd.components[0]
    A = rd.A
    GF = rd.GF

    def jetpair_to_vec(P, Q):
        jp, jq = local_jet(P, c0, rd), local_jet(Q, c0, rd)
        return [jp[a] for a in A] + [jq[a] for a in A]

    mine_rows = [jetpair_to_vec(P, Q) for (P, Q) in LX]
    mine_reduced = GF(mine_rows).row_reduce()
    mine_span = [row for row in mine_reduced if any(int(x) != 0 for x in row)]

    # paper's stated K1..K4: (u+v,0),(uv,0),(1+u,1),(u,u), in (1,u,v,uv) order for both P and Q
    def mk(pcoef, qcoef):
        # pcoef/qcoef given as dict over (1,u,v,uv) i.e. rd.A order (0,0),(1,0),(0,1),(1,1)
        return [pcoef.get(a, 0) for a in A] + [qcoef.get(a, 0) for a in A]

    paper_rows = [
        mk({(1, 0): 1, (0, 1): 1}, {}),                       # (u+v, 0)
        mk({(1, 1): 1}, {}),                                   # (uv, 0)
        mk({(0, 0): 1, (1, 0): 1}, {(0, 0): 1}),               # (1+u, 1)
        mk({(1, 0): 1}, {(1, 0): 1}),                          # (u, u)
    ]
    paper_reduced = GF(paper_rows).row_reduce()
    paper_span = [row for row in paper_reduced if any(int(x) != 0 for x in row)]

    same_span = (len(mine_span) == len(paper_span) and
                 all((r1 == r2).all() for r1, r2 in zip(mine_span, paper_span)))
    print(f"my LX span == paper's stated LX span? {same_span}")
    assert same_span
    print("  PASS -- Algorithm 1 + global assembly reproduces §5.10 exactly")
