"""Logical operators: the Corollary 2 dimension count, Algorithm 1's local kernel/quotient basis, and global assembly."""

from __future__ import annotations
from bb_core import RingData
from bb_ring_elt import RingElement, local_jet
from bb_local_ring import local_mul, _zero
from bb_automorphisms import local_ideal_basis, full_fold
from bb_crt import CRTBasis, kappa_generator


def code_dimension(f1: RingElement, f2: RingElement, rd: RingData) -> int:
    """Corollary 2: k = 2 * sum_c dim_F2(J_c / I_c)."""
    total = 0
    for c in rd.components:
        Ic = local_ideal_basis([local_jet(f1, c, rd), local_jet(f2, c, rd)], rd)
        dim_Jc = c.dim
        dim_Ic = Ic.shape[0] * c.Dc
        total += dim_Jc - dim_Ic
    return 2 * total


def _rank(rows, rd):
    """kappa_c-rank of a list of equal-length GF(2^N) vectors."""
    if not rows:
        return 0
    stacked = rd.GF(rows)
    R = stacked.row_reduce()
    return sum(1 for row in R if any(int(x) != 0 for x in row))


def local_kernel_quotient_basis(f1c: dict, f2c: dict, rd: RingData, Dc: int):
    """Algorithm 1: an F2-basis of ker(Lambda_c)/MX,c at one component, as a list of (p,q) pairs."""
    GF = rd.GF
    A = rd.A
    n = len(A)

    # step 1: the matrix of Lambda_c(p,q) = f2c*p + f1c*q.
    M = GF.Zeros((n, 2 * n))
    for k, a in enumerate(A):
        e = _zero(rd)
        e[a] = GF(1)
        col_p = local_mul(f2c, e, rd)
        col_q = local_mul(f1c, e, rd)
        for i, aa in enumerate(A):
            M[i, k] = col_p[aa]
            M[i, n + k] = col_q[aa]

    # step 2: the kernel of Lambda_c.
    ker_basis = M.null_space()

    # step 3: a basis of MX,c = {(u^a f1c, u^a f2c) : a in A}.
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

    # step 4: greedily accept kernel vectors that increase rank beyond the running span.
    running = [GF(row) for row in mx_basis]
    quotient_reps = []
    for v in ker_basis:
        trial = running + [v]
        if _rank(trial, rd) > _rank(running, rd):
            running.append(v)
            quotient_reps.append(v)

    # step 5: restrict scalars to F2 via the power basis {1,beta,...,beta^{Dc-1}}.
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
    """construct L_X and L_Z: the local quotient basis at every component, assembled by CRT reconstruction."""
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

    # L=(2,2), f1=x+y, f2=1+xy, single component: expect k=4 and dim LX = dim LZ = 4.
    rd = ring_setup((2, 2))
    f1 = RingElement.from_string((2, 2), "x + y")
    f2 = RingElement.from_string((2, 2), "1 + x*y")

    k = code_dimension(f1, f2, rd)
    print(f"code_dimension: k = {k}  (expect 4)")
    assert k == 4
    print("  pass\n")

    crt = CRTBasis(rd)
    LX, LZ = logical_basis(f1, f2, rd, crt)
    print(f"dim LX = {len(LX)}  (expect 4)")
    print(f"dim LZ = {len(LZ)}  (expect 4, by symmetry)")
    assert len(LX) == 4 and len(LZ) == 4
    print("  pass\n")

    # compare the row space of the computed LX against the paper's stated basis.
    c0 = rd.components[0]
    A = rd.A
    GF = rd.GF

    def jetpair_to_vec(P, Q):
        jp, jq = local_jet(P, c0, rd), local_jet(Q, c0, rd)
        return [jp[a] for a in A] + [jq[a] for a in A]

    mine_rows = [jetpair_to_vec(P, Q) for (P, Q) in LX]
    mine_reduced = GF(mine_rows).row_reduce()
    mine_span = [row for row in mine_reduced if any(int(x) != 0 for x in row)]

    def mk(pcoef, qcoef):
        return [pcoef.get(a, 0) for a in A] + [qcoef.get(a, 0) for a in A]

    paper_rows = [
        mk({(1, 0): 1, (0, 1): 1}, {}),
        mk({(1, 1): 1}, {}),
        mk({(0, 0): 1, (1, 0): 1}, {(0, 0): 1}),
        mk({(1, 0): 1}, {(1, 0): 1}),
    ]
    paper_reduced = GF(paper_rows).row_reduce()
    paper_span = [row for row in paper_reduced if any(int(x) != 0 for x in row)]

    same_span = (len(mine_span) == len(paper_span) and
                 all((r1 == r2).all() for r1, r2 in zip(mine_span, paper_span)))
    print(f"my LX span == paper's stated LX span? {same_span}")
    assert same_span
    print("  pass")
