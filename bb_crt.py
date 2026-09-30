"""CRT reconstruction: the inverse of R_L -> prod_c J_c, built as an explicit F2-linear map."""

from __future__ import annotations
from itertools import product as iproduct
import galois
from bb_core import RingData
from bb_ring_elt import RingElement, local_jet


def kappa_generator(Dc: int, rd: RingData):
    """a generator beta of GF(2^Dc)^*, embedded in GF(2^N), so {1,beta,...,beta^{Dc-1}} is an F2-basis."""
    GF = rd.GF
    order = GF.order - 1
    sub_order = 2 ** Dc - 1
    assert order % sub_order == 0
    return GF.primitive_element ** (order // sub_order)


def _decompose(val, beta, Dc, rd):
    """val in GF(2^Dc) -> its Dc F2 coefficients in the power basis {1,beta,...,beta^{Dc-1}}."""
    GF2 = galois.GF(2)
    powers = [beta ** t for t in range(Dc)]
    Mat = GF2.Zeros((rd.N, Dc))
    for t, p in enumerate(powers):
        Mat[:, t] = p.vector()
    target = val.vector()
    aug = GF2.Zeros((rd.N, Dc + 1))
    aug[:, :Dc] = Mat
    aug[:, Dc] = target
    R = aug.row_reduce()
    coeffs = [0] * Dc
    r = 0
    for col in range(Dc):
        if r < R.shape[0] and int(R[r, col]) == 1 and all(int(R[r, c]) == 0 for c in range(col)):
            coeffs[col] = int(R[r, Dc])
            r += 1
    return coeffs


def _recompose(coeffs, beta, rd):
    GF = rd.GF
    val = GF(0)
    for t, b in enumerate(coeffs):
        if b:
            val = val + beta ** t
    return val


class CRTBasis:
    """the linear map between R_L's monomial coefficients and the flattened local-jet data, and its inverse."""

    def __init__(self, rd: RingData):
        self.rd = rd
        self.betas = {c.index: kappa_generator(c.Dc, rd) for c in rd.components}
        self.layout = []
        for c in rd.components:
            for a in rd.A:
                self.layout.append((c.index, a))
        self.DL = rd.DL()
        assert sum(c.Dc for c in rd.components) * len(rd.A) == self.DL

        self._build_matrix()

    def _flatten(self, jets_by_component: dict) -> list:
        """{c.index: {a: GF elt}} (missing entries treated as zero) -> a flat list of DL bits."""
        out = []
        for (cidx, a) in self.layout:
            jet = jets_by_component.get(cidx)
            val = jet[a] if jet is not None else self.rd.GF(0)
            Dc = self.rd.components[cidx].Dc
            out.extend(_decompose(val, self.betas[cidx], Dc, self.rd))
        return out

    def _build_matrix(self):
        rd = self.rd
        GF2 = galois.GF(2)
        DL = self.DL
        M = GF2.Zeros((DL, DL))
        monomials = [tuple(m) for m in iproduct(*[range(li) for li in rd.L])]
        assert len(monomials) == DL
        for col, m in enumerate(monomials):
            f = RingElement(rd.L, [m])
            jets = {c.index: local_jet(f, c, rd) for c in rd.components}
            flat = self._flatten(jets)
            for row, bit in enumerate(flat):
                M[row, col] = bit
        self._monomials = monomials
        self._M = M
        self._Minv = None

    def inverse_matrix(self):
        if self._Minv is None:
            self._Minv = _f2_inverse(self._M)
        return self._Minv

    def reconstruct(self, jets_by_component: dict) -> RingElement:
        """reconstruct the unique global element with this local-jet data, zero where unmentioned."""
        flat = self._flatten(jets_by_component)
        GF2 = galois.GF(2)
        vec = GF2(flat).reshape((self.DL, 1))
        Minv = self.inverse_matrix()
        sol = (Minv @ vec).flatten()
        mons = [self._monomials[i] for i in range(self.DL) if int(sol[i]) == 1]
        return RingElement(self.rd.L, mons)

    def idempotent(self, c_index: int) -> RingElement:
        """1_c: value 1 at component c_index, 0 elsewhere."""
        rd = self.rd
        one_local = {tuple(0 for _ in rd.E): rd.GF(1)}
        for a in rd.A:
            one_local.setdefault(a, rd.GF(0))
        return self.reconstruct({c_index: one_local})


def _f2_inverse(M):
    """invert a square matrix over GF(2) via augmented row reduction."""
    GF2 = galois.GF(2)
    n = M.shape[0]
    aug = GF2.Zeros((n, 2 * n))
    aug[:, :n] = M
    for i in range(n):
        aug[i, n + i] = GF2(1)
    R = aug.row_reduce()
    return R[:, n:]


if __name__ == "__main__":
    from bb_core import ring_setup

    # round-trip check on a ring with a nontrivial residue field.
    rd = ring_setup((3, 4, 6))
    crt = CRTBasis(rd)
    print(f"L=(3,4,6): DL={rd.DL()}, {len(rd.components)} components, "
          f"degrees {[c.Dc for c in rd.components]}")

    target_c = next(c for c in rd.components if c.Dc == 2)
    target_jet = {a: rd.GF(1) if sum(a) % 2 == 0 else rd.GF(0) for a in rd.A}
    beta = kappa_generator(target_c.Dc, rd)
    target_jet[tuple(0 for _ in rd.E)] = beta

    f = crt.reconstruct({target_c.index: target_jet})
    recovered = local_jet(f, target_c, rd)
    ok = all(recovered[a] == target_jet[a] for a in rd.A)
    print(f"round-trip reconstruction at a Dc=2 component matches target? {ok}")
    assert ok
    print("  pass\n")

    # confirm the result is zero at every other component.
    other_zero = True
    for c in rd.components:
        if c.index == target_c.index:
            continue
        j = local_jet(f, c, rd)
        if any(int(v) != 0 for v in j.values()):
            other_zero = False
    print(f"zero at every other component? {other_zero}")
    assert other_zero
    print("  pass\n")

    # idempotent sanity check: 1_c0 squared equals itself.
    c0 = rd.components[0]
    e0 = crt.idempotent(c0.index)
    e0_sq = e0 * e0
    print(f"1_c0 == 1_c0^2 (idempotent)? {e0 == e0_sq}")
    assert e0 == e0_sq
    print("  pass")
