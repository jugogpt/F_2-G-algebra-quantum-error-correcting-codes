"""RingElement: elements of R_L = F2[x_1,...,x_k]/<x_i^{l_i}-1>, stored as the set of "on" monomials."""

from __future__ import annotations
from itertools import product as iproduct
from bb_core import RingData, Component, ring_setup


class RingElement:
    """an F2-linear combination of monomials, stored as a frozenset of exponent tuples with coefficient 1."""

    __slots__ = ("L", "mons")

    def __init__(self, L: tuple, mons):
        self.L = L
        self.mons = frozenset(tuple(m) for m in mons)

    @staticmethod
    def from_string(L: tuple, s: str, varnames=None) -> "RingElement":
        """parse strings like '1 + x + x*y' (k<=26 uses x,y,z,... by default)."""
        k = len(L)
        if varnames is None:
            varnames = list("xyzwvutsrqponmlkjihgfedcba")[:k] if k <= 26 else [f"x{i}" for i in range(k)]
        mons = set()
        for term in s.replace(" ", "").split("+"):
            expo = [0] * k
            if term == "1":
                mons.add(tuple(expo))
                continue
            for factor in term.split("*"):
                if "^" in factor:
                    name, p = factor.split("^")
                    p = int(p)
                else:
                    name, p = factor, 1
                idx = varnames.index(name)
                expo[idx] = (expo[idx] + p) % L[idx]
            mons.add(tuple(expo))
        return RingElement(L, mons)

    def __add__(self, other: "RingElement") -> "RingElement":
        assert self.L == other.L
        return RingElement(self.L, self.mons.symmetric_difference(other.mons))

    def __mul__(self, other: "RingElement") -> "RingElement":
        """group-algebra convolution with F2 coefficients accumulated by xor."""
        assert self.L == other.L
        k = len(self.L)
        acc = set()
        for a in self.mons:
            for b in other.mons:
                c = tuple((a[i] + b[i]) % self.L[i] for i in range(k))
                if c in acc:
                    acc.discard(c)
                else:
                    acc.add(c)
        return RingElement(self.L, acc)

    def __eq__(self, other):
        return isinstance(other, RingElement) and self.L == other.L and self.mons == other.mons

    def __repr__(self):
        if not self.mons:
            return "0"
        terms = []
        varnames = list("xyzwvutsrqponmlkjihgfedcba")[:len(self.L)]
        for m in sorted(self.mons):
            if all(e == 0 for e in m):
                terms.append("1")
            else:
                parts = [f"{varnames[i]}^{m[i]}" if m[i] > 1 else varnames[i]
                         for i in range(len(m)) if m[i] != 0]
                terms.append("*".join(parts))
        return " + ".join(terms)

    def evaluate(self, alpha, GF):
        """evaluate f(alpha_1,...,alpha_k) in GF, the residue term of the jet."""
        total = GF(0)
        for m in self.mons:
            term = GF(1)
            for ai, ei in zip(alpha, m):
                term = term * (ai ** ei)
            total = total + term
        return total

    def apply_automorphism(self, Amat) -> "RingElement":
        """psi(x^j) = x^{A j mod L}, with Amat[i][j] = a_ij."""
        k = len(self.L)
        new_mons = set()
        for j in self.mons:
            newexp = tuple(sum(Amat[i][t] * j[t] for t in range(k)) % self.L[i] for i in range(k))
            if newexp in new_mons:
                new_mons.discard(newexp)
            else:
                new_mons.add(newexp)
        return RingElement(self.L, new_mons)


def _lucas_submask(t: int, j: int) -> bool:
    """C(j, t) mod 2 == 1 iff t is a submask of j in binary (Lucas' theorem)."""
    return (j & t) == t


def local_jet(f: RingElement, comp: Component, rd: RingData) -> dict:
    """the Hasse-Taylor jet of f at comp, returned as {a: coeff} with a in rd.A and coeff in kappa_c."""
    GF = rd.GF
    E = rd.E
    alpha = comp.rep

    jet = {a: GF(0) for a in rd.A}

    for j in f.mons:
        const = GF(1)
        for i in range(rd.k):
            if i not in E:
                const = const * (alpha[i] ** j[i])

        # per even variable, the truncated local polynomial in u_i via Lucas' theorem.
        per_var = []
        for pos, i in enumerate(E):
            si = rd.s[i]
            maxt = 2 ** si
            local = {t: alpha[i] ** (j[i] - t) for t in range(maxt) if _lucas_submask(t, j[i])}
            per_var.append(local)

        # tensor the per-variable local polynomials together, then scale by const.
        for combo in iproduct(*[list(pv.items()) for pv in per_var]) if per_var else [()]:
            if per_var:
                a = tuple(t for (t, _) in combo)
                coeff = const
                for (_, c) in combo:
                    coeff = coeff * c
            else:
                a = ()
                coeff = const
            jet[a] = jet[a] + coeff

    return jet


def jet_to_str(jet: dict, rd: RingData) -> str:
    """pretty-print a jet as a sum of u_i monomials, assuming kappa_c = F2."""
    uvars = [f"u{i}" for i in rd.E]
    terms = []
    for a, c in sorted(jet.items()):
        if int(c) == 0:
            continue
        if all(t == 0 for t in a):
            terms.append("1" if int(c) == 1 else str(int(c)))
        else:
            parts = [f"{uvars[pos]}^{a[pos]}" if a[pos] > 1 else uvars[pos]
                     for pos in range(len(a)) if a[pos] != 0]
            coeff_str = "" if int(c) == 1 else f"{int(c)}*"
            terms.append(coeff_str + "*".join(parts))
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    # L=(2,2), f1=x+y, f2=1+xy, unique component c0=(1,1): expect f1,c0 = u+v, f2,c0 = u+v+uv.
    rd = ring_setup((2, 2))
    assert len(rd.components) == 1
    c0 = rd.components[0]

    f1 = RingElement.from_string((2, 2), "x + y")
    f2 = RingElement.from_string((2, 2), "1 + x*y")

    jet1 = local_jet(f1, c0, rd)
    jet2 = local_jet(f2, c0, rd)

    print("f1 =", f1, " -> jet at c0:", jet_to_str(jet1, rd), " (expect u0 + u1)")
    print("f2 =", f2, " -> jet at c0:", jet_to_str(jet2, rd), " (expect u0 + u1 + u0^1*u1^1 i.e. u+v+uv)")

    expect1 = {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 0}
    expect2 = {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 1}
    assert all(int(jet1[a]) == v for a, v in expect1.items()), jet1
    assert all(int(jet2[a]) == v for a, v in expect2.items()), jet2
    print("  pass\n")

    # cross-check: the residue (degree-0 term) matches direct evaluation.
    GF = rd.GF
    r1 = f1.evaluate(c0.rep, GF)
    r2 = f2.evaluate(c0.rep, GF)
    print("direct residues:", int(r1), int(r2), " jet degree-0 terms:", int(jet1[(0, 0)]), int(jet2[(0, 0)]))
    assert int(r1) == int(jet1[(0, 0)]) and int(r2) == int(jet2[(0, 0)])
    print("  pass")
