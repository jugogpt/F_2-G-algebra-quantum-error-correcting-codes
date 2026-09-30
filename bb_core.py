"""Core algebra for k-variate bicycle (kB) codes: ring setup and CRT components."""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product as iproduct
from functools import reduce
from math import gcd
import galois


# ring setup: L = (l_1,...,l_k) -> s_i, l_i', e, S, exponent lattice A

def _val2(n: int) -> int:
    """2-adic valuation: largest s with 2^s | n."""
    s = 0
    while n % 2 == 0:
        n //= 2
        s += 1
    return s


def multiplicative_order(a: int, n: int) -> int:
    """smallest positive t with a^t == 1 (mod n), for odd n and gcd(a,n)=1."""
    if n == 1:
        return 1
    t, x = 1, a % n
    while x != 1:
        x = (x * a) % n
        t += 1
    return t


@dataclass
class RingData:
    L: tuple              # (l_1, ..., l_k)
    k: int
    s: tuple               # s_i = 2-adic valuation of l_i
    lprime: tuple           # l_i' = l_i / 2^{s_i}
    E: tuple                # indices i with s_i > 0
    e: int                  # |E|
    S: int                  # sum of s_i
    A: list                 # exponent lattice for the nilpotent part of every J_c
    N: int                  # degree of the splitting field GF(2^N)
    GF: object
    components: list

    def DL(self) -> int:
        r = 1
        for li in self.L:
            r *= li
        return r


@dataclass
class Component:
    """one CRT component J_c of R_L, i.e. one orbit of the diagonal Frobenius on root tuples."""
    index: int
    rep: tuple
    orbit: tuple
    Dc: int                 # = [kappa_c : F2]
    dim: int                # dim_F2 J_c = Dc * 2^S


def ring_setup(L: tuple) -> RingData:
    """build the static ring data for R_L."""
    k = len(L)
    s = tuple(_val2(li) for li in L)
    lprime = tuple(L[i] // (2 ** s[i]) for i in range(k))
    E = tuple(i for i in range(k) if s[i] > 0)
    e = len(E)
    S = sum(s)
    A = [tuple(a) for a in iproduct(*[range(2 ** s[i]) for i in E])] if E else [()]

    # N is the lcm of the multiplicative orders of 2 mod each l_i', so every l_i'-th root of unity lives in GF(2^N).
    orders = [multiplicative_order(2, lp) if lp > 1 else 1 for lp in lprime]
    N = reduce(lambda a, b: a * b // gcd(a, b), orders, 1)

    GF = galois.GF(2 ** N)
    components = _find_components(L, lprime, S, GF, N)

    return RingData(L=L, k=k, s=s, lprime=lprime, E=E, e=e, S=S, A=A,
                     N=N, GF=GF, components=components)


def _find_components(L, lprime, S, GF, N) -> list:
    """root-tuple orbits of the diagonal Frobenius, giving the component set of R_L."""
    order = GF.order - 1
    g = GF.primitive_element

    root_groups = []
    for lp in lprime:
        assert order % lp == 0, f"l_i'={lp} does not divide 2^N-1={order}"
        gen = g ** (order // lp)
        pts = [int(GF(1))]
        cur = GF(1)
        for _ in range(lp - 1):
            cur = cur * gen
            pts.append(int(cur))
        root_groups.append(pts)

    P = list(iproduct(*root_groups))

    def frob(tup):
        return tuple(int(GF(x) ** 2) for x in tup)

    seen = set()
    components = []
    for t in P:
        if t in seen:
            continue
        orbit = []
        cur = t
        while cur not in seen:
            seen.add(cur)
            orbit.append(cur)
            cur = frob(cur)
        Dc = len(orbit)
        rep = tuple(GF(x) for x in orbit[0])
        orbit_gf = tuple(tuple(GF(x) for x in tup) for tup in orbit)
        components.append(Component(index=len(components), rep=rep,
                                     orbit=orbit_gf, Dc=Dc, dim=Dc * (2 ** S)))
    return components


if __name__ == "__main__":
    # L=(3,4,6): expect S=3, five components of dims 8,16,16,16,16.
    rd = ring_setup((3, 4, 6))
    dims = sorted(c.dim for c in rd.components)
    print("L=(3,4,6):  s =", rd.s, " S =", rd.S, " N =", rd.N)
    print("  component dims:", dims, " sum =", sum(dims), " expect 72")
    assert rd.S == 3
    assert dims == [8, 16, 16, 16, 16]
    assert sum(dims) == 72
    print("  pass\n")

    # L=(3,3): odd x odd, expect 5 components of dims 1,2,2,2,2.
    rd2 = ring_setup((3, 3))
    dims2 = sorted(c.dim for c in rd2.components)
    print("L=(3,3):  s =", rd2.s, " S =", rd2.S, " N =", rd2.N)
    print("  component dims:", dims2, " sum =", sum(dims2), " expect 9")
    assert rd2.S == 0
    assert dims2 == [1, 2, 2, 2, 2]
    print("  pass\n")

    # L=(2,2): even x even, expect one component of dimension 4.
    rd3 = ring_setup((2, 2))
    print("L=(2,2):  s =", rd3.s, " e =", rd3.e, " S =", rd3.S)
    print("  num components:", len(rd3.components), "dim:", rd3.components[0].dim)
    assert rd3.e == 2 and rd3.S == 2 and len(rd3.components) == 1
    assert rd3.components[0].dim == 4
    print("  pass")
