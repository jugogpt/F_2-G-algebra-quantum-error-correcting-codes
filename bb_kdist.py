"""Exact distribution of the code dimension k over every code of a solution space that splits over component orbits."""
from __future__ import annotations
import collections
from bb_ring_elt import local_jet
from bb_local_ring import local_mul, _zero
from bb_crt import kappa_generator, _decompose


def _flat(jet, c, beta, rd):
    bits = 0; pos = 0
    for a in rd.A:
        for b in _decompose(jet[a], beta, c.Dc, rd):
            if b: bits |= 1 << pos
            pos += 1
    return bits


def _rank_bits(rows):
    basis = {}; r = 0
    for x in rows:
        while x:
            h = x.bit_length() - 1
            if h in basis: x ^= basis[h]
            else: basis[h] = x; r += 1; break
    return r


def _prepare(block_basis, orbit, rd):
    """for each basis vector, the bit-vectors of (t u^a) * f_s at each component of the orbit."""
    GF = rd.GF; comps = [rd.components[i] for i in orbit]; out = []
    for f1, f2 in block_basis:
        per_comp = []
        for c in comps:
            beta = kappa_generator(c.Dc, rd) if c.Dc > 1 else GF(1)
            j1, j2 = local_jet(f1, c, rd), local_jet(f2, c, rd)
            gens = []
            for a in rd.A:
                for t in range(c.Dc):
                    g = _zero(rd); g[a] = beta ** t; gens.append(g)
            row = [_flat(local_mul(g, j1, rd), c, beta, rd) for g in gens] + \
                  [_flat(local_mul(g, j2, rd), c, beta, rd) for g in gens]
            per_comp.append(row)
        out.append(per_comp)
    return comps, out


def block_distribution(block_basis, orbit, rd):
    """exact counter {k_O: number of codes in the block with that k_O}, over all 2^d elements."""
    d = len(block_basis)
    comps, img = _prepare(block_basis, orbit, rd)
    rows = [[0] * len(img[0][ci]) for ci in range(len(comps))] if d else [[0] * 2 for _ in comps]
    dimJ = [c.dim for c in comps]
    def kval():
        return sum(2 * (dimJ[ci] - _rank_bits(rows[ci])) for ci in range(len(comps)))
    dist = collections.Counter(); dist[kval()] += 1
    for m in range(1, 1 << d):
        i = (m & -m).bit_length() - 1
        for ci in range(len(comps)):
            r = rows[ci]; im = img[i][ci]
            for j in range(len(r)): r[j] ^= im[j]
        dist[kval()] += 1
    return dist


def block_k_of(f1, f2, orbit, rd):
    """k_O of one code (f1,f2) restricted to the orbit, by the same local rank formula."""
    comps, img = _prepare([(f1, f2)], orbit, rd)
    return sum(2 * (comps[ci].dim - _rank_bits(img[0][ci])) for ci in range(len(comps)))


def convolve(dists):
    """the generating function of the sum of independent k-distributions."""
    total = collections.Counter({0: 1})
    for D in dists:
        new = collections.Counter()
        for k1, n1 in total.items():
            for k2, n2 in D.items(): new[k1 + k2] += n1 * n2
        total = new
    return total
