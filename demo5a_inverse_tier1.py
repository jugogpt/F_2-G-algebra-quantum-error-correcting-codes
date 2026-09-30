# demo 5a: inverse construction, Tier 1, codes with psi(f) = f exactly (R_{4,3}).

from common import *
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import multiplier, shear
from bb_theta import form_C_theta
from bb_script_b import joint_target_space, construct_code, construct_code_theta
from bb_kdist import block_distribution, convolve
from bb_logical import code_dimension

L = (4, 3); rd = ring_setup(L); one = RE(L, [(0, 0)])
ix, iy = [[3, 0], [0, 1]], [[1, 0], [0, 2]]
for nm, targets in (("iota_x", [(ix, one)]), ("iota_x and iota_y", [(ix, one), (iy, one)])):
    V, mons = joint_target_space(rd, targets)
    print(f"  targets {nm:<18}: solution space dim {V.shape[0]}  -> 2^{2*V.shape[0]} pairs (f1,f2)")
res = construct_code(rd, [(ix, one)], max_pairs=40)
import collections
best = min(res, key=lambda r: (r["k"], len(r["f1"].mons) + len(r["f2"].mons)))
print("  construct_code returned", len(res), "verified codes with k>0; k-values:", dict(sorted(collections.Counter(r["k"] for r in res).items())))
print("  sparsest code with the smallest k:\n    f1 =", best["f1"], "\n    f2 =", best["f2"], "\n    k  =", best["k"])

