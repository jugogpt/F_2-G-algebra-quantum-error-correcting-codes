# demo 5b: inverse construction, Tier 2, all codes for a prescribed (psi, Theta) set (R_{3,4,6}).

from common import *
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import multiplier, shear
from bb_theta import form_C_theta
from bb_script_b import joint_target_space, construct_code, construct_code_theta
from bb_kdist import block_distribution, convolve
from bb_logical import code_dimension

L3 = (3, 4, 6); rd3 = ring_setup(L3)
CN = np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]]); Th = form_C_theta(L3, (0,0,0), CN)
one3 = RE(L3, [(0,0,0)]); zero3 = RE(L3, []); W = [[one3, zero3], [zero3, one3]]
gens = [(multiplier(L3, (1,1,5)), Th, W), (shear(L3, 2, 1, 1), Th, W)]
t = time.time(); R = construct_code_theta(rd3, gens)
print(f"  solution space dim {R['dim']}  (2^{R['dim']} of 2^144 pairs);  orbit blocks {R['orbits']} of dims {[len(b) for b in R['blocks']]}")
dists = [block_distribution(b, orb, rd3) for b, orb in zip(R["blocks"], R["orbits"])]
tot = convolve(dists)
print(f"  exact k-distribution of all codes: k=0: {tot[0]:,}   smallest k>0: {min(k for k in tot if k>0)} ({tot[min(k for k in tot if k>0)]:,} codes)   ({time.time()-t:.0f}s)")
