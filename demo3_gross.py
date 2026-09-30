# demo 3: forward detection on the gross code [[144,12,12]], via Z12 x Z6 == Z3 x Z4 x Z6.
from common import *
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure
from bb_script_a import find_symmetries
from bb_logical import code_dimension

L = (3, 4, 6); rd = ring_setup(L)              # variables x,y,z of orders 3,4,6
# gross code: A = X^3+Y+Y^2, B = Y^3+X+X^2 with X = x*y (order 12), Y = z (order 6).
f1 = RE.from_string(L, "y^3 + z + z^2")
f2 = RE.from_string(L, "z^3 + x*y + x^2*y^2")
print("n =", 2 * rd.DL(), " k =", code_dimension(f1, f2, rd), " components:", [c.dim for c in rd.components])
G = automorphism_group(rd)
surv = [A for A in G if preserves_local_ideal_structure(f1, f2, A, rd)]
print("|Aut(G)| =", len(G), "   survive Prop 4.13 ideal-pair filter:", len(surv))
t = time.time()
res = find_symmetries(f1, f2, rd, psi_candidates=surv, include_depth1=False)
print(f"scalar-twist symmetries found: {len(res)}   ({time.time()-t:.0f}s)")
for r in res[:6]:
    print("   psi =", name_psi(r["psi"]))
