# demo 1: general setup, ring, components, local jets, CRT.
from common import *
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE, local_jet
from bb_crt import CRTBasis
from bb_logical import code_dimension

L = (4, 3)                                   # R_L = F2[x,y]/(x^4-1, y^3-1)
rd = ring_setup(L)
print("L =", L, " D_L =", rd.DL(), " #components =", len(rd.components))
for c in rd.components:
    print(f"  c{c.index}: root pair {tuple(int(t) for t in c.rep)}  D_c={c.Dc}  dim J_c={c.dim}")

f1 = RE.from_string(L, "1 + x*y")
f2 = RE.from_string(L, "y^2 + x*y + x^2 + x^3")
print("f1 =", f1, "\nf2 =", f2)
for c in rd.components:
    print(f"  local jets at c{c.index}:  f1 -> {pretty(local_jet(f1, c, rd), rd)}   f2 -> {pretty(local_jet(f2, c, rd), rd)}")

crt = CRTBasis(rd)
print("idempotents: 1_c0 =", crt.idempotent(0), "   1_c1 =", crt.idempotent(1))
back = crt.reconstruct({c.index: local_jet(f1, c, rd) for c in rd.components})
print("CRT round trip  f1 == reconstruct(jets):", back == f1)
print("code dimension k = 2*sum_c dim(J_c/I_c) =", code_dimension(f1, f2, rd))
