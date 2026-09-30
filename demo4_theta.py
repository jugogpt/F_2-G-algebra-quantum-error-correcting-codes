# demo 4: forward detection with a two-qubit layer Theta (Theorem 4.4, Lemma 4.11 form C).
from common import *
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import multiplier, shear
from bb_theta import sp4_f2, form_C_theta, verify_symplectic
from bb_script_a import check_general_matching
from gate_naming import name_gate

L = (3, 4, 6); rd = ring_setup(L)
f1 = RE.from_string(L, "x*y*z + x*y*z^2 + x*y*z^4 + x*y*z^5")     # the [[144,12]] code of Example 3
f2 = RE.from_string(L, "x + x*y*z + x*y*z^2")
psi1, psi2 = multiplier(L, (1, 1, 5)), shear(L, 2, 1, 1)          # z->z^-1 ; y->y z^3
t = time.time()
for nm, psi in (("psi1: z->z^-1", psi1), ("psi2: y->y z^3", psi2)):
    hits = []
    for g in sp4_f2():                                              # all 720 two-qubit Cliffords mod Pauli
        Th = form_C_theta(L, (0, 0, 0), g)                          # transversal layer (delta = 0)
        if check_general_matching(f1, f2, psi, Th, rd): hits.append(name_gate(g, 2))
    print(f"{nm}: {len(hits)} of 720 transversal layers work ->", hits)
print(f"({time.time()-t:.0f}s)")
