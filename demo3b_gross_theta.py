# demo 3b: gross code, which (psi, transversal Theta) symmetries exist, form C with delta=0, all 720 two-qubit layers.
from common import *
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure
from bb_theta import sp4_f2, form_C_theta
from bb_script_a import check_general_matching
from gate_naming import name_gate
L = (3, 4, 6); rd = ring_setup(L)
f1 = RE.from_string(L, "y^3 + z + z^2"); f2 = RE.from_string(L, "z^3 + x*y + x^2*y^2")
surv = [A for A in automorphism_group(rd) if preserves_local_ideal_structure(f1, f2, A, rd)]
sp4 = sp4_f2(); t = time.time(); tot = 0
for A in surv:
    hits = [name_gate(g, 2) for g in sp4 if check_general_matching(f1, f2, A, form_C_theta(L, (0, 0, 0), g), rd)]
    if hits:
        tot += len(hits); print("psi =", name_psi(A), "->", hits, flush=True)
print(f"{len(surv)} psi x 720 layers checked, {tot} symmetries found ({time.time()-t:.0f}s)")
