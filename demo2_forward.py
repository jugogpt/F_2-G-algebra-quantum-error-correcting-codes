# demo 2: forward detection (Algorithm 3.12) on R_{4,3}.
from common import *
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure
from bb_local_ring import check_cyclic_matching
from bb_script_a import find_symmetries

L = (4, 3); rd = ring_setup(L)
names = {((1,0),(0,1)): "id", ((3,0),(0,1)): "iota_x (x->x^-1)", ((1,0),(0,2)): "iota_y (y->y^-1)", ((3,0),(0,2)): "iota (both)"}
def run(tag, f1, f2):
    print(f"--- code {tag}: f1 = {f1},  f2 = {f2}")
    for A in automorphism_group(rd):
        key = tuple(tuple(int(x) for x in r) for r in A)
        p413 = preserves_local_ideal_structure(f1, f2, A, rd)
        ok, info = check_cyclic_matching(f1, f2, A, rd)
        tail = ("accepted, w_c = " + "  ".join("c%d: %s" % (k, pretty(v, rd)) for k, v in sorted(info.items()))) if ok else ("REJECTED: " + str(info))
        print(f"  {names[key]:<18} Prop4.13={'pass' if p413 else 'fail'}  {tail}")
run("A", RE.from_string(L, "1 + x*y"), RE.from_string(L, "y^2 + x*y + x^2 + x^3"))
run("A'", RE.from_string(L, "1 + x*y"), RE.from_string(L, "y^2 + x + x*y^2 + x^2 + x^3*y + x^3*y^2"))
res = find_symmetries(RE.from_string(L, "1 + x*y"), RE.from_string(L, "y^2 + x*y + x^2 + x^3"), rd, include_depth1=False)
print("find_symmetries (scalar-twist tier) returns", len(res), "symmetries:", [names[tuple(tuple(int(x) for x in r) for r in d['psi'])] for d in res])
