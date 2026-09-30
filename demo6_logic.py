# demo 6: logic construction (Algorithm 1) and the induced logical action.
from common import *
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_crt import CRTBasis
from bb_logical import code_dimension, logical_basis
from bb_logical_action import induced_logical_action
from bb_theta import identity4
from gate_naming import name_gate
import numpy as np, galois
GF2 = galois.GF(2)

L = (4, 3); rd = ring_setup(L); crt = CRTBasis(rd); ix = [[3, 0], [0, 1]]
def report(tag, f1, f2):
    k = code_dimension(f1, f2, rd); LX, LZ = logical_basis(f1, f2, rd, crt)
    print(f"--- {tag}:  f1 = {f1},  f2 = {f2}\n    k = {k},  |L_X| = {len(LX)},  |L_Z| = {len(LZ)}")
    for i, (P, Q) in enumerate(LX): print(f"    L_X[{i}] = (P, Q) with P = {P}   Q = {Q}")
    A = induced_logical_action(ix, identity4(L), f1, f2, LX, LZ, rd)
    print("    action of iota_x on L_X (+) L_Z, X-block:", [[int(x) for x in row] for row in A[:k, :k]])
    print("    gate:", name_gate(A, k))
    N = GF2(A[:k, :k]) + GF2.Identity(k)
    print(f"    X-block minus identity: rank {np.linalg.matrix_rank(N)}, square is zero: {not (N @ N).any()}")
report("forward-example code", RE.from_string(L, "1 + x*y"), RE.from_string(L, "y^2 + x*y + x^2 + x^3"))
report("inverse-example code", RE.from_string(L, "1 + x^2 + x^2*y + x^2*y^2"), RE.from_string(L, "x + x^3"))
