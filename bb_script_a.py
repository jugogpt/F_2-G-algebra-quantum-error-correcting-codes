"""Script A: given a bicycle code, find every symmetry sigma = Phi_psi o Theta over scalar and depth-one Theta."""

from __future__ import annotations
from bb_core import RingData
from bb_ring_elt import RingElement, local_jet
from bb_local_ring import check_cyclic_matching, local_exists_r
from bb_automorphisms import automorphism_group, preserves_local_ideal_structure, full_fold
from bb_theta import depth1_theta_catalog, mat4_mul, zero4, identity4


def _mat4_vec_mul(Theta, gvec, L):
    """Theta . gvec, a 4x4 ring matrix times a 4-vector of RingElement."""
    out = []
    for i in range(4):
        acc = RingElement(L, [])
        for j in range(4):
            acc = acc + (Theta[i][j] * gvec[j])
        out.append(acc)
    return out


def _apply_sigma(gvec, psi_matrix, Theta, L):
    """sigma = Phi_psi o Theta: first the ring-linear part Theta, then psi entrywise."""
    Tv = _mat4_vec_mul(Theta, gvec, L)
    return [comp.apply_automorphism(psi_matrix) for comp in Tv]


def check_general_matching(f1: RingElement, f2: RingElement, psi_matrix, Theta, rd: RingData) -> bool:
    """Theorem 4.4 on M = RL.gX + RL.gZ: sigma(M)=M iff sigma(gX) and sigma(gZ) both lie in M."""
    L = rd.L
    fold = full_fold(L)
    f1bar = f1.apply_automorphism(fold)
    f2bar = f2.apply_automorphism(fold)
    zero = RingElement(L, [])

    gX = [f1, f2, zero, zero]
    gZ = [zero, zero, f2bar, f1bar]

    def in_M(v):
        for c in rd.components:
            v1c, v2c, v3c, v4c = (local_jet(v[i], c, rd) for i in range(4))
            f1c, f2c = local_jet(f1, c, rd), local_jet(f2, c, rd)
            f2bc, f1bc = local_jet(f2bar, c, rd), local_jet(f1bar, c, rd)
            if not local_exists_r(v1c, v2c, f1c, f2c, rd):
                return False
            if not local_exists_r(v3c, v4c, f2bc, f1bc, rd):
                return False
        return True

    sigma_gX = _apply_sigma(gX, psi_matrix, Theta, L)
    sigma_gZ = _apply_sigma(gZ, psi_matrix, Theta, L)
    return in_M(sigma_gX) and in_M(sigma_gZ)


def find_symmetries(f1: RingElement, f2: RingElement, rd: RingData,
                     psi_candidates=None, include_depth1: bool = True,
                     max_form_c: int = 2000, max_form_w: int = 2000,
                     verbose: bool = False):
    """full forward search, returning a list of {psi, theta_name, theta_meta, tier} dicts."""
    L = rd.L
    if psi_candidates is None:
        psi_candidates = automorphism_group(rd)

    results = []
    survivors = [psi for psi in psi_candidates if preserves_local_ideal_structure(f1, f2, psi, rd)]
    if verbose:
        print(f"Prop 4.13 cull: {len(survivors)}/{len(psi_candidates)} candidates survive")

    for psi in survivors:
        ok, _ = check_cyclic_matching(f1, f2, psi, rd)
        if ok:
            results.append({"psi": psi, "theta_name": "scalar", "theta_meta": None, "tier": 1})

        if include_depth1:
            for name, meta, Theta in depth1_theta_catalog(rd, max_form_c=max_form_c, max_form_w=max_form_w):
                if check_general_matching(f1, f2, psi, Theta, rd):
                    results.append({"psi": psi, "theta_name": name, "theta_meta": meta, "tier": 2})
    return results


if __name__ == "__main__":
    from bb_core import ring_setup
    from bb_automorphisms import identity_matrix, multiplier, full_fold as ff

    # re-validate the L=(3,3) forward example through find_symmetries itself.
    rd = ring_setup((3, 3))
    f1 = RingElement.from_string((3, 3), "x + y")
    f2 = RingElement.from_string((3, 3), "1 + x*y")

    A_id = identity_matrix(2)
    A_12 = multiplier((3, 3), (1, 2))
    A_fold = ff((3, 3))

    res = find_symmetries(f1, f2, rd, psi_candidates=[A_id, A_12, A_fold],
                           include_depth1=False, verbose=True)
    found_psis = [r["psi"] for r in res]
    print("Tier-1 symmetric psi's found:", found_psis)

    assert A_id in found_psis, "identity must always be found"
    assert A_fold in found_psis, "full fold was independently confirmed earlier"
    assert A_12 not in found_psis, "psi=(1,2) must be correctly rejected"
    print("  pass\n")

    # Tier 2 smoke test: identity psi with identity Theta must preserve M.
    rd22 = ring_setup((2, 2))
    g1 = RingElement.from_string((2, 2), "x + y")
    g2 = RingElement.from_string((2, 2), "1 + x*y")
    L = rd22.L
    id4 = identity4(L)
    ok_identity = check_general_matching(g1, g2, identity_matrix(2), id4, rd22)
    print("Tier 2 sanity: identity psi + identity Theta preserves M?", ok_identity)
    assert ok_identity is True
    print("  pass")
