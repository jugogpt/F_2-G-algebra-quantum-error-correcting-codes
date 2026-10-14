# Paper map

Paper section, example or claim -> code -> data. Section numbers follow the current draft.

| Paper | Content | Code | Data / check |
|---|---|---|---|
| V-A | Ambient ring R, CRT components, bar involution | `bb_core.ring_setup`, `bb_orbits.bar_perm` | `data/csv/crt_components.csv` |
| V-A1 | Aut(G) and its action on components | `bb_automorphisms`, `bb_orbits.component_perm` | `tests/test_core.py` |
| V-B1 | Local component data (I_c, jets, n_c) | `bb_ring_elt.local_jet`, `bb_local_ring` | `verify_examples.py` |
| V-B2 | Logical operators, Algorithm 1 | `bb_logical`, `bb_crt` | `verify_examples.py` |
| VI-A, Prop. VI.2 | Local ideal filter | `bb_automorphisms.preserves_local_ideal_structure` | `verify_examples.py` |
| VI-B | Forward detection of semilinear maps | `bb_script_a`, `bb_symcheck` | `verify_examples.py` |
| VI-C | Inverse design, component-seeded construction | `bb_script_b`, `bb_theta_inverse`, `bb_gamma` | `verify_examples.py`: solution space dimension 42, k-distribution sums to 2^42 |
| VI-D, Example 4 | Fold gate on Z4 x Z4 | `bb_automorphisms.full_fold` | `verify_examples.py` |
| VI-D, Example 5 | [[18,4,4]] on Z3 x Z3, full Sp(8,2) | `bb_fullgroup`, `bb_prove_full` | `tests/test_slow.py`; row `same_as_example5 = True` in `logical_groups_k_gt2.csv` |
| IV-B, MCR codes | Validation against the earlier cyclic-submodule paper's Table 4 | `bb_sweep`, `bb_fold_s`, `bb_distance` | `data/csv/mcr_validation.csv` |
| I, Fig. 1 | Gross code [[144,12,12]]: components, d | `bb_core`, `bb_distance` | `crt_components.csv` (Z12xZ6), `code_screen.csv` (`source = exact_survivors.csv`) |
| Supplement | Sp(2k,2) generation: which codes qualify | `bb_logical_group`, `bb_prove_full`, `scripts/find_full.py` | `logical_groups_k_gt2.csv`, `docs/obstruction.pdf` |
| Supplement | Scan of code classes at gross scale | `bb_sweep` | `code_screen.csv` |
| Supplement | Wide class (psi_L != psi_R) on R_9 | `bb_wide` | `r9_wide_class.csv` |

Gross-code gate counts (51 generators: 13 class, 2 translations, 36 fold actions) and the nilradical certificate that its group is not Sp(24,2) come from `bb_fullgroup.nilradical_certificate`. They are not stored as a table.
