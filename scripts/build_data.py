"""Build the tidy tables in data/csv/ from the raw run outputs in data/raw/.
Usage: python3 scripts/build_data.py"""
import os, sys, csv, json, glob, ast
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
RAW, OUT = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "csv")

def write(name, cols, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in cols})
    print(f"{name}: {len(rows)} rows")

def rd_csv(name): return list(csv.DictReader(open(os.path.join(RAW, name))))
def rd_jsonl(name): return [json.loads(l) for l in open(os.path.join(RAW, name))]
def tf(x): return {"True": True, "False": False}.get(x, x) if isinstance(x, str) else x

# ---- 1. CRT components (computed) ----
def components():
    from bb_core import ring_setup
    from bb_orbits import bar_perm
    rows = []
    for L in [(7,), (9,), (3, 3), (4, 4), (3, 6), (6, 6), (5, 5), (9, 9), (12, 6), (12, 12)]:
        rd = ring_setup(L); bp = bar_perm(rd)
        for c in rd.components:
            rows.append(dict(ring="x".join(f"Z{l}" for l in L), L=str(L), n_qubits=2 * rd.DL(),
                             component=c.index, D_c=c.Dc, local_dim=c.dim,
                             conjugate_component=bp[c.index], self_conjugate=bp[c.index] == c.index))
    return rows

# ---- 2. screens of bicycle-code classes ----
OUTCOME = {"rejected_distance": "below_target_d", "rejected_check_weight": "over_weight_budget",
           "screen_unresolved": "screen_timeout", "evaluated": "evaluated"}
def d_status(r):
    st = OUTCOME.get(r["status"], r["status"])
    if st == "below_target_d": return "upper_bound"
    if tf(r.get("exact")) is True: return "exact"
    return "unresolved"
def screen_rows():
    rows = []; seen = set()
    files = ["fs_k2_9.csv", "gross_family_9x9.csv", "gross_family_10x6.csv", "gross_family_12x6_v2.csv",
             "gross_family_12x6.csv", "gross_family_14x6.csv", "gross_family_15x6.csv", "k6_33.csv", "k6_44.csv", "k6_36.csv", "k6_66.csv"]
    for fn in files:
        for r in rd_csv(fn):
            key = (r["L"], r["module_key"])
            if key in seen: continue
            seen.add(key)
            L = ast.literal_eval(r["L"])
            rows.append(dict(ring="x".join(f"Z{l}" for l in L), n=r["n"], k=r["k"], f1=r["f1"], f2=r["f2"],
                             check_weight=r["check_weight"], outcome=OUTCOME.get(r["status"], r["status"]),
                             d=r["d"], d_status=d_status(r), kd2_over_n=r["merit"],
                             beats_gross=r["beats_gross"], beats_two_gross=r["beats_two_gross"],
                             n_symmetry_gens=r["design_syms"], logical_group_order=r["group_order"],
                             group_status=r["group_status"], acts_as_full_Sp=r["full_clifford"],
                             class_id=r["module_key"], source=fn))
    for e in rd_csv("exact_survivors.csv"):      # exact distance of the gross code (overrides the screen row)
        for r in rows:
            if r["ring"] == "Z12xZ6" and (r["f1"], r["f2"]) == (e["f1"], e["f2"]):
                r.update(outcome="evaluated", d=e["d"], d_status="exact", kd2_over_n=e["merit"], source="exact_survivors.csv")
                break
        else:
            rows.append(dict(ring="Z12xZ6", n=e["n"], k=e["k"], f1=e["f1"], f2=e["f2"], outcome="evaluated", d=e["d"],
                             d_status="exact", kd2_over_n=e["merit"], source="exact_survivors.csv"))
    return rows

# ---- 3. logical groups for k>2 ----
def verdict(r):
    if r.get("full") is True: return "full"
    nc = r.get("nilcert", "")
    if "'proper_certificate': True" in str(nc): return "not_full:nilradical_certificate"
    if r.get("order") not in (None, "None", ""): return "not_full:exact_order"
    return "undecided"
def group_rows():
    rows = []
    for fn, cls in [("ff_33_k4.jsonl", "narrow"), ("k6_44_groups.jsonl", "wide"), ("k6_36_groups.jsonl", "wide"),
                    ("k6_66_narrow.jsonl", "narrow"), ("ff_55_k8s.jsonl", "narrow")]:
        for r in rd_jsonl(fn):
            L = ast.literal_eval(r["L"])
            if fn == "ff_55_k8s.jsonl" and r.get("order") == "None": r["order"] = ""
            rows.append(dict(ring="x".join(f"Z{l}" for l in L), n=r["n"], k=r["k"], f1=r["f1"], f2=r["f2"], d=r.get("d"),
                             symmetry_class=cls, n_generators=r.get("generators"),
                             logical_group_order=r.get("order", ""), verdict=verdict(r),
                             same_as_example5=r.get("same_as_example5", ""), seconds=r.get("seconds")))
    # commutant certificates (scripts/commutant.py), first three codes of ff_55_k8s
    cert = {0: 16, 1: 16, 2: 4}
    idx = [i for i, r in enumerate(rows) if r["ring"] == "Z5xZ5"]
    for j, i in enumerate(idx[:3]):
        rows[i]["verdict"] = "not_full:commutant_certificate"; rows[i]["commutant_dim"] = cert[j]
    return rows

def main():
    os.makedirs(OUT, exist_ok=True)
    write("crt_components.csv", ["ring", "L", "n_qubits", "component", "D_c", "local_dim", "conjugate_component", "self_conjugate"], components())
    write("code_screen.csv", ["ring", "n", "k", "f1", "f2", "check_weight", "outcome", "d", "d_status", "kd2_over_n", "beats_gross",
          "beats_two_gross", "n_symmetry_gens", "logical_group_order", "group_status", "acts_as_full_Sp", "class_id", "source"], screen_rows())
    write("logical_groups_k_gt2.csv", ["ring", "n", "k", "f1", "f2", "d", "symmetry_class", "n_generators", "logical_group_order",
          "verdict", "commutant_dim", "same_as_example5", "seconds"], group_rows())
    write("r9_wide_class.csv", ["f1", "f2", "d", "order_semilinear", "n_wide_maps", "n_nonsemilinear", "order_wide", "order_wide_plus_foldS", "full"],
          [dict(f1=r["f1"], f2=r["f2"], d=r["d"], order_semilinear=r["order_before"], n_wide_maps=r["n_wide"], n_nonsemilinear=r["n_nonsemilinear"],
                order_wide=r["order_wide"], order_wide_plus_foldS=r["order_wide_foldS"], full=r["full"]) for r in rd_csv("wide_r9_v2.csv")])
    write("mcr_validation.csv", ["l", "n", "k", "d_paper", "d_computed", "d_exact", "gates_paper", "n_semilinear_syms", "order_semilinear",
          "n_foldS", "order_with_foldS", "full_with_foldS", "seconds"],
          [dict(r, d_computed=r["d_ours"]) for r in rd_csv("table4_validation.csv")])
if __name__ == "__main__": main()
