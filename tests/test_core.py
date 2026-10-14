"""Fast checks (seconds). Run: pytest tests -q"""
import csv, os
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_logical import code_dimension
from bb_distance import code_distance

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def test_gross_components():
    rd = ring_setup((12, 6))
    assert sorted(c.Dc for c in rd.components) == [1, 2, 2, 2, 2]
    assert sum(c.dim for c in rd.components) == 72

def test_gross_dimension():
    rd = ring_setup((12, 6))
    f1, f2 = RE.from_string((12, 6), "x^3 + y + y^2"), RE.from_string((12, 6), "y^3 + x + x^2")
    assert code_dimension(f1, f2, rd) == 12

def test_example5_parameters():
    rd = ring_setup((3, 3))
    f1, f2 = RE.from_string((3, 3), "1 + x + y"), RE.from_string((3, 3), "1 + y + x^2*y")
    assert code_dimension(f1, f2, rd) == 4
    d = code_distance(f1, f2, rd)
    assert d["d"] == 4 and d["exact"]

def _rows(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "data", "csv", name))))

def test_tables_consistent():
    g = _rows("logical_groups_k_gt2.csv")
    full = [r for r in g if r["verdict"] == "full"]
    assert len(full) == 1 and full[0]["same_as_example5"] == "True"
    assert all(r["k"] in {"4", "6", "8"} for r in g)
    z33 = [r for r in g if r["ring"] == "Z3xZ3"]
    assert len(z33) == 7
    gross = [r for r in _rows("code_screen.csv") if r["source"] == "exact_survivors.csv"]
    assert gross and gross[0]["d"] == "12" and gross[0]["d_status"] == "exact"
