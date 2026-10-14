"""Slower checks (minutes). Run: pytest tests -q -m slow"""
import pytest
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE
from bb_fullgroup import narrow_group_mats
from bb_prove_full import prove_full

pytestmark = pytest.mark.slow

def test_example5_group_is_full_Sp8():
    rd = ring_setup((3, 3))
    f1, f2 = RE.from_string((3, 3), "1 + x + y"), RE.from_string((3, 3), "1 + y + x^2*y")
    mats, k, _ = narrow_group_mats(f1, f2, rd)
    assert k == 4 and prove_full(mats, k)["verdict"] == "full"

def test_other_z3z3_class_not_full():
    rd = ring_setup((3, 3))
    f1, f2 = RE.from_string((3, 3), "1 + y"), RE.from_string((3, 3), "1 + x + x^2")
    mats, k, _ = narrow_group_mats(f1, f2, rd)
    assert prove_full(mats, k)["verdict"] != "full"
