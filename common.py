import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/home/claude")
def name_psi(A):
    """readable name of a multiplier/fold/shear matrix."""
    return "[" + "; ".join(",".join(str(int(x)) for x in row) for row in A) + "]"

def pretty(jet, rd):
    """local element as text: u for the uniformizer, w/w2 for the F4 elements."""
    nm = {1: "", 2: "w", 3: "w2"} if int(rd.GF.order) == 4 else {}
    out = []
    for a in rd.A:
        c = int(jet[a])
        if not c: continue
        mon = "*".join(("u%d" % i) + ("" if e == 1 else "^%d" % e) for i, e in enumerate(a) if e)
        co = nm.get(c, "[%d]" % c) if c != 1 else ""
        out.append((co + ("*" if co and mon else "") + mon) or "1")
    return " + ".join(out) if out else "0"
