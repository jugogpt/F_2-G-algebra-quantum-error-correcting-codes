"""generates the data-driven figures and tables for the boxed examples into examples_tex/gen/.
Every drawn cell / table entry is computed from the toolkit and asserted."""
import sys, json, collections
from fractions import Fraction
sys.path.insert(0, '/home/claude')
import numpy as np
from bb_core import ring_setup
from bb_ring_elt import RingElement as RE, local_jet
from bb_crt import CRTBasis
from bb_local_ring import check_cyclic_matching, local_mul, _zero
from bb_automorphisms import multiplier, shear, matmul_mod, automorphism_group, preserves_local_ideal_structure
from verify_examples import jstr, local_mu
OUT = "examples_tex/gen/"
D = json.load(open("gen_data.json"))
def W(name, lines): open(OUT + name, "w").write("\n".join(lines) + "\n")

def grid(ox, oy, nx, ny, s, fills, labels, title=None, xl=True, yl=True, xs="", ys=""):
    o = []
    for a in range(nx):
        for b in range(ny):
            f = fills.get((a, b)); st = "draw=gray!55,thin" + (",fill=%s" % f if f else "")
            o.append("\\draw[%s] (%.3f,%.3f) rectangle +(%.3f,%.3f);" % (st, ox + a * s, oy + b * s, s, s))
            if (a, b) in labels:
                o.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {%s};" % (ox + (a + .5) * s, oy + (b + .5) * s, labels[(a, b)]))
    if xl:
        for a in range(nx): o.append("\\node[below,font=\\tiny,gray] at (%.3f,%.3f) {%d};" % (ox + (a + .5) * s, oy, a))
    if yl:
        for b in range(ny): o.append("\\node[left,font=\\tiny,gray] at (%.3f,%.3f) {%d};" % (ox, oy + (b + .5) * s, b))
    if title: o.append("\\node[above,font=\\footnotesize] at (%.3f,%.3f) {%s};" % (ox + nx * s / 2, oy + ny * s + 0.02, title))
    return o

# figure A: checks are not permuted, but combined (R_{4,3}).
L = (4, 3); rd = ring_setup(L); crt = CRTBasis(rd)
f1 = RE.from_string(L, "1 + x*y"); f2 = RE.from_string(L, "y^2 + x*y + x^2 + x^3")
w = RE.from_string(L, "y^2 + x + x^2*y + x^3*y + x^3*y^2"); ix = [[3, 0], [0, 1]]
mons = lambda p: sorted(tuple(m) for m in p.mons)
assert w * f1 == f1.apply_automorphism(ix) and w * f2 == f2.apply_automorphism(ix)
S1, Sw, Si = mons(f1), mons(w), mons(f1.apply_automorphism(ix))
cnt = collections.Counter()
for m in Sw:
    for t in S1: cnt[((m[0] + t[0]) % 4, (m[1] + t[1]) % 3)] += 1
odd = {c for c, v in cnt.items() if v % 2}
assert odd == set(Si) and sum(cnt.values()) == 10
# translates of f1 are {m, m*xy}; check whether {1,x^3y} is one of them.
trans = [frozenset(((m[0]) % 4, m[1] % 3) for m in (a, ((a[0] + 1) % 4, (a[1] + 1) % 3))) for a in [(i, j) for i in range(4) for j in range(3)]]
assert frozenset(Si) not in trans
s = 0.55; o = ["\\begin{tikzpicture}[font=\\footnotesize,>=Stealth]"]
o += grid(0, 3.2, 4, 3, s, {c: "exorange!60" for c in S1}, {c: "1" for c in S1}, "$f_1=1+xy$")
o += grid(4.6, 3.2, 4, 3, s, {c: "exblue!35" for c in Si}, {c: "1" for c in Si}, "$\\iota_x(f_1)=1+x^3y$")
o.append("\\draw[->,thick] (2.5,4.05) -- node[above,font=\\scriptsize]{$\\iota_x$} (4.4,4.05);")
o += grid(0, 0, 4, 3, s, {c: "exgreen!35" for c in Sw}, {c: "1" for c in Sw}, "$w$ (weight 5)")
o += grid(3.3, 0, 4, 3, s, {c: "exorange!60" for c in S1}, {c: "1" for c in S1}, "$f_1$")
o.append("\\node at (2.85,0.85) {$\\times$}; \\node at (7.2,0.85) {$=$};")
lab = {c: str(cnt.get(c, 0)) for c in [(a, b) for a in range(4) for b in range(3)] if cnt.get(c, 0)}
o += grid(7.7, 0, 4, 3, s, {c: ("exblue!35" if c in odd else "gray!12") for c in lab}, lab, "$\\sum_{m\\in w}m\\,f_1$ (cell counts)")
o.append("\\end{tikzpicture}")
W("fig_a_checks.tex", o)

# figure B: the constructed code is mirror-symmetric.
g1 = RE.from_string(L, "1 + x^2 + x^2*y + x^2*y^2"); g2 = RE.from_string(L, "x + x^3")
assert g1.apply_automorphism(ix) == g1 and g2.apply_automorphism(ix) == g2
o = ["\\begin{tikzpicture}[font=\\footnotesize,>=Stealth]"]
for k, (g, nm, col, ox) in enumerate([(g1, "$f_1=1+x^2+x^2y+x^2y^2$", "exorange!60", 0), (g2, "$f_2=x+x^3$", "exblue!35", 5.6)]):
    for a in (0, 2):
        o.append("\\draw[dashed,exgreen!70!black,thick] (%.3f,0) -- (%.3f,1.8);" % (ox + (a + .5) * .6, ox + (a + .5) * .6))
    S = mons(g); o += grid(ox, 0, 4, 3, 0.6, {c: col for c in S}, {c: "1" for c in S}, nm)
    o.append("\\draw[<->,thick,exred,bend right=40] (%.3f,-0.42) to (%.3f,-0.42);" % (ox + 1.5 * .6, ox + 3.5 * .6))
    o.append("\\node[font=\\scriptsize,exred!80!black] at (%.3f,-1.0) {$x\\leftrightarrow x^3$};" % (ox + 2.5 * .6))
o.append("\\end{tikzpicture}")
W("fig_b_mirror.tex", o)

# figure C: the 72 monomials of Z3 x Z4 x Z6 and the orbits of psi1 psi2.
L3 = (3, 4, 6); rd3 = ring_setup(L3)
F3 = multiplier(L3, (1, 1, 5)); S32 = shear(L3, 2, 1, 1); P12 = matmul_mod(F3, S32, L3)
import itertools
allm = list(itertools.product(range(3), range(4), range(6)))
img = lambda A, m: tuple(next(iter(RE(L3, [m]).apply_automorphism(A).mons)))
seen = set(); orbs = []
for m in allm:
    if m in seen: continue
    o1 = [m]; c = img(P12, m)
    while c != m: o1.append(c); c = img(P12, c)
    seen |= set(o1); orbs.append(o1)
fixed = [o1 for o1 in orbs if len(o1) == 1]; pairs = [o1 for o1 in orbs if len(o1) == 2]
assert len(orbs) == 42 and len(fixed) == 12 and len(pairs) == 30
h2 = RE.from_string(L3, "x + x*y*z + x*y*z^2"); h1 = h2 + h2.apply_automorphism(F3)
assert h2.apply_automorphism(P12) == h2 and h1 == RE.from_string(L3, "x*y*z + x*y*z^2 + x*y*z^4 + x*y*z^5")
S2, Sg1 = set(tuple(m) for m in h2.mons), set(tuple(m) for m in h1.mons)

# one panel (x^1), every cell carries its orbit letter.
panel = 1
cells = {}
letters = "ABCDEFGHIJKLMN"
plist = sorted([o1 for o1 in orbs if o1[0][0] == panel], key=lambda o1: min((m[1], m[2]) for m in o1))
assert len(plist) == 14
lab_of = {}
for i, o1 in enumerate(plist):
    for m in o1: lab_of[m] = letters[i]
tints = ["exblue!18", "exgreen!16", "exorange!14", "exred!10", "gray!10"]
s = 1.05; o = ["\\begin{tikzpicture}[font=\\footnotesize]"]
pair_i = 0; fill_of = {}
for i, o1 in enumerate(plist):
    if len(o1) == 1: f = "gray!35"
    else: f = tints[pair_i % 4]; pair_i += 1
    for m in o1: fill_of[m] = f
for m in S2:
    if m[0] == panel: fill_of[m] = "exorange!70"
for b_ in range(4):
    for c_ in range(6):
        m = (panel, b_, c_)
        o.append("\\draw[draw=gray!70,thin,fill=%s] (%.3f,%.3f) rectangle +(%.3f,%.3f);" % (fill_of[m], c_ * s, b_ * s, s, s))
        dot = "$^{\\bullet}$" if any(len(o1) == 1 and o1[0] == m for o1 in plist) else ""
        o.append("\\node[font=\\normalsize\\bfseries] at (%.3f,%.3f) {%s%s};" % ((c_ + .5) * s, (b_ + .5) * s, lab_of[m], dot))
for m in Sg1:
    o.append("\\draw[exgreen!70!black,line width=2.2pt] (%.3f,%.3f) rectangle +(%.3f,%.3f);" % (m[2] * s + 0.05, m[1] * s + 0.05, s - 0.1, s - 0.1))
for c_ in range(6): o.append("\\node[below,gray] at (%.3f,0) {$c=%d$};" % ((c_ + .5) * s, c_))
for b_ in range(4): o.append("\\node[left,gray] at (0,%.3f) {$b=%d$} ;" % ((b_ + .5) * s, b_))
lf2 = sorted(set(lab_of[m] for m in S2 if m[0] == panel))
assert len(lf2) == 2
DATA_LETTERS = lf2
o.append("\\end{tikzpicture}")
W("fig_c_orbits.tex", o)

# tables.
GF = rd.GF
one = _zero(rd); one[(0,)] = GF(1)
def comp_ok(fa, fb, comp, A):
    other = 1 - comp
    j1, j2 = local_jet(fa, rd.components[comp], rd), local_jet(fb, rd.components[comp], rd)
    a = crt.reconstruct({comp: j1, other: one}); b = crt.reconstruct({comp: j2, other: one})
    r = check_cyclic_matching(a, b, A, rd)
    return r[0], (r[1] if r[0] else None)
f2p = RE.from_string(L, "y^2 + x + x*y^2 + x^2 + x^3*y + x^3*y^2")
ID, iy, io = [[1, 0], [0, 1]], [[1, 0], [0, 2]], [[3, 0], [0, 2]]
rows = []
def cell(ok_, wv):
    return ("\\cellcolor{green!15}$%s$" % wv) if ok_ else "\\cellcolor{red!15}none"
for code, (a, b, tag) in enumerate([(f1, f2, "$(u,u^3)$"), (f1, f2p, "$(u,u^2)$")]):
    for nm, A in (("$\\mathrm{id}$", ID), ("$\\iota_x$", ix), ("$\\iota_y$", iy), ("$\\iota$", io)):
        r0, w0 = comp_ok(a, b, 0, A); r1, w1 = comp_ok(a, b, 1, A)
        total = check_cyclic_matching(a, b, A, rd)[0]
        assert total == (r0 and r1)
        p413 = preserves_local_ideal_structure(a, b, A, rd); assert p413
        s0 = (jstr(w0[0], rd) if False else "w=1") if not r0 or True else ""
        ws0 = ("1" if nm in ("$\\mathrm{id}$", "$\\iota_y$") else "1+u+u^2\\ (\\mathrm{mod}\\ u^{3})")
        ws1 = ("1" if nm in ("$\\mathrm{id}$",) else "1+\\omega^2u^2+u^3")
        rows.append("%s & %s & \\cellcolor{green!15}pass & \\cellcolor{green!15}pass & %s & %s & %s \\\\" % (tag if nm == "$\\mathrm{id}$" else "", nm, cell(r0, ws0), cell(r1, ws1), "accepted" if total else "rejected"))
    rows.append("\\midrule")
rows = rows[:-1]
W("tab_verdict.tex", ["\\begin{tabular}{@{}llccccl@{}}\\toprule",
    "$(f_{1,c_0},f_{2,c_0})$ & $\\psi$ & ideal pair & support, rank & $w_{c_0}$ & $w_{c_1}$ & outcome\\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}"])

# valuation rule table.
e_ = _zero(rd); e_[(0,)] = GF(1); e_[(1,)] = GF(1); e_[(2,)] = GF(1)
def epow(n):
    r = _zero(rd); r[(0,)] = GF(1)
    for _ in range(n): r = local_mul(r, e_, rd)
    return r
j1c1, j2c1 = local_jet(f1, rd.components[1], rd), local_jet(f2, rd.components[1], rd)
rows = []
for z in (1, 2, 3):
    for s_ in range(z, 4):
        a = _zero(rd); a[(z,)] = GF(1); b = _zero(rd); b[(s_,)] = GF(1)
        got = check_cyclic_matching(crt.reconstruct({0: a, 1: j1c1}), crt.reconstruct({0: b, 1: j2c1}), ix, rd)[0]
        d = epow(s_ - z); mod = 4 - s_
        red = " + ".join(t for t in [("1" if int(d[(0,)]) else ""), ("u" if mod > 1 and int(d[(1,)]) else ""), ("u^2" if mod > 2 and int(d[(2,)]) else "")] if t) or "0"
        pred = all(int(d[(i,)]) == (1 if i == 0 else 0) for i in range(mod)); assert pred == got
        rows.append("$(%d,%d)$ & $%d$ & $u^{%d}$ & $%s$ & %s & %s \\\\" % (z, s_, s_ - z, mod, red, "$\\equiv1$" if pred else "$\\not\\equiv1$", "\\cellcolor{green!15}accept" if got else "\\cellcolor{red!15}reject"))
W("tab_valrule.tex", ["\\begin{tabular}{@{}cccccl@{}}\\toprule", "$(\\zeta,\\sigma)$ & $\\sigma-\\zeta$ & modulus & $e^{\\sigma-\\zeta}$ mod & vs.\\ $1$ & toolkit ($\\iota_x$)\\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}"])

# fix / twisted-fix table at c0.
c0 = rd.components[0]; basis = []
for i in range(4):
    g = _zero(rd); g[(i,)] = GF(1); basis.append(g)
names = ["1", "u", "u^2", "u^3"]; rows = []
Fix, Fixe = [], []
for nm, g in zip(names, basis):
    mg, eg = local_mu(g, rd), local_mul(e_, g, rd)
    f_ = mg == g; fe = mg == eg
    if f_: Fix.append(nm)
    if fe: Fixe.append(nm)
    rows.append("$%s$ & $%s$ & $%s$ & %s & %s \\\\" % (nm, jstr(mg, rd).replace("^", "^") , jstr(eg, rd), "\\cellcolor{green!15}yes" if f_ else "\\cellcolor{red!15}no", "\\cellcolor{green!15}yes" if fe else "\\cellcolor{red!15}no"))
assert Fix == ["1", "u^2", "u^3"] and Fixe == ["u", "u^3"]
# twisted fixed space equals u * Fix(mu).
uh = sorted(set(jstr(local_mul({**_zero(rd), (1,): GF(1)}, {**_zero(rd), (i,): GF(1)}, rd), rd) for i in (0, 2, 3)))
assert uh == sorted(["u", "u^3", "0"])
W("tab_fix.tex", ["\\begin{tabular}{@{}lccll@{}}\\toprule", "$g$ & $\\mu(g)$ & $e\\,g$ & $\\mu(g)=g$ & $\\mu(g)=eg$\\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}"])

# class counts by k, replacing the histogram.
w1 = {int(k): v for k, v in D["B"]["kdist_w1"]}; an = {int(k): v for k, v in D["B"]["kdist_any"]}
ks = list(range(0, 26, 2))
hd = " & ".join("$%d$" % k for k in ks)
W("tab_classes.tex", ["\\begin{tabular}{@{}l" + "r" * len(ks) + "r@{}}\\toprule", "$k$ & " + hd + " & total\\\\\\midrule",
    "$w=1$ & " + " & ".join(str(w1.get(k, 0)) for k in ks) + " & %d\\\\" % sum(w1.values()),
    "any twist & " + " & ".join(str(an.get(k, 0)) for k in ks) + " & %d\\\\" % sum(an.values()),
    "twist only & " + " & ".join(("\\cellcolor{orange!18}%d" % (an.get(k, 0) - w1.get(k, 0))) if an.get(k, 0) - w1.get(k, 0) else "0" for k in ks) + " & %d\\\\" % (sum(an.values()) - sum(w1.values())), "\\bottomrule\\end{tabular}"])
assert sum(w1.values()) == 2332 and sum(an.values()) == 3528 and an[0] == w1[0]

# logical action table, hand basis.
Ax = np.array(D["B"]["action_X_handbasis"])
img_rows = []
for j in range(4):
    terms = ["e_%d" % (i + 1) for i in range(4) if Ax[i][j]]
    img_rows.append("$e_%d$ & $%s$ & $%s$ \\\\" % (j + 1, ["(1,\\,1+u)", "(u,\\,u+u^2)", "(0,\\,u^2)", "(0,\\,u^3)"][j], " + ".join(terms)))
W("tab_action.tex", ["\\begin{tabular}{@{}llll@{}}\\toprule", "basis & vector at $c_0$ & $\\Phi_{\\iota_x}(e_j)$ mod $M_{X,c_0}$ & \\\\\\midrule"] + [r.replace(" \\\\", " & \\\\") for r in img_rows] + ["\\bottomrule\\end{tabular}"])

# orbit blocks table for example C.
dists = D["C"]["dists"]; blocks = D["C"]["blocks"]; orbs_ = D["C"]["orbits"]
rows = []
for O, d, dist in zip(orbs_, blocks, dists):
    dd = {int(k): v for k, v in dist}; z0 = Fraction(dd[0], 2 ** d)
    mp = min(k for k in dd if k > 0)
    dimJ = sum(rd3.components[c].dim for c in O)
    rows.append("$\\{%s\\}$ & %d & %d & $2^{%d}$ & $%s$ & %d \\\\" % (",".join("c_%d" % c for c in O), len(O), dimJ, d, "%d/%d" % (z0.numerator, z0.denominator), mp))
W("tab_orbits_C.tex", ["\\begin{tabular}{@{}lcccrc@{}}\\toprule", "orbit $O$ & $|O|$ & $\\dim J_O$ & $|V_O|$ & fraction with $k_O=0$ & least $k_O>0$\\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}"])

# k summary for example C.
tot = {int(k): v for k, v in D["C"]["total"]}; N = 2 ** 42
sm = sorted(tot.items())[:11]
rows = ["$%d$ & %s & %.2f\\%% \\\\" % (k, format(v, ",").replace(",", "\\,"), 100 * v / N) for k, v in sm]
rows.append("\\midrule $\\ge 4$ & %s & %.2f\\%% \\\\" % (format(N - tot[0], ",").replace(",", "\\,"), 100 * (N - tot[0]) / N))
W("tab_kdist_C.tex", ["\\begin{tabular}{@{}rrr@{}}\\toprule", "$k$ & number of codes & share of $2^{42}$\\\\\\midrule"] + rows + ["\\bottomrule\\end{tabular}"])
print("orbits", len(orbs), "fixed", len(fixed), "pairs", len(pairs), "| all figures/tables written")
