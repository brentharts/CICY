#!/usr/bin/env python3
"""
Generate figures/sixsphere_facts.tex, figures/sixsphere_laves.pdf and
figures/sixsphere_funnel.pdf for paper/supplementary_material_sixsphere.tex.

Every number quoted in that paper is written here, so the prose and the
computations cannot drift apart. The script refuses to write anything if a
status check of pyCICY.sixsphere_search fails, and it records whether the
Lean kernel actually ran: the paper's machine-checked claims are typeset
from that record, so a build without a toolchain cannot claim a check that
did not happen.

Run:  python3 paper/make_sixsphere_facts.py     (from the repository root;
      about two minutes, most of it the stability boxes)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import sixsphere as S
from pyCICY import sixsphere_search as X
from pyCICY import sixsphere_links as LK

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BOX = 5
STABILITY_BOXES = (3, 4, 5, 8, 11)


def _mat(M):
    """A sympy matrix as a LaTeX pmatrix."""
    rows = [" & ".join(str(int(x)) for x in M.row(i)) for i in range(M.rows)]
    return "\\begin{pmatrix}" + " \\\\ ".join(rows) + "\\end{pmatrix}"


def _set(xs):
    return "\\{" + ",".join(str(x) for x in sorted(xs)) + "\\}"


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine"}


def survey():
    f = {}

    # --- the guards: refuse to write if anything fails ---------------------
    st = X.status_search(BOX)
    sc = X.status_connections()
    for name, res in (("status_search", st), ("status_connections", sc)):
        bad = [k for k, v in res["checks"].items() if not v]
        if bad:
            raise SystemExit("%s failed: %s -- nothing written" % (name, bad))

    # --- triangle groups ---------------------------------------------------
    sv = X.triangle_group_survey()
    rows = {r["m"]: r for r in sv["rows"]}
    f["SurveyPairs"] = len(sv["rows"])
    f["SurveyPairsWord"] = WORDS[len(sv["rows"])]
    f["NoCuspCount"] = sum(1 for r in sv["rows"]
                           if r["verdict"] == "no unipotent cusp")
    f["StarCuspCount"] = sum(1 for r in sv["rows"]
                             if r["verdict"].startswith("cusp only"))
    f["MinPThreeSix"] = rows[(3, 6)]["min_|p|"]
    f["MinPFourSix"] = rows[(4, 6)]["min_|p|"]
    f["CuspNThreeSix"] = rows[(3, 6)]["cusp_n"][0]
    f["SurveyTable"] = survey_table(sv)

    # --- funnel and stability -----------------------------------------------
    fn = X.funnel(BOX)
    c = fn["counts"]
    f["Box"] = BOX
    f["Tried"], f["Orders"], f["Cusp"] = c["tried"], c["orders"], c["cusp"]
    f["Coinv"], f["Closure"] = c["coinvariants_Z"], c["closure_ker_gamma"]
    f["Unimod"] = c["unimodular_B0"]
    f["MOneChoices"] = len(fn["M1_choices"])
    stab = {b: (c if b == BOX else X.funnel(b)["counts"])
            for b in STABILITY_BOXES}
    f["StabBoxes"] = ", ".join("\\pm%d" % b for b in STABILITY_BOXES)
    f["StabBoxMin"], f["StabBoxMax"] = STABILITY_BOXES[0], STABILITY_BOXES[-1]
    f["StabFinal"] = ", ".join(str(stab[b]["unimodular_B0"])
                               for b in STABILITY_BOXES)
    f["StabCusp"] = ", ".join(str(stab[b]["cusp"]) for b in STABILITY_BOXES)
    finals = {stab[b]["unimodular_B0"] for b in STABILITY_BOXES}
    if len(finals) != 1:
        raise SystemExit("final count not stable across boxes: %s" % stab)

    # --- classes and the conjugator -----------------------------------------
    cls = X.classify(fn["survivors"])
    f["Classes"] = len(cls)
    f["ClassesWord"] = WORDS[len(cls)]
    f["ClassSize"] = len(cls[0]["members"])
    good = [cl for cl in cls if cl["reaches_|p|=1"]]
    f["GoodInvariant"] = _set(good[0]["invariant"])
    f["OtherInvariants"] = ", ".join(
        "$" + _set(cl["invariant"]) + "$" for cl in cls
        if not cl["reaches_|p|=1"])
    f["NearMissMins"] = ", ".join(str(min(v for v in cl["invariant"] if v))
                                  for cl in cls if not cl["reaches_|p|=1"])
    pc = X.paper_conjugator(cls)
    f["PaperConj"] = _mat(pc["P"])
    f["PaperConjDet"] = int(pc["det"])
    f["RepOne"], f["RepTwo"] = _mat(X.CLASS_REP[0]), _mat(X.CLASS_REP[1])

    # --- the Laves identity -------------------------------------------------
    lv = X.laves_is_cusp_hasse()
    hp = X.hat_point_only()
    f["LavesSites"], f["LavesBonds"] = lv["sites"], lv["bonds"]
    f["LavesIncidences"] = lv["incidences"]
    f["SpectreSites"] = hp["Spectre (1 : 1)"]["sites"]
    f["SpectreVertexValence"] = hp["Spectre (1 : 1)"]["vertex_valences"][0]
    f["NearHatSites"] = hp["near Hat (17/10 : 1)"]["sites"]
    cf = S.cusp_fibre()
    f["WComponents"], f["WCurves"] = cf["components"], cf["double_curves"]
    f["WPoints"] = cf["triple_points"]
    f["EW"] = S.euler_W_stratified()

    # --- mirror and duality -------------------------------------------------
    mir = X.lattice_mirror()
    f["MirrorConj"] = _mat(mir["P"])
    f["MirrorDet"] = int(mir["det"])
    du = X.dual_lattice()
    f["DetQZero"] = int(du["det_Q0"])
    f["DualDetForm"] = str(du["det_form"]).replace("**", "^").replace("*", "")

    # --- arithmetic ----------------------------------------------------------
    fc = X.field_comparison()
    cl_ = fc["classes"]
    f["ClassesPeriods"] = _set(cl_["S6 periods Q(sqrt-3, i)"])
    f["ClassesSpectre"] = _set(cl_["Spectre inflation Q(sqrt6, sqrt10)"])
    f["ShareGeometry"] = _set(fc["periods & geometry"])
    f["ShareLatticeSpectre"] = _set(fc["lattice & Spectre"])

    # --- Lean: what actually ran --------------------------------------------
    from pyCICY import sixsphere_lean as L
    ls = L.status()
    f["LeanFacts"] = ls["n_facts"]
    f["LeanChoiceFree"] = ls["n_choice_free"]
    new = ("gcd_obstruction_3_6", "gcd_obstruction_4_6", "paper_conjugator",
           "lattice_achiral")
    f["LeanNew"] = len(new)
    if ls["toolchain"] and ls["all_machine_checked"]:
        f["LeanStatus"] = ("machine-checked by the Lean~4 kernel in this "
                           "build")
        f["LeanChecked"] = "true"
    else:
        f["LeanStatus"] = ("emitted as Lean~4 but \\emph{not} kernel-checked "
                           "in this build (no toolchain was found)")
        f["LeanChecked"] = "false"
    f.update(link_facts())
    return f, stab, cls


def link_facts():
    """Numbers for the section on connections to the rest of the package.

    Computed by pyCICY.sixsphere_links, each by two or more routes (see
    tests/test_sixsphere_links.py)."""
    f = {}
    b = LK.b3_facts()
    f["DPVolume"] = b["volume"]
    f["DPBoundary"] = b["boundary_points"]
    f["DPDual"] = b["dual_volume"]
    f["DPProduct"] = b["product"]
    f["DPBoundSym"] = b["bound_symmetric"]
    f["DPKTwo"] = b["K2_noether"]
    f["DPRays"] = b["rays"]
    k = LK.kodaira_facts(3, 4, 1)
    f["KodairaCandidates"] = len(k["all"])
    f["KodairaTotals"] = ", ".join(str(t) for t in sorted(x[3] for x in k["all"]))
    (a, bb, c, tot), = k["allowed"]
    def fmt(t):
        return "\\mathrm{%s}%s" % (t.rstrip("*"), "^*" if t.endswith("*") else "")

    f["KodairaTriple"] = "%s+%s+\\mathrm{I}_1" % (fmt(a), fmt(bb))
    r = LK.root_lattice_facts()
    f["ERootsEight"] = r["n_roots_E8"]
    f["ERootsSeven"] = r["n_roots_E7"]
    f["ERootsSix"] = r["n_roots_E6"]
    f["EDiscSeven"] = r["disc_E7"]
    f["EDiscSix"] = r["disc_E6"]
    f["EDiscSixA"] = r["disc_E6_plus_A1"]
    f["MWRank"] = r["mw_rank_IVstar_III_I1"]
    g = LK.central_charge_guard()
    f["GuardRefused"] = "refuses" if g["refused"] else "does \\emph{not} refuse"
    return f


def survey_table(sv):
    """The body of the triangle-group table, as LaTeX rows."""
    why = {"no unipotent cusp": "no unipotent cusp",
           "cusp only I_n* (sign)": "cusp only $\\mathrm{I}_2^*$"}
    lines = []
    for r in sv["rows"]:
        v = r["verdict"]
        v = why.get(v, v)
        if v.startswith("|p| >="):
            v = "$|p| \\ge %d$ (gcd)" % r["min_|p|"]
        if v == "candidate":
            v = "\\textbf{candidate}"
        cusp = ("$\\mathrm{I}_{%d}$" % r["cusp_n"][0]) if r["cusp_n"] else "---"
        lines.append("$(%d,%d,\\infty)$ & %d & %s & %d & %s \\\\"
                     % (r["m"][0], r["m"][1], r["unipotent_pairs"], cusp,
                        r["gcd"], v))
    return "\n".join(lines)


def write_facts(facts):
    lines = ["% generated by paper/make_sixsphere_facts.py -- do not edit",
             "% providecommand+renewcommand so this works whether or not",
             "% the fallback definitions in the .tex have been read first"]
    for k, v in sorted(facts.items()):
        lines.append("\\providecommand{\\SS%s}{}\\renewcommand{\\SS%s}{%s}"
                     % (k, k, v))
    path = os.path.join(OUT, "sixsphere_facts.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "sixsphere_facts.json"), "w") as fh:
        json.dump(facts, fh, indent=1, default=str)
    print("wrote", path)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _q(x):
    return float(x.p) + float(x.q) * 3 ** 0.5


def figure_laves():
    """Panel (a): the Hat-point substrate with its A2 cells and one
    fundamental domain of B0 Z^2. Panel (b): the Spectre's proportions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from pyCICY import monotile as Mo
    from pyCICY.monotile import Quad

    style = {"centre": dict(marker="o", s=34, c="#1f3b73", zorder=4),
             "edge": dict(marker="s", s=20, c="#c0392b", zorder=4),
             "vertex": dict(marker="^", s=26, c="#2e7d32", zorder=4)}
    label = {"centre": "hexagon centre = component ($dP_6$)",
             "edge": "edge midpoint = double curve",
             "vertex": "hexagon vertex = triple point"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    cases = [("(a) Hat point  $a:b=\\sqrt{3}:1$", Quad(0, 1, 3), Quad(1, 0, 3)),
             ("(b) Spectre  $a:b=1:1$", Quad(1, 0, 3), Quad(1, 0, 3))]
    for ax, (title, a, b) in zip(axes, cases):
        P = Mo.laves_patch(a=a, b=b, rings=1)
        pts = [(_q(x), _q(y)) for x, y in P["sites"]]
        for i, j, _ in P["bonds"]:
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color="0.55", lw=0.7, zorder=1)
        for kind in ("centre", "edge", "vertex"):
            xs = [p[0] for p, k in zip(pts, P["kind"]) if k == kind]
            ys = [p[1] for p, k in zip(pts, P["kind"]) if k == kind]
            ax.scatter(xs, ys, label=label[kind], **style[kind])
        if b.p == 1 and a.q == 1:
            # one fundamental domain: spanned by the images of e1, e2
            s3 = 3 ** 0.5
            e1, e2 = (s3 * s3, s3), (0.0, 2 * s3)
            poly = [(0, 0), e1, (e1[0] + e2[0], e1[1] + e2[1]), e2]
            ax.add_patch(Polygon(poly, closed=True, fc="#f5d76e", ec="#b7950b",
                                 alpha=0.35, lw=1.0, zorder=0))
            # the caption explains the shaded fundamental domain
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower center",
                   bbox_to_anchor=(1.05, -0.16), ncol=3)
    fig.tight_layout()
    path = os.path.join(OUT, "sixsphere_laves.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def figure_funnel(stab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    boxes = sorted(stab)
    stages = [("tried", "candidates tried", "o"),
              ("orders", "$T_1^3=T_2^4=I$", "s"),
              ("cusp", "unipotent cusp, $N^2=0$", "^"),
              ("coinvariants_Z", "coinvariants $\\mathbb{Z}$", "D"),
              ("unimodular_B0", "$|\\det B_0|=1$", "*")]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, lab, mk in stages:
        ax.semilogy(boxes, [stab[b][key] for b in boxes], mk + "-", ms=5,
                    lw=1.0, label=lab)
    ax.set_xlabel("box: $T_1$'s parameters in $[-B, B]^4$")
    ax.set_ylabel("survivors")
    ax.set_xticks(boxes)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    fig.tight_layout()
    path = os.path.join(OUT, "sixsphere_funnel.pdf")
    fig.savefig(path)
    print("wrote", path)


def links_only():
    """Refresh only the connection macros in the existing facts file."""
    new = link_facts()
    path = os.path.join(OUT, "sixsphere_facts.tex")
    keep = []
    if os.path.exists(path):
        for line in open(path).read().splitlines():
            if not any("\\SS%s}" % k in line for k in new):
                keep.append(line)
    for k, v in sorted(new.items()):
        keep.append("\\providecommand{\\SS%s}{}\\renewcommand{\\SS%s}{%s}"
                    % (k, k, v))
    with open(path, "w") as fh:
        fh.write("\n".join(keep) + "\n")
    print("updated", path, "with", len(new), "macros")


def main():
    if "--links-only" in sys.argv:
        return links_only()
    facts, stab, _ = survey()
    write_facts(facts)
    for fig in (figure_laves, lambda: figure_funnel(stab)):
        try:
            fig()
        except Exception as e:                                   # noqa: BLE001
            print("figure skipped:", e)
    print(json.dumps({k: v for k, v in facts.items()
                      if "\\" not in str(v)}, indent=1))


if __name__ == "__main__":
    main()
