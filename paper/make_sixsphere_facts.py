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
from pyCICY import chair44_nested as CH
from pyCICY import sixsphere_cosmology as CO
from pyCICY import sixsphere_gauge as GA
from pyCICY import cusp_hofstadter as HO
from pyCICY import sixsphere_theta as TH
from pyCICY import sixsphere_search as SRCH

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


def _thin(n):
    """1835008 -> 1\\,835\\,008"""
    return "{:,}".format(n).replace(",", "\\,")


def chair_facts():
    """Numbers for the section on Chair44 and the nested tori (pyCICY.chair44_nested).

    Refuses to return anything if a status check fails."""
    st = CH.status()
    bad = [k for k, v in st["checks"].items() if not v]
    if bad:
        raise SystemExit("chair44_nested.status failed: %s -- nothing written" % bad)
    f = {}
    tab = st["table"]
    f["ChChecks"] = len(st["checks"])
    f["ChLabels"] = 168
    f["ChCoincDepth"] = next(r["M"] for r in tab if r["coincident"] > 0)
    f["ChCoincAddr"] = tab[3]["coincident"]
    f["ChCoincTotal"] = tab[3]["addresses"]
    order = CH.subset_automaton()[0]
    f["ChSets"] = len(order)
    f["ChNonSingleton"] = sum(1 for A in order if len(A) > 1)
    cf = st["closed_forms"]
    assert cf["proved"]
    # c(M) = a 2^M - b and E(M) = a' 2^M + b'; read off the table, then confirmed by the certificate
    a = (tab[4]["non_coincident"] - tab[3]["non_coincident"]) // 8
    b = a * 8 - tab[3]["non_coincident"]
    a2 = (tab[4]["excess"] - tab[3]["excess"]) // 8
    b2 = tab[3]["excess"] - a2 * 8
    assert all(tab[m]["non_coincident"] == a * 2 ** m - b and tab[m]["excess"] == a2 * 2 ** m + b2
               for m in range(1, len(tab)))
    f["ChCCoef"], f["ChCConst"], f["ChECoef"], f["ChEConst"] = a, b, a2, b2
    f["ChCharPoly"] = CH.transfer_charpoly()[2]
    f["ChTermsNeeded"], f["ChTermsChecked"] = cf["terms_needed"], cf["terms_checked"]
    f["ChRowAddr"] = " & ".join(str(tab[m]["addresses"]) for m in range(3, 9))
    f["ChRowC"] = " & ".join(str(tab[m]["non_coincident"]) for m in range(3, 9))
    f["ChRowE"] = " & ".join(str(tab[m]["excess"]) for m in range(3, 9))
    na = st["nested"]
    f["ChPatchCells"] = _thin(na["cells"])
    f["ChAgreeRows"] = len(na["rows"])
    assert all(r["holds"] for r in na["rows"])
    x = {r["M"]: r for r in na["rows"] if r["v"][1:] == (0, 0)}
    for M, w in ((2, "Two"), (3, "Three"), (4, "Four"), (5, "Five")):
        f["ChAgree" + w] = "%.3f" % x[M]["fraction"]
        f["ChBound" + w] = "%.3f" % float(x[M]["bound"])
    sc = st["scan"]
    th = sc["by_threshold"]
    f["ChScanCells"] = _thin(sc["cells"])
    f["ChScanHalf"], f["ChScanNinety"], f["ChScanFull"] = th[0.5]["count"], th[0.9]["count"], th[0.99]["count"]
    val = lambda d: ", ".join("%d: %d" % (k, v) for k, v in sorted(d.items()))
    f["ChScanHalfVal"] = val(th[0.5]["by_valuation"])
    f["ChScanNinetyVal"] = val(th[0.9]["by_valuation"])
    f["ChScanFullVal"] = val(th[0.99]["by_valuation"])
    prof = CH.undetermined_profile(12)
    f["ChUndetM"] = 12
    f["ChUndet"] = _thin(sum(prof.values()))
    f["ChSix"], f["ChTwentyFour"] = _thin(prof[6]), prof[24]
    tb = st["tubes"]
    assert all(r["equal"] and r["tube_formula"] and r["c_formula"] and r["excess_formula"] for r in tb)
    f["ChTubeM"], f["ChTubeOverlap"] = tb[-1]["M"], tb[-1]["overlap"]
    # --- two defects of one hierarchy: the interface (area) and the coincidence (line) ----------
    ic = st["interface"]
    assert all(r["filled"] and r["overlap"] == 0 and r["F_formula"] and r["B_formula"] and r["C_formula"] for r in ic)
    assert all(r["defect_formula"] for r in ic[:-1])
    f["ChIfaceN"] = ic[-1]["n"]
    f["ChIfaceRowChairs"] = " & ".join(_thin(r["chairs"]) for r in ic)
    f["ChIfaceRowF"] = " & ".join(_thin(r["F"]) for r in ic)
    f["ChIfaceRowB"] = " & ".join(_thin(r["B"]) for r in ic)
    f["ChIfaceRowC"] = " & ".join(_thin(r["C"]) for r in ic)
    f["ChFaceCoef"] = ic[0]["F"] // (8 - 4)                        # F = a (8^n - 4^n)
    f["ChBoundCoef"] = ic[0]["B"] // 4                              # B = b 4^n
    assert all(r["F"] == f["ChFaceCoef"] * (8 ** r["n"] - 4 ** r["n"]) and r["B"] == f["ChBoundCoef"] * 4 ** r["n"]
               for r in ic)
    # C = a 8^n + b 4^n + c, solved on n = 1, 2, 3 and then asserted on all rows
    import fractions
    A = [[fractions.Fraction(8 ** n), fractions.Fraction(4 ** n), fractions.Fraction(1)] for n in (1, 2, 3)]
    y = [fractions.Fraction(ic[n - 1]["C"]) for n in (1, 2, 3)]
    for i in range(3):                                              # Gauss-Jordan, exact
        p = next(r for r in range(i, 3) if A[r][i] != 0)
        A[i], A[p], y[i], y[p] = A[p], A[i], y[p], y[i]
        piv = A[i][i]
        A[i], y[i] = [x / piv for x in A[i]], y[i] / piv
        for r in range(3):
            if r != i:
                m = A[r][i]
                A[r], y[r] = [a - m * b for a, b in zip(A[r], A[i])], y[r] - m * y[i]
    ca, cb, cc = y
    assert all(r["C"] == ca * 8 ** r["n"] + cb * 4 ** r["n"] + cc for r in ic)
    f["ChContactVol"] = "\\frac{%d}{%d}" % (ca.numerator, ca.denominator)
    f["ChContactArea"], f["ChContactConst"] = int(-cb), int(-cc)
    f["ChDefectCoef"] = -(ic[0]["defect"] + 7) // 4                 # 8 C_n - C_{n+1} = -(a 4^n + 7)
    f["ChDefectConst"] = 7
    assert all(r["defect"] == -(f["ChDefectCoef"] * 4 ** r["n"] + f["ChDefectConst"]) for r in ic[:-1])
    de = st["exponents"]
    assert de["no_area_root"] and sorted(de["roots"]) == [0, 1, 2]
    lim = fractions.Fraction(7 * 28, 24)                            # (c/8^M) / (surface fraction) * 2^M -> 49/6
    f["ChLineToSurface"] = "\\frac{%d}{%d}" % (lim.numerator, lim.denominator)
    assert abs(float(de["rows"][-1]["ratio"]) * 2 ** de["rows"][-1]["M"] / float(lim) - 1) < 0.01
    # --- periodic approximants: tight binding and diffraction on the nested tori ----------------
    ap, sx, df = st["approximants"], st["six"], st["diffraction"]
    assert all(r["S_formula"] and r["S_within_bound"] for r in ap)
    f["ChApproxM"] = ap[-1]["M"]
    f["ChSCoef"], f["ChSConst"] = 140, 232
    f["ChIdsCoef"] = 2 * 28                                         # 2 c(M)/8^M <= 56 * 4^-M
    f["ChRankCoef"] = 12
    rk = [r for r in ap if "rank" in r]
    assert all(r["rank_formula"] and r["rank_within_bound"] for r in rk)
    f["ChRankM"] = rk[-1]["M"]
    dn = [r for r in ap if "max_dN" in r]
    assert all(r["dN_within_bound"] and r["lift_is_twisted_union"] for r in dn)
    f["ChDenseM"] = dn[-1]["M"]
    f["ChDNRow"] = " & ".join(str(r["max_dN"]) for r in dn)
    f["ChDenseSites"] = _thin(dn[-1]["sites"])
    f["ChDNFrac"] = " & ".join("%.4f" % (r["max_dN"] / r["sites"]) for r in dn)
    f["ChDNBound"] = " & ".join("%.2f" % float(r["ids_bound"]) for r in dn)
    f["ChRankRow"] = " & ".join(str(r["rank"]) for r in rk)
    f["ChSRow"] = " & ".join(_thin(r["S"]) for r in rk)
    f["ChSBoundRow"] = " & ".join(_thin(r["S_bound"]) for r in rk)
    assert all(r["pair_is_tube"] and r["distinct_on_tube"] and r["tube_formula"] for r in sx)
    f["ChSixM"] = sx[-1]["M"]
    assert all(abs(r["total"] - r["density"]) < 1e-12 for r in df) and all(r["amp_within_bound"] for r in df if "amp" in r)
    last = df[-1]
    f["ChBraggM"] = last["M"]
    f["ChBraggTotal"] = "%.5f" % last["total"]
    f["ChBraggZero"] = "%.5f" % last["grid"][0]
    for M0, w in ((2, "Two"), (3, "Three"), (4, "Four"), (5, "Five")):
        f["ChBragg" + w] = "%.4f" % last["grid"][M0]
    gaps = [last["total"] - last["grid"][m] for m in range(2, 6)]
    fall = sorted(a / b for a, b in zip(gaps, gaps[1:]))
    f["ChBraggFall"] = int(round(fall[len(fall) // 2]))             # per level; the paper says "about"
    assert f["ChBraggFall"] == 8
    am = [r["amp"] for r in df if "amp" in r]
    afall = sorted(a / b for a, b in zip(am[3:], am[4:]))
    f["ChAmpFall"] = int(round(afall[len(afall) // 2]))
    assert f["ChAmpFall"] == 8
    def _sci(x):
        m, e = ("%.1e" % x).split("e")
        return "%s\\times10^{%d}" % (m, int(e))
    f["ChAmpLast"] = _sci(am[-1])
    f["ChAmpBoundLast"] = _sci([r["amp_bound"] for r in df if "amp" in r][-1])
    fm = st["mismatch"]
    assert all(r["support_in_tubes"] and r["six_in_dihedral"] and r["six_all_values"] and r["cocycle"]
               and not r["periodic"] and r["junction_values"] == 24 and r["half_period_scale"] for r in fm)
    lastm = fm[-1]
    f["ChMisMs"] = ", ".join(str(r["M"]) for r in fm)
    f["ChMisFirst"], f["ChMisLast"] = fm[0]["M"], lastm["M"]
    f["ChMisNonId"] = "%.0f" % (100 * lastm["mismatch_cells"] / lastm["tube_cells"])
    f["ChMisHalf"] = "%.3f" % lastm["half_period_disagree"]
    f["ChMisBound"] = "%.3f" % lastm["bound"]
    f["ChMisBoundPrev"] = "%.3f" % lastm["bound_prev"]
    kv = st["kuhn"]
    f["ChKuhnPreserve"], f["ChKuhnGroup"] = kv["cube_preserving"], kv["cube_group"]
    f["ChKuhnRot"] = kv["rotations_preserving"]
    f["ChKuhnAut"], f["ChKuhnMeet"] = kv["automorphism_group"], kv["intersection"]
    oc = CH.orbifold_cover()
    f["ChCoverCusps"], f["ChCoverGenus"] = oc["cusps"], int(oc["genus"])
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


def cosmo_facts():
    """Numbers for the two further matches (pyCICY.sixsphere_cosmology), macros \\SSCo...

    Refuses to return anything if a check fails."""
    st = CO.status()
    bad = [k for k, v in st["checks"].items() if not v]
    if bad:
        raise SystemExit("sixsphere_cosmology.status failed: %s -- nothing written" % bad)
    f = {"Co" + k: v for k, v in st["facts"].items()}
    f["CoChecks"] = len(st["checks"])
    return f


def gauge_facts():
    """Numbers for the fibre structure and its holonomy (pyCICY.sixsphere_gauge),
    macros \\SSGa...  Refuses to return anything if status_gauge fails."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        good = GA.status_gauge()
    if not good:
        raise SystemExit("sixsphere_gauge.status_gauge failed -- nothing written")
    it = GA.invariant_tensors()
    hi = GA.holonomy_index()
    lc = GA.levi_lift_control()
    za = GA.zariski_lie_algebra()
    sd = GA.stabiliser_dimension()
    cd = GA.class_divisibility(3)
    cls = SRCH.classify(SRCH.funnel(3)["survivors"])
    central = sorted(GA.central_index(*c["rep"], max_elements=60000)["gcd"]
                     for c in cls)
    f = {
        "GaChecks": 10,
        "GaIndex": hi["index"],
        "GaChiOne": hi["chi(T1)"], "GaChiTwo": hi["chi(T2)"],
        "GaChiZ": hi["chi(z)"],
        "GaChiLeviOne": lc["chi"][0], "GaChiLeviTwo": lc["chi"][1],
        "GaLeviRank": lc["rank N"],
        "GaWordHOne": GA.WORDS["h(1,0,6)"], "GaWordHTwo": GA.WORDS["h(0,1,6)"],
        "GaWordHMinus": GA.WORDS["h(-1,0,6)"],
        "GaWordLenOne": len(GA.WORDS["h(1,0,6)"]),
        "GaWordLenTwo": len(GA.WORDS["h(0,1,6)"]),
        "GaWordLenMinus": len(GA.WORDS["h(-1,0,6)"]),
        "GaTensorDims": ", ".join(str(it["dims"][k])
                                  for k in ("V", "V*", "VV", "V*V*", "End")),
        "GaZariskiDim": za["dim generated"], "GaStabDim": za["dim stabiliser"],
        "GaOrbitDim": sd["orbit_dim"],
        "GaSurvivors": cd["survivors"],
        "GaPerClass": cd["count_by_d"][1],
        "GaDValues": ", ".join(str(d) for d in cd["d_values"]),
        "GaCentralIndex": ", ".join(str(g) for g in central),
    }
    if central != [12, 12, 12, 12]:
        raise SystemExit("central index of the four classes is %s, not 12 each"
                         % central)
    return f


def _signed(xs):
    return ", ".join(("$%d$" % x) for x in xs)


def hofstadter_facts():
    """Numbers for the electron on the cusp substrate (pyCICY.cusp_hofstadter),
    macros \\SSHo...  Refuses to return anything if a check fails."""
    import contextlib
    import io
    from fractions import Fraction as Fr
    with contextlib.redirect_stdout(io.StringIO()):
        good = HO.status_cusp_hofstadter()
    if not good:
        raise SystemExit("cusp_hofstadter.status failed -- nothing written")
    third = HO.gap_cherns(Fr(1, 3), nk=12, nkgap=12)
    quarter = HO.gap_cherns(Fr(1, 4), nk=12, nkgap=12)
    sixth = HO.gap_cherns(Fr(1, 6), nk=12, nkgap=12)
    pos, b = HO.finite_patch(24.0)
    k3 = [round(HO.kitaev_chern(HO.finite_hamiltonian(pos, b, Fr(1, 3)), pos,
                                sum(g["window"]) / 2, radius=12.0))
          for g in third]
    wide = [g for g in quarter if g["window"][1] - g["window"][0] > 0.3]
    H4 = HO.finite_hamiltonian(pos, b, Fr(1, 4))
    k4 = [round(HO.kitaev_chern(H4, pos, sum(g["window"]) / 2, radius=12.0))
          for g in wide]
    f3 = [g["chern_int"] for g in third]
    f4 = [g["chern_int"] for g in wide]
    if k3 != f3 or k4 != f4:
        raise SystemExit("Kitaev and FHS disagree: %s %s / %s %s" % (k3, f3, k4, f4))
    allgaps = third + quarter + sixth
    if not all(g["tknn_integer"] for g in allgaps):
        raise SystemExit("TKNN relation fails")
    per = HO.flux_periods()
    return {
        "HoChecks": 8,
        "HoGapsThird": len(third), "HoChernThird": _signed(f3),
        "HoGapsQuarter": len(quarter), "HoGapsQuarterWide": len(wide),
        "HoGapsQuarterNarrow": len(quarter) - len(wide),
        "HoChernQuarterWide": _signed(f4),
        "HoGapsSixth": len(sixth),
        "HoChernQuarter": _signed([g["chern_int"] for g in quarter]),
        "HoChernSixth": _signed([g["chern_int"] for g in sixth]),
        "HoGapsTotal": len(allgaps),
        "HoPatchSites": len(pos), "HoPatchBonds": len(b),
        "HoPeriodSub": per["substrate"]["period"],
        "HoPeriodHat": per["hat"]["period"],
    }


def _tex_sci(x):
    """A float as a LaTeX power of ten, upper-rounded: 1.5e-26 -> 10^{-25}."""
    import math
    return "10^{%d}" % math.ceil(math.log10(x)) if x > 0 else "0"


def theta_facts():
    """Numbers for the holonomy character as a theta multiplier
    (pyCICY.sixsphere_theta), macros \\SSTh...  Refuses to return anything if a
    check fails."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        good = TH.status_theta()
    if not good:
        raise SystemExit("sixsphere_theta.status_theta failed -- nothing written")
    mc = TH.multiplier_check()
    def _dev(g):
        d = TH.multiplier(g)[0] - TH.chi(g)
        return abs(d - 12 * TH.mp.nint(d / 12))
    dev = max(_dev(g) for g in TH.GENERATORS.values())
    at = TH.alpha_two_ways()
    cs = TH.candidate_search()
    pr = TH.proof_by_generators()
    cl = TH.classical_laws_check()
    zd = TH.zero_divisor()
    sh = TH.stabiliser_is_holonomy()
    if not (all(v["ok"] for v in pr.values()) and mc["words_ok"] == mc["words"]):
        raise SystemExit("theta multiplier check failed")
    return {
        "ThChecks": 11,
        "ThDigits": int(TH.mp.mp.dps),
        "ThWords": mc["words"],
        "ThAlphaAgree": at["agree"], "ThAlphaOf": at["of"],
        "ThCandidates": cs["candidates"], "ThSurvivors": len(cs["survivors"]),
        "ThClassicalErr": _tex_sci(cl["max_rel_err"]),
        "ThMaxDev": _tex_sci(float(dev)),
        "ThCosets": len(sh["cosets"]),
    }


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


def chair_only():
    """Refresh only the Chair44 macros in the existing facts file."""
    new = chair_facts()
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


def cosmo_only():
    """Refresh only the cosmology macros in the existing facts file."""
    new = cosmo_facts()
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


def gauge_only():
    """Refresh only the gauge/holonomy macros in the existing facts file."""
    new = gauge_facts()
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


def hofstadter_only():
    """Refresh only the Hofstadter macros in the existing facts file."""
    new = hofstadter_facts()
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


def surface_facts():
    """Numbers for the cusp fibre and curves on surfaces
    (pyCICY.sixsphere_surface), macros \\SSSf...  Refuses to return anything if
    a check fails."""
    from pyCICY import sixsphere_surface as SF
    r = SF.all_checks()
    m, lk, lu, ht = r["markov"], r["link"], r["lucas"], r["hat"]
    ok = (m["markov_quiver"] and m["constant_preserved"] and m["laurent_positive"]
          and m["markov_triples_ok"] and lk["line"] and lk["hexagon"] == 6
          and lu["ngon_power_sum"] and lu["blocks"] and lu["block_product"]
          and all(lu["traces"].values()) and ht["trace_is_L4n"])
    pr, lo, th = r["principal"], r["loop"], r["theta"]
    ok = (ok and all(v for k, v in pr.items() if isinstance(v, bool))
          and lo["F"] == 1 and lo["genus"] == 1 and lo["loops_graph"] == lo["loops_surface"] == 2
          and lo["control_genus"] == 0 and th["routes_agree"] and th["equals"])
    if not ok:
        raise SystemExit("sixsphere_surface check failed: %r" % (r,))
    from pyCICY import sixsphere_u as SU
    hx, sq = SU.hexagon_match(), SU.punctured_square()
    ta, tb = SU.torus_ansatz(depth=3, grid=3), SU.torus_ansatz(depth=4, grid=3)
    rep = sq["report"]
    if not (hx["match"] and rep["u_equations_exact"] and rep["unique"] and rep["C_zero_iff_compatible"]
            and sq["plain_flips_tag_and_shifts"] and sq["notched_pattern"]
            and ta["initial_unsolved"] and tb["initial_unsolved"]
            and ta["fan_only_rank"] < ta["fan_only_unknowns"]
            and tb["fan_only_rank"] < tb["fan_only_unknowns"]):
        raise SystemExit("sixsphere_u check failed")
    return {
        "SfUHexVars": hx["variables"],
        "SfUDVars": rep["variables"],
        "SfUDClusters": rep["clusters"],
        "SfUDMaxC": max(rep["C_values"]),
        "SfUDChords": sq["chords"],
        "SfUTorusDepthA": ta["depth"], "SfUTorusVarsA": ta["variables"],
        "SfUTorusUnsolvedA": ta["off_fan_unsolved"],
        "SfUTorusDepthB": tb["depth"], "SfUTorusVarsB": tb["variables"],
        "SfUTorusUnsolvedB": tb["off_fan_unsolved"],
        "SfUTorusRankB": tb["fan_only_rank"], "SfUTorusUnkB": tb["fan_only_unknowns"],
        **_surface_base(r, m, lk, lu, ht, lo, th, pr),
    }


def _surface_base(r, m, lk, lu, ht, lo, th, pr):
    return {
        "SfArrows": m["arrows"],
        "SfSeeds": m["seeds"],
        "SfVariables": m["variables"],
        "SfDepth": 4,
        "SfTriples": ", ".join("(%d,%d,%d)" % t for t in m["markov_triples"][1:5]),
        "SfLinkDepth": lk["depth"],
        "SfLinkClusters": lk["clusters"],
        "SfLinkThrough": lk["through_x0"],
        "SfHexCones": lk["hexagon"],
        "SfLucasN": lu["nmax"],
        "SfHatX": ht["x"],
        "SfLoops": lo["loops_graph"],
        "SfGenus": lo["genus"],
        "SfFaces": lo["F"],
        "SfTrees": th["trees"],
        "SfPExchange": pr["exchange_pairs"],
        "SfCrossTen": r["crossing_counts"][0],
        "SfCrossTwenty": r["crossing_counts"][1],
    }


def surface_only():
    """Refresh only the surfaceology macros in the existing facts file."""
    new = surface_facts()
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


def theta_only():
    """Refresh only the theta macros in the existing facts file."""
    new = theta_facts()
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
    if "--surface-only" in sys.argv:
        return surface_only()
    if "--theta-only" in sys.argv:
        return theta_only()
    if "--hofstadter-only" in sys.argv:
        return hofstadter_only()
    if "--gauge-only" in sys.argv:
        return gauge_only()
    if "--cosmo-only" in sys.argv:
        return cosmo_only()
    if "--links-only" in sys.argv:
        return links_only()
    if "--chair44-only" in sys.argv:
        return chair_only()
    facts, stab, _ = survey()
    facts.update(chair_facts())
    facts.update(cosmo_facts())
    facts.update(gauge_facts())
    facts.update(hofstadter_facts())
    facts.update(theta_facts())
    facts.update(surface_facts())
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
