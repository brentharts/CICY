#!/usr/bin/env python3
"""
Emit every number quoted in the strings and twistor papers.

    python3 paper/make_strings_facts.py --outdir paper/figures
    python3 paper/make_strings_facts.py --sections twistor

The strings paper contains no figures, only numbers, and the point of this
script is the same as the point of make_figures.py: the prose and the code
must not be able to drift apart. Every quantity the paper states is recomputed
here from pyCICY and written to figures/strings_facts.tex as a LaTeX macro,
which the document \\input{}s. If a computation changes, the paper changes with
it or the build fails.

Nothing here is a table lookup. The matter-free algebras are derived from the
anomaly conditions, the CICY fibration count is a scan of the published list,
and the two routes to each Euler characteristic are computed separately and
compared.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import CICY
from pyCICY import twistor as TW
from pyCICY.theories import ftheory as FT
from pyCICY.theories import orientifold as OR


def collect(cicy_list=None):
    f = {}

    # --- the derived matter-free algebras -----------------------------------
    derived = FT.matter_free_algebras()
    f["MatterFreeCount"] = len(derived)
    order = sorted(derived.items(), key=lambda kv: -kv[1])
    f["MatterFreeList"] = ", ".join(
        "$%s$ at $%d$" % (_tex_alg(a), n) for a, n in order)
    for a, n in derived.items():
        f["MF" + _key(a)] = -n

    # --- matter from the anomaly conditions ---------------------------------
    m = FT.matter_content("su(5)", -1)["matter"]
    f["SUFiveMinusOneFund"] = int(m["fund"])
    f["SUFiveMinusOneAnti"] = int(m["antisym"])
    f["SUFiveMinusTwoFund"] = int(FT.matter_content("su(5)", -2)["matter"]["fund"])
    f["SUFiveZeroFund"] = int(FT.matter_content("su(5)", 0)["matter"]["fund"])
    f["SUFiveZeroAnti"] = int(FT.matter_content("su(5)", 0)["matter"]["antisym"])
    f["ESevenHalf"] = str(FT.matter_content("e7", -7)["matter"]["56"])
    f["ESevenCharged"] = int(FT.matter_content("e7", -7)["charged_dim"])

    # --- so(N) spinors, derived from the weights ---------------------------
    d, A, B, C = FT.algebra_data("so(10)")["reps"]["spinor"]
    f["SOTenSpinorDim"] = d
    f["SOTenSpinorA"] = str(A)
    sp = FT.matter_content("so(10)", -3)["matter"]
    f["SOTenMinusThreeVec"] = int(sp["vector"])
    f["SOTenMinusThreeSpin"] = int(sp["spinor"])
    f["SOTwelveHalfSpinor"] = str(
        FT.matter_content("so(12)", -3)["matter"]["spinor"])
    f["SOSevenIndexOne"] = ", ".join(FT.index_one_reps("so(7)"))

    # --- the multi-curve clusters ------------------------------------------
    f["ClusterCount"] = len(FT.NON_HIGGSABLE_CLUSTERS)
    c = FT.cluster([(-3, "g2"), (-2, "su(2)")], {(0, 1): 1})
    f["GTwoSevens"] = str(c["matter"][0]["7"])
    f["GTwoSUTwoDoublets"] = str(c["matter"][1]["fund"])
    f["GTwoSharedMult"] = str(c["shared"][0]["multiplicity"])
    c3 = FT.cluster([(-2, "su(2)"), (-3, "so(7)"), (-2, "su(2)")],
                    {(0, 1): 1, (1, 2): 1})
    f["SOSevenSpinors"] = str(c3["matter"][1]["spinor"])
    f["ClusterThreeCharged"] = str(c3["charged"])

    # --- bases: two routes to the moduli count ------------------------------
    P2 = FT.Base.P2()
    f["PTwoHfour"] = P2.h0_anticanonical(4)
    f["PTwoHsix"] = P2.h0_anticanonical(6)
    f["PTwoChiT"] = P2.chi_tangent()
    f["PTwoModuli"] = FT.weierstrass_moduli(P2)
    f["FTwoChiT"] = FT.Base.hirzebruch(2).chi_tangent()
    f["FTwoModuli"] = FT.weierstrass_moduli(FT.Base.hirzebruch(2))
    agree = all(FT.weierstrass_moduli(b) == 272 - 29 * b.T
                for b in [FT.Base.P2()]
                + [FT.Base.hirzebruch(n) for n in range(3)]
                + [FT.Base.del_pezzo(k) for k in range(9)])
    f["ModuliAgree"] = "yes" if agree else "NO"

    # --- whole models -------------------------------------------------------
    for spec, key in (("P2", "PTwo"), ("F0", "FZero"), ("F12", "FTwelve"),
                      ("F7", "FSeven")):
        mm = FT.FTheory6D(spec)
        h11, h21 = mm.hodge_numbers()
        f[key + "Hone"] = h11
        f[key + "Htwo"] = h21
        f[key + "Chi"] = mm.euler_characteristic()
        f[key + "Alg"] = mm.gauge_group()
    f["FSevenCharged"] = FT.FTheory6D("F7").spectrum()["H_charged"]

    # --- three routes to chi ------------------------------------------------
    f["PTwoChern"] = FT.weierstrass_euler(FT.Base.P2())
    f["FTwelveChern"] = FT.weierstrass_euler(FT.Base.hirzebruch(12))
    f["FTwelveGap"] = (FT.FTheory6D("F12").euler_characteristic()
                       - FT.weierstrass_euler(FT.Base.hirzebruch(12)))

    # --- Mordell-Weil -------------------------------------------------------
    mw = FT.FTheory6D("P2", abelian=[{"height": [6]}])
    f["MWCharge"] = int(mw.abelian[0]["matter"][1])
    f["MWHone"], f["MWHtwo"] = mw.hodge_numbers()
    f["MWGood"] = ", ".join(
        str(h) for h in range(1, 13)
        if _u1_ok(P2, h))

    # --- fourfolds ----------------------------------------------------------
    for dims, key in (([3], "PThree"), ([1, 2], "POneTwo")):
        B = FT.ProductBase(dims)
        h = FT.fourfold_hodge(B)
        f[key + "CoCt"] = B.chern_number([1, 2])
        f[key + "CoCube"] = B.chern_number([1, 1, 1])
        f[key + "Hone"] = h["h11"]
        f[key + "Hthree"] = h["h31"]
        f[key + "Chi"] = h["euler"]
        f[key + "ChiChern"] = h["euler_chern"]
        f[key + "Tadpole"] = str(h["d3_tadpole"])
        f[key + "Agree"] = "yes" if h["agree"] else "NO"
    f["PThreeFluxBound"] = str(FT.FTheory4D.over([3]).flux()["max_GG"])

    # --- orientifolds -------------------------------------------------------
    inv = OR.SignInvolution([[4, 5]], [[4]])
    f["QuinticFixedChi"] = inv.fixed_euler()
    hs = inv.hodge_split()
    f["QuinticHtwoPlus"], f["QuinticHtwoMinus"] = hs["h21"]
    mono = OR.hypersurface_moduli_split([4], [5], [[4]])
    f["QuinticDefPlus"], f["QuinticDefMinus"] = mono["deformations"]
    f["QuinticRoutesAgree"] = "yes" if mono["h21"] == hs["h21"] else "NO"
    o = OR.Orientifold(inv)
    f["QuinticChiral"] = o.spectrum()["chiral"]
    f["QuinticVectors"] = o.spectrum()["vectors"]
    inv5 = OR.SignInvolution([[4, 5]], [[3, 4]])
    s5 = OR.Orientifold(inv5).spectrum()
    f["QuinticFiveChiral"] = s5["chiral"]
    f["QuinticFiveVectors"] = s5["vectors"]
    f["QuinticFiveHtwoPlus"] = s5["h21_plus"]
    f["QuinticFiveHtwoMinus"] = s5["h21_minus"]

    # --- Sen's limit --------------------------------------------------------
    covers = set()
    for spec in (["P2"] + ["F%d" % n for n in range(9)] + ["F12"]
                 + ["dP%d" % k for k in range(1, 9)]):
        covers.add(OR.SenLimit(spec).double_cover_euler())
    f["SenCoverChi"] = ", ".join(str(c) for c in sorted(covers))
    f["SenBases"] = 1 + 10 + 8
    f["SenGenusPTwo"] = OR.SenLimit("P2").branch_genus()
    f["SenGenusDPEight"] = OR.SenLimit("dP8").branch_genus()
    f["SenTadpoleAll"] = "yes" if all(
        OR.SenLimit(s).d7_tadpole()["cancels"]
        for s in ["P2"] + ["F%d" % n for n in range(9)]) else "NO"
    agree = all(OR.brane_stack(n, on_o7=True)["agree"] for n in range(4, 13))
    f["BraneKodairaAgree"] = "yes" if agree else "NO"

    # --- CICY fibrations ----------------------------------------------------
    if cicy_list and os.path.exists(cicy_list):
        with open(cicy_list) as fh:
            entries = json.load(fh)["entries"]
        f["CicyTotal"] = len(entries)
        f["CicyElliptic"] = sum(
            1 for e in entries if FT.obvious_fibrations(e["conf"]))
        f["CicyKThree"] = sum(
            1 for e in entries if FT.obvious_fibrations(e["conf"], 2))
    X = CICY([[2, 3], [2, 3]])
    f["BicubicHone"] = int(X.h[2])
    f["BicubicHtwo"] = int(X.h[1])
    f["BicubicFibrations"] = len(FT.obvious_fibrations([[2, 3], [2, 3]]))
    f["TetraFibrations"] = len(
        FT.obvious_fibrations([[1, 2], [1, 2], [1, 2], [1, 2]]))

    return f


def collect_twistor():
    """Numbers quoted in paper/supplementary_material_twistor.tex.

    A tree amplitude is a rational function of spinor brackets, so with
    rational spinors every quantity below is an exact rational number and the
    agreements reported are equalities rather than tolerances. The seeds are
    fixed so the paper quotes reproducible values.
    """
    f = {}

    # --- BCFW against the closed forms ------------------------------------
    mhv_ok, anti_ok = [], []
    for n in range(4, 10):
        k = TW.Kinematics.random(n, seed=n)
        b = TW.tree_amplitude(k.lam, k.lamt, [-1, -1] + [1] * (n - 2))
        p = TW.parke_taylor(k, (1, 2))
        mhv_ok.append(b == p and p != 0)
        if n <= 7:
            b2 = TW.tree_amplitude(k.lam, k.lamt, [1, 1] + [-1] * (n - 2))
            anti_ok.append(b2 == TW.anti_mhv(k, (1, 2)))
    f["BcfwMax"] = 9
    f["BcfwAgree"] = "yes" if all(mhv_ok) else "NO"
    f["BcfwAntiAgree"] = "yes" if all(anti_ok) else "NO"

    # A displayed value, with a fixed seed so the paper can quote it.
    k6 = TW.Kinematics.random(6, seed=6)
    a6 = TW.parke_taylor(k6, (1, 2))
    f["SixPointMHV"] = "%d/%d" % (a6.numerator, a6.denominator)
    f["SixPointBCFW"] = "%d/%d" % tuple(
        (lambda x: (x.numerator, x.denominator))(
            TW.tree_amplitude(k6.lam, k6.lamt, [-1, -1, 1, 1, 1, 1])))

    # --- the three-point degeneracy ---------------------------------------
    kh = TW.Kinematics.random(3, seed=2, kind="holomorphic")
    ka = TW.Kinematics.random(3, seed=2, kind="antiholomorphic")
    f["ThreePtSquares"] = sum(
        1 for i, j in [(1, 2), (2, 3), (1, 3)] if kh.square(i, j) != 0)
    f["ThreePtAngles"] = sum(
        1 for i, j in [(1, 2), (2, 3), (1, 3)] if kh.angle(i, j) != 0)
    f["ThreePtHol"] = str(TW.tree_amplitude(kh.lam, kh.lamt, [-1, -1, 1]))
    f["ThreePtAnti"] = str(TW.tree_amplitude(ka.lam, ka.lamt, [-1, 1, 1]))

    # --- NMHV, checked by symmetry ----------------------------------------
    kn = TW.Kinematics.random(6, seed=11)
    hel = [-1, -1, -1, 1, 1, 1]
    A = TW.tree_amplitude(kn.lam, kn.lamt, hel)
    f["NmhvValue"] = "%d/%d" % (A.numerator, A.denominator)

    def rot(v, r):
        return [v[(i + r) % 6] for i in range(6)]
    f["NmhvCyclic"] = "yes" if all(
        TW.tree_amplitude(rot(kn.lam, r), rot(kn.lamt, r),
                          rot(hel, r)) == A for r in range(6)) else "NO"
    f["NmhvReflect"] = "yes" if TW.tree_amplitude(
        kn.lam[::-1], kn.lamt[::-1], hel[::-1]) == A else "NO"

    # --- relations ---------------------------------------------------------
    u1, bcj = [], []
    for n in range(4, 10):
        k = TW.Kinematics.random(n, seed=n)
        u1.append(TW.u1_decoupling_residual(k, (1, 2)) == 0)
        bcj.append(TW.bcj_residual(k, (1, 2)) == 0)
    f["RelationMax"] = 9
    f["UoneZero"] = "yes" if all(u1) else "NO"
    f["BcjZero"] = "yes" if all(bcj) else "NO"

    for n in (5, 6):
        pts = [TW.Kinematics.random(n, seed=300 + i) for i in range(40)]
        r = TW.ordering_rank(pts, (1, 2))
        key = {5: "Five", 6: "Six"}[n]
        f["Rank" + key] = r["rank"]
        f["Orderings" + key] = r["orderings"]
        f["KK" + key] = _fact(n - 2)
        f["BCJcount" + key] = _fact(n - 3)
        f["RankAgrees" + key] = "yes" if r["agrees"] else "NO"

    # --- the positive Grassmannian ----------------------------------------
    # LaTeX macro names cannot contain digits, so the suffixes are spelled.
    for n, word in ((3, "three"), (4, "four"), (5, "five")):
        row = [TW.positroid_cells(k, n) for k in range(n + 1)]
        f["Cells" + word] = ", ".join(str(x) for x in row)
        counted, closed = TW.positroid_total_check(n)
        f["CellTotal" + word] = counted
        f["CellClosed" + word] = closed
    f["CellsGTwoFour"] = TW.positroid_cells(2, 4)
    f["CellsSymmetric"] = "yes" if all(
        [TW.positroid_cells(k, n) for k in range(n + 1)]
        == [TW.positroid_cells(n - k, n) for k in range(n + 1)]
        for n in range(1, 7)) else "NO"
    f["TopCellGTwoFour"] = TW.cell_dimension(2, 4)

    # --- the double fibration ----------------------------------------------
    g = {x["name"]: x for x in TW.twistor_geometry()}
    f["ChiPT"] = g["twistor space PT"]["euler"]
    f["ChiIncidence"] = g["incidence F(1,3;4)"]["euler"]
    f["ChiMinkowski"] = g["Minkowski G(2,4)"]["euler"]
    f["KleinConfig"] = str(g["Minkowski G(2,4)"]["configuration"])

    # --- twistor-space degrees and the Penrose transform -------------------
    f["DegreeMHV"] = TW.mhv_degree(2)["degree"]
    f["DegreeNMHV"] = TW.mhv_degree(3)["degree"]
    f["DegreeNMHVLoop"] = TW.mhv_degree(3, 1)["degree"]
    f["PenroseGraviton"] = TW.penrose_helicity(2)["bundle"]
    f["PenrosePhoton"] = TW.penrose_helicity(1)["bundle"]
    f["PenroseScalar"] = TW.penrose_helicity(0)["bundle"]
    f["PenroseHOne"] = TW.penrose_helicity(1)["h1_on_P3"]

    return f


def _fact(k):
    out = 1
    for i in range(2, k + 1):
        out *= i
    return out


def _u1_ok(base, h):
    try:
        FT.u1_matter((1, 2), -base.dot(base.K, [h]), base.dot([h], [h]))
        return True
    except ValueError:
        return False


def _tex_alg(name):
    """LaTeX for an algebra name: su(3) -> \\mathfrak{su}(3), e6 -> \\mathfrak{e}_6."""
    if "(" in name:
        head, rest = name.split("(", 1)
        return r"\mathfrak{%s}(%s" % (head, rest)
    return r"\mathfrak{%s}_%s" % (name[0], name[1:])


def _key(name):
    return (name.replace("(", "").replace(")", "").replace("_", "")
            .replace("2", "Two").replace("3", "Three").replace("4", "Four")
            .replace("6", "Six").replace("7", "Seven").replace("8", "Eight"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--sections", default="both",
                   choices=["strings", "twistor", "both"])
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"))
    p.add_argument("--list", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cicylist.json"))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    jobs = []
    if a.sections in ("strings", "both"):
        jobs.append(("strings_facts", "SF", collect(a.list)))
    if a.sections in ("twistor", "both"):
        jobs.append(("twistor_facts", "TF", collect_twistor()))

    for stem, prefix, facts in jobs:
        with open(os.path.join(a.outdir, stem + ".json"), "w") as fh:
            json.dump(facts, fh, indent=2, sort_keys=True)
        with open(os.path.join(a.outdir, stem + ".tex"), "w") as fh:
            fh.write("% Generated by paper/make_strings_facts.py. "
                     "Do not edit.\n")
            for k in sorted(facts):
                fh.write("\\renewcommand{\\%s%s}{%s}\n"
                         % (prefix, k, facts[k]))
        print("wrote %d facts to %s/%s.tex" % (len(facts), a.outdir, stem))
        for k in sorted(facts):
            print("  %-24s %s" % (k, facts[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
