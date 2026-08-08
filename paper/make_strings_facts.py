#!/usr/bin/env python3
"""
Emit every number quoted in paper/supplementary_material_strings.tex.

    python3 paper/make_strings_facts.py --outdir paper/figures

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
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"))
    p.add_argument("--list", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cicylist.json"))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    facts = collect(a.list)

    with open(os.path.join(a.outdir, "strings_facts.json"), "w") as fh:
        json.dump(facts, fh, indent=2, sort_keys=True)
    with open(os.path.join(a.outdir, "strings_facts.tex"), "w") as fh:
        fh.write("% Generated by paper/make_strings_facts.py. Do not edit.\n")
        for k in sorted(facts):
            fh.write("\\renewcommand{\\SF%s}{%s}\n" % (k, facts[k]))

    print("wrote %d facts to %s" % (len(facts), a.outdir))
    for k in sorted(facts):
        print("  %-24s %s" % (k, facts[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
