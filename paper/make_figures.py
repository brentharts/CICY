#!/usr/bin/env python3
"""
Generate every figure used by supplementary_material.tex.

All heavy computation goes through pyCICY.cache, so the first run is slow and
subsequent runs are nearly free. This is what makes it practical to iterate on
the paper: change a label, rebuild, and the cohomology is not recomputed.

Each figure written here is described in the corresponding section of
supplementary_material.tex; the label used in the LaTeX source is given in the
comment above each function.

Usage
-----
    python3 paper/make_figures.py --outdir paper/figures
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyCICY import cache as C
from pyCICY import chirality as CH
from pyCICY import cicylist as L
from pyCICY import enumerative as EN
from pyCICY import additivity as AD
from pyCICY import phenomenology as PH
from pyCICY import symmetries as SY
from pyCICY import transitions as T
from pyCICY import viz

# Manifolds called out by name in the paper. Hodge numbers are the values
# pyCICY computes; they agree with the literature (see Table 1 of the paper).
LANDMARKS = [
    (1.0, 101.0, "quintic"),
    (2.0, 86.0, "split quintic"),
    (19.0, 19.0, "Schoen"),
]

# Base configurations used for the node-count cross-validation. These are the
# five single-equation seeds plus a couple of their splits, which gives a
# spread of matrix shapes rather than only the simplest ones.
CROSSCHECK_BASES = L.SEEDS + [
    [[1, 1, 1], [4, 1, 4]],
    [[1, 1, 1], [2, 1, 2], [2, 1, 2]],
]


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % path)
    return path


# --------------------------------------------------------------- fig:hodge
def figure_hodge(web, outdir):
    """Hodge plot of the generated web, shaded by splitting depth."""
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    viz.plot_hodge(L.web_nodes(web), ax=ax, color_by="depth",
                   annotate=LANDMARKS, title=None)
    ax.set_title("")
    return _save(fig, outdir, "hodge_depth.pdf")


# ---------------------------------------------------------- fig:favourable
def figure_favourable(web, outdir):
    """The same points, separated by whether the description is favourable."""
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    viz.plot_hodge(L.web_nodes(web), ax=ax, color_by="favourable",
                   annotate=LANDMARKS, title=None)
    ax.set_title("")
    return _save(fig, outdir, "hodge_favourable.pdf")


# --------------------------------------------------------------- fig:nodes
def figure_nodes(web, outdir):
    """Distribution of the node count N over the split edges."""
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    viz.plot_node_counts(web["edges"], ax=ax, bins=32, title=None)
    ax.set_title("")
    return _save(fig, outdir, "node_counts.pdf")


# ---------------------------------------------------------- fig:validation
def figure_validation(outdir):
    """N from ambient intersection theory against N from chi.

    This is the central consistency check of the package: two computations
    with nothing in common must land on the same integer for every split.
    """
    pairs = []
    seen = set()
    for base in CROSSCHECK_BASES:
        for col, part, child in T.splits(base):
            key = T.canonical_key(child)
            if key in seen:
                continue
            seen.add(key)
            try:
                n_geom = T.nodes_expected(base, col, part)
            except ValueError:
                continue
            d = C.hodge(base)
            r = C.hodge(child)
            if d.get("error") or r.get("error"):
                continue
            if d.get("euler") is None or r.get("euler") is None:
                continue
            n_chi = (r["euler"] - d["euler"]) // 2
            pairs.append((n_geom, n_chi))

    geom = np.array([p[0] for p in pairs])
    chi = np.array([p[1] for p in pairs])
    agree = int((geom == chi).sum())

    fig = plt.figure(figsize=(6.4, 6))
    ax = fig.add_subplot(111)
    lim = max(geom.max(), chi.max()) * 1.06 + 2
    ax.plot([0, lim], [0, lim], color="0.65", lw=1.0, zorder=0)
    # Jitter only for display: many splits share the same (N, N) pair and
    # would otherwise overplot into a single dot.
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(geom)) - 0.5) * 0.6
    ax.scatter(geom + jitter, chi + jitter, s=30, alpha=0.55,
               edgecolors="none")
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.set_xlabel(r"$N$ from ambient intersection theory")
    ax.set_ylabel(r"$N$ from $\chi(X_R) = \chi(X_D) + 2N$")
    ax.set_aspect("equal")
    ax.text(0.04, 0.95,
            "%d / %d splits agree" % (agree, len(pairs)),
            transform=ax.transAxes, va="top", fontsize=11)
    return _save(fig, outdir, "node_validation.pdf"), {
        "pairs": len(pairs), "agree": agree,
        "max_nodes": int(geom.max()) if len(geom) else 0,
    }


# ----------------------------------------------------------------- fig:ch2
def figure_ch2(outdir):
    """Two things about ch_2 across a conifold transition.

    Left: the adjunction formula for ch_2 against pyCICY's independently
    implemented second Chern class, via ch_2 = -c_2 for a Calabi-Yau.
    Right: the exceptional class does not vanish for ineffective splits,
    even though the node count does.
    """
    web = L.split_web(max_depth=2, max_configs=400)

    # -- independent check, coefficient by coefficient
    xs, ys = [], []
    disagree = 0
    for rec in L.web_nodes(web):
        adj = T.chern_character_2(rec["conf"])
        c2 = T.chern_character_2_from_c2(rec["conf"])
        if adj != c2:
            disagree += 1
        for key in set(adj) | set(c2):
            xs.append(float(adj.get(key, 0)))
            ys.append(float(c2.get(key, 0)))

    # -- exceptional class vs node count
    nodes, sizes, ineffective = [], [], []
    for base in L.SEEDS + [[[1, 1, 1], [4, 1, 4]]]:
        for col, part, _ in T.splits(base):
            n = T.nodes_expected(base, col, part)
            t = T.transition_ch2(base, col, part)
            size = sum(abs(float(v)) for v in t["exceptional"].values())
            nodes.append(n)
            sizes.append(size)
            ineffective.append(n == 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    lo = min(xs + ys) - 3
    hi = max(xs + ys) + 3
    ax1.plot([lo, hi], [lo, hi], color="0.65", lw=1.0, zorder=0)
    rng = np.random.default_rng(1)
    jx = (rng.random(len(xs)) - 0.5) * 0.8
    jy = (rng.random(len(ys)) - 0.5) * 0.8
    ax1.scatter(np.array(xs) + jx, np.array(ys) + jy, s=22, alpha=0.5,
                edgecolors="none")
    ax1.set_xlabel(r"coefficient of $\mathrm{ch}_2$ from adjunction")
    ax1.set_ylabel(r"coefficient of $-c_2$ from \texttt{pyCICY}"
                   if False else r"coefficient of $-c_2$ from pyCICY")
    ax1.set_aspect("equal")
    ax1.text(0.04, 0.95,
             "%d configurations,\n%d disagreements" % (len(L.web_nodes(web)),
                                                       disagree),
             transform=ax1.transAxes, va="top", fontsize=10)

    nodes = np.array(nodes, dtype=float)
    sizes = np.array(sizes, dtype=float)
    ineff = np.array(ineffective)
    ax2.scatter(nodes[~ineff], sizes[~ineff], s=30, alpha=0.6,
                edgecolors="none", label="effective ($N>0$)")
    ax2.scatter(nodes[ineff], sizes[ineff], s=44, alpha=0.85,
                color="C3", marker="s", edgecolors="none",
                label="ineffective ($N=0$)")
    ax2.axhline(0, color="0.65", lw=1.0)
    ax2.set_xlabel("node count $N$")
    ax2.set_ylabel(r"$\sum |{\rm coefficients}|$ of $[\mathbb{P}^1_s]$")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylim(bottom=-0.5)

    fig.tight_layout()
    return _save(fig, outdir, "ch2_check.pdf"), {
        "configs": len(L.web_nodes(web)),
        "coefficients": len(xs),
        "disagreements": disagree,
        "edges": len(nodes),
        "ineffective_edges": int(ineff.sum()),
        "ineffective_with_nonzero_class": int(
            sum(1 for s, i in zip(sizes, ineff) if i and s > 0)),
    }



# ------------------------------------------------------------------ fig:gv
def figure_gv(outdir):
    """Genus-zero Gopakumar-Vafa invariants of the five one-parameter models."""
    max_d = 8
    results = {}
    for conf in ([[4, 5]], [[5, 3, 3]], [[5, 2, 4]], [[6, 2, 2, 3]],
                 [[7, 2, 2, 2, 2]]):
        g = EN.gv_invariants(conf, max_degree=max_d)
        results[g["name"]] = g

    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_subplot(111)
    markers = ["o", "s", "^", "D", "v"]
    for (name, g), mk in zip(sorted(results.items()), markers):
        ds = sorted(g["invariants"])
        ns = [g["invariants"][d] for d in ds]
        ax.plot(ds, ns, mk + "-", ms=5, lw=1.2, label="%s  ($n_1=%d$)"
                % (name, g["invariants"][1]))
    ax.set_yscale("log")
    ax.set_xlabel("degree $d$")
    ax.set_ylabel("$n_d$")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout()
    return _save(fig, outdir, "gv_invariants.pdf"), {
        "models": len(results),
        "max_degree": max_d,
        "quintic": [results["P^4[5]"]["invariants"][d] for d in (1, 2, 3)],
    }



# ---------------------------------------------------------- fig:additivity
def figure_additivity(web, outdir):
    """Path dependence of the node count: totals rigid, decompositions not.

    Prompted by the failure of additivity for unknotting number
    (arXiv:2506.24088). Here the total is forced by chi and the decomposition
    is not, which is the opposite failure mode.
    """
    survey = AD.survey(web)
    seeds = AD.seed_keys()

    spreads = []
    per_depth = {}
    for key, rec in web["nodes"].items():
        if key in seeds:
            continue
        ms = AD.decompositions(web, key)
        if not ms:
            continue
        d = rec["depth"]
        per_depth.setdefault(d, [0, 0])
        per_depth[d][1] += 1
        if len(ms) > 1:
            per_depth[d][0] += 1
            flat = [x for m in ms for x in m]
            spreads.append(max(flat) - min(flat))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    depths = sorted(per_depth)
    multi = [per_depth[d][0] for d in depths]
    total = [per_depth[d][1] for d in depths]
    width = 0.38
    xs = np.arange(len(depths))
    ax1.bar(xs - width / 2, total, width, label="configurations", alpha=0.85)
    ax1.bar(xs + width / 2, multi, width,
            label="with $>1$ decomposition", alpha=0.9, color="C3")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.set_xlabel("split depth $=K-1$")
    ax1.set_ylabel("configurations")
    ax1.legend(frameon=False, fontsize=9)

    if spreads:
        ax2.hist(spreads, bins=24, alpha=0.85, color="C3", edgecolor="none")
    ax2.set_xlabel(r"spread of a single-step $N$ between routes")
    ax2.set_ylabel("configurations")
    ax2.text(0.97, 0.95,
             "totals disagree in\n%d of %d cases"
             % (len(survey["total_mismatches"]), survey["checked"]),
             transform=ax2.transAxes, ha="right", va="top", fontsize=10)

    fig.tight_layout()
    return _save(fig, outdir, "additivity.pdf"), {
        "checked": survey["checked"],
        "multiple": survey["multiple_decompositions"],
        "total_mismatches": len(survey["total_mismatches"]),
        "max_spread": survey["max_single_step_spread"],
    }



# ------------------------------------------------------- fig:generations
def figure_generations(outdir):
    """Three-generation search: what the Euler characteristic permits, and
    what the freely acting symmetries actually allow.

    Returns (path, facts) or (None, None) when the data files are absent, so
    that the paper can still be built without them.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cicy_path = os.path.join(here, "data", "cicylist.json")
    sym_path = os.path.join(here, "data", "symmetries.json")
    if not (os.path.exists(cicy_path) and os.path.exists(sym_path)):
        print("  skipping generations figure: data/ not present")
        return None, None

    entries = L.load_published_list(cicy_path)
    syms = SY.load_symmetries(sym_path)
    by_num = {r["num"]: r for r in syms}

    survey = PH.generation_survey(entries, generations=3)
    required = survey["required_orders"]

    with_data = [r for r in syms
                 if isinstance(r["symmetries"], list) and r["symmetries"]]
    records = 0
    for rec in with_data:
        parsed, _ = SY.parse_symmetry_records(rec)
        records += len(parsed)

    hits = []
    for rec in entries:
        need = PH.required_symmetry_order(rec["euler"], 3)
        if need is None:
            continue
        sym = by_num.get(rec["num"])
        if not sym:
            continue
        parsed, _ = SY.parse_symmetry_records(sym)
        if need in {x.order for x in parsed}:
            hits.append(rec["num"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ks = sorted(k for k in required if k <= 20)
    ax1.bar([str(k) for k in ks], [required[k] for k in ks], alpha=0.85)
    ax1.set_xlabel(r"required $|\Gamma| = |\chi|/6$")
    ax1.set_ylabel("configurations")

    stages = ["all\nCICYs", r"$|\chi|$ permits" + "\n3 generations",
              "carry a\nsymmetry", "of the\nrequired order"]
    counts = [len(entries), survey["candidates"], len(with_data), len(hits)]
    bars = ax2.bar(stages, counts, alpha=0.85,
                   color=["C0", "C0", "C0", "C3"])
    ax2.set_yscale("log")
    ax2.set_ylabel("configurations (log scale)")
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2, c, str(c),
                 ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    path = _save(fig, outdir, "generations.pdf")
    return path, {
        "cicys": len(entries),
        "candidates": survey["candidates"],
        "with_symmetry": len(with_data),
        "records": records,
        "hits": sorted(hits),
        "n_hits": len(hits),
        "min_abs_euler": survey["min_abs_euler"],
    }


# -------------------------------------------------------------- fig:growth
def figure_growth(web, outdir):
    """How the web grows with splitting depth, in configurations and in
    distinct Hodge pairs."""
    recs = L.web_nodes(web)
    depths = sorted({r["depth"] for r in recs})
    counts = [sum(1 for r in recs if r["depth"] == d) for d in depths]
    cumulative_pairs = []
    pairs = set()
    for d in depths:
        for r in recs:
            if r["depth"] == d and r.get("h11") is not None:
                pairs.add((r["h11"], r["h21"]))
        cumulative_pairs.append(len(pairs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(depths, counts, color="C0", alpha=0.85)
    ax1.set_xlabel("splitting depth")
    ax1.set_ylabel("new configurations")
    ax1.set_xticks(depths)
    for d, c in zip(depths, counts):
        ax1.text(d, c, str(c), ha="center", va="bottom", fontsize=9)

    ax2.plot(depths, cumulative_pairs, "o-", color="C1")
    ax2.axhline(266, color="0.5", ls="--", lw=1.0)
    ax2.text(depths[0], 266, " 266 in the published list", va="bottom",
             fontsize=9, color="0.35")
    ax2.set_xlabel("splitting depth")
    ax2.set_ylabel("distinct Hodge pairs found")
    ax2.set_xticks(depths)
    ax2.set_ylim(0, 300)
    fig.tight_layout()
    return _save(fig, outdir, "web_growth.pdf")


# ----------------------------------------------------------------- fig:cy
def figure_quintic(outdir):
    """The Fermat cross-section of the quintic, labelled with its invariants."""
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    viz.plot(5, res=60, ax=ax, cmap="viridis", color_by="z", axis_off=True,
             elev=25, azim=50)
    ax.set_title("")
    return _save(fig, outdir, "quintic_surface.pdf")


# ------------------------------------------------------------ cache timing
def _measure_timing(web_kwargs):
    """Wall time for the survey with and without the cache.

    Reported in the build log. The paper no longer carries a figure for this;
    the number is quoted in the text instead.
    """
    scratch = "/tmp/pycicy-cache/paper-timing.sqlite"
    if os.path.exists(scratch):
        os.remove(scratch)

    cold_cache = C.Cache(path=scratch)
    t0 = time.time()
    L.split_web(cache=cold_cache, **web_kwargs)
    cold = time.time() - t0

    t0 = time.time()
    L.split_web(cache=cold_cache, **web_kwargs)
    warm = time.time() - t0

    return {"cold": cold, "warm": warm}


# ----------------------------------------------------------------- driver


# ----------------------------------------------------------- fig:polygons
def figure_polygons(outdir):
    """The sixteen reflexive polygons with their polar duals.

    Each panel is the toric diagram of a local Calabi-Yau K_S together with
    its Batyrev mirror P*, and the titles record the twelve theorem.
    """
    fig = viz.plot_polygon_grid()
    return _save(fig, outdir, "fig_polygons.pdf")


# --------------------------------------------------------- fig:butterflies
def figure_butterflies(outdir, qmax=40):
    """Hofstadter spectra of the local F_0 and local B_3 mirror curves.

    Quantizing the mirror curve of a local toric Calabi-Yau gives an electron
    on a 2d lattice in a magnetic field (Sugimoto, arXiv:1701.01561). The
    square Newton polygon of local F_0 gives the square lattice and the
    classic butterfly; the hexagonal polygon of local B_3 gives the
    triangular lattice, whose spectrum is not symmetric in E and so is
    visibly slanted, since E(Phi) = -E(1-Phi) holds regardless.
    """
    fig = viz.plot_butterfly_grid(["F0", "B3"], qmax=qmax, nk=6,
                                  gaps_at=(1, 3))
    return _save(fig, outdir, "fig_butterflies.pdf")


# --------------------------------------------------------------- fig:knots
def figure_knots(outdir):
    """K15n81556 against its mirror, and 7_1 # m7_1 as a braid closure.

    Wang and Zhang, arXiv:2507.14265, observed that the two diagrams of
    K15n81556 in the Brittenham-Hermiller argument are a chiral knot and its
    mirror image, which the Jones polynomial detects; the left panel is that
    comparison. The right panel is the connected sum whose unknotting number
    breaks additivity, drawn as the juxtaposition of the two braid words.
    """
    # stacked rather than side by side: the braid is fourteen crossings wide
    # and three strands tall, so it wants the full width
    fig = plt.figure(figsize=(9.5, 7.0))
    ax1 = fig.add_subplot(211)
    viz.plot_jones("K15n81556", ax=ax1)
    ax2 = fig.add_subplot(212)
    viz.plot_braid([1] * 7 + [-2] * 7, strands=3, ax=ax2,
                   title=r"$7_1 \,\#\, m7_1$, the knot of arXiv:2506.24088")
    fig.tight_layout()
    return _save(fig, outdir, "fig_knots.pdf")


# ----------------------------------------------------------- fig:chirality
def figure_chirality(outdir):
    """One mirror operation over knots, reflexive polygons and threefolds.

    Horizontally the combination of each object's invariant pair that the
    mirror negates, vertically the one it preserves. Mirror partners sit
    symmetrically about zero and objects fixed by their involution sit on the
    axis. The right-hand panel is the conventional Hodge plot in disguise.
    """
    fig = viz.plot_chirality_grid()
    return _save(fig, outdir, "fig_chirality.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-configs", type=int, default=1200)
    ap.add_argument("--qmax", type=int, default=40,
                    help="butterfly resolution: largest flux denominator")
    ap.add_argument("--skip-timing", action="store_true",
                    help="skip the cache timing figure, which by design "
                         "cannot use the cache and so is the slow one")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(here, "figures")
    os.makedirs(outdir, exist_ok=True)

    web_kwargs = dict(max_depth=args.depth, max_configs=args.max_configs)

    print("building the split web (depth %d)" % args.depth)
    web = L.split_web(**web_kwargs)
    survey = L.survey(web)
    stats = web["stats"]
    print("  %d configurations, %d edges" % (survey["configurations"],
                                             stats["edges"]))

    print("generating figures into %s" % outdir)
    figure_hodge(web, outdir)
    figure_favourable(web, outdir)
    figure_nodes(web, outdir)
    _, validation = figure_validation(outdir)
    _, ch2 = figure_ch2(outdir)
    _, gv = figure_gv(outdir)
    _, add = figure_additivity(web, outdir)
    _, gen = figure_generations(outdir)
    figure_growth(web, outdir)
    figure_quintic(outdir)
    figure_polygons(outdir)
    figure_butterflies(outdir, qmax=args.qmax)
    figure_knots(outdir)
    figure_chirality(outdir)

    # The cache timing figure was dropped from the paper; the timing is
    # still measured and reported here, but no longer plotted.
    timing = None
    if not args.skip_timing:
        timing = _measure_timing(web_kwargs)

    # Numbers quoted in the text are written out here so the paper and the
    # figures can never drift apart.
    qk = T.canonical_key([[4, 5]])
    sk = T.canonical_key([[1, 1, 1], [4, 1, 4]])
    quintic_edge = [e for e in web["edges"]
                    if e["parent"] == qk and e["child"] == sk]

    facts = {
        "depth": args.depth,
        "configurations": survey["configurations"],
        "edges": stats["edges"],
        "effective_edges": stats["effective_edges"],
        "ineffective_edges": stats["ineffective_edges"],
        "distinct_hodge_pairs": survey["distinct_hodge_pairs"],
        "favourable": survey["favourable"],
        "by_depth": survey["by_depth"],
        "h11_range": survey["h11_range"],
        "h21_range": survey["h21_range"],
        "euler_range": survey["euler_range"],
        "quintic_split_nodes": quintic_edge[0]["nodes"] if quintic_edge else None,
        "validation": validation,
        "ch2": ch2,
        "gv": gv,
        "additivity": add,
        "generations": gen,
        "timing": timing,
    }
    with open(os.path.join(outdir, "facts.json"), "w") as f:
        json.dump(facts, f, indent=2, sort_keys=True)
    print("  wrote %s" % os.path.join(outdir, "facts.json"))

    # The same numbers as LaTeX macros. supplementary_material.tex \input's
    # this, so a quantity quoted in the prose is the one that was actually
    # computed; there is no opportunity for the text to go stale.
    macros = {
        "FactDepth": args.depth,
        "FactConfigs": survey["configurations"],
        "FactEdges": stats["edges"],
        "FactEffective": stats["effective_edges"],
        "FactIneffective": stats["ineffective_edges"],
        "FactHodgePairs": survey["distinct_hodge_pairs"],
        "FactFavourable": survey["favourable"],
        "FactQuinticNodes": facts["quintic_split_nodes"],
        "FactValidationPairs": validation["pairs"],
        "FactValidationAgree": validation["agree"],
        "FactChTwoConfigs": ch2["configs"],
        "FactChTwoCoeffs": ch2["coefficients"],
        "FactChTwoDisagree": ch2["disagreements"],
        "FactChTwoEdges": ch2["edges"],
        "FactChTwoIneffective": ch2["ineffective_edges"],
        "FactChTwoIneffNonzero": ch2["ineffective_with_nonzero_class"],
        "FactGVModels": gv["models"],
        "FactGVMaxDegree": gv["max_degree"],
        "FactGVQuinticOne": gv["quintic"][0],
        "FactGVQuinticTwo": gv["quintic"][1],
        "FactGVQuinticThree": gv["quintic"][2],
        "FactAddChecked": add["checked"],
        "FactAddMultiple": add["multiple"],
        "FactAddMismatch": add["total_mismatches"],
        "FactAddSpread": add["max_spread"],
        "FactHminOne": int(survey["h11_range"][0]),
        "FactHmaxOne": int(survey["h11_range"][1]),
        "FactHminTwo": int(survey["h21_range"][0]),
        "FactHmaxTwo": int(survey["h21_range"][1]),
    }
    if timing:
        macros["FactCold"] = "%.0f" % timing["cold"]
        macros["FactWarm"] = "%.1f" % timing["warm"]
    if gen:
        macros.update({
            "FactGenCicys": gen["cicys"],
            "FactGenCandidates": gen["candidates"],
            "FactGenWithSymmetry": gen["with_symmetry"],
            "FactGenRecords": gen["records"],
            "FactGenHits": gen["n_hits"],
            "FactGenMinEuler": gen["min_abs_euler"],
            "FactGenList": ", ".join(str(h) for h in gen["hits"]),
        })

    tex_path = os.path.join(outdir, "facts.tex")
    with open(tex_path, "w") as f:
        f.write("% Generated by paper/make_figures.py -- do not edit.\n")
        f.write("% providecommand+renewcommand so this works whether or not\n"
                "% the document already defines a placeholder.\n")
        for k, v in sorted(macros.items()):
            f.write("\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}\n"
                    % (k, k, v))
    print("  wrote %s" % tex_path)

    print("\nfigures done. Numbers quoted in the paper:")
    for k in ("configurations", "edges", "effective_edges",
              "ineffective_edges", "distinct_hodge_pairs", "favourable",
              "quintic_split_nodes"):
        print("  %-22s %s" % (k, facts[k]))
    print("  %-22s %s" % ("validation", validation))
    print("  %-22s %s" % ("ch2", ch2))
    print("  %-22s %s" % ("gv", gv))
    print("  %-22s %s" % ("additivity", add))
    print("  %-22s %s" % ("generations", gen))
    if timing:
        print("  %-22s %s" % ("timing",
                              {k: round(v, 1) for k, v in timing.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
