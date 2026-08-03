#!/usr/bin/env python3
"""
Phenomenological analysis of a CICY configuration.

What this reports is what the topology determines: the charged spectrum of
the heterotic standard embedding, the net number of chiral generations, and
the order a freely acting symmetry would need for three generations.

What it does not report is masses or couplings. The proton-to-electron mass
ratio is not a function of the configuration matrix. Physical Yukawa
couplings need the Kahler normalisation of the fields, hence the Ricci-flat
metric, which is not known in closed form on any compact Calabi-Yau
threefold; the overall scale needs the supersymmetry breaking mechanism; the
gauge coupling needs the dilaton vacuum expectation value. All of these need
the moduli stabilised, which is unsolved. Passing
--proton-electron-mass-ratio therefore prints the reason rather than a
number. See pyCICY.phenomenology for the full statement.

Usage
-----
    python3 scripts/stdmodel.py --cicy '[[4,5]]'
    python3 scripts/stdmodel.py --cicy '[[4,5]]' --symmetry-order 1
    python3 scripts/stdmodel.py --survey data/cicylist.json --out /tmp
"""

import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import phenomenology as P
from pyCICY import transitions as T


def analyse(conf, symmetry_order=1):
    print("configuration: %r" % (conf,))
    info = T.check_configuration(conf)
    print("  dim X            %d" % info["dim_X"])
    print("  Calabi-Yau       %s" % info["calabi_yau"])
    for w in info["warnings"]:
        print("  warning          %s" % w)
    if not info["calabi_yau"] or info["dim_X"] != 3:
        print("\nGeneration counting applies to Calabi-Yau threefolds only.")
        return None

    spec = P.standard_embedding_spectrum(conf, symmetry_order=symmetry_order)
    print("\nheterotic standard embedding, V = TX")
    print("  gauge group      %s" % spec["gauge_group"])
    print("  chi              %d" % spec["euler"])
    print("  h^{1,1}          %d" % spec["h11"])
    print("  h^{2,1}          %d" % spec["h21"])
    print("  n(27)            %s" % spec["n_27"])
    print("  n(27-bar)        %s" % spec["n_27bar"])
    print("  |Gamma|          %d" % spec["symmetry_order"])
    print("  net generations  %d" % spec["net_generations"])
    for n in spec["notes"]:
        print("  note             %s" % n)

    need = P.required_symmetry_order(spec["euler"], 3)
    print("\nfor three generations")
    if need is None:
        print("  no integer |Gamma| works: |chi|/6 is not an integer")
    else:
        print("  would need a freely acting Gamma of order |chi|/6 = %d" % need)
        print("  (existence of such an action is a separate question, settled")
        print("   for the CICY list by Braun, arXiv:1003.3235, not here)")
    return spec


def survey(path, outdir=None, generations=3):
    from pyCICY import cicylist as L

    entries = L.load_published_list(path)
    result = P.generation_survey(entries, generations=generations)

    print("surveying %d published CICYs for %d-generation candidates"
          % (result["entries"], generations))
    print("  chi = 0                      %d" % result["zero_euler"])
    print("  |chi| divisible by 2*%d       %d"
          % (generations, result["candidates"]))
    print("  smallest |chi| (non-zero)    %d" % result["min_abs_euler"])
    print("  |chi| = %d ever occurs?       %s"
          % (2 * generations, not result["needs_quotient"]))
    if result["needs_quotient"]:
        print("  -> a non-trivial free quotient is always required")

    orders = result["required_orders"]
    print("\n  required |Gamma|   CICYs")
    for k in sorted(orders)[:12]:
        print("    %-14d %d" % (k, orders[k]))

    if outdir:
        _plot_survey(result, outdir)
    return result


def _plot_survey(result, outdir):
    """Plot what was computed: the distribution of required |Gamma|.

    There is deliberately no plot of a predicted mass ratio. A bar chart
    comparing a computed value with a measured constant is only meaningful
    when the computed value exists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orders = result["required_orders"]
    ks = sorted(k for k in orders if k <= 24)
    vals = [orders[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar([str(k) for k in ks], vals, alpha=0.85)
    ax.set_xlabel(r"required $|\Gamma| = |\chi|/6$")
    ax.set_ylabel("CICYs")
    ax.set_title("Three-generation candidates among %d CICYs "
                 "(standard embedding)" % result["entries"])
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "generation_candidates.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote %s" % path)


def report_mass_ratio():
    """Print why the mass ratio is not available, instead of inventing one."""
    why = P.why_not_masses()
    print("\nrequested: %s" % why["quantity"])
    print("computable from the configuration matrix: NO\n")
    print("why not:")
    for r in why["reasons"]:
        print("  - %s" % r)
    print("\nwhat this package can compute instead:")
    for r in why["what_is_computable"]:
        print("  - %s" % r)
    print("\nNo number is printed and no plot is produced, deliberately: a")
    print("placeholder value shown beside the measured ratio would be")
    print("indistinguishable from a derived one.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cicy", type=str,
                    help="configuration matrix, e.g. '[[4,5]]'")
    ap.add_argument("--symmetry-order", type=int, default=1,
                    help="|Gamma| of a freely acting symmetry (default 1)")
    ap.add_argument("--survey", type=str, default=None,
                    help="path to data/cicylist.json; survey the whole list")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--out", default=None, help="directory for plots")
    ap.add_argument("--proton-electron-mass-ratio", action="store_true",
                    help="explain why this is not computable from topology")
    args = ap.parse_args(argv)

    if args.proton_electron_mass_ratio:
        report_mass_ratio()
        if not args.cicy:
            return 0

    if args.survey:
        survey(args.survey, outdir=args.out, generations=args.generations)
        return 0

    if not args.cicy:
        ap.error("give --cicy or --survey")

    try:
        conf = ast.literal_eval(args.cicy)
    except (ValueError, SyntaxError):
        print("Error: invalid configuration matrix. Use e.g. '[[4,5]]'")
        return 2

    analyse(conf, symmetry_order=args.symmetry_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
