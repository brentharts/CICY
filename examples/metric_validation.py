#!/usr/bin/env python3
"""
Is the metric any good?

Everything else in this package is exact: indices, cohomology, intersection
numbers, group characters. The metric is not. It is a neural network fit, and
the only honest way to talk about it is to measure how far it is from
satisfying the equations it is supposed to satisfy.

This script runs the whole bridge end to end -- configuration matrix, invariant
polynomial, stability moduli, orbit-augmented points, trained metric -- and
then reports three numbers:

    sigma measure    how far det(g) is from being proportional to |Omega|^2,
                     which is the Monge-Ampere equation the Ricci-flat metric
                     solves. Zero for the exact answer.
    Ricci measure    the residual scalar curvature, normalised. Zero for the
                     exact answer.
    Gamma deviation  how far the learned metric is from being invariant under
                     the group. Zero by construction with a symmetrised
                     network, and *not* zero with orbit augmentation alone.

The first two say whether the metric has converged. The third says whether it
descends to the quotient, which is what a heterotic model actually needs.

What this does NOT do
---------------------
It does not compute the Euler characteristic from the metric. That would be
the strongest possible check -- Chern-Gauss-Bonnet gives chi from the curvature
of any Hermitian metric, so comparing it against the exact chi = -128 that
:meth:`pyCICY.CICY.euler_characteristic` computes from the configuration matrix
would validate the entire chain rather than its interfaces. cymyc implements
that integral in ``chern_gauss_bonnet.euler_characteristic``, but it expects
local coordinates, pullbacks and a volume form in its own conventions, and
this bridge produces cymetric's. Matching them is real work and is not done.
Until it is, "the metric is good" means "these three residuals are small", not
"the topology comes out right".

Nor does it compute a Yukawa coupling or a fermion mass. Those need the metric
*and* harmonic representatives of the cohomology classes, which is another
project again. :mod:`pyCICY.phenomenology` raises rather than guessing, and
nothing here changes that.

Usage
-----
    python3 examples/metric_validation.py
    python3 examples/metric_validation.py --epochs 20 --points 5000
    make metric-validation EPOCHS=20
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--points", type=int, default=1200)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--outdir", default="/tmp/pycicy_metric")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)

    from pyCICY import CICY
    from pyCICY import equivariant as E
    from pyCICY import export as X_

    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        from cymetric.pointgen import EquivariantCICYPointGenerator
        from cymetric.pointgen.pointgen_equivariant import make_symmetrised_net
        from cymetric.pointgen.nphelper import (prepare_dataset,
                                                prepare_basis_pickle)
        from cymetric.jax.models.models import PhiFSModel
        from cymetric.jax.models.helper import prepare_basis, train_model
        from cymetric.jax.models import measures
    except ImportError as e:
        print("This example needs cymetric with its JAX extras and the "
              "equivariant point generator:")
        print("    pip install jax equinox optax")
        print("    git clone https://github.com/brentharts/cymetric")
        print("missing: %s" % e)
        return 0

    conf = [[1, 2], [1, 2], [1, 2], [1, 2]]
    model = [[-2, -2, -1, 2], [-2, 1, 0, 0], [1, -2, 1, 0],
             [1, 1, -1, 0], [2, 2, 1, -2]]
    A = E.TETRAQUADRIC_Z2()
    X = CICY(conf)

    banner("The model, exactly")
    print("  configuration      %s" % conf)
    print("  chi                %d   (exact, from the configuration matrix)"
          % X.euler_characteristic())
    print("  h^{1,1}, h^{2,1}   %d, %d" % (X.h[2], X.h[1]))
    rep = X_.quotient_report(conf, A, summands=model)
    print("  |Gamma|            %d" % rep["order"])
    print("  preserves Omega    %s   (else X/Gamma is not Calabi-Yau)"
          % rep["preserves_omega"])
    print("  acts freely        %s" % rep["looks_free"])
    print("  kmoduli            %s" % np.round(rep["kmoduli"], 5).tolist())
    print("     the point where the bundle is poly-stable; at any other point")
    print("     the model does not exist")

    banner("Building the dataset")
    export = X_.to_cymetric(conf, action=A, kmoduli="stability",
                            summands=model, seed=3, include_group=True)
    pg = EquivariantCICYPointGenerator(**export, verbose=3)
    t0 = time.time()
    kappa = prepare_dataset(pg, args.points, args.outdir)
    prepare_basis_pickle(pg, args.outdir, kappa)
    data = dict(np.load(os.path.join(args.outdir, "dataset.npz")))
    BASIS = prepare_basis(os.path.join(args.outdir, "basis.pickle"))
    print("  %d orbits -> %d training points   kappa = %.4f   [%.0fs]"
          % (args.points, len(data["X_train"]), kappa, time.time() - t0))

    ncoords = data["X_train"].shape[1] // 2

    def make_net(symmetrise):
        # gelu, not relu: this model uses the *second* derivative of the
        # network, and a piecewise-linear activation has none.
        inner = eqx.nn.MLP(in_size=2 * ncoords, out_size=1,
                           width_size=args.width, depth=args.depth,
                           activation=jax.nn.gelu,
                           key=jax.random.PRNGKey(0))
        if not symmetrise:
            return inner
        return make_symmetrised_net(inner, export["group_matrices"],
                                    export["ambient"], ncoords)

    banner("Training")
    runs = {}
    for tag, sym in (("plain", False), ("symmetrised", True)):
        t0 = time.time()
        m, hist = train_model(PhiFSModel(make_net(sym), BASIS), data,
                              epochs=args.epochs, batch_sizes=(64, 2000),
                              verbose=0)
        runs[tag] = (m, hist)
        print("  %-12s sigma %.5f -> %.5f   [%.0fs]"
              % (tag, hist["sigma_loss"][0], hist["sigma_loss"][-1],
                 time.time() - t0))

    banner("How good is it?")
    Xv = jnp.array(data["X_val"], dtype=jnp.float32)
    yv = jnp.array(data["y_val"], dtype=jnp.float32)
    pbs = jnp.array(data["val_pullbacks"])

    z = pg.generate_point_weights(150)["point"][:150]
    g = export["group_matrices"][1]
    gz = pg._rescale_to_patch(z @ g.T)

    def feats(w):
        return jnp.array(np.concatenate([w.real, w.imag], axis=-1),
                         dtype=jnp.float32)

    print("  %-14s %14s %14s %16s"
          % ("network", "sigma measure", "Ricci measure", "Gamma deviation"))
    for tag, (m, hist) in runs.items():
        try:
            sm = float(measures.sigma_measure(m, Xv, yv))
        except Exception:                                        # noqa: BLE001
            sm = float("nan")
        try:
            rm = float(measures.ricci_measure(m, Xv, yv, pullbacks=pbs))
        except Exception:                                        # noqa: BLE001
            rm = float("nan")
        a = np.array(m(feats(z)))
        b = np.array(m(feats(gz)))
        dev = float(np.mean(np.abs(a - b)) / np.mean(np.abs(a)))
        print("  %-14s %14.5f %14.5f %16.3e" % (tag, sm, rm, dev))

    print("""
The first two columns are the residuals of the Monge-Ampere equation and of
Ricci flatness, and they are what "converged" means: both should fall towards
zero as epochs and points increase. A handful of CPU epochs will not get them
there, and this script makes no claim that they do -- run it with --epochs 100
--points 20000 on a GPU before believing any number that depends on the metric.

The third column is different in kind. It is zero by construction for the
symmetrised network and stays non-zero for the plain one however long either
trains, because orbit augmentation symmetrises the training distribution and
not the learned function. That is the one thing here that is exact.""")

    banner("Not done")
    print("""\
Computing chi from the trained metric via Chern-Gauss-Bonnet and comparing it
against the exact %d above would validate the whole chain rather than its
interfaces, since chi is topological and any Hermitian metric should give it.
cymyc implements the integral; matching its coordinate and pullback
conventions to cymetric's data is not done, so that check does not yet exist.

Until it does, nothing here supports a claim about masses or couplings. Those
need harmonic representatives as well as a metric, and
pyCICY.phenomenology.why_not_masses still raises rather than guessing."""
          % X.euler_characteristic())
    return 0


if __name__ == "__main__":
    sys.exit(main())
