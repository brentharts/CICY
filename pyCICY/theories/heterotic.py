r"""
pyCICY.theories.heterotic -- heterotic E_8 x E_8 constructions.

Two of them, at opposite ends of what this package can say.

:class:`StandardEmbedding` is the case where a Yukawa coupling comes out
*exactly*. Setting the gauge bundle equal to the tangent bundle leaves E_6 in
four dimensions, with the 27-bar families counted by h^{1,1}, and the coupling
of three of them is

    Y_rst = int_X J_r ^ J_s ^ J_t = d_rst ,

the triple intersection numbers -- integers, already computed by
:meth:`pyCICY.CICY.triple_intersection`. This is the classical limit; the
coupling receives worldsheet instanton corrections,

    Y(q) = d_rst + sum_beta n_beta beta_r beta_s beta_t q^beta / (1 - q^beta) ,

with n_beta the genus-zero Gopakumar-Vafa invariants, which
:func:`pyCICY.enumerative.gv_invariants` supplies for the five one-parameter
models. On the quintic that gives

    Y(q) = 5 + 2875 q/(1-q) + 8*609250 q^2/(1-q^2) + ... ,

every coefficient an integer, no metric anywhere. It is a genuine Yukawa
coupling and it is exact.

:class:`LineBundleModel` is the case where it is not. An SU(5) sum of line
bundles gives an SU(5) GUT and, after a Wilson line, the Standard Model; the
spectrum is exact via :mod:`pyCICY.breaking`. But the couplings are cup
products H^1(V) x H^1(V) x H^1(Lambda^2 V) -> H^3(Lambda^3 V) = C, and
evaluating them needs explicit cohomology representatives rather than
dimensions. This package computes dimensions. So the holomorphic coupling is
declared not-implemented with the reason, which is a different statement from
the physical coupling being unavailable, and the class keeps them distinct.

The distinction that runs through both: **holomorphic couplings are exact when
they can be computed at all; physical couplings need the metric.** Neither
class returns a physical number.
"""

import numpy as np

from .base import NeedsMetric, Theory, register

__all__ = ["StandardEmbedding", "LineBundleModel"]


@register
class StandardEmbedding(Theory):
    r"""V = TX, giving E_6 in four dimensions.

    The oldest heterotic compactification and the one where everything is
    determined by the geometry alone -- no bundle to choose, no stability to
    check. :mod:`pyCICY.phenomenology` computes its spectrum; this class adds
    the Yukawa couplings.
    """

    key = "heterotic-standard-embedding"

    def gauge_group(self):
        return "E_6"

    def spectrum(self):
        """Exact, from Hodge numbers.

        The 27 families are counted by h^{2,1} and the 27-bar by h^{1,1}, so
        the net chirality is h^{2,1} - h^{1,1} = -chi/2, which is what
        :mod:`pyCICY.phenomenology` reports.
        """
        h11 = int(self.X.h[2])
        h21 = int(self.X.h[1])
        return {"27": h21, "27bar": h11,
                "generations": h21 - h11,
                "singlets": "h^1(End TX), not computed here"}

    def holomorphic_yukawa(self, q=None, max_degree=6):
        r"""
        The (1,1)-type Yukawa couplings, exactly.

        With ``q`` omitted, returns the classical couplings ``d_rst``: integers,
        no approximation, no metric.

        With ``q`` given -- a single number for a one-parameter model -- the
        instanton-corrected coupling is returned as well, summing the
        genus-zero Gopakumar-Vafa invariants up to ``max_degree``. The series
        converges for ``|q| < 1``; the truncation error is not estimated here,
        so treat the value as the partial sum it is.

        Returns a dict with ``classical``, and when ``q`` is given ``quantum``,
        ``invariants``, ``terms``, ``last_term`` and ``converging``.

        The convergence flag is not decoration. The Gopakumar-Vafa invariants
        grow very fast -- on the quintic n_5 is 2.3 x 10^14 -- so the series
        only converges for genuinely small ``q``. At ``q = 0.01`` the
        degree-five term alone is about 3 x 10^6 and the partial sum is
        meaningless; at ``q = 0.001`` it is about 0.03 and the sum is the
        coupling. The flag compares the last term against the total and says
        which regime you are in.
        """
        d = np.asarray(self.X.triple_intersection(), dtype=float)
        out = {"classical": d,
               "summary": "d_rst, %d nonzero entries"
                          % int(np.count_nonzero(d))}
        if q is None:
            return out

        h11 = int(self.X.h[2])
        if h11 != 1:
            raise NotImplementedError(
                "instanton corrections are implemented only for one-parameter "
                "models, where the mirror map is a one-variable problem. This "
                "manifold has h^{1,1} = %d. The classical couplings above are "
                "still exact." % h11)

        from .. import enumerative
        gv = enumerative.gv_invariants(self.X.M.tolist(), max_degree=max_degree)
        inv = gv["invariants"]
        total = float(d.ravel()[0])
        terms = {}
        for deg, n in sorted(inv.items()):
            t = n * (deg ** 3) * (q ** deg) / (1.0 - q ** deg)
            terms[deg] = t
            total += t
        # The invariants grow faster than q^d shrinks unless q is genuinely
        # small: on the quintic n_5 = 2.3e14, so at q = 0.01 the degree-five
        # term alone is 2.9e6 and the partial sum means nothing. Report the
        # size of the last term so the caller can see whether the truncation
        # is a series or a coincidence, and refuse to call it converged when
        # it plainly is not.
        last = abs(terms[max(terms)]) if terms else 0.0
        converging = last < 0.01 * abs(total) if total else False
        out.update({"quantum": total, "invariants": inv, "terms": terms,
                    "max_degree": max_degree, "q": q,
                    "last_term": last, "converging": bool(converging)})
        out["summary"] = (
            "classical %g, instanton-corrected %g at q=%g "
            "(last term %.3g, %s)"
            % (d.ravel()[0], total, q, last,
               "converging" if converging
               else "NOT converging -- reduce q or distrust this"))
        return out


@register
class LineBundleModel(Theory):
    r"""V a sum of line bundles, giving SU(5) and then the Standard Model.

    Parameters
    ----------
    X : CICY or configuration matrix
    summands : list of charge vectors
        The bundle, as :class:`pyCICY.bundles.LineBundleSum` takes it.
    action : an action from :mod:`pyCICY.equivariant`, optional
        The freely acting group, needed for the quotient spectrum.
    wilson : (p, q), optional
        The Wilson line breaking SU(5).
    """

    key = "heterotic-line-bundle"

    def __init__(self, X, summands, action=None, wilson=None, name=None):
        Theory.__init__(self, X, name=name)
        self.summands = [list(map(int, s)) for s in summands]
        self.action = action
        self.wilson = wilson

    def gauge_group(self):
        from .. import breaking
        from .. import bundles
        rank = bundles.LineBundleSum(self.X, self.summands).rank
        g = breaking.gut_group(rank)
        if self.wilson is not None and rank == 5:
            return "SU(3) x SU(2) x U(1)  (from %s by a Wilson line)" % g
        return g

    def spectrum(self):
        """Exact, from the equivariant index when an action is given."""
        from .. import breaking
        from .. import bundles

        V = bundles.LineBundleSum(self.X, self.summands)
        if self.action is None:
            return {"index": int(V.index()),
                    "generations upstairs": -int(V.index()),
                    "note": "no group action given, so this is the spectrum "
                            "on the cover"}
        r = breaking.chiral_spectrum(self.action, self.summands,
                                     wilson=self.wilson)
        out = {"%s %s" % (k[0], k[1]): v for k, v in r["spectrum"].items()}
        out["generations"] = r["generations"]
        out["anomaly"] = r["anomaly"]
        return out

    def holomorphic_yukawa(self, **kw):
        r"""Not implemented, and the reason is not the metric.

        The couplings are cup products

            H^1(X, V) x H^1(X, V) x H^1(X, Lambda^2 V) -> H^3(X, Lambda^3 V) = C

        which are quasi-topological -- they do not need the Ricci-flat metric
        -- but they do need explicit representatives of the cohomology classes,
        via a Koszul or Cech resolution. This package computes the *dimensions*
        of those groups, exactly, and not the classes themselves. So this is a
        gap in what is implemented rather than a statement about what is
        knowable, and it is worth keeping the two apart: with representatives,
        these couplings would be exact integers like the standard-embedding
        ones. The physical couplings would still need the metric.
        """
        raise NotImplementedError(
            "holomorphic Yukawa couplings for a line bundle sum are cup "
            "products H^1(V) x H^1(V) x H^1(Lambda^2 V) -> C. They are "
            "quasi-topological and would be exact, but evaluating them needs "
            "explicit cohomology representatives, which this package does not "
            "construct -- it computes dimensions. This is a missing feature, "
            "not a fundamental obstruction; contrast physical_yukawa(), which "
            "is obstructed.")

    def yukawa_texture(self, kind="both", SpaSM=False):
        """The pattern of allowed and forbidden couplings. Exact.

        Delegates to :func:`pyCICY.theories.yukawa.texture`. This is what can
        be said about the couplings without representatives: which are allowed
        by the charges, which are additionally killed by a vanishing
        cohomology group, and which survive. Not the values.
        """
        from . import yukawa
        return yukawa.texture(self.X, self.summands, kind=kind, SpaSM=SpaSM)

    def beta_coefficients(self, n_higgs_pairs=1, extra=None):
        """One-loop coefficients from this model's exact generation count.

        The generation count comes from the equivariant index and is exact;
        ``n_higgs_pairs`` does not, because the Higgs is vector-like and lives
        in the sector an index cannot see. See
        :mod:`pyCICY.theories.running`.
        """
        from . import running
        ng = self.spectrum().get("generations")
        if ng is None:
            raise ValueError("this model has no determined generation count; "
                             "supply a group action")
        return running.beta_coefficients(ng, n_higgs_pairs, extra)

    def missing_for_physical(self):
        return ["explicit cohomology representatives, to evaluate the "
                "holomorphic cup product at all (not required for the "
                "standard embedding, where it is a triple intersection "
                "number)"] + Theory.missing_for_physical(self)
