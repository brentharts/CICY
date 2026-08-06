r"""
pyCICY.export -- handing verified models to the numerical-metric packages.

The gap this closes
-------------------
This package works with *configuration matrices*. It never needs the defining
polynomials: every Hodge number, index, and spectrum it computes depends only
on the degrees. That is the source of its exactness and also its ceiling --
:mod:`pyCICY.phenomenology` says plainly that physical Yukawa couplings and
mass ratios need the Ricci-flat metric, which is not a function of the
topology.

The packages that do compute those metrics need the opposite thing. Both

    cymetric   https://github.com/ruehlef/cymetric
    cymyc      https://github.com/Justin-Tan/cymyc

take an explicit polynomial: a list of monomial exponent vectors and a list of
coefficients, together with the ambient dimensions and the Kahler moduli. They
cannot start from a configuration matrix.

So the bridge is: generate a defining polynomial consistent with a given
configuration matrix, and hand it over. That sounds trivial and is not, because
of the third requirement below.

The three conditions on the polynomial
--------------------------------------
1. *Right multidegree.* The a-th polynomial must have degree ``d[i][a]`` in the
   coordinates of ambient factor i. :func:`monomials` enumerates exactly those.

2. *Smooth.* A random choice is smooth with probability one, but "probability
   one" is not a check. :func:`is_smooth_over_Fp` tests the Jacobian criterion
   over a finite field, reusing the same idea as :mod:`pyCICY.smoothness`.

3. *Equivariant, if the model needs a quotient.* This is the one that matters
   and the one a generic random polynomial fails. A heterotic model from
   :mod:`pyCICY.bundles` has three generations only after quotienting by a
   freely acting Gamma, so the metric that model needs is the metric on X/Gamma
   -- and that requires X itself to be Gamma-invariant, i.e. every defining
   polynomial to be an eigenvector of the group action. Coefficients drawn at
   random give a manifold with no symmetry at all, and a metric computed on it
   is a metric on the wrong space.

   :func:`defining_polynomials` therefore takes an action from
   :mod:`pyCICY.equivariant` and restricts the coefficients to the monomials
   carrying the declared charge. That is the same monomial-charge computation
   :meth:`pyCICY.equivariant.CyclicAction.admissible_polynomial_charges`
   already performs, used here to build rather than to check.

Output format
-------------
Both target packages want essentially the same tuple, which is convenient:

    cymetric   CICYPointGenerator(monomials, coefficients, kmoduli, ambient)
    cymyc      (monomials, cy_dim, kmoduli, ambient) plus coefficients

with ``monomials`` a list of ``(nMonomials, ncoords)`` integer arrays, one per
defining equation, and coordinates ordered factor by factor -- factor i
contributing ``n_i + 1`` of them. :func:`to_cymetric` and :func:`to_cymyc`
produce each, and :func:`poly_spec_source` emits a ``poly_spec.py``-style
function that can be pasted into cymyc's examples directory.

What this does not do
---------------------
It does not compute a metric, and it does not check that the *quotient* is
smooth or that Gamma acts freely on the particular X produced -- freeness is
diagnosed on the degrees by :mod:`pyCICY.equivariant`, which is necessary and
not sufficient. What it produces is an input file, correct by construction on
points 1 and 3 and checked on point 2.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ambient_vector", "coordinate_blocks", "monomials", "monomial_charges",
    "invariant_monomials", "defining_polynomials", "is_smooth_over_Fp",
    "kmoduli_from_stability", "kahler_volume",
    "omega_character", "preserves_holomorphic_form", "group_matrices",
    "kmoduli_are_invariant", "quotient_report",
    "to_cymetric", "to_cymyc", "poly_spec_source",
    "metric_backend_support",
]


def _conf(X):
    from .pyCICY import CICY
    if isinstance(X, CICY):
        return np.asarray(X.M, dtype=int)
    return np.asarray(X, dtype=int)


def ambient_vector(X):
    """The ambient dimensions ``[n_1, ..., n_m]``, in the order both packages want."""
    return _conf(X)[:, 0].astype(np.int64)


def coordinate_blocks(X):
    """Index ranges of each ambient factor in the concatenated coordinates.

    Factor i occupies ``n_i + 1`` consecutive coordinates. Both target
    packages use this ordering -- cymetric builds the same thing as
    ``coord_to_ambient`` -- so it is the one convention that must not drift.
    """
    dims = ambient_vector(X)
    out = []
    start = 0
    for n in dims:
        out.append((start, start + int(n) + 1))
        start += int(n) + 1
    return out


def monomials(X, a):
    """Every monomial of the multidegree of the ``a``-th defining polynomial.

    Returned as an integer array of shape ``(nMonomials, ncoords)`` holding the
    exponents, which is exactly the format both packages read.
    """
    conf = _conf(X)
    dims = conf[:, 0]
    deg = conf[:, 1:].T[a]
    blocks = []
    for i, n in enumerate(dims):
        rows = []
        for combo in itertools.combinations_with_replacement(
                range(int(n) + 1), int(deg[i])):
            e = [0] * (int(n) + 1)
            for j in combo:
                e[j] += 1
            rows.append(e)
        blocks.append(rows)
    out = []
    for pick in itertools.product(*blocks):
        out.append([x for row in pick for x in row])
    return np.array(out, dtype=np.int64)


def monomial_charges(X, a, action):
    """The Gamma-charge of every monomial of the ``a``-th polynomial.

    A monomial with exponent vector ``e`` picks up ``sum_j e_j w_j`` where the
    ``w_j`` are the weights of the coordinates. Only defined for actions that
    do not permute the ambient factors, since a permuting action does not act
    diagonally on monomials.
    """
    mon = monomials(X, a)
    weights = []
    for w in action.weights:
        weights.extend(list(w))
    weights = np.array(weights, dtype=np.int64)
    if len(weights) != mon.shape[1]:
        raise ValueError(
            "the action has %d coordinate weights but the configuration has "
            "%d coordinates" % (len(weights), mon.shape[1]))
    n = getattr(action, "n", getattr(action, "N", None))
    if n is None:
        raise ValueError("the action does not expose a modulus")
    return (mon @ weights) % n


def invariant_monomials(X, a, action, charge=None):
    """Monomials of the ``a``-th polynomial carrying a given Gamma-charge.

    With ``charge=None`` the action's own declared
    ``polynomial_charges[a]`` is used, which is the consistent choice: it is
    the charge :mod:`pyCICY.equivariant` assumed when it computed the index.

    Returns ``(monomials, mask)``.
    """
    if charge is None:
        pc = action.polynomial_charges
        charge = int(pc[a] if not isinstance(pc[0], (list, tuple))
                     else pc[0][a])
    ch = monomial_charges(X, a, action)
    mask = (ch == (int(charge) % len(set(range(max(2, int(ch.max()) + 1))))
                   if False else int(charge) % _modulus(action)))
    return monomials(X, a)[mask], mask


def _modulus(action):
    return int(getattr(action, "n", getattr(action, "N", 1)))


def defining_polynomials(X, action=None, seed=0, real=False, scale=1.0,
                         check_smooth=True, p=101, samples=8000,
                         integer_coefficients=True, spread=12):
    r"""
    A defining polynomial for each hypersurface, as ``(monomials, coefficients)``.

    With ``action`` given, only monomials carrying the action's declared charge
    are used, so the resulting X is invariant and the quotient exists. Without
    it, all monomials of the right multidegree are used and X has no symmetry
    -- which is fine for a metric on X itself and wrong for a metric on
    X/Gamma.

    Coefficients are drawn from a fixed seed so that a run is reproducible;
    both target packages treat them as given data.

    Coefficients are Gaussian integers by default. That is not cosmetic: the
    smoothness check reduces them modulo a prime, and reducing a float by
    rounding its real part discards the imaginary part and collapses most
    coefficients to 0 or +-1, so the check would be testing a different and far
    more degenerate polynomial than the one exported. With integer
    coefficients the reduction is exact. Pass
    ``integer_coefficients=False`` for floats, in which case the smoothness
    check refuses to run rather than run on the wrong thing.

    With ``check_smooth`` the result is passed through
    :func:`is_smooth_over_Fp` and a detected singularity raises. That test is
    one-sided -- it can prove singularity and never smoothness -- so this is a
    filter against a bad draw rather than a guarantee, and it is cheap. The
    default sample count is set so that a hypersurface over F_101 yields a few
    dozen points to test on, which is enough to catch the degenerate draws;
    raise ``samples`` for more confidence, at linear cost. It is on by default because a singular
    X reaching a metric package produces a training run that looks fine and
    converges to nothing in particular.
    """
    conf = _conf(X)
    K = conf.shape[1] - 1
    rng = np.random.default_rng(seed)
    mons, coeffs = [], []
    for a in range(K):
        if action is None:
            m = monomials(X, a)
        else:
            m, _ = invariant_monomials(X, a, action)
            if len(m) == 0:
                raise ValueError(
                    "no monomial of the multidegree of polynomial %d carries "
                    "the declared charge, so no invariant hypersurface of this "
                    "shape exists. Check admissible_polynomial_charges()." % a)
        if integer_coefficients:
            # Gaussian integers, so that reduction mod p is exact and the
            # smoothness check tests the polynomial actually being exported.
            # Zero coefficients are avoided: a monomial with coefficient zero
            # is a monomial that is not there, which changes the variety.
            lo = rng.integers(1, spread + 1, size=len(m))
            sg = rng.choice([-1, 1], size=len(m))
            re = lo * sg
            if real:
                c = scale * re.astype(float)
            else:
                lo2 = rng.integers(1, spread + 1, size=len(m))
                sg2 = rng.choice([-1, 1], size=len(m))
                c = scale * (re + 1j * lo2 * sg2).astype(complex)
        elif real:
            c = scale * rng.normal(size=len(m))
        else:
            c = scale * (rng.normal(size=len(m))
                         + 1j * rng.normal(size=len(m)))
        mons.append(m)
        coeffs.append(c)
    if check_smooth:
        ok, on_X, drops = is_smooth_over_Fp(X, mons, coeffs, p=p,
                                            samples=samples, seed=seed)
        if not ok:
            raise ValueError(
                "this defining polynomial is singular: the Jacobian drops "
                "rank at %d of the %d sampled points of X over F_%d. Try "
                "another seed. (The test is one-sided: it can prove "
                "singularity, never smoothness.)" % (drops, on_X, p))
        if on_X == 0:
            logger.warning(
                "the smoothness check found no points of X over F_%d in %d "
                "samples, so it checked nothing. Raise samples or lower p.",
                p, samples)
    return mons, coeffs


def is_smooth_over_Fp(X, mons, coeffs, p=101, samples=20000, seed=0):
    r"""
    A Jacobian-criterion smoothness check over ``F_p``, by sampling.

    The variety is singular where the Jacobian of the defining polynomials
    drops rank on X. Sampling points of the ambient over ``F_p``, keeping those
    on X, and testing the rank there is **one-sided**: finding a rank drop
    proves singularity, finding none proves nothing. Returns
    ``(no_singularity_found, n_points_on_X, n_rank_drops)``.

    Coefficients must be Gaussian integers, and ``p`` must be congruent to 1
    modulo 4 so that ``F_p`` contains a square root of ``-1`` and the reduction
    ``a + b i -> a + b i_p`` is exact. An earlier version reduced floats by
    rounding the real part, which threw the imaginary part away and turned most
    coefficients into 0 or +-1 -- it was testing a much sparser polynomial than
    the one exported, and reported a false singularity on the very first seed
    it was pointed at. Faithfulness of the reduction is the whole content of
    this function being worth running.
    """
    conf = _conf(X)
    dims = conf[:, 0]
    ncoord = int(sum(dims + 1))
    K = len(mons)
    if p % 4 != 1:
        raise ValueError(
            "p must be 1 mod 4 so that F_p contains a square root of -1 and "
            "Gaussian integer coefficients reduce exactly; got p = %d" % p)
    i_p = next(x for x in range(p) if (x * x) % p == p - 1)

    ic = []
    for c in coeffs:
        c = np.asarray(c)
        re, im = np.real(c), np.imag(c)
        if (np.max(np.abs(re - np.rint(re))) > 1e-9
                or np.max(np.abs(im - np.rint(im))) > 1e-9):
            raise ValueError(
                "the coefficients are not Gaussian integers, so reducing them "
                "modulo %d would not be faithful and this check would test a "
                "different polynomial. Use integer_coefficients=True, or pass "
                "check_smooth=False and accept that smoothness is unverified."
                % p)
        ic.append((np.rint(re).astype(np.int64)
                   + i_p * np.rint(im).astype(np.int64)) % p)

    rng = np.random.default_rng(seed)
    blocks = coordinate_blocks(X)
    maxdeg = int(max(m.max() for m in mons)) + 1
    on_X = 0
    drops = 0
    for _ in range(samples):
        z = rng.integers(0, p, size=ncoord)
        if any(not np.any(z[s0:e0] % p) for (s0, e0) in blocks):
            continue
        # Powers table: pw[j, e] = z_j^e mod p, for e up to the largest
        # exponent that occurs. Degrees are small, so this replaces a Python
        # pow() per (monomial, coordinate) with an array lookup and takes the
        # check from ~90s to a couple of seconds over the suite.
        pw = np.ones((ncoord, maxdeg + 1), dtype=np.int64)
        for e in range(1, maxdeg + 1):
            pw[:, e] = (pw[:, e - 1] * z) % p
        idx = np.arange(ncoord)

        vals = []
        for a in range(K):
            terms = pw[idx[None, :], mons[a]]
            prod = np.ones(len(mons[a]), dtype=np.int64)
            for j in range(ncoord):
                prod = (prod * terms[:, j]) % p
            vals.append(int((ic[a] * prod).sum() % p))
        if any(v != 0 for v in vals):
            continue
        on_X += 1
        J = np.zeros((K, ncoord), dtype=np.int64)
        for a in range(K):
            for j in range(ncoord):
                e = mons[a][:, j]
                nz = e > 0
                if not np.any(nz):
                    continue
                sub = mons[a][nz].copy()
                coef = (ic[a][nz] * e[nz]) % p
                sub[:, j] -= 1
                terms = pw[idx[None, :], sub]
                prod = coef
                for jj in range(ncoord):
                    prod = (prod * terms[:, jj]) % p
                J[a, j] = int(prod.sum() % p)
        if _rank_mod_p(J, p) < K:
            drops += 1
    return (drops == 0), on_X, drops


def _rank_mod_p(M, p):
    A = np.array(M, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c] % p:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        for i in range(rows):
            if i != r and A[i, c] % p:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
        if r == rows:
            break
    return r



# ---------------------------------------------------------------------------
# Kahler moduli
# ---------------------------------------------------------------------------

def kahler_volume(X, t):
    r"""``kappa(t) = d_rst t^r t^s t^t / 6``, the volume of X at the Kahler point ``t``.

    Note the factor. cymetric's ``get_volume_from_intersections`` returns
    ``int_X J^3 = d_rst t^r t^s t^t`` without the ``1/3!``, so the two differ
    by exactly six. They agree to the last bit otherwise, on every manifold
    tested, which is a genuine cross-check: our intersection numbers come from
    the Leray spectral sequence on the configuration matrix and theirs from
    an independent computation on the same ambient data.
    """
    from .pyCICY import CICY
    XX = X if isinstance(X, CICY) else CICY(_conf(X).tolist())
    d = np.asarray(XX.triple_intersection(), dtype=float)
    t = np.asarray(t, dtype=float)
    return float(np.einsum("rst,r,s,t->", d, t, t, t) / 6.0)


def kmoduli_from_stability(X, summands, normalisation="volume", target=1.0,
                           tries=40, tol=1e-7):
    r"""
    The Kahler moduli at which the given bundle is poly-stable.

    A sum of line bundles is poly-stable exactly where every slope
    ``mu(L_a) = d_rst L_a^r t^s t^t`` vanishes, and
    :func:`pyCICY.bundles.stability_locus` finds that point. It is the
    physically correct Kahler point for the model: at any other point in the
    cone the bundle is not stable, the Hermitian-Yang-Mills equation has no
    solution, and the four-dimensional theory is not supersymmetric. Handing a
    metric package ``kmoduli = ones`` instead asks it for a metric at a point
    in moduli space where the model does not exist -- silently, since nothing
    downstream knows what the bundle was.

    **The slope condition fixes a ray, not a point.** Every slope is homogeneous
    of degree two in ``t``, so if ``t`` is a solution then so is every positive
    multiple. The *direction* is determined by the bundle; the overall scale is
    the volume modulus and is a genuine free parameter that stability says
    nothing about. So this function returns a direction and a scale you chose,
    and ``normalisation`` names the choice:

    ``'volume'``   scale so that ``kappa(t) = target`` (default 1)
    ``'sum'``      scale so that ``sum_r t_r`` equals the number of moduli,
                   which puts it on the same footing as ``ones``
    ``'max'``      scale so that the largest modulus is ``target``
    ``'raw'``      whatever the optimiser returned, normalised to unit norm

    Returns a dict with ``kmoduli``, the ``volume`` there, the ``slopes`` (all
    of which should vanish), and the ``residual`` from the search.

    Raises when the bundle is not poly-stable anywhere in the cone, rather than
    falling back to a default -- a bundle with no stability locus has no
    correct Kahler point, and returning one would be inventing it.
    """
    from . import bundles as _bundles

    loc = _bundles.stability_locus(X, summands, tries=tries)
    if not loc["found"]:
        raise ValueError(
            "this bundle is not poly-stable anywhere in the Kahler cone (%s), "
            "so there is no Kahler point at which the model exists and no "
            "correct kmoduli to return"
            % loc.get("reason", "no common zero of the slopes was found"))
    t = np.asarray(loc["t"], dtype=float)
    if np.any(t <= 0):
        raise ValueError(
            "the stability point sits on the boundary of the Kahler cone, "
            "where the manifold degenerates: t = %s" % t.tolist())

    if normalisation == "volume":
        v = kahler_volume(X, t)
        if v <= 0:
            raise ValueError(
                "the volume at the stability point is %g, which is not "
                "positive; the point is not in the Kahler cone as this basis "
                "assumes" % v)
        t = t * (float(target) / v) ** (1.0 / 3.0)
    elif normalisation == "sum":
        t = t * (len(t) / t.sum())
    elif normalisation == "max":
        t = t * (float(target) / t.max())
    elif normalisation == "raw":
        t = t / np.linalg.norm(t)
    else:
        raise ValueError("normalisation must be 'volume', 'sum', 'max' or "
                         "'raw', got %r" % (normalisation,))

    V = _bundles.LineBundleSum(X, summands)
    slopes = np.asarray(V.slopes(t), dtype=float)
    if np.max(np.abs(slopes)) > tol * max(1.0, float(np.abs(slopes).max() + 1)):
        # rescaling is exact, so this should never fire; it is here because a
        # silent failure would put the metric at the wrong point.
        raise ArithmeticError(
            "the slopes do not vanish after rescaling: %s" % slopes.tolist())

    return {"kmoduli": t, "volume": kahler_volume(X, t),
            "slopes": slopes, "residual": loc["residual"],
            "normalisation": normalisation}



# ---------------------------------------------------------------------------
# exporting the group itself
# ---------------------------------------------------------------------------

def omega_character(X, action):
    r"""
    The character by which Gamma acts on the holomorphic (3,0)-form.

    Omega is built from the holomorphic volume form of the ambient divided by
    the Jacobian of the defining polynomials, so under a diagonal action it
    picks up

        chi(Omega) = sum over all coordinates of the weights
                     - sum over the defining polynomials of their charges,

    modulo the order. Returned as an integer in ``[0, n)``.
    """
    n = _modulus(action)
    w = sum(sum(row) for row in action.weights)
    pc = action.polynomial_charges
    if pc and isinstance(pc[0], (list, tuple)):
        pc = pc[0]
    return int((w - sum(pc)) % n)


def preserves_holomorphic_form(X, action):
    r"""
    Whether X/Gamma is Calabi-Yau at all.

    A quotient of a Calabi-Yau by a freely acting group is again Calabi-Yau
    only if the group preserves the holomorphic form -- equivalently sits in
    SU(3) rather than merely U(3). If it does not, Omega does not descend,
    the quotient has no covariantly constant spinor, and asking a metric
    package for "the Ricci-flat metric on X/Gamma" is asking for something
    that does not exist.

    This is **independent** of freeness and neither implies the other. On the
    tetraquadric, the order-4 factor-permuting action preserves Omega and is
    not free; the Z_3 diagonal action is neither. The classic free Z_5 on the
    quintic, ``x_i -> zeta^i x_i``, satisfies both -- weights summing to 10,
    which vanishes mod 5, against a charge-0 quintic.
    """
    return omega_character(X, action) == 0


def group_matrices(action):
    r"""
    Every group element as an explicit matrix on the homogeneous coordinates.

    Returned as a list of complex ``(ncoords, ncoords)`` arrays, in the
    coordinate ordering of :func:`coordinate_blocks`, so that ``M @ z`` is the
    action on a point. Diagonal for actions within factors, a permutation with
    phases when factors are exchanged.

    This is what a metric package needs in order to work equivariantly: it
    never has to know anything about configuration matrices or characters, only
    how to move a point.
    """
    n = _modulus(action)
    conf = np.asarray(action.conf, dtype=int)
    dims = conf[:, 0]
    ncoord = int(sum(dims + 1))
    starts = []
    acc = 0
    for d in dims:
        starts.append(acc)
        acc += int(d) + 1
    zeta = np.exp(2j * np.pi / n)

    perms_weights = []
    if hasattr(action, "perm"):                      # PermutationAction
        from .equivariant import _permutation_power
        for j in range(n):
            sj = _permutation_power(list(action.perm), j)
            w = []
            for i in range(len(dims)):
                u = [0] * (int(dims[i]) + 1)
                x = i
                for _ in range(j):
                    for t in range(int(dims[i]) + 1):
                        u[t] += action.weights[x][t]
                    x = action.perm[x]
                w.append(u)
            perms_weights.append((sj, w))
    else:                                            # CyclicAction
        for j in range(n):
            perms_weights.append(
                (list(range(len(dims))),
                 [[j * t for t in row] for row in action.weights]))

    out = []
    for perm, w in perms_weights:
        M = np.zeros((ncoord, ncoord), dtype=complex)
        for i in range(len(dims)):
            for t in range(int(dims[i]) + 1):
                src = starts[i] + t
                dst = starts[perm[i]] + t
                M[dst, src] = zeta ** (w[i][t] % n)
        out.append(M)
    return out


def kmoduli_are_invariant(action, kmoduli, tol=1e-9):
    """Whether the Kahler class is fixed by the group.

    A permuting action moves the Kahler parameters around with the factors, so
    unless ``t`` is constant on each orbit the class is not invariant, does not
    descend, and the quotient metric is not the one being asked for. Trivially
    true for an action that does not permute factors.
    """
    t = np.asarray(kmoduli, dtype=float)
    perm = list(getattr(action, "perm", range(len(t))))
    return bool(np.all(np.abs(t[perm] - t) < tol))


def quotient_report(X, action, kmoduli=None, summands=None):
    """The conditions for X/Gamma to be a Calabi-Yau the model can live on.

    Returns a dict of independent checks. None implies the others, and a
    metric computed when any of them fails is a metric on the wrong space:

    ``preserves_omega``   Gamma in SU(3), so Omega descends and the quotient
                          is Calabi-Yau
    ``looks_free``        no fixed points, so the quotient is smooth
    ``kmoduli_invariant`` the Kahler class descends
    """
    out = {"order": _modulus(action),
           "omega_character": omega_character(X, action),
           "preserves_omega": preserves_holomorphic_form(X, action)}
    try:
        out["looks_free"] = bool(action.looks_free()[0])
    except Exception:                                            # noqa: BLE001
        out["looks_free"] = None
    if kmoduli is None and summands is not None:
        kmoduli = kmoduli_from_stability(X, summands)["kmoduli"]
    if kmoduli is not None:
        out["kmoduli"] = np.asarray(kmoduli)
        out["kmoduli_invariant"] = kmoduli_are_invariant(action, kmoduli)
    out["ok"] = bool(out["preserves_omega"]
                     and (out["looks_free"] is not False)
                     and out.get("kmoduli_invariant", True))
    return out


# ---------------------------------------------------------------------------
# the two output formats
# ---------------------------------------------------------------------------

def _resolve_kmoduli(X, kmoduli, summands, normalisation, n):
    """Turn the ``kmoduli`` argument into an array, or explain why it cannot."""
    if isinstance(kmoduli, str):
        if kmoduli != "stability":
            raise ValueError("the only named kmoduli option is 'stability'")
        if summands is None:
            raise ValueError(
                "kmoduli='stability' needs the bundle: pass summands=[...]. "
                "The Kahler point is a property of the bundle, not of the "
                "manifold alone")
        return kmoduli_from_stability(X, summands,
                                      normalisation=normalisation)["kmoduli"]
    if kmoduli is None:
        return np.ones(n)
    return np.asarray(kmoduli)


def to_cymetric(X, action=None, kmoduli=None, seed=0, summands=None,
                normalisation="volume", include_group=False, **kw):
    """Arguments for ``cymetric.pointgen.pointgen_cicy.CICYPointGenerator``.

    Returns a dict with ``monomials``, ``coefficients``, ``kmoduli`` and
    ``ambient``, ready to splat into the constructor::

        from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
        pg = CICYPointGenerator(**export.to_cymetric(X, action))

    With ``include_group`` the result also carries ``group_matrices``, which
    the stock ``CICYPointGenerator`` does not accept -- it is for the
    equivariant subclass, and is left out by default so that the dict can be
    splatted straight into the upstream constructor.
    """
    mons, coeffs = defining_polynomials(X, action=action, seed=seed, **kw)
    amb = ambient_vector(X)
    km = _resolve_kmoduli(X, kmoduli, summands, normalisation, len(amb))
    out = {"monomials": mons, "coefficients": coeffs,
           "kmoduli": km, "ambient": amb}
    if action is not None and include_group:
        out["group_matrices"] = group_matrices(action)
    return out


def to_cymyc(X, action=None, kmoduli=None, seed=0, summands=None,
             normalisation="volume", **kw):
    """A ``poly_spec``-style tuple for cymyc, plus the coefficients.

    Returns ``((monomials, cy_dim, kmoduli, ambient), coefficients)``, matching
    the signature of the functions in ``cymyc/examples/poly_spec.py``.
    """
    mons, coeffs = defining_polynomials(X, action=action, seed=seed, **kw)
    conf = _conf(X)
    amb = ambient_vector(X)
    cy_dim = int(sum(conf[:, 0]) - (conf.shape[1] - 1))
    km = _resolve_kmoduli(X, kmoduli, summands, normalisation, len(amb))
    return (mons, cy_dim, np.asarray(km, dtype=np.complex64), amb), coeffs


def poly_spec_source(X, action=None, name="pycicy_spec", seed=0, **kw):
    """Source text of a cymyc ``poly_spec`` function, ready to paste.

    cymyc's examples are literal Python files holding monomial arrays, so the
    most useful handover is the file rather than the objects.
    """
    (mons, cy_dim, kmoduli, amb), coeffs = to_cymyc(
        X, action=action, seed=seed, **kw)
    lines = ["import numpy as np", "", "", "def %s():" % name,
             '    """Generated by pyCICY.export from the configuration %s."""'
             % _conf(X).tolist()]
    names = []
    for a, m in enumerate(mons):
        nm = "monomials_%d" % (a + 1)
        names.append(nm)
        lines.append("    %s = np.asarray([" % nm)
        for row in m:
            lines.append("        %s," % list(map(int, row)))
        lines.append("    ], dtype=np.int64)")
    lines.append("    monomials = [%s]" % ", ".join(names))
    lines.append("    cy_dim = %d" % cy_dim)
    lines.append("    kmoduli = np.ones(%d, dtype=np.complex64)" % len(amb))
    lines.append("    ambient = np.array(%s)" % list(map(int, amb)))
    lines.append("    return monomials, cy_dim, kmoduli, ambient")
    lines.append("")
    lines.append("")
    lines.append("def %s_coefficients():" % name)
    lines.append("    return [")
    for c in coeffs:
        lines.append("        np.array(%s, dtype=np.complex128),"
                     % [complex(round(x.real, 6), round(x.imag, 6))
                        if np.iscomplexobj(c) else float(round(x, 6))
                        for x in c])
    lines.append("    ]")
    return "\n".join(lines) + "\n"


def metric_backend_support(X, backend="jax"):
    r"""
    Whether a metric package can actually train on this configuration.

    The export is correct for any CICY, but the downstream packages are not
    uniformly able to consume it, and it is better to say so here than to let
    a user hit a framework traceback after building a dataset.

    **cymetric's JAX backend cannot train a CICY with more than one defining
    polynomial.** The transition loss needs the patch transitions of each
    point, and for ``nhyper = 1`` those are precomputed once at construction,
    as a static array. For ``nhyper > 1`` they are generated per point at run
    time from data-dependent indices, by a routine whose own comment says it
    "runs eagerly" -- and eager numpy on traced values is exactly what ``jit``
    forbids. The failure is a ``TracerArrayConversionError`` deep inside
    ``fubinistudy._generate_patches``, and it fires whatever weight the
    transition loss is given, because the loss is computed before the weight
    is applied.

    Two contributing bugs in the same file are genuinely one-line fixes and
    are worth reporting separately: ``self.degrees`` and ``self._proj_indices``
    are stored as ``jnp`` arrays and then passed to ``np.array`` inside jitted
    code, although the class already keeps a static ``_degrees_list`` for
    exactly this purpose. Fixing those is necessary and not sufficient.

    So a model on a hypersurface (``nhyper = 1``) can be handed to the JAX
    backend today; a complete intersection cannot, and this is a limitation of
    the backend rather than of the export. The TensorFlow and PyTorch backends
    are untested here.

    Returns a dict with ``supported``, ``nhyper`` and the reason.
    """
    conf = _conf(X)
    nhyper = conf.shape[1] - 1
    if backend != "jax":
        return {"supported": None, "nhyper": nhyper,
                "reason": "only the JAX backend has been tested from here"}
    if nhyper == 1:
        return {"supported": True, "nhyper": nhyper,
                "reason": "a single defining polynomial: patch transitions are "
                          "precomputed statically and jit is happy"}
    return {"supported": False, "nhyper": nhyper,
            "reason": "cymetric's JAX backend generates patch transitions per "
                      "point for nhyper > 1, eagerly, which fails under jit "
                      "with a TracerArrayConversionError. The export is fine; "
                      "the backend cannot consume it."}
