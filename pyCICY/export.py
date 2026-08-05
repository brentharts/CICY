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

import numpy as np

__all__ = [
    "ambient_vector", "coordinate_blocks", "monomials", "monomial_charges",
    "invariant_monomials", "defining_polynomials", "is_smooth_over_Fp",
    "to_cymetric", "to_cymyc", "poly_spec_source",
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


def defining_polynomials(X, action=None, seed=0, real=False, scale=1.0):
    r"""
    A defining polynomial for each hypersurface, as ``(monomials, coefficients)``.

    With ``action`` given, only monomials carrying the action's declared charge
    are used, so the resulting X is invariant and the quotient exists. Without
    it, all monomials of the right multidegree are used and X has no symmetry
    -- which is fine for a metric on X itself and wrong for a metric on
    X/Gamma.

    Coefficients are drawn from a fixed seed so that a run is reproducible;
    both target packages treat them as given data.
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
        if real:
            c = scale * rng.normal(size=len(m))
        else:
            c = scale * (rng.normal(size=len(m))
                         + 1j * rng.normal(size=len(m)))
        mons.append(m)
        coeffs.append(c)
    return mons, coeffs


def is_smooth_over_Fp(X, mons, coeffs, p=101, samples=20000, seed=0):
    r"""
    A Jacobian-criterion smoothness check over ``F_p``, by sampling.

    The variety is singular at a point of X where the Jacobian of the defining
    polynomials drops rank. Sampling points of the ambient over ``F_p``,
    keeping those on X, and testing the rank there is a one-sided test: finding
    a rank drop proves singularity, finding none over a sample proves nothing.
    Returns ``(no_singularity_found, n_points_on_X, n_rank_drops)`` and the
    docstring is the disclaimer -- this is a filter against a bad random draw,
    not a proof of smoothness.

    Exact smoothness over a finite field for a whole configuration is what
    :mod:`pyCICY.smoothness` does exhaustively; this is the cheap version for
    one explicit polynomial with coefficients reduced mod ``p``.
    """
    conf = _conf(X)
    dims = conf[:, 0]
    ncoord = int(sum(dims + 1))
    K = len(mons)
    rng = np.random.default_rng(seed)
    ic = [np.rint(np.real(c)).astype(np.int64) % p for c in coeffs]

    on_X = 0
    drops = 0
    for _ in range(samples):
        z = rng.integers(0, p, size=ncoord)
        # avoid the origin of any factor, which is not a point of P^n
        ok = True
        for (s, e) in coordinate_blocks(X):
            if not np.any(z[s:e] % p):
                ok = False
                break
        if not ok:
            continue
        vals = []
        for a in range(K):
            powers = np.ones(len(mons[a]), dtype=np.int64)
            for j in range(ncoord):
                powers = (powers * pow(int(z[j]), 1, p) ** mons[a][:, j]) % p
            vals.append(int((ic[a] * powers).sum() % p))
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
                term = ic[a][nz] * e[nz] % p
                for jj in range(ncoord):
                    ex = mons[a][nz, jj] - (1 if jj == j else 0)
                    term = (term * np.array(
                        [pow(int(z[jj]), int(x), p) for x in ex],
                        dtype=np.int64)) % p
                J[a, j] = int(term.sum() % p)
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
# the two output formats
# ---------------------------------------------------------------------------

def to_cymetric(X, action=None, kmoduli=None, seed=0, **kw):
    """Arguments for ``cymetric.pointgen.pointgen_cicy.CICYPointGenerator``.

    Returns a dict with ``monomials``, ``coefficients``, ``kmoduli`` and
    ``ambient``, ready to splat into the constructor::

        from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
        pg = CICYPointGenerator(**export.to_cymetric(X, action))
    """
    mons, coeffs = defining_polynomials(X, action=action, seed=seed, **kw)
    amb = ambient_vector(X)
    if kmoduli is None:
        kmoduli = np.ones(len(amb))
    return {"monomials": mons, "coefficients": coeffs,
            "kmoduli": np.asarray(kmoduli), "ambient": amb}


def to_cymyc(X, action=None, kmoduli=None, seed=0, **kw):
    """A ``poly_spec``-style tuple for cymyc, plus the coefficients.

    Returns ``((monomials, cy_dim, kmoduli, ambient), coefficients)``, matching
    the signature of the functions in ``cymyc/examples/poly_spec.py``.
    """
    mons, coeffs = defining_polynomials(X, action=action, seed=seed, **kw)
    conf = _conf(X)
    amb = ambient_vector(X)
    cy_dim = int(sum(conf[:, 0]) - (conf.shape[1] - 1))
    if kmoduli is None:
        kmoduli = np.ones(len(amb), dtype=np.complex64)
    return (mons, cy_dim, np.asarray(kmoduli, dtype=np.complex64), amb), coeffs


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
