r"""
pyCICY.cicylist -- the CICY threefold list, and systematic exploration of it
by splitting.

Background
----------
The classification of complete intersection Calabi-Yau threefolds is due to

    P. Candelas, A. M. Dale, C. A. Lutken, R. Schimmrigk,
    "Complete Intersection Calabi-Yau Manifolds",
    Nucl. Phys. B298 (1988) 493.

It contains 7890 configuration matrices. That classification was built by
*splitting*: starting from the small number of configurations with a single
defining equation and repeatedly applying the P^1 split, then discarding
duplicates. :func:`split_web` reproduces that construction, and records the
conifold transition data for every edge it generates.

Splits are not merely a bookkeeping device. A P^1 split is the resolution
side of a conifold transition, and across it

    h^{1,1}(X_R) = h^{1,1}(X_D) + 1
    chi(X_R)     = chi(X_D) + 2N

with N the number of nodes of the shared singular variety. See section 1.1 of

    L. B. Anderson, J. Gray, S. A. Patil, C. Scanlon,
    "Mapping moduli across heterotic conifolds", arXiv:2512.18124.

That paper's central example, the quintic and its P^1 split, appears here as
the edge from ``[[4, 5]]`` to ``[[1,1,1],[4,1,4]]`` with N = 16.

Normal forms and equivalence of configurations follow the CINormal / CIEquiv
functionality of

    L. B. Anderson, A. Constantin, J. Gray, Y.-H. He, S.-J. Lee, A. Lukas,
    "CIPro Package: Complete Intersections in Products of Projective Spaces
    and Line Bundles", arXiv:2606.27588, section 2.4.

Note on the published list
--------------------------
The 7890-entry list is distributed as a data file by its authors and is not
bundled here. :func:`load_list` will read it if you have a copy;
:func:`split_web` does not need it, since it generates configurations itself.
Where a count below refers to the published list it is quoted from the
literature, not computed here.
"""

import itertools as it
import os
import time

from . import transitions as T
from .cache import default_cache, hodge

__all__ = [
    "SEEDS", "split_web", "load_list", "contract_to_seed", "survey",
    "web_edges", "web_nodes", "check_list",
    "load_published_list", "compare_to_published", "published_coverage",
]


# The five Calabi-Yau threefolds cut out by a single equation. For K = 1 the
# threefold condition sum_r n_r - K = 3 forces sum_r n_r = 4, and the
# Calabi-Yau condition then fixes the single column to q^r = n_r + 1. These
# are the roots of the splitting construction of Candelas et al.
SEEDS = [
    [[4, 5]],                                  # the quintic in P^4
    [[3, 4], [1, 2]],                          # P^3 x P^1
    [[2, 3], [2, 3]],                          # P^2 x P^2, the bicubic
    [[2, 3], [1, 2], [1, 2]],                  # P^2 x P^1 x P^1
    [[1, 2], [1, 2], [1, 2], [1, 2]],          # the tetraquadric
]


def _shape_ok(conf, max_rows, max_cols):
    return len(conf) <= max_rows and (len(conf[0]) - 1) <= max_cols


def split_web(seeds=None, max_depth=2, max_configs=2000, max_rows=8,
              max_cols=10, cache=None, include_ineffective=True,
              with_topology=True, progress=None):
    """Generate configurations by repeatedly splitting, recording the edges.

    Breadth-first from ``seeds``. Every configuration reached is reduced to
    its normal form before being recorded, so relabellings collapse to one
    node, exactly as duplicates were removed when the published list was
    compiled.

    Parameters
    ----------
    seeds : list of configurations or None
        Starting points; :data:`SEEDS` by default.
    max_depth : int
        How many successive splits to apply.
    max_configs : int
        Stop once this many distinct configurations are known. Splitting
        grows fast, so this is a hard budget rather than a hint.
    max_rows, max_cols : int
        Discard configurations whose matrix exceeds these dimensions. The
        published threefold list tops out at 12 x 15.
    cache : Cache or None
        Where to memoise the Hodge computations.
    include_ineffective : bool
        Whether to keep splits with N = 0. These are the ineffective splits:
        they add no nodes and change no topology, giving another description
        of the same manifold. They are what makes many unfavourable
        configurations favourable, so they are kept by default.
    with_topology : bool
        Compute Hodge data for each configuration. Turning this off makes
        the enumeration purely combinatorial and much faster.
    progress : callable or None
        Called as ``progress(n_known, n_frontier, depth)`` after each depth.

    Returns
    -------
    dict with keys
        nodes : dict
            normal-form key -> record with the configuration, depth, and
            (if requested) Hodge data.
        edges : list
            One record per split performed, with parent and child keys, the
            column and partition used, the node count N, and whether the
            split was effective.
        stats : dict
            Counts and timing, including cache behaviour.
    """
    seeds = SEEDS if seeds is None else seeds
    cache = cache if cache is not None else default_cache()
    started = time.time()

    nodes = {}
    edges = []
    truncated = False

    def record(conf, depth):
        key, rperm, cperm = T.normal_form(conf)
        canonical = _apply(conf, rperm, cperm)
        if key in nodes:
            return key, False
        rec = {"conf": canonical, "depth": depth}
        if with_topology:
            rec.update(hodge(canonical, cache=cache))
        nodes[key] = rec
        return key, True

    frontier = []
    for s in seeds:
        k, _ = record(s, 0)
        frontier.append((k, nodes[k]["conf"]))

    for depth in range(1, max_depth + 1):
        nxt = []
        for parent_key, parent_conf in frontier:
            for col, part, child in T.splits(parent_conf):
                if not _shape_ok(child, max_rows, max_cols):
                    continue
                if len(nodes) >= max_configs:
                    truncated = True
                    break

                n_nodes = T.nodes_expected(parent_conf, col, part)
                effective = n_nodes > 0
                if not effective and not include_ineffective:
                    continue

                child_key, is_new = record(child, depth)
                edges.append({
                    "parent": parent_key,
                    "child": child_key,
                    "column": col,
                    "partition": list(part),
                    "nodes": int(n_nodes),
                    "effective": bool(effective),
                    "new": bool(is_new),
                })
                if is_new:
                    nxt.append((child_key, nodes[child_key]["conf"]))
            if truncated:
                break
        if progress is not None:
            progress(len(nodes), len(nxt), depth)
        frontier = nxt
        if truncated or not frontier:
            break

    stats = {
        "configurations": len(nodes),
        "edges": len(edges),
        "effective_edges": sum(1 for e in edges if e["effective"]),
        "ineffective_edges": sum(1 for e in edges if not e["effective"]),
        "truncated": truncated,
        "seconds": time.time() - started,
        "cache": cache.stats(),
    }
    return {"nodes": nodes, "edges": edges, "stats": stats}


def _apply(conf, rperm, cperm):
    """Rewrite a configuration under the given row/column permutation."""
    rows = []
    for i in rperm:
        row = conf[i]
        rows.append([int(row[0])] + [int(row[1 + c]) for c in cperm])
    return rows


def web_nodes(web):
    """The configurations of a web as a list of records."""
    return list(web["nodes"].values())


def web_edges(web):
    """The split edges of a web."""
    return web["edges"]


def contract_to_seed(conf, max_steps=50):
    """Contract a configuration repeatedly until no P^1 split remains.

    The inverse of the splitting construction: this walks back down towards
    the single-equation configurations the list was built from. Returns the
    chain of configurations, starting with the input.
    """
    chain = [[list(map(int, r)) for r in conf]]
    for _ in range(max_steps):
        try:
            nxt = T.contract(chain[-1])
        except ValueError:
            break
        chain.append(nxt)
    return chain


def survey(web):
    """Aggregate a web into counts that are easy to plot or tabulate."""
    recs = [r for r in web_nodes(web) if r.get("error") is None]
    usable = [r for r in recs if r.get("h11") is not None]

    hodge_pairs = {}
    for r in usable:
        pair = (r["h11"], r["h21"])
        hodge_pairs[pair] = hodge_pairs.get(pair, 0) + 1

    by_depth = {}
    for r in recs:
        by_depth[r["depth"]] = by_depth.get(r["depth"], 0) + 1

    node_counts = {}
    for e in web["edges"]:
        node_counts[e["nodes"]] = node_counts.get(e["nodes"], 0) + 1

    return {
        "configurations": len(recs),
        "with_hodge": len(usable),
        "failed": len(web_nodes(web)) - len(recs),
        "distinct_hodge_pairs": len(hodge_pairs),
        "hodge_pairs": hodge_pairs,
        "favourable": sum(1 for r in usable if r.get("favourable")),
        "by_depth": by_depth,
        "euler_range": (min((r["euler"] for r in usable), default=None),
                        max((r["euler"] for r in usable), default=None)),
        "h11_range": (min((r["h11"] for r in usable), default=None),
                      max((r["h11"] for r in usable), default=None)),
        "h21_range": (min((r["h21"] for r in usable), default=None),
                      max((r["h21"] for r in usable), default=None)),
        "node_counts": node_counts,
    }


# ------------------------------------------------------------- list loading

def load_published_list(path):
    """Load the published CICY three-fold list produced by the fetch script.

    ``scripts/fetch_cicy_list.py`` downloads the list from

        https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/

    and writes ``data/cicylist.json``. The data is redistributed by its
    authors and is not bundled with pyCICY, so this raises a helpful error
    rather than a bare FileNotFoundError when the file is absent.

    Returns a list of records with keys ``num``, ``conf``, ``h11``, ``h21``,
    ``euler``, ``c2``, where ``conf`` is in pyCICY form.
    """
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found. The published CICY list is distributed by its "
            "authors and is not bundled with pyCICY; run\n"
            "    python3 scripts/fetch_cicy_list.py\n"
            "to download and convert it." % path)
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload["entries"]
    return payload


class _EntryTimeout(BaseException):
    """Raised when an entry exceeds its wall-clock budget.

    Deliberately derived from BaseException, not Exception: the Hodge
    computation wraps failures in a broad ``except Exception`` and caches
    them as permanent errors, so a timeout derived from Exception would be
    swallowed and recorded as if the entry were unevaluable. It would then
    never be retried, even with a larger budget.
    """


def _time_limited(seconds, fn):
    """Run ``fn`` under a wall-clock limit, raising _EntryTimeout if exceeded.

    Uses SIGALRM, so it only works on the main thread of a POSIX process. If
    that is unavailable the call simply runs unbounded rather than failing.
    """
    import signal

    if seconds is None or not hasattr(signal, "SIGALRM"):
        return fn()

    def handler(signum, frame):
        raise _EntryTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def compare_to_published(entries, limit=None, cache=None, progress=None,
                         skip_products=True, time_limit=None):
    """Recompute Hodge data for published entries and compare.

    This is the validation the generated web of :func:`split_web` cannot
    provide: an entry-by-entry check of pyCICY's cohomology against the
    values of Green, Hubsch and Lutken carried in the published list.

    Parameters
    ----------
    entries : list
        Records from :func:`load_published_list`.
    limit : int or None
        Check only the first ``limit`` entries. The full list takes a long
        time on first run; results are cached, so a later full run only pays
        for what it has not seen.
    skip_products : bool
        The list records the direct-product manifolds with
        h^{1,1} = h^{2,1} = 0 as a placeholder rather than as their actual
        Hodge numbers. Comparing against those would manufacture spurious
        disagreements, so they are reported separately by default.
    time_limit : float or None
        Wall-clock budget per entry, in seconds. The cost of a single entry
        varies by orders of magnitude -- the median is a few hundredths of a
        second, but the tail runs to minutes -- so without a budget one
        pathological configuration stalls the whole comparison. Entries that
        exceed it are counted in ``timed_out`` and are deliberately **not**
        cached, so a later run with a larger budget will attempt them again.

    Returns
    -------
    dict with ``checked``, ``agree``, ``disagree`` (a list of records),
    ``errors``, ``skipped_products``, and ``by_field`` counting which of
    h^{1,1}, h^{2,1}, chi disagreed.
    """
    cache = cache if cache is not None else default_cache()
    if limit is not None:
        entries = entries[:limit]

    agree = 0
    disagree = []
    errors = []
    timed_out = []
    skipped = 0
    by_field = {"h11": 0, "h21": 0, "euler": 0}

    for i, rec in enumerate(entries):
        if skip_products and rec["h11"] == 0 and rec["h21"] == 0:
            skipped += 1
            continue
        try:
            got = _time_limited(time_limit,
                                lambda: hodge(rec["conf"], cache=cache))
        except _EntryTimeout:
            timed_out.append(rec["num"])
            continue
        if got.get("error"):
            errors.append({"num": rec["num"], "error": got["error"]})
            continue
        mismatch = {}
        if got["h11"] != rec["h11"]:
            mismatch["h11"] = (rec["h11"], got["h11"])
            by_field["h11"] += 1
        if got["h21"] != rec["h21"]:
            mismatch["h21"] = (rec["h21"], got["h21"])
            by_field["h21"] += 1
        if got["euler"] != rec["euler"]:
            mismatch["euler"] = (rec["euler"], got["euler"])
            by_field["euler"] += 1
        if mismatch:
            disagree.append({"num": rec["num"], "conf": rec["conf"],
                             "mismatch": mismatch})
        else:
            agree += 1
        if progress is not None and (i + 1) % 100 == 0:
            progress(i + 1, len(entries), agree, len(disagree))

    return {
        "checked": agree + len(disagree),
        "agree": agree,
        "disagree": disagree,
        "errors": errors,
        "timed_out": timed_out,
        "skipped_products": skipped,
        "by_field": by_field,
    }


def published_coverage(entries, web):
    """How much of the published list the generated web reproduces.

    Matches on normal form, so a configuration found by splitting counts as
    covering a published entry only if the two are related by a relabelling
    of projective factors and defining equations.
    """
    published = {}
    for rec in entries:
        try:
            published[T.canonical_key(rec["conf"])] = rec["num"]
        except ValueError:
            continue
    found = set(web["nodes"])
    hit = found & set(published)
    return {
        "published": len(published),
        "generated": len(found),
        "in_both": len(hit),
        "fraction_of_published": len(hit) / len(published) if published else 0.0,
        "generated_not_published": len(found - set(published)),
    }


def load_list(path, limit=None):
    """Read a CICY configuration list from a text file.

    The published list is not bundled with pyCICY; point this at your own
    copy. Two layouts are accepted:

    * whitespace separated rows, configurations separated by blank lines,
      each row being ``n_r q^r_1 ... q^r_K``;
    * one configuration per line in Python literal form, e.g.
      ``[[4,5]]`` or ``[[1,1,1],[4,1,4]]``.

    Returns a list of configurations. Entries that do not parse, or that are
    not Calabi-Yau threefolds, are reported through :func:`check_list` rather
    than being silently dropped here.
    """
    import ast

    text = open(path).read()
    confs = []

    stripped = text.strip()
    if stripped.startswith("["):
        for line in stripped.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            confs.append([list(map(int, r)) for r in ast.literal_eval(line)])
    else:
        block = []
        for line in text.splitlines():
            if not line.strip():
                if block:
                    confs.append(block)
                    block = []
                continue
            block.append([int(x) for x in line.split()])
        if block:
            confs.append(block)

    if limit is not None:
        confs = confs[:limit]
    return confs


def check_list(confs):
    """Validate a list of configurations, returning a report.

    Checks the Calabi-Yau condition and the threefold condition, and counts
    how many entries are duplicates of each other up to relabelling. The
    published list has no such duplicates by construction, so a non-zero
    count means the file is not what it claims to be.
    """
    ok, bad, seen, dupes = [], [], set(), []
    for i, c in enumerate(confs):
        try:
            info = T.check_configuration(c)
        except ValueError as exc:
            bad.append((i, str(exc)))
            continue
        if not info["calabi_yau"] or info["dim_X"] != 3:
            bad.append((i, "dim_X=%d calabi_yau=%s"
                        % (info["dim_X"], info["calabi_yau"])))
            continue
        key = T.canonical_key(c)
        if key in seen:
            dupes.append(i)
        seen.add(key)
        ok.append(i)
    return {
        "total": len(confs),
        "valid": len(ok),
        "invalid": bad,
        "duplicates": dupes,
        "distinct": len(seen),
    }
