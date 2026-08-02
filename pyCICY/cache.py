r"""
pyCICY.cache -- disk-backed cache for expensive computations.

Hodge data for a configuration comes out of the Leray spectral sequence and
gets expensive quickly with matrix size. When surveying many configurations,
or re-running a script to adjust a plot, recomputing all of it each time
dominates the wall clock. This module keeps results in a small sqlite file so
that only genuinely new configurations cost anything.

Cache location, in order of precedence:

  1. the ``path`` argument to :class:`Cache`
  2. the ``PYCICY_CACHE`` environment variable
  3. ``/tmp/pycicy-cache/pycicy.sqlite``

/tmp is the default deliberately: this is a scratch cache for speed, not
durable data, and it should not accumulate in the working tree.

Keying
------
Entries are keyed on the *normal form* of the configuration
(:func:`pyCICY.transitions.canonical_key`), not on the matrix as written.
Configurations differing only by a relabelling of the projective factors or
the defining equations describe the same variety, so they share one entry.
In a split survey, where the same manifold turns up repeatedly under
different labellings, this is where most of the hits come from.

sqlite is used rather than a pickle so that a crashed or killed run leaves a
valid cache behind, and so several processes can read the same file.
"""

import json
import os
import sqlite3
import threading
import time

__all__ = ["Cache", "default_cache", "hodge", "cache_info", "clear_cache"]

_DEFAULT_DIR = "/tmp/pycicy-cache"
_DEFAULT_NAME = "pycicy.sqlite"


def _default_path():
    env = os.environ.get("PYCICY_CACHE")
    if env:
        return env
    return os.path.join(_DEFAULT_DIR, _DEFAULT_NAME)


class Cache(object):
    """A small key/value store on disk, with JSON values.

    Parameters
    ----------
    path : str or None
        sqlite file to use. None selects the default location.
    enabled : bool
        When False every lookup misses and nothing is written. Useful for
        timing comparisons and for verifying that cached results agree with
        freshly computed ones.
    """

    def __init__(self, path=None, enabled=True):
        self.path = path or _default_path()
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        self._conn = None
        if self.enabled:
            self._open()

    # -- plumbing ---------------------------------------------------------

    def _open(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  namespace TEXT NOT NULL,"
            "  key       TEXT NOT NULL,"
            "  value     TEXT NOT NULL,"
            "  seconds   REAL,"
            "  stamp     REAL,"
            "  PRIMARY KEY (namespace, key))")
        self._conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- api --------------------------------------------------------------

    def get(self, namespace, key):
        """Return the stored value, or None on a miss."""
        if not self.enabled:
            self.misses += 1
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM entries WHERE namespace=? AND key=?",
                (namespace, _text(key))).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, namespace, key, value, seconds=None):
        """Store a JSON-serialisable value."""
        if not self.enabled:
            return value
        payload = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(namespace, key, value, seconds, stamp) VALUES (?,?,?,?,?)",
                (namespace, _text(key), payload, seconds, time.time()))
            self._conn.commit()
        self.writes += 1
        return value

    def call(self, namespace, key, fn):
        """Return the cached value for ``key``, computing it with ``fn`` if absent.

        The wall time of a miss is recorded alongside the value, which makes
        it possible to report how much time the cache is saving.
        """
        got = self.get(namespace, key)
        if got is not None:
            return got
        start = time.time()
        value = fn()
        self.set(namespace, key, value, seconds=time.time() - start)
        return value

    # -- reporting --------------------------------------------------------

    def stats(self):
        total = self.hits + self.misses
        return {
            "path": self.path,
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    def info(self):
        """Summarise what is on disk: entries and stored compute time."""
        if not self.enabled or self._conn is None:
            return {"entries": 0, "namespaces": {}, "seconds_stored": 0.0}
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*), COALESCE(SUM(seconds), 0) "
                "FROM entries GROUP BY namespace").fetchall()
        namespaces = {r[0]: {"entries": r[1], "seconds": r[2]} for r in rows}
        return {
            "path": self.path,
            "entries": sum(v["entries"] for v in namespaces.values()),
            "namespaces": namespaces,
            "seconds_stored": sum(v["seconds"] for v in namespaces.values()),
        }

    def clear(self, namespace=None):
        """Drop cached entries, all of them or just one namespace."""
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM entries")
            else:
                cur = self._conn.execute(
                    "DELETE FROM entries WHERE namespace=?", (namespace,))
            self._conn.commit()
        return cur.rowcount


def _text(key):
    if isinstance(key, str):
        return key
    return json.dumps(key, separators=(",", ":"), sort_keys=True)


# --------------------------------------------------------------- singleton

_DEFAULT = None


def default_cache():
    """The process-wide cache, created on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Cache()
    return _DEFAULT


def cache_info():
    """Summary of the default cache."""
    return default_cache().info()


def clear_cache(namespace=None):
    """Empty the default cache."""
    return default_cache().clear(namespace)


# ------------------------------------------------------------ cached hodge

def hodge(conf, cache=None, log=3):
    """Hodge and Euler data for a configuration, cached on disk.

    Returns a dict with ``h11``, ``h21``, ``euler``, ``favourable``,
    ``nfold``, matching what a fresh CICY construction would give. A failed
    construction (an unsupported dimension, say) is cached too, as
    ``{"error": ...}``, so a survey does not retry it on every run.

    >>> hodge([[4, 5]])["h11"]
    1.0
    """
    from . import transitions as _T

    cache = cache if cache is not None else default_cache()
    key = _T.canonical_key(conf)

    def compute():
        return _hodge_uncached(conf, log=log)

    return cache.call("hodge", key, compute)


def _hodge_uncached(conf, log=3):
    import logging

    try:
        from .pyCICY import CICY
    except ImportError:  # running outside the package
        from pyCICY import CICY

    cy_logger = logging.getLogger("pyCICY")

    def mute(record):
        return False

    cy_logger.addFilter(mute)
    try:
        M = CICY(conf, log=log)
        return {
            "h11": float(M.h[2]) if M.nfold == 3 else None,
            "h21": float(M.h[1]) if M.nfold == 3 else None,
            "euler": int(M.euler_characteristic()),
            "favourable": bool(M.fav),
            "nfold": int(M.nfold),
        }
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    finally:
        cy_logger.removeFilter(mute)
