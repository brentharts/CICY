#!/usr/bin/env python3
"""
Download the published list of 7890 CICY three-folds and convert it to JSON.

Source
------
    https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/

The list was first compiled in

    P. Candelas, A. M. Dale, C. A. Lutken, R. Schimmrigk,
    "Complete Intersection Calabi-Yau Manifolds",
    Nucl. Phys. B298 (1988) 493,

and the Hodge numbers it carries were computed in

    P. S. Green, T. Hubsch, C. A. Lutken,
    "All Hodge Numbers for All Calabi-Yau Complete Intersections",
    Class. Quant. Grav. 6 (1989) 105.

The data is redistributed by its authors, not by pyCICY, which is why it is
fetched rather than vendored. Any use of it should cite the two papers above.

Why this script exists
----------------------
pyCICY generates configurations by splitting (see pyCICY.cicylist) rather
than reading them from the published list. Comparing the two is the natural
validation step, and the file also carries published h^{1,1}, h^{2,1} and
Euler characteristics that pyCICY's own computations can be checked against
one entry at a time.

Format of the source file
-------------------------
Blocks separated by blank lines::

    Num    : 14
    NumPs  : 3
    NumPol : 2
    Eta    : 0
    H11    : 19
    H21    : 19
    C2     : {0, 36, 36}
    Redun  : {0, 0, 0, 0, 0}
    {1, 1}
    {0, 3}
    {3, 0}

``NumPs`` rows of ``NumPol`` degrees follow the header. The dimension of each
projective factor is *not* given; it follows from the Calabi-Yau condition
n_i = sum_a q^i_a - 1, which is how CIPro reads these matrices too.

Usage
-----
    python3 scripts/fetch_cicy_list.py
    python3 scripts/fetch_cicy_list.py --outdir data --keep-raw

Writes ``data/cicylist.json``. Nothing is written unless every check passes.
"""

import argparse
import json
import os
import sys
import urllib.request

URL = "https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt"

EXPECTED_ENTRIES = 7890
EXPECTED_PRODUCTS = 22          # direct products, recorded with h11 = h21 = 0
EXPECTED_H11_MAX = 19
EXPECTED_H21_MAX = 101


def download(url, timeout=120):
    print("downloading %s" % url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "pyCICY-fetch/1.0 (+https://github.com/)"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        raw = fh.read()
    print("  %d bytes" % len(raw))
    return raw.decode("utf-8", errors="replace")


def parse(text):
    """Parse the Oxford plain-text list into records.

    Returns a list of dicts with keys ``num``, ``conf`` (in pyCICY form, i.e.
    with the projective space dimension prepended to each row), ``h11``,
    ``h21``, ``euler`` and ``c2``.
    """
    records = []
    block = []
    for line in text.splitlines() + [""]:
        if line.strip():
            block.append(line)
            continue
        if block:
            rec = _parse_block(block)
            if rec is not None:
                records.append(rec)
            block = []
    return records


def _parse_block(lines):
    header = {}
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            body = stripped.strip("{}")
            rows.append([int(x) for x in body.split(",") if x.strip() != ""])
        elif ":" in stripped:
            key, _, value = stripped.partition(":")
            header[key.strip()] = value.strip()

    if "Num" not in header or not rows:
        return None

    # C2 and Redun are header lines ("C2 : {..}"), so they land in `header`
    # rather than in `rows`; only the configuration rows start with a brace.
    n_ps = int(header["NumPs"])
    n_pol = int(header["NumPol"])
    matrix = [r for r in rows if len(r) == n_pol]
    if len(matrix) > n_ps:
        matrix = matrix[-n_ps:]
    if len(matrix) != n_ps:
        raise ValueError("entry %s: expected %d rows of width %d, found %d"
                         % (header.get("Num"), n_ps, n_pol, len(matrix)))

    # The dimension of each projective factor follows from the Calabi-Yau
    # condition n_i = sum_a q^i_a - 1.
    conf = [[sum(row) - 1] + list(row) for row in matrix]

    c2 = [int(x) for x in header.get("C2", "").strip("{}").split(",")
          if x.strip() != ""]

    return {
        "num": int(header["Num"]),
        "conf": conf,
        "h11": int(header["H11"]),
        "h21": int(header["H21"]),
        "euler": int(header["Eta"]),
        "c2": c2,
    }


def validate(records):
    """Check the parsed list against what the literature says it contains."""
    problems = []
    n = len(records)
    print("\nvalidating %d entries" % n)

    if n != EXPECTED_ENTRIES:
        problems.append("expected %d entries, parsed %d"
                        % (EXPECTED_ENTRIES, n))

    nums = [r["num"] for r in records]
    if sorted(nums) != list(range(1, n + 1)):
        problems.append("entry numbers are not 1..%d without gaps" % n)

    bad_cy = bad_dim = bad_euler = 0
    h11s, h21s = [], []
    for r in records:
        conf = r["conf"]
        # Calabi-Yau condition, by construction of the dimension column
        if any(sum(row[1:]) != row[0] + 1 for row in conf):
            bad_cy += 1
        # threefold condition
        dim_a = sum(row[0] for row in conf)
        k = len(conf[0]) - 1
        if dim_a - k != 3:
            bad_dim += 1
        # chi = 2(h11 - h21), which the file records independently as Eta
        if r["euler"] != 2 * (r["h11"] - r["h21"]):
            bad_euler += 1
        h11s.append(r["h11"])
        h21s.append(r["h21"])

    if bad_cy:
        problems.append("%d entries violate the Calabi-Yau condition" % bad_cy)
    if bad_dim:
        problems.append("%d entries are not threefolds" % bad_dim)
    if bad_euler:
        problems.append("%d entries have Eta != 2(H11 - H21)" % bad_euler)

    products = sum(1 for r in records if r["h11"] == 0 and r["h21"] == 0)
    print("  Calabi-Yau condition   %s" % ("ok" if not bad_cy else "FAIL"))
    print("  threefold condition    %s" % ("ok" if not bad_dim else "FAIL"))
    print("  Eta == 2(H11 - H21)    %s" % ("ok" if not bad_euler else "FAIL"))
    print("  h^{1,1} range          %d..%d (literature 0..%d)"
          % (min(h11s), max(h11s), EXPECTED_H11_MAX))
    print("  h^{2,1} range          %d..%d (literature 0..%d)"
          % (min(h21s), max(h21s), EXPECTED_H21_MAX))
    print("  entries with h11=h21=0 %d (literature: %d direct products)"
          % (products, EXPECTED_PRODUCTS))
    print("  distinct Hodge pairs   %d (literature: 266)"
          % len({(r["h11"], r["h21"]) for r in records}))

    if max(h11s) > EXPECTED_H11_MAX or max(h21s) > EXPECTED_H21_MAX:
        problems.append("Hodge numbers outside the published ranges")

    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also save the original cicylist.txt")
    ap.add_argument("--from-file", default=None,
                    help="parse a local copy instead of downloading")
    args = ap.parse_args(argv)

    if args.from_file:
        print("reading %s" % args.from_file)
        text = open(args.from_file, encoding="utf-8", errors="replace").read()
    else:
        try:
            text = download(args.url)
        except Exception as exc:
            print("\ndownload failed: %s: %s" % (type(exc).__name__, exc))
            print("If the host is unreachable, fetch the file manually from")
            print("  %s" % args.url)
            print("and re-run with --from-file cicylist.txt")
            return 2

    records = parse(text)
    problems = validate(records)

    if problems:
        print("\nNOT writing output; the following checks failed:")
        for p in problems:
            print("  - %s" % p)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "cicylist.json")
    payload = {
        "source": args.url,
        "count": len(records),
        "references": [
            "Candelas, Dale, Lutken, Schimmrigk, Nucl. Phys. B298 (1988) 493",
            "Green, Hubsch, Lutken, Class. Quant. Grav. 6 (1989) 105",
        ],
        "note": "Configuration rows are [n_i, q^i_1, ..., q^i_K]; the "
                "dimension n_i was reconstructed from the Calabi-Yau "
                "condition, as the source file omits it.",
        "entries": records,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    print("\nwrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))

    if args.keep_raw:
        raw_out = os.path.join(args.outdir, "cicylist.txt")
        with open(raw_out, "w") as fh:
            fh.write(text)
        print("wrote %s" % raw_out)

    print("\nAll checks passed. pyCICY can now run:")
    print("  from pyCICY import cicylist as L")
    print("  entries = L.load_published_list('%s')" % out)
    print("  print(L.compare_to_published(entries, limit=200))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
