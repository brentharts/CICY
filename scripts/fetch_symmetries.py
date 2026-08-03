#!/usr/bin/env python3
"""
Download the Mathematica CICY file and extract the freely acting symmetries.

Source
------
    https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.m

Only the Mathematica file carries the freely acting discrete symmetries. They
were classified in

    V. Braun, "On Free Quotients of Complete Intersection Calabi-Yau
    Manifolds", JHEP 04 (2011) 005, arXiv:1003.3235,

and Hodge numbers for the resulting quotients were computed in

    A. Constantin, J. Gray, A. Lukas, "Hodge Numbers for All CICY Quotients",
    JHEP 01 (2017) 001, arXiv:1607.01830.

The data is redistributed by its authors, not by pyCICY, so it is fetched
rather than vendored. Cite the papers above if you use it.

Why this script is cautious
---------------------------
The layout of the Mathematica file could not be inspected while this script
was written: the server returns it as a binary Wolfram package, so the
structure below is inferred rather than confirmed. The script therefore
parses the file generically with pyCICY.mathematica, then *looks* for the
symmetry data instead of assuming where it is. If it cannot find it, it
prints a structural report and exits non-zero rather than writing a file that
might be wrong.

If that happens, send the output of

    python3 scripts/fetch_symmetries.py --probe

which prints the shape of the parsed file without dumping its contents, and
the extraction can be adjusted to match.

Usage
-----
    python3 scripts/fetch_symmetries.py                 # download and extract
    python3 scripts/fetch_symmetries.py --probe         # report structure only
    python3 scripts/fetch_symmetries.py --from-file cicylist.m
"""

import argparse
import json
import os
import sys
import urllib.request

sys.setrecursionlimit(20000)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import mathematica as MM

URL = ("https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/"
       "cicylist/cicylist.m")

EXPECTED_ENTRIES = 7890
# Lukas and Mishra, Commun. Math. Phys. 379 (2020) 847, refer to "the 1695
# known quotients of complete intersection manifolds by freely-acting
# discrete symmetries" arising from Braun's classification. Used as a sanity
# check on the extraction, not as a hard requirement, since exactly what the
# file records may be counted differently.
REFERENCE_QUOTIENTS = 1695

# Key names the symmetry data might plausibly sit under. Matching is done
# case-insensitively and by substring, since the exact spelling is unknown.
SYMMETRY_HINTS = ("symmetr", "freeact", "freelyacting", "quotient", "group")
CONF_HINTS = ("conf", "matrix")
NUM_HINTS = ("num", "index", "id")


def download(url, timeout=300):
    print("downloading %s" % url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "pyCICY-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        raw = fh.read()
    print("  %d bytes" % len(raw))
    return raw.decode("utf-8", errors="replace")


def report_comments(text):
    """Print every Mathematica comment, plus the head of the file.

    The JSON output preserves the data but not the comments, and comments are
    where a data file usually documents what its fields mean. The boolean
    leading each symmetry record is not explained by the data itself, so this
    is the cheapest place to look before reaching for the paper.
    """
    print("\n--- first 600 characters ---")
    print(text[:600])

    comments = []
    i = 0
    while True:
        start = text.find("(*", i)
        if start < 0:
            break
        depth = 1
        j = start + 2
        while j < len(text) and depth:
            if text.startswith("(*", j):
                depth += 1
                j += 2
            elif text.startswith("*)", j):
                depth -= 1
                j += 2
            else:
                j += 1
        comments.append((start, text[start:j]))
        i = j

    print("\n--- %d comment(s) ---" % len(comments))
    for offset, body in comments[:40]:
        condensed = " ".join(body.split())
        if len(condensed) > 300:
            condensed = condensed[:297] + "..."
        print("  @%d: %s" % (offset, condensed))
    if len(comments) > 40:
        print("  ... %d more" % (len(comments) - 40))
    if not comments:
        print("  (none; the conventions are not documented in the file)")


def parse_file(text):
    """Parse the whole file, tolerating several statements."""
    print("parsing Mathematica syntax (%d chars)" % len(text))
    try:
        exprs = MM.loads(text, all_expressions=True)
    except MM.MathematicaSyntaxError as exc:
        print("\nparse failed: %s" % exc)
        return None
    print("  parsed %d top-level expression(s)" % len(exprs))
    return exprs


def _match(name, hints):
    low = str(name).lower()
    return any(h in low for h in hints)


def find_records(exprs):
    """Locate the list of per-manifold records, wherever it lives.

    Looks for the longest list whose elements are lists of rules, which is
    what a table of 7890 records should look like however it is wrapped.
    """
    best = None

    def visit(node, depth=0):
        nonlocal best
        if depth > 6:
            return
        if isinstance(node, list):
            if node and all(isinstance(x, MM.Rule) for x in node):
                return                        # a single record, not the table
            rule_lists = [x for x in node
                          if isinstance(x, list) and x
                          and all(isinstance(y, MM.Rule) for y in x)]
            if len(rule_lists) > (len(best) if best else 0):
                best = rule_lists
            for item in node[:50]:
                visit(item, depth + 1)
        elif isinstance(node, MM.Rule):
            visit(node.rhs, depth + 1)
        elif isinstance(node, MM.Expr):
            for a in node.args[:50]:
                visit(a, depth + 1)

    for e in exprs:
        visit(e)
    return best


def extract(records):
    """Pull number, configuration and symmetry data out of the records."""
    out = []
    sym_keys = set()
    for rec in records:
        fields = {str(r.lhs): r.rhs for r in rec}
        num = None
        conf = None
        syms = None
        for key, value in fields.items():
            if num is None and _match(key, NUM_HINTS) and isinstance(value, int):
                num = value
            elif conf is None and _match(key, CONF_HINTS) \
                    and isinstance(value, list):
                conf = value
            elif _match(key, SYMMETRY_HINTS):
                sym_keys.add(key)
                syms = value
        out.append({"num": num, "conf": conf, "symmetries": syms,
                    "fields": sorted(fields)})
    return out, sorted(sym_keys)


def summarise_symmetries(records):
    """Describe what the symmetry field actually contains."""
    import collections

    kinds = collections.Counter()
    nonempty = 0
    samples = []
    for rec in records:
        s = rec["symmetries"]
        if s is None:
            kinds["absent"] += 1
            continue
        if isinstance(s, list):
            if not s:
                kinds["empty list"] += 1
                continue
            nonempty += 1
            kinds["list of %s" % type(s[0]).__name__] += 1
            if len(samples) < 5:
                samples.append((rec["num"], s))
        else:
            nonempty += 1
            kinds[type(s).__name__] += 1
            if len(samples) < 5:
                samples.append((rec["num"], s))
    return {"kinds": dict(kinds), "nonempty": nonempty, "samples": samples}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="report the structure and stop, writing nothing")
    ap.add_argument("--no-keep-raw", action="store_true",
                    help="do not save the downloaded cicylist.m alongside "
                         "the JSON (it is saved by default, since the raw "
                         "file is the only record of conventions the JSON "
                         "does not preserve, such as comments)")
    ap.add_argument("--comments", action="store_true",
                    help="print the comments and header of the file and "
                         "stop. Mathematica comments are the most likely "
                         "place for the field conventions to be documented, "
                         "and the parser discards them.")
    ap.add_argument("--context", type=int, default=None,
                    help="print the raw text around this byte offset, for "
                         "inspecting a construct the parser rejected")
    args = ap.parse_args(argv)

    if args.from_file:
        print("reading %s" % args.from_file)
        text = open(args.from_file, encoding="utf-8", errors="replace").read()
    else:
        try:
            text = download(args.url)
        except Exception as exc:
            print("\ndownload failed: %s: %s" % (type(exc).__name__, exc))
            print("Fetch it manually from %s" % args.url)
            print("then re-run with --from-file cicylist.m")
            return 2

    if args.context is not None:
        lo = max(0, args.context - 400)
        hi = min(len(text), args.context + 400)
        print("\n--- raw text around offset %d ---" % args.context)
        print(text[lo:hi])
        return 0

    if args.comments:
        report_comments(text)
        return 0

    exprs = parse_file(text)
    if exprs is None:
        return 1

    if args.probe:
        print("\n--- structure ---")
        for e in exprs[:3]:
            print(MM.describe(e, max_depth=4))
        return 0

    records = find_records(exprs)
    if not records:
        print("\nCould not find a table of records in the parsed file.")
        print("Structure follows; please send this output.\n")
        for e in exprs[:3]:
            print(MM.describe(e, max_depth=4))
        return 1

    print("  found %d records" % len(records))
    extracted, sym_keys = extract(records)
    print("  field names present: %s"
          % sorted({f for r in extracted for f in r["fields"]})[:20])
    print("  fields matched as symmetry data: %s" % (sym_keys or "NONE"))

    summary = summarise_symmetries(extracted)
    print("\nsymmetry field contents: %s" % summary["kinds"])
    print("records with a non-empty symmetry field: %d" % summary["nonempty"])
    for num, sample in summary["samples"]:
        text = repr(sample)
        if len(text) > 160:
            text = text[:157] + "..."
        print("   entry %s: %s" % (num, text))

    # The file carries more records than the plain text list: the parse finds
    # 7921 where the text file has 7890. What matters is that entries 1..7890
    # are all present exactly once; anything beyond that is reported and kept
    # separately rather than silently mixed in.
    nums = [r["num"] for r in extracted if isinstance(r["num"], int)]
    seen = {}
    for n in nums:
        seen[n] = seen.get(n, 0) + 1
    expected_range = set(range(1, EXPECTED_ENTRIES + 1))
    missing = sorted(expected_range - set(seen))
    duplicated = sorted(n for n, c in seen.items() if c > 1)
    extra = sorted(n for n in seen if n not in expected_range)

    print("\nrecord numbering")
    print("  records parsed        %d" % len(records))
    print("  distinct Num values   %d" % len(seen))
    print("  covering 1..%d        %s"
          % (EXPECTED_ENTRIES, "yes" if not missing else
             "NO, %d missing" % len(missing)))
    print("  duplicated Num        %d %s"
          % (len(duplicated), duplicated[:10] if duplicated else ""))
    print("  Num outside range     %d %s"
          % (len(extra), extra[:10] if extra else ""))
    if len(records) != EXPECTED_ENTRIES:
        print("  note: %d extra record(s) beyond the %d of the text list"
              % (len(records) - EXPECTED_ENTRIES, EXPECTED_ENTRIES))

    # Report the records whose symmetry field is a bare string, since those
    # are a different shape from the matrix data and need explaining.
    strings = [(r["num"], r["symmetries"]) for r in extracted
               if isinstance(r["symmetries"], str)]
    if strings:
        print("\nrecords whose symmetry field is a string: %d" % len(strings))
        for num, text in strings[:10]:
            print("   entry %s: %r" % (num, text[:70]))
        if len(strings) > 10:
            print("   ... %d more" % (len(strings) - 10))

    problems = []
    if missing:
        problems.append("entries %s missing from 1..%d"
                        % (missing[:5], EXPECTED_ENTRIES))
    if duplicated:
        problems.append("%d duplicated Num values" % len(duplicated))
    if not sym_keys:
        problems.append("no field looked like symmetry data; the hints in "
                        "SYMMETRY_HINTS may need adjusting")
    if summary["nonempty"] == 0:
        problems.append("every symmetry field was empty or absent")

    if problems:
        print("\nNOT writing output:")
        for p in problems:
            print("  - %s" % p)
        print("\nRun with --probe and send the structural report so the")
        print("extraction can be matched to the actual layout.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "symmetries.json")
    payload = {
        "source": args.url,
        "count": len(extracted),
        "expected_entries": EXPECTED_ENTRIES,
        "extra_records": len(records) - EXPECTED_ENTRIES,
        "references": [
            "Braun, JHEP 04 (2011) 005, arXiv:1003.3235",
            "Constantin, Gray, Lukas, JHEP 01 (2017) 001, arXiv:1607.01830",
        ],
        "note": "Symmetry field extracted heuristically; see "
                "scripts/fetch_symmetries.py. Verify before relying on it.",
        "entries": [{"num": r["num"], "symmetries": _jsonable(r["symmetries"])}
                    for r in extracted],
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    print("\nwrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
    print("reference: Braun's classification is usually quoted as giving "
          "%d quotients" % REFERENCE_QUOTIENTS)

    if not args.no_keep_raw and not args.from_file:
        raw_out = os.path.join(args.outdir, "cicylist.m")
        with open(raw_out, "w") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)"
              % (raw_out, os.path.getsize(raw_out) / 1e6))
    return 0


def _jsonable(value):
    if isinstance(value, MM.Rule):
        return {"rule": [_jsonable(value.lhs), _jsonable(value.rhs)]}
    if isinstance(value, MM.Expr):
        return {"head": value.head, "args": [_jsonable(a) for a in value.args]}
    if isinstance(value, MM.Symbol):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
