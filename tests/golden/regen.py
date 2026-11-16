"""
Generate or check the golden parse-tree snapshots.

    .venv/bin/python tests/golden/regen.py            rewrite every snapshot
    .venv/bin/python tests/golden/regen.py --check    list snapshots that differ (exit 1)

Groups (sources -> tests/golden/<group>/<name>.sexp):
    language   tests/kgtests/language/*.kg (recursive)
    lib        klongpy/lib/*.kg
    fixtures   tests/golden/fixtures/*.kg
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from klongpy.dump import dump_text  # noqa: E402

GROUPS = {
    "language": os.path.join(ROOT, "tests", "kgtests", "language"),
    "lib": os.path.join(ROOT, "klongpy", "lib"),
    "fixtures": os.path.join(HERE, "fixtures"),
}


def cases():
    """Yield (group, name, source_path, golden_path), sorted."""
    for group, base in GROUPS.items():
        found = []
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".kg"):
                    p = os.path.join(dp, fn)
                    found.append((os.path.relpath(p, base)[:-3].replace(os.sep, "__"), p))
        for name, p in sorted(found):
            yield group, name, p, os.path.join(HERE, group, name + ".sexp")


def render(path):
    with open(path, "r", newline="") as f:
        text = f.read()
    return dump_text(text, parser="token")


def main(argv):
    check = "--check" in argv
    bad = []
    for group, name, src, golden in cases():
        out = render(src)
        if check:
            old = open(golden).read() if os.path.exists(golden) else None
            if old != out:
                bad.append(f"{group}/{name}")
        else:
            os.makedirs(os.path.dirname(golden), exist_ok=True)
            with open(golden, "w") as f:
                f.write(out)
    if check:
        print("\n".join(bad) or "all golden files match")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
