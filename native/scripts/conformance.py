#!/usr/bin/env python3
"""
Per-file conformance counts of the native binary (uses conformance/run.py as a library).

    .venv/bin/python native/scripts/conformance.py [--tags a,b] [--all] [-- extra klong options]

Default tags: what the spike of ticket 102 declares. --all adds strings,dicts,unicode (ticket 103)
and lists every failure. Extra options go to the binary (for example `-- --heap-limit 1M`).
"""
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPIKE = "core,float,string-fmt,width64,strict1,strict2,numpy-only,device"


def main(argv):
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    tags = SPIKE
    show = "--all" in argv
    if show:
        tags += ",strings,dicts,unicode"
    if "--tags" in argv:
        tags = argv[argv.index("--tags") + 1]
    tags = set(tags.split(","))
    binary = os.environ.get("KLONG_NATIVE") or os.path.join(ROOT, "native", "target", "release", "klong")
    spec = importlib.util.spec_from_file_location("conformance_run", os.path.join(ROOT, "conformance", "run.py"))
    conf = importlib.util.module_from_spec(spec)
    sys.modules["conformance_run"] = conf
    spec.loader.exec_module(conf)
    executor = conf.External([binary, *extra, "--conformance"])
    total = passed = 0
    kinds = {}
    start = time.perf_counter()
    print(f"{'file':28} {'pass':>5} {'run':>5} {'in file':>8}")
    for path in conf.case_files(conf.CASES_DIR):
        every = conf.load_file(path)
        cases = conf.selected(every, tags)
        rows = conf.run_cases(cases, executor, os.cpu_count() or 2)
        ok = sum(1 for _, _, p in rows if not p)
        total += len(cases)
        passed += ok
        print(f"{os.path.basename(path):28} {ok:5d} {len(cases):5d} {len(every):8d}")
        for case, result, problem in rows:
            if problem:
                got = problem.strip().splitlines()[-1].strip()
                kinds[got] = kinds.get(got, 0) + 1
                if show or len(extra) == 0:
                    print(f"    FAIL {case.where} {case.name}: {got}")
    print(f"{'total':28} {passed:5d} {total:5d}   in {time.perf_counter() - start:.1f} s")
    if kinds and extra:
        print("failures by outcome:")
        for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {k}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
