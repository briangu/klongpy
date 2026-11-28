"""Verify the AoC ports and measure repeatable workloads with a selected native kg."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent
UPSTREAM = "defeb2e63ade7aed486936e5de18dfb7f2db7907"


@dataclass(frozen=True)
class Case:
    year: int
    day: int
    part: int
    fixture: Path
    answer: object
    group: str
    parameter: int = 10

    @property
    def key(self):
        return f"{self.group}/{self.year}/{self.day:02}/{self.part}"

    def command(self, kg, runs):
        module = ROOT / str(self.year) / f"day{self.day:02}.kg"
        return [kg, "--strict", "2", ROOT / "bench.kg", module, self.fixture,
                str(self.part), str(runs), str(self.parameter)]


def sample_cases():
    for manifest in sorted(ROOT.glob("20??/cases.json")):
        year = int(manifest.parent.name)
        for task, spec in json.loads(manifest.read_text()).items():
            day, part = map(int, task.split("/"))
            fixture = ROOT / "samples" / str(year) / spec.get("fixture", f"{day}.txt")
            yield Case(year, day, part, fixture, spec["answer"], "sample", spec.get("parameter", 10))


def scale_cases(kg, directory):
    execute([kg, "--strict", "2", ROOT / "generate.kg", directory])
    for year, day, name, answers in [
        (2022, 2, "rounds", (45 * 4096, 45 * 4096)),
        (2022, 6, "marker", (32768 + 3, 32768 + 13)),
        (2022, 12, "hill", (1050, 26)),
        (2023, 4, "cards", (4095, 4096 * 4097 // 2)),
        (2023, 5, "interval", (1000000000, 10)),
    ]:
        for part, answer in enumerate(answers, start=1):
            yield Case(year, day, part, directory / f"{name}.txt", answer, "scale")


def execute(command, cwd=None):
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f"{command}: {proc.stderr.strip()}")
    return proc.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(case, answer):
    if answer != case.answer:
        raise ValueError(f"{case.key}: expected {case.answer!r}, got {answer!r}")


def measure(case, kg, runs):
    result = json.loads(execute(case.command(kg, runs)))
    verify(case, result["answer"])
    times = result["seconds"]
    if len(times) != runs or any(t <= 0 for t in times):
        raise ValueError(f"{case.key}: invalid timings {times}")
    result["parameter"] = case.parameter
    result["input_sha256"] = digest(case.fixture)
    if times:
        result["median_seconds"] = statistics.median(times)
    return result


def compare_classic(case, kg, checkout, directory, runs):
    # These three original programs print just the answer and expect DAY.txt in cwd.
    suffix = "b" if case.part == 2 else ""
    source = f"22/{case.day}{suffix}.kg"
    script = directory / "classic.kg"
    script.write_bytes(subprocess.check_output(["git", "-C", checkout, "show", f"{UPSTREAM}:{source}"]))
    shutil.copyfile(case.fixture, directory / f"{case.day}.txt")
    commands = {"classic": [kg, script], "v2": case.command(kg, 0)}
    samples = {key: [] for key in commands}
    for trial in range(runs + 1):
        order = list(commands) if trial % 2 == 0 else list(reversed(commands))
        for name in order:
            start = time.perf_counter()
            output = execute(commands[name], directory)
            elapsed = time.perf_counter() - start
            answer = int(output.strip()) if name == "classic" else json.loads(output)["answer"]
            verify(case, answer)
            if trial:
                samples[name].append(elapsed)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", type=Path, default=ROOT.parents[1] / "native/target/release/kg")
    parser.add_argument("--runs", type=int, default=7, help="timed batches after warmup")
    parser.add_argument("--check", action="store_true", help="check every answer without timing")
    parser.add_argument("--workload", choices=["all", "sample", "scale"], default="all")
    parser.add_argument("--year", type=int, choices=range(2022, 2026), help="run cases for one year")
    parser.add_argument("--day", type=int, choices=range(1, 26), help="run cases for one puzzle day")
    parser.add_argument("--output", type=Path, help="write machine-readable results")
    parser.add_argument("--baseline", type=Path, help="compare medians against an earlier report")
    parser.add_argument("--classic", type=Path, help="local briangu/aoc checkout for matched process timings")
    args = parser.parse_args()
    if args.runs < 1 or (args.check and (args.baseline or args.classic)):
        parser.error("timing comparisons need a positive run count and timing enabled")
    if args.classic and args.workload == "sample":
        parser.error("--classic uses the scale fixtures")
    kg = args.kg.resolve()
    runs = 0 if args.check else args.runs
    prior = json.loads(args.baseline.read_text())["cases"] if args.baseline else None
    report = {"kg_version": execute([kg, "--version"]).strip(), "kg_sha256": digest(kg),
              "platform": f"{platform.system()} {platform.machine()}", "runs": runs,
              "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(ROOT.rglob("*.kg"))},
              "cases": {}}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cases = list(sample_cases()) if args.workload != "scale" else []
        if args.workload != "sample":
            cases.extend(scale_cases(kg, directory))
        if args.year:
            cases = [case for case in cases if case.year == args.year]
        if args.day:
            cases = [case for case in cases if case.day == args.day]
        if not cases:
            parser.error("no cases match the selected year, day and workload")
        for case in cases:
            result = measure(case, kg, runs)
            report["cases"][case.key] = result
            timing = f"  {result['median_seconds'] * 1000:9.3f} ms" if runs else ""
            if prior is not None:
                old = prior.get(case.key, {})
                if (old.get("input_sha256") != result["input_sha256"] or
                        old.get("parameter") != case.parameter or old.get("answer") != case.answer):
                    raise ValueError(f"{case.key}: baseline missing or input, parameter or answer changed")
                if old.get("median_seconds", 0) <= 0:
                    raise ValueError(f"{case.key}: baseline contains no measured time")
                timing += f"  {old['median_seconds'] / result['median_seconds']:.2f}x baseline/current"
            print(f"{case.key}  PASS{timing}", flush=True)
            if args.classic and case.group == "scale" and (case.day, case.part) in {(2, 1), (6, 1), (6, 2)} and case.year == 2022:
                wall = compare_classic(case, kg, args.classic.resolve(), directory, runs)
                result["process_seconds"] = wall
                print(f"  process median: classic {statistics.median(wall['classic']):.4f}s,"
                      f" v2 {statistics.median(wall['v2']):.4f}s", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
