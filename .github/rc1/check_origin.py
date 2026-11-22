"""Match the trusted publisher's upload receipt to the original GitHub build jobs."""
import json
from pathlib import Path
import sys


receipt, run, jobs = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
if not (run["id"] == receipt["github_run"] and run["head_sha"] == receipt["source_commit"]
        and run["head_branch"] == receipt["source_tag"] and run["repository"]["full_name"] == "briangu/klongpy"
        and run["path"] == ".github/workflows/wheels.yml" and run["event"] == "workflow_dispatch"
        and run["status"] in {"in_progress", "completed"}):
    raise ValueError("Source build identity does not match the qualification receipt")
results = {job["id"]: job for job in jobs["jobs"]}
for distribution in receipt["distributions"]:
    job = results[distribution["github_job"]]
    artifact = distribution["artifact"]
    expected = "Source distribution" if artifact == "sdist" else "Wheel " + artifact.removeprefix("wheel-")
    if (job["name"] != expected or job["status"] != "completed"
            or job["conclusion"] != distribution["github_conclusion"]):
        raise ValueError(f"Build job differs from the verified result: {expected}")
    if artifact.startswith("wheel-"):
        steps = {step["name"]:step["conclusion"] for step in job["steps"]}
        for required in ("Install the wheel into a fresh environment", "Installed commands, and the package comes from the wheel",
                         "Conformance suite through the installed package", "Save installed-wheel test reports"):
            if steps.get(required) != "success":
                raise ValueError(f"Required step did not pass: {expected} / {required}")
        failures = {name for name, conclusion in steps.items() if conclusion == "failure"}
        if failures - {"Test suite against the installed wheel"}:
            raise ValueError(f"Unexpected wheel job failures for {expected}: {failures}")
    if job["conclusion"] != "success" and artifact != "wheel-macos-arm64":
        raise ValueError(f"Missing passing platform job: {expected}")
    print(expected, job["conclusion"], job["html_url"])
