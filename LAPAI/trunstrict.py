#!/usr/bin/env python3
"""Run the LAPAI strict regression tests.

Examples (run from the LAPAI repository root):
    python tools/run_strict_tests.py                  # normal suite
    python tools/run_strict_tests.py --stress         # + property/concurrency tests
    python tools/run_strict_tests.py --repo D:\\LAPAI  # repo is somewhere else
    python tools/run_strict_tests.py -k faiss -x      # extra args go straight to pytest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_marker_expression(stress: bool, live: bool) -> str | None:
    """Return the pytest -m expression (None = run everything)."""
    if stress and live:
        return None
    if stress:
        return "not live"
    if live:
        return "not stress"
    return "not stress and not live"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LAPAI strict regression tests")
    parser.add_argument("--repo", type=Path, default=None, help="LAPAI repository root (default: current directory, then the suite folder)")
    parser.add_argument("--stress", action="store_true", help="include property/concurrency tests")
    parser.add_argument("--live", action="store_true", help="include live backend/model tests")
    args, extra = parser.parse_known_args()

    repo = (args.repo or Path.cwd()).resolve()
    if not (repo / "MainCore").is_dir() and ("MainCore").is_dir():
        repo = ROOT
    if not (repo / "MainCore").is_dir():
        print(
            f"ERROR: folder 'MainCore' not found in {repo}\n"
            "Run this from the LAPAI repository root, or pass --repo <path-to-LAPAI>.",
            file=sys.stderr,
        )
        return 2

    report_dir = repo / "test-reports"
    report_dir.mkdir(exist_ok=True)
    junit = report_dir / "junit.xml"

    env = os.environ.copy()
    env["LAPAI_REPO_ROOT"] = str(repo)

    cmd = [
        sys.executable, "-m", "pytest", str("tests"),
        "-c", str("pytest.ini"),
        "--rootdir", str(ROOT),
        f"--junitxml={junit}",
    ]
    marker = build_marker_expression(args.stress, args.live)
    if marker:
        cmd += ["-m", marker]
    cmd += extra

    print("LAPAI STRICT TESTS")
    print(f"Repository root: {repo}")
    print(f"Stress: {'ON' if args.stress else 'OFF'}")
    print(f"Live: {'ON' if args.live else 'OFF'}")
    print()

    # cwd = the LAPAI repo, so LAPAI code that uses relative paths behaves as it does in real use.
    result = subprocess.run(cmd, cwd=repo, env=env)
    print()
    print(f"JUnit report: {junit}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
