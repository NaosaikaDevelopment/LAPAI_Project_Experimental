from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MainCore.trace.tracer import DynamicTracer, TraceConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Python program while streaming its dynamic call tree to trace_listener.py"
    )
    parser.add_argument("script", help="Target Python script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Root path to trace. Repeat for multiple roots. Defaults to the target script directory.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="fnmatch pattern for absolute filenames or function names to exclude.",
    )
    parser.add_argument("--capture-args", action="store_true")
    parser.add_argument("--max-value-repr", type=int, default=240)
    args = parser.parse_args()

    target = Path(args.script).resolve()
    if not target.is_file():
        parser.error(f"Target script not found: {target}")

    include_paths = [str(Path(p).resolve()) for p in args.include] if args.include else [str((target.parent / "MainCore").resolve())]

    config = TraceConfig(
        host=args.host,
        port=args.port,
        include_paths=include_paths,
        exclude_patterns=args.exclude,
        capture_args=args.capture_args,
        max_value_repr=max(40, args.max_value_repr),
    )
    tracer = DynamicTracer(config)

    sys.argv = [str(target), *args.script_args]
    sys.path.insert(0, str(target.parent))
    os.chdir(target.parent)

    tracer.start()
    exit_code = 0
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1
    except BaseException:
        exit_code = 1
        raise
    finally:
        tracer.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())