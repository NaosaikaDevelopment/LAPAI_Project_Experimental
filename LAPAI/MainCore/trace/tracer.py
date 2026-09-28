from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Optional


TRACE_EVENTS = {"call", "return", "exception"}


def _safe_repr(value: Any, max_len: int = 240) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive path
        text = f"<repr failed: {type(exc).__name__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def summarize_value(value: Any, max_repr: int = 240) -> dict[str, Any]:
    """Return a JSON-safe, compact summary without serializing arbitrary objects."""
    result: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > max_repr:
            result["value"] = value[: max_repr - 3] + "..."
            result["truncated"] = True
        else:
            result["value"] = value
        return result

    # Avoid importing optional packages just for tracing.
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            result["shape"] = list(shape)
        except Exception:
            result["shape"] = _safe_repr(shape)

    try:
        result["len"] = len(value)
    except Exception:
        pass

    if isinstance(value, dict):
        keys = list(value.keys())[:20]
        result["keys"] = [_safe_repr(k, 100) for k in keys]
        if len(value) > len(keys):
            result["keys_truncated"] = True
    elif isinstance(value, (list, tuple, set, frozenset)):
        result["preview"] = [_safe_repr(v, 80) for v in list(value)[:8]]
        result["preview_truncated"] = len(value) > 8

    result["repr"] = _safe_repr(value, max_repr)
    return result


def summarize_exception(exc_info: tuple[type[BaseException], BaseException, Any]) -> dict[str, Any]:
    exc_type, exc_value, _ = exc_info
    return {
        "type": f"{exc_type.__module__}.{exc_type.__qualname__}",
        "message": _safe_repr(str(exc_value), 500),
    }


@dataclass
class CallFrame:
    call_id: str
    started_ns: int


class SocketEmitter:
    """Tiny newline-delimited JSON client. The tracer never waits forever on the listener."""

    def __init__(self, host: str, port: int, session_id: str, connect_timeout: float = 0.75) -> None:
        self.host = host
        self.port = port
        self.session_id = session_id
        self.connect_timeout = connect_timeout
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()
        self.enabled = False

    def connect(self) -> bool:
        if self.sock is not None:
            return True
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            sock.settimeout(0.25)
            self.sock = sock
            self.enabled = True
            return True
        except OSError:
            self.enabled = False
            return False

    def emit(self, payload: dict[str, Any]) -> None:
        if not self.enabled and not self.connect():
            return

        payload = dict(payload)
        payload["session_id"] = self.session_id
        payload["protocol"] = 1
        raw = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

        with self.lock:
            try:
                assert self.sock is not None
                self.sock.sendall(raw)
            except OSError:
                self.close()

    def close(self) -> None:
        sock = self.sock
        self.sock = None
        self.enabled = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


@dataclass
class TraceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    include_paths: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    capture_args: bool = False
    max_value_repr: int = 240


class DynamicTracer:
    """In-process tracer used by the wrapper. It discovers Python call frames dynamically."""

    def __init__(self, config: TraceConfig) -> None:
        import uuid

        self.config = config
        self.session_id = uuid.uuid4().hex
        self.emitter = SocketEmitter(config.host, config.port, self.session_id)
        self._local = threading.local()
        self._enabled = False
        self._call_counter = 0
        self._counter_lock = threading.Lock()

    def _next_call_id(self) -> str:
        with self._counter_lock:
            self._call_counter += 1
            return f"{self._call_counter:08d}"

    def _stack(self) -> list[CallFrame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _normalize_path(self, path: str) -> str:
        try:
            return os.path.abspath(path)
        except Exception:
            return path

    def _should_trace(self, frame: FrameType) -> bool:
        raw_filename = str(frame.f_code.co_filename)

        
        if raw_filename.startswith("<") and raw_filename.endswith(">"):
            return False
        filename = self._normalize_path(raw_filename)

        # Never trace the tracer implementation itself.
        tracer_root = self._normalize_path(os.path.dirname(__file__))
        if filename == tracer_root or filename.startswith(tracer_root + os.sep):
            return False

        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(frame.f_code.co_name, pattern):
                return False

        if not self.config.include_paths:
            return True

        for root in self.config.include_paths:
            root = self._normalize_path(root)
            if filename == root or filename.startswith(root + os.sep):
                return True
        return False

    def _function_name(self, frame: FrameType) -> str:
        module_name = frame.f_globals.get("__name__", "<unknown_module>")
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        return f"{module_name}.{qualname}"

    def _args_summary(self, frame: FrameType) -> dict[str, Any] | None:
        if not self.config.capture_args:
            return None
        try:
            code = frame.f_code
            positional_count = code.co_argcount + code.co_posonlyargcount
            kwonly_count = code.co_kwonlyargcount
            names = list(code.co_varnames[: positional_count + kwonly_count])
            out = {}
            for name in names:
                if name in frame.f_locals:
                    out[name] = summarize_value(frame.f_locals[name], self.config.max_value_repr)
            return out
        except Exception:
            return None

    def start(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self.emitter.emit(
            {
                "type": "session_start",
                "timestamp_ns": time.time_ns(),
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "cwd": os.getcwd(),
                "command": sys.argv,
            }
        )
        sys.settrace(self._trace)
        threading.settrace(self._trace)

    def stop(self) -> None:
        if not self._enabled:
            return
        sys.settrace(None)
        threading.settrace(None)
        self.emitter.emit(
            {
                "type": "session_end",
                "timestamp_ns": time.time_ns(),
            }
        )
        self.emitter.close()
        self._enabled = False

    def _trace(self, frame: FrameType, event: str, arg: Any):
        if event not in TRACE_EVENTS:
            return self._trace

        if event == "call":
            if not self._should_trace(frame):
                # Returning None tells CPython not to trace descendants of this frame.
                return None

            call_id = self._next_call_id()
            now = time.time_ns()
            stack = self._stack()
            stack.append(CallFrame(call_id=call_id, started_ns=now))

            payload = {
                "type": "call",
                "timestamp_ns": now,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "call_id": call_id,
                "depth": len(stack) - 1,
                "module": frame.f_globals.get("__name__", "<unknown_module>"),
                "function": frame.f_code.co_qualname,
                "file": self._normalize_path(frame.f_code.co_filename),
                "line": frame.f_lineno,
            }
            args = self._args_summary(frame)
            if args is not None:
                payload["args"] = args
            self.emitter.emit(payload)
            return self._trace

        stack = self._stack()
        if not stack:
            return self._trace

        current = stack[-1]

        if event == "exception":
            exc = summarize_exception(arg)
            self.emitter.emit(
                {
                    "type": "exception",
                    "timestamp_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "call_id": current.call_id,
                    "exception": exc,
                }
            )
            return self._trace

        # return
        duration_ns = max(0, time.time_ns() - current.started_ns)
        self.emitter.emit(
            {
                "type": "return",
                "timestamp_ns": time.time_ns(),
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "call_id": current.call_id,
                "duration_ns": duration_ns,
                "result": summarize_value(arg, self.config.max_value_repr),
            }
        )
        stack.pop()
        return self._trace
