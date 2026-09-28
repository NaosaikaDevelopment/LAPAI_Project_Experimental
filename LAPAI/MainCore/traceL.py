from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Node:
    call_id: str
    module: str
    function: str
    file: str
    line: int
    thread_id: int
    depth: int
    started_ns: int
    children: list["Node"] = field(default_factory=list)
    result: dict[str, Any] | None = None
    duration_ms: float | None = None
    exception: dict[str, Any] | None = None
    args: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "call_id": self.call_id,
            "module": self.module,
            "function": self.function,
            "file": self.file,
            "line": self.line,
            "thread_id": self.thread_id,
            "depth": self.depth,
            "started_ns": self.started_ns,
            "children": [child.as_dict() for child in self.children],
        }
        if self.args is not None:
            data["args"] = self.args
        if self.result is not None:
            data["result"] = self.result
        if self.duration_ms is not None:
            data["duration_ms"] = round(self.duration_ms, 3)
        if self.exception is not None:
            data["exception"] = self.exception
        return data


class TraceSession:
    def __init__(self, traces_dir: Path, metadata: dict[str, Any]) -> None:
        self.session_id = metadata["session_id"]
        now = datetime.now().astimezone()
        day_dir = traces_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        safe_time = now.strftime("%H-%M-%S")
        self.path = day_dir / f"trace_{safe_time}_{metadata.get('pid', 'unknown')}_{self.session_id[:8]}.json"
        self.metadata = {
            "session_id": self.session_id,
            "started_at": now.isoformat(),
            "pid": metadata.get("pid"),
            "python": metadata.get("python"),
            "cwd": metadata.get("cwd"),
            "command": metadata.get("command"),
        }
        self.roots: list[dict[str, Any]] = []
        self.nodes: dict[str, Node] = {}
        self.stacks: dict[int, list[str]] = {}
        self.event_count = 0
        self.lock = threading.RLock()
        self.finished = False
        self.last_flush = 0.0
        self.flush_interval = 0.5

    def _thread_stack(self, tid: int) -> list[str]:
        return self.stacks.setdefault(tid, [])

    def apply(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.event_count += 1
            etype = event.get("type")
            tid = int(event.get("thread_id", 0))

            if etype == "call":
                node = Node(
                    call_id=str(event["call_id"]),
                    module=str(event.get("module", "<unknown_module>")),
                    function=str(event.get("function", "<unknown_function>")),
                    file=str(event.get("file", "")),
                    line=int(event.get("line", 0)),
                    thread_id=tid,
                    depth=int(event.get("depth", 0)),
                    started_ns=int(event.get("timestamp_ns", 0)),
                    args=event.get("args"),
                )
                self.nodes[node.call_id] = node
                stack = self._thread_stack(tid)
                if stack:
                    parent = self.nodes.get(stack[-1])
                    if parent:
                        parent.children.append(node)
                else:
                    self.roots.append(node)
                stack.append(node.call_id)

            elif etype == "exception":
                call_id = str(event.get("call_id", ""))
                node = self.nodes.get(call_id)
                if node:
                    node.exception = event.get("exception")

            elif etype == "return":
                call_id = str(event.get("call_id", ""))
                node = self.nodes.get(call_id)
                if node:
                    node.duration_ms = int(event.get("duration_ns", 0)) / 1_000_000
                    node.result = event.get("result")
                stack = self._thread_stack(tid)
                if stack and stack[-1] == call_id:
                    stack.pop()
                elif call_id in stack:
                    # Recover from unusual trace ordering, e.g. async/thread edge cases.
                    stack.remove(call_id)

            elif etype == "session_end":
                self.finished = True

            now = time.monotonic()
            if self.finished or now - self.last_flush >= self.flush_interval:
                self.flush()

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema": "lapai-trace-tree/v1",
                "session": self.metadata,
                "event_count": self.event_count,
                "finished": self.finished,
                "roots": [node.as_dict() for node in self.roots],
            }

    def flush(self) -> None:
        payload = self.payload()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        self.last_flush = time.monotonic()


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        session: TraceSession | None = None
        try:
            for raw in self.rfile:
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if session is None:
                    if event.get("type") != "session_start":
                        continue
                    session = TraceSession(self.server.traces_dir, event)
                    with self.server.sessions_lock:
                        self.server.sessions[event["session_id"]] = session
                    print(f"\n[TRACE START] {session.session_id[:8]} -> {session.path}", flush=True)

                session.apply(event)
                self.server.render_live(event, session)

        finally:
            if session is not None:
                session.finished = True
                session.flush()
                print(f"[TRACE END]   events={session.event_count} file={session.path}", flush=True)


class TraceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, traces_dir: Path):
        super().__init__(server_address, handler_cls)
        self.traces_dir = traces_dir
        self.sessions: dict[str, TraceSession] = {}
        self.sessions_lock = threading.RLock()

    def render_live(self, event: dict[str, Any], session: TraceSession) -> None:
        etype = event.get("type")
        if etype == "call":
            depth = int(event.get("depth", 0))
            indent = "  " * depth
            print(
                f"{indent}▶ {event.get('module')}.{event.get('function')} "
                f"({os.path.basename(event.get('file', ''))}:{event.get('line', 0)})",
                flush=True,
            )
        elif etype == "return":
            depth = max(0, int(event.get("depth", 0)) if "depth" in event else 0)
            result = event.get("result", {})
            duration = int(event.get("duration_ns", 0)) / 1_000_000
            print(
                f"{'  ' * depth}◀ result={result.get('type', '<unknown>')} "
                f"{result.get('value', result.get('repr', ''))} [{duration:.3f} ms]",
                flush=True,
            )
        elif etype == "exception":
            exc = event.get("exception", {})
            print(
                f"  ✖ {exc.get('type', '<Exception>')}: {exc.get('message', '')}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live socket listener for dynamic Python execution traces")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--traces-dir", default="traces")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir).resolve()
    traces_dir.mkdir(parents=True, exist_ok=True)

    server = TraceServer((args.host, args.port), Handler, traces_dir)
    print(f"[LISTENER ACTIVE] tcp://{args.host}:{args.port}")
    print(f"[OUTPUT] {traces_dir}")
    print("Waiting for traced Python processes...\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LISTENER STOPPED]")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
