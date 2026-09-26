from __future__ import annotations

from types import SimpleNamespace

import pytest

statecore = pytest.importorskip(
    "MainCore.statecore",
    reason="Full LAPAI runtime dependencies are required for agent-loop integration.",
)


class FakeRanker:
    def search(self, *args, **kwargs):
        return []


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Fake model called more times than expected")
        message = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _tool_call(name, args="{}", call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _base_registry():
    class Registry:
        def __init__(self):
            self.tools = {}

    registry = Registry()

    def alpha():
        """
        @required_each_turn
        @provides_state: ready
        """
        return {"success": True, "_state": {"ready": True}}

    registry.tools["alpha"] = {
        "function": alpha,
        "description": alpha.__doc__ or "",
        "required_each_turn": True,
        "requires_state": [],
        "provides_state": ["ready"],
        "required_when_state": [],
    }
    return registry


def test_zero_argument_dependency_chain_is_auto_executed(monkeypatch):
    registry = _base_registry()

    def beta():
        """
        @requires_state: ready
        @required_when_state: ready
        """
        return {"success": True}

    registry.tools["beta"] = {
        "function": beta,
        "description": beta.__doc__ or "",
        "required_each_turn": False,
        "requires_state": ["ready"],
        "provides_state": [],
        "required_when_state": ["ready"],
    }

    schemas = [statecore.build_tool_schema(name, info) for name, info in registry.tools.items()]
    fake_client = FakeClient([_message(content="final")])
    cache = statecore.cache
    monkeypatch.setattr(cache, "tool_registry", registry)
    monkeypatch.setattr(cache, "tool_ranker", FakeRanker())
    monkeypatch.setattr(cache, "tool_schemas", schemas)
    monkeypatch.setattr(cache, "model_name", "fake")
    monkeypatch.setattr(cache, "client", fake_client)
    monkeypatch.setattr(cache, "agent_state", {}, raising=False)

    result = statecore.run_agent_turn([], current_user="hello", max_tool_iters=6)

    assert result == "final"
    assert len(fake_client.chat.completions.requests) == 1


def test_argument_required_dependency_is_left_to_model(monkeypatch):
    registry = _base_registry()

    def beta(value: str):
        """
        @requires_state: ready
        @required_when_state: ready
        """
        return {"success": True, "value": value}

    registry.tools["beta"] = {
        "function": beta,
        "description": beta.__doc__ or "",
        "required_each_turn": False,
        "requires_state": ["ready"],
        "provides_state": [],
        "required_when_state": ["ready"],
    }

    schemas = [statecore.build_tool_schema(name, info) for name, info in registry.tools.items()]
    fake_client = FakeClient([
        _message(content="", tool_calls=[_tool_call("beta", '{"value":"ok"}')]),
        _message(content="done"),
    ])

    cache = statecore.cache
    monkeypatch.setattr(cache, "tool_registry", registry)
    monkeypatch.setattr(cache, "tool_ranker", FakeRanker())
    monkeypatch.setattr(cache, "tool_schemas", schemas)
    monkeypatch.setattr(cache, "model_name", "fake")
    monkeypatch.setattr(cache, "client", fake_client)
    monkeypatch.setattr(cache, "agent_state", {}, raising=False)

    result = statecore.run_agent_turn([], current_user="hello", max_tool_iters=6)

    assert result == "done"
    assert fake_client.chat.completions.requests[0]["tool_choice"] == "required"
    assert [
        s["function"]["name"] for s in fake_client.chat.completions.requests[0]["tools"]
    ] == ["beta"]
