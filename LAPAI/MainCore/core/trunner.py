from .state import cache

import inspect
import json


class ToolRunner:
    def __init__(self, registry=None):
        self.registry = registry

    def set_registry(self, registry):
        self.registry = registry

    def _get_registry(self):
        registry = self.registry or getattr(cache, "tool_registry", None)
        if registry is None:
            raise RuntimeError(
                "Tool registry is not initialized. "
                "Call initialize_tools() or initialize_core() first."
            )
        return registry

    def execute(self, tool_name: str, arguments=None, context_state=None):
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")

        if arguments is None:
            arguments = {}
        elif isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("Tool arguments must contain valid JSON") from exc

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary")

        registry = self._get_registry()
        entry = registry.tools.get(tool_name)

        if entry is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        function = entry["function"]

        context_state = context_state or {}
        requirements = entry.get("requires_state", [])
        missing_state = [
            state_name
            for state_name in requirements
            if not bool(context_state.get(state_name))
        ]

        if missing_state:
            raise RuntimeError(
                f"Tool '{tool_name}' requires inactive state(s): "
                + ", ".join(missing_state)
            )

        signature = inspect.signature(function)
        signature.bind(**arguments)

        result = function(**arguments)

        return result


runner = ToolRunner()
