from .state import *


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
    @staticmethod
    def _parse_arguments(arguments):
        if arguments is None:
            return {}, None
        if isinstance(arguments, dict):
            return dict(arguments), None

        if not isinstance(arguments, str):
            return None, "Tool arguments must be a JSON object or dictionary"

        raw = arguments.strip()
        if not raw:
            return {}, None

        # Some local models wrap otherwise-valid JSON in markdown fences.
        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2:
                raw = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            return None, f"Invalid tool JSON: {exc}"

        if not isinstance(parsed, dict):
            return None, "Tool arguments JSON must decode to an object"

        return parsed, None

    @staticmethod
    def _drop_null_optional_arguments(function, arguments):
        """Remove explicit nulls only for parameters that already have defaults.

        This lets an optional parameter such as confidence: float = 0.95 use its
        function default when a local model emits "confidence": null.
        Required parameters are never silently removed.
        """
        signature = inspect.signature(function)
        cleaned = dict(arguments)

        for name, parameter in signature.parameters.items():
            if name in cleaned and cleaned[name] is None:
                if parameter.default is not inspect.Parameter.empty:
                    cleaned.pop(name, None)

        return cleaned
    def execute(self, tool_name: str, arguments=None, context_state=None):
        """Execute a tool as a fault-isolated operation.

        Tool failures become structured results so one broken tool cannot
        terminate the agent turn. The caller decides whether the failure
        should trigger a fallback/retry.
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            return {"success": False, "error": "tool_name must be a non-empty string"}

        parsed, parse_error = self._parse_arguments(arguments)
        if parse_error:
            return {"success": False, "error": parse_error}
        arguments = parsed
        try:
            registry = self._get_registry()
            entry = registry.tools.get(tool_name)
            if entry is None:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            function = entry["function"]
            context_state = context_state if isinstance(context_state, dict) else {}
            requirements = entry.get("requires_state", [])
            missing_state = [
                state_name for state_name in requirements
                if not bool(context_state.get(state_name))
            ]
            if missing_state:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' requires inactive state(s): {', '.join(missing_state)}",
                }
            arguments = self._drop_null_optional_arguments(function, arguments)
            signature = inspect.signature(function)
            try:
                signature.bind(**arguments)
            except TypeError as exc:
                return {"success": False, "error": f"Invalid arguments for '{tool_name}': {exc}"}

            try:
                result = function(**arguments)
            except Exception as exc:
                print(f"[WARNING] Tool '{tool_name}' raised: {exc}")
                return {"success": False, "error": f"Tool '{tool_name}' failed: {exc}"}

            if result is None:
                result = {"success": True}

            print(
                "[TRUNNER DEBUG]",
                "tool=", tool_name,
                "module=", function.__module__,
                "doc_has_required=", "@required_each_turn" in (function.__doc__ or ""),
                "result=", repr(result),
            )
            return result
        except Exception as exc:
            print(f"[WARNING] Tool runner internal failure for '{tool_name}': {exc}")
            return {"success": False, "error": f"Tool runner failure: {exc}"}



runner = ToolRunner()
