from __future__ import annotations

import inspect
import json
import re
from typing import Any
from .state import cache


def is_required_tool(tool_name: str,registry: Any,context_state: dict[str, Any] | None = None,) -> bool:
    if registry is None or not isinstance(tool_name, str):
        return False

    info = getattr(registry, "tools", {}).get(tool_name)
    if not isinstance(info, dict):
        return False

    if info.get("required_each_turn", False):
        return True
    function = info.get("function")
    doc = inspect.getdoc(function) if callable(function) else ""
    if any(line.strip() == "@required_each_turn" for line in (doc or "").splitlines()):
        return True

    # Support state-conditioned required tools as well as @required_each_turn.
    conditions = info.get("required_when_state", []) or []
    state = context_state if isinstance(context_state, dict) else {}
    return bool(conditions) and all(bool(state.get(name)) for name in conditions)


def run_required_tool_fallback(tool_name: str,registry: Any,runner: Any,context_state: dict[str, Any] | None = None,user_text: str | None = None,failure_reason: str | None = None,) -> dict[str, Any]:
    if not is_required_tool(tool_name, registry, context_state):
        return {
            "success": False,
            "error": f"SUM fallback denied: '{tool_name}' is not required in this state",
        }

    info = registry.tools[tool_name]
    function = info.get("function")
    if not callable(function):
        return {"success": False, "error": f"Tool '{tool_name}' has no callable function"}

    schema = _find_schema(tool_name, registry)
    parameters_schema = (
        schema.get("function", {}).get("parameters", {})
        if isinstance(schema, dict)
        else {}
    )
    if not isinstance(parameters_schema, dict):
        parameters_schema = {}

    tool_description = str(info.get("description") or inspect.getdoc(function) or tool_name)
    current_user_text = str(
        user_text if user_text is not None else getattr(cache, "current_user_msg", "")
    )
    state_snapshot = context_state if isinstance(context_state, dict) else {}

    prompt_data = {
        "tool_name": tool_name,
        "tool_description_and_rules": tool_description,
        "parameter_schema": parameters_schema,
        "current_user_message": current_user_text,
        "active_runtime_state": state_snapshot,
        "previous_failure": str(failure_reason or "The primary model did not successfully execute this required tool."),
    }

    system_prompt = (
        "You are LAPAI's SUM fallback for a required tool. "
        "Your only task is to produce the keyword arguments for the named Python tool.\n"
        "Return exactly one valid JSON object and nothing else. The object must contain "
        "only parameters accepted by the tool.\n"
        "Follow the tool description, its explicit rules, and its parameter schema.\n"
        "Use the current user message and provided runtime state as evidence. Do not invent "
        "missing facts, IDs, paths, amounts, or other values. If a required parameter cannot "
        "be safely determined, return this JSON instead: "
        '{"_fallback_error":"insufficient_information","_missing":["parameter_name"]}.\n'
        "Do not execute the tool, explain your answer, or return a natural-language response."
    )

    try:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False, default=str)},
            ],
            temperature=0,
        )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return {"success": False, "error": "SUM fallback returned no choices"}
        raw_content = getattr(choices[0].message, "content", "") or ""
    except Exception as exc:
        print(f"[SUM REQUIRE FALLBACK] model request failed for {tool_name}: {exc}")
        return {"success": False, "error": f"SUM fallback model request failed: {exc}"}

    arguments, parse_error = _parse_arguments_object(raw_content)
    if parse_error:
        print(f"[SUM REQUIRE FALLBACK] invalid arguments for {tool_name}: {parse_error}")
        return {"success": False, "error": parse_error}

    if "_fallback_error" in arguments:
        reason = str(arguments.get("_fallback_error") or "SUM reported insufficient information")
        missing = arguments.get("_missing") or []
        suffix = f"; missing={missing}" if missing else ""
        return {"success": False, "error": f"SUM fallback declined '{tool_name}': {reason}{suffix}"}

    validation_error = _validate_arguments(function, arguments)
    if validation_error:
        return {"success": False, "error": f"SUM fallback produced invalid arguments for '{tool_name}': {validation_error}"}

    print(
        "[SUM REQUIRE FALLBACK] generated arguments",
        "tool=", tool_name,
        "keys=", sorted(arguments),
    )

    try:
        result = runner.execute(
            tool_name,
            arguments,
            context_state=context_state if isinstance(context_state, dict) else {},
        )
    except Exception as exc:
        return {"success": False, "error": f"SUM fallback execution failed for '{tool_name}': {exc}"}

    if isinstance(result, dict):
        result = dict(result)
        if result.get("success") is False:
            return result
        result.setdefault("success", True)
        result["_sum_fallback_used"] = True
        result["_fallback_arguments"] = arguments
        return result

    if isinstance(result, str):
        return {
            "success": True,
            "_text": result,
            "_sum_fallback_used": True,
            "_fallback_arguments": arguments,
        }

    return {
        "success": True,
        "result": result,
        "_sum_fallback_used": True,
        "_fallback_arguments": arguments,
    }


def _find_schema(tool_name: str, registry: Any) -> dict[str, Any] | None:
    for source in (getattr(cache, "tool_schemas", None),getattr(registry, "schemas", None),):
        for schema in source or []:
            if not isinstance(schema, dict):
                continue
            function = schema.get("function")
            if isinstance(function, dict) and function.get("name") == tool_name:
                return schema

    info = getattr(registry, "tools", {}).get(tool_name, {})
    function = info.get("function") if isinstance(info, dict) else None
    if not callable(function):
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name in {"self", "cls"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        properties[name] = {"description": str(parameter.annotation)} if parameter.annotation is not inspect.Parameter.empty else {}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": str(info.get("description") or tool_name),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _parse_arguments_object(raw: str) -> tuple[dict[str, Any], str | None]:
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if set(value) <= {"arguments"} and isinstance(value.get("arguments"), dict):
            value = value["arguments"]
        return value, None

    return {}, "SUM fallback did not return a valid JSON object"


def _validate_arguments(function: Any, arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    if any(not isinstance(key, str) for key in arguments):
        return "all argument names must be strings"

    try:
        signature = inspect.signature(function)
        signature.bind(**arguments)
        for name, parameter in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            if (
                parameter.default is inspect.Parameter.empty
                and name in arguments
                and arguments[name] is None
            ):
                return f"required parameter '{name}' cannot be null"
    except TypeError as exc:
        return str(exc)
    except (ValueError, AttributeError) as exc:
        return f"unable to inspect tool signature: {exc}"

    return None
