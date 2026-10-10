from .statecore import *
def run_agent_turn(messages,top_k=cache.conf.get('Tool_Topk'),threshold=cache.conf.get('ToolThrCalling'),max_tool_iters=6,current_user=None,):
    """Run the tool loop with strict fault isolation and a non-blocking finish."""
    if current_user is None:
        current_user = getattr(cache, "current_user_msg", "")
    current_user = str(current_user or "")
    cache.current_user_msg = current_user
    cache.crntTime = datetime.now().astimezone().isoformat()
    cache.agent_state = {}
    try:
        if cache.tool_registry is None or cache.tool_ranker is None:
            initialize_tools()
    except Exception as exc:
        print(f"[WARNING] Tool system initialization failed: {exc}")

    registry = getattr(cache, "tool_registry", None)
    local_runner = ToolRunner(registry)
    context_state: dict[str, object] = {}
    completed_required: set[str] = set()
    fail_counts: dict[str, int] = {}
    max_iters = max(1, int(max_tool_iters))
    max_fallback_failures = max(1, int(cache.conf.get("memoryCommitMaxFailures") or 2))

    if registry is None:
        for required_name in sorted(_discover_fallbacks()):
            result = _safe_fallback(required_name, context_state, None)
            messages.append({
                "role": "tool",
                "tool_call_id": f"runtime_fallback_{required_name}",
                "name": required_name,
                "content": _tool_result_text({k: v for k, v in result.items() if k not in {"_state"}})
                if isinstance(result, dict) else _tool_result_text(result),
            })
        return safe_final_model_call(messages)

    candidates = []
    try:
        if current_user and cache.tool_ranker is not None:
            candidates = cache.tool_ranker.search(
                current_user,
                top_k=top_k,
                threshold=threshold,
            ) or []
    except Exception as exc:
        print(f"[WARNING] Tool ranking failed; required tools will still run: {exc}")
        candidates = []

    if candidates:
        try:
            candidates = reranka(
                query=current_user,
                candidates=[
                    {
                        **candidate,
                        "content": registry.tools[candidate["tool"]]["description"],
                    }
                    for candidate in candidates
                    if candidate.get("tool") in registry.tools
                ],
                top_k=5,
                text_key="content",
            ) or []
        except Exception as exc:
            print(f"[WARNING] Tool reranking failed; using ranker order: {exc}")

    schema_map = {
        schema["function"]["name"]: schema
        for schema in getattr(cache, "tool_schemas", [])
        if isinstance(schema, dict) and "function" in schema
    }

    for iteration in range(max_iters):
        try:
            eligible_names = get_eligible_tool_names(registry, context_state)
            required_names = get_required_tool_names(registry, context_state)
        except Exception as exc:
            print(f"[WARNING] Tool-state evaluation failed: {exc}")
            required_names = _required_each_turn_names(registry)
            eligible_names = set(registry.tools)

        missing_required = required_names - completed_required
        print(
            "[TOOL LOOP]",
            "iteration=", iteration,
            "max_tool_iters=", max_iters,
            "completed=", completed_required,
            "missing=", missing_required,
        )

        if not missing_required:
            break

        next_required = sorted(missing_required,key=_required_order_key(registry),)[0]

        if fail_counts.get(next_required, 0) >= max_fallback_failures:
            fb_result = _run_required_fallback(
                next_required,
                registry,
                local_runner,
                context_state,
                current_user,
                failure_reason="Primary model did not complete the required tool within the retry limit.",
            )
            fb_success = _tool_execution_succeeded(fb_result)
            if fb_success:
                completed_required.add(next_required)
            else:
                fail_counts[next_required] = fail_counts.get(next_required, 0) + 1

            fb_id = f"fallback_{iteration}_{next_required}"
            payload = dict(fb_result) if isinstance(fb_result, dict) else {"result": fb_result}
            fb_args = payload.pop("_fallback_arguments", {})
            payload.pop("_sum_fallback_used", None)
            payload.pop("_state", None)
            payload.pop("success", None)
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": fb_id,
                    "type": "function",
                    "function": {"name": next_required, "arguments": json.dumps(fb_args, ensure_ascii=False, default=str),},
                }],
            })
            payload = dict(fb_result) if isinstance(fb_result, dict) else {"result": fb_result}
            payload.pop("_state", None)
            messages.append({
                "role": "tool",
                "tool_call_id": fb_id,
                "name": next_required,
                "content": _tool_message_content(payload),
            })
            if fb_success:
                fail_counts[next_required] = 0
            continue

        try:
            candidate_schemas = get_candidate_schemas(candidates, allowed_names=eligible_names)
        except Exception as exc:
            print(f"[WARNING] Candidate schema build failed: {exc}")
            candidate_schemas = []

        included = {
            schema["function"]["name"]
            for schema in candidate_schemas
            if isinstance(schema, dict) and "function" in schema
        }
        for name in required_names:
            schema = schema_map.get(name)
            if schema is not None and name in eligible_names and name not in included:
                candidate_schemas.append(schema)
                included.add(name)

        request_kwargs = {
            "model": cache.model_name,
            "messages": build_messages(messages),
        }
        if candidate_schemas:
            forced = schema_map.get(next_required)
            if forced is not None and next_required in eligible_names:
                request_kwargs["tools"] = [forced]
                request_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": next_required},
                }
            else:
                request_kwargs["tools"] = candidate_schemas

        try:
            completion = cache.client.chat.completions.create(**request_kwargs)
            choices = getattr(completion, "choices", None) or []
            if not choices:
                raise RuntimeError("model returned no choices")
            message = choices[0].message
        except Exception as exc:
            print(f"[WARNING] Tool-call model request failed: {exc}")
            fail_counts[next_required] = fail_counts.get(next_required, 0) + 1
            continue

        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            print(
                "[NO-TOOL DEBUG]",
                "finish=", getattr(choices[0], "finish_reason", None),
                "forced=", request_kwargs.get("tool_choice"),
                "content=", repr(getattr(message, "content", ""))[:300],
            )
            fail_counts[next_required] = fail_counts.get(next_required, 0) + 1
            continue

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        completed_before = set(completed_required)
        for tc in tool_calls:
            name = getattr(tc.function, "name", "")
            result = local_runner.execute(
                name,
                getattr(tc.function, "arguments", "{}"),
                context_state=context_state,
            )
            _apply_tool_state(name, result, registry, context_state)

            succeeded = _tool_execution_succeeded(result)
            if name in required_names and succeeded:
                completed_required.add(name)
                fail_counts[name] = 0

            payload = dict(result) if isinstance(result, dict) else result
            if isinstance(payload, dict):
                payload.pop("_state", None)
                payload.pop("success", None)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": _tool_message_content(payload),
            })

        if completed_required == completed_before:
            fail_counts[next_required] = fail_counts.get(next_required, 0) + 1
    try:
        required_names = get_required_tool_names(registry, context_state)
    except Exception:
        required_names = _required_each_turn_names(registry)
    missing_required = required_names - completed_required

    for name in sorted(missing_required, key=_required_order_key(registry)):
        result = _run_required_fallback(
            name,
            registry,
            local_runner,
            context_state,
            current_user,
            failure_reason="Required tool remained incomplete after the tool loop ended.",
        )
        fallback_id = f"final_fallback_{name}"
        payload = dict(result) if isinstance(result, dict) else {"result": result}
        fallback_args = payload.pop("_fallback_arguments", {})
        sum_fallback_used = bool(payload.pop("_sum_fallback_used", False))
        succeeded = _tool_execution_succeeded(result)
        if succeeded:
            completed_required.add(name)
        if sum_fallback_used:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": fallback_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(fallback_args, ensure_ascii=False, default=str),
                    },
                }],
            })
        if isinstance(payload, dict):
            payload.pop("_state", None)
            payload.pop("success", None)
        messages.append({
            "role": "tool",
            "tool_call_id": fallback_id,
            "name": name,
            "content": _tool_message_content(payload),
        })
    
    try:
        required_now = get_required_tool_names(
            registry, context_state
        )
        eligible_now = get_eligible_tool_names(
            registry, context_state
        )
        optional_names = eligible_now - required_now
        optional_schemas = [
            schema
            for schema in (getattr(cache, "tool_schemas", []) or [])
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and schema["function"].get("name") in optional_names
        ]

        print(
            "[OPTIONAL PASS] unresolved_required =",
            sorted(required_now - completed_required),
        )
        print(
            "[OPTIONAL PASS] offered =",
            [
                schema["function"]["name"]
                for schema in optional_schemas
            ],
        )

        if optional_schemas:
            completion = cache.client.chat.completions.create(
                model=cache.model_name,
                messages=build_messages(messages),
                tools=optional_schemas,
                tool_choice="auto",
            )

            choices = getattr(completion, "choices", None) or []
            if not choices:
                raise RuntimeError(
                    "Optional pass returned no choices"
                )

            message = choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                print(
                    "[OPTIONAL PASS] model did not call a tool;",
                    "content =",
                    repr(getattr(message, "content", ""))[:300],
                )
            else:
                valid_calls = [
                    tc for tc in tool_calls
                    if getattr(tc.function, "name", "")
                    in optional_names
                ]

                if valid_calls:
                    messages.append({
                        "role": "assistant",
                        "content": getattr(message, "content", None) or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": (
                                        getattr(tc.function, "arguments", None)
                                        or "{}"
                                    ),
                                },
                            }
                            for tc in valid_calls
                        ],
                    })

                    for tc in valid_calls:
                        name = tc.function.name
                        result = local_runner.execute(
                            name,
                            getattr(tc.function, "arguments", None) or "{}",
                            context_state=context_state,
                        )

                        _apply_tool_state(
                            name,
                            result,
                            registry,
                            context_state,
                        )

                        payload = (
                            dict(result)
                            if isinstance(result, dict)
                            else result
                        )
                        if isinstance(payload, dict):
                            payload.pop("_state", None)
                            payload.pop("success", None)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": _tool_message_content(payload),
                        })

                        print(
                            "[OPTIONAL TOOL EXECUTED]",
                            name,
                            "result =",
                            repr(payload)[:300],
                        )

    except Exception as exc:
        print(
            f"[WARNING] Optional tool pass failed: {exc}"
        )

    return safe_final_model_call(messages)
def _tool_execution_succeeded(result):
    """A tool succeeded unless it explicitly returned success=False."""
    if isinstance(result, dict):
        return result.get("success", True) is not False
    return result is not None


def _run_required_fallback(tool_name, registry, runner, context_state, current_user, failure_reason=""):
    result = run_required_tool_fallback(tool_name=tool_name,registry=registry,runner=runner,context_state=context_state,user_text=current_user,failure_reason=failure_reason,)
    if _tool_execution_succeeded(result):
        _apply_tool_state(tool_name, result, registry, context_state)
        print("[REQUIRED FALLBACK] SUM fallback succeeded for", tool_name)
        return result

    if isinstance(result, dict):
        print(
            "[REQUIRED FALLBACK] SUM fallback unavailable for",
            tool_name,
            "reason=",
            result.get("error", "unknown error"),
        )

    legacy_result = _safe_fallback(tool_name, context_state, registry)
    if _tool_execution_succeeded(legacy_result):
        print("[REQUIRED FALLBACK] legacy fallback succeeded for", tool_name)
        return legacy_result

    # Preserve the most informative failure if neither fallback worked.
    if isinstance(legacy_result, dict) and legacy_result.get("error"):
        legacy_result.setdefault("sum_fallback_error", result.get("error") if isinstance(result, dict) else "SUM fallback failed")
        return legacy_result
    return result


def _discover_fallbacks():
    from .core import mfallcom
    return {
        attr[:-4]: fn
        for attr, fn in vars(mfallcom).items()
        if len(attr) > 4
        and attr.endswith("Fall")
        and inspect.isfunction(fn)
        and fn.__module__ == mfallcom.__name__
    }
def _safe_fallback(tool_name, context_state, registry):
    fallback = _discover_fallbacks().get(tool_name)
    if fallback is None:
        return {"success": False, "error": f"No fallback registered for {tool_name}"}
    try:
        result = fallback()
    except Exception as exc:
        result = {"success": False, "error": f"Fallback '{tool_name}' failed: {exc}"}
    if isinstance(result, dict):
        _apply_tool_state(tool_name, result, registry, context_state)
    return result
def _required_each_turn_names(registry):
    return {name for name, info in registry.tools.items() if info.get("required_each_turn")}


def _required_order_key(registry):
    tools = registry.tools if registry is not None else {}
    return lambda name: (not tools.get(name, {}).get("required_each_turn", False), name)
def _apply_tool_state(tool_name, result, registry, context_state):
    if not isinstance(result, dict) or result.get("success") is False:
        return
    provided_state = result.get("_state")
    if not isinstance(provided_state, dict) or registry is None:
        return
    info = registry.tools.get(tool_name, {})
    declared_state = info.get("provides_state", [])
    for state_name in declared_state:
        if state_name in provided_state:
            context_state[state_name] = provided_state[state_name]
            cache.agent_state[state_name] = provided_state[state_name]
def _tool_message_content(payload):
    if isinstance(payload, dict):
        payload = dict(payload)
        text = payload.pop("_text", None)
        if isinstance(text, str) and text.strip():
            return text
    return _tool_result_text(payload)

def _tool_result_text(result):
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


