from .toolsState import *
def _sqlite_timestamp_to_epoch(timestamp_text):
    if not timestamp_text:
        return time.time()

    try:
        dt = datetime.strptime(
            timestamp_text,
            "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

        return dt.timestamp()
    except (TypeError, ValueError):
        return time.time()

def decsn(user_msg):
    completion = cache.client.chat.completions.create(
        model=cache.Sum_model,
        messages=[{"role":"system", "content": f"just Answer Yes or No. Is there any Personal Information in this input text: {user_msg}"}]
    )
    Decision = parse_yesno(completion.choices[0].message.content)
    if "Yes" in Decision:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[{"role":"system", "content":f"just Answer short as possible, take the personal info from this input text: {user_msg}"}]
        )
        reject = ["No","no","No.","no.","NO","NO."]
        if any(q in completion.choices[0].message.content for q in reject):
            pass
        else:
            append_txt(completion.choices[0].message.content)
def append_txt(items):
    data_extract = f"[UserDataInformation: {items}]"

    with open(
        cache.persnoal_file,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(data_extract + "\n")

    try:
        add_personal_memory(data_extract)
    except Exception as e:
        print(f"[WARNING] Failed to index personal memory: {e}")
def read_txt():

    with open(cache.persnoal_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
def append_message(session_id, json_file, role, content):
    history = []
    cache.faiss_index, cache.id_map
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append({"role": role, "content": content})
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    conn = sqlite3.connect(cache.DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, datetime('now'))", (session_id, role, content))
    conn.commit()
    conn.close()

    #Extended FAISS Function

    importance = 1.0
    meta_type = "conversation"

    if role == "summary":
        add_to_faiss(
            cache.faiss_index,
            cache.id_map,
            content,
            f"{session_id}:{role}:{len(content)}",
            meta_type="summary",
            importance=compute_importance(content) * 1.5
        )

    add_to_faiss(
        cache.faiss_index,
        cache.id_map,
        content,
        f"{session_id}:{role}:{len(content)}",
        meta_type=meta_type,
        importance=compute_importance(content)
    )
def recall_relevant_memory(user_input, limit=10, threshold=cache.conf.get('thresholdrrm')):
    """Recall only validated/classified long-term memories."""
    return recall_long_term_memory(
        user_input=user_input,
        limit=limit,
        threshold=threshold,
    )

def initialize_core():
    init_db()
    init_memory_state()
    init_memory_store()
    init_learning_db()
    title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
    title_learn = "Learning"+title_hint
    cache.seid, cache.jsfile = create_session_Learning(title_learn)
    cache.faiss_index, cache.id_map = init_faiss()
    cache.session_id, cache.session_file = create_session(title_hint)
    initialize_tools()


def run_agent_turn(messages, top_k=5, threshold=None, max_tool_iters=6, current_user=None):
    """Run the model/tool loop with generic tool lifecycle dependencies."""
    if cache.tool_registry is None or cache.tool_ranker is None:
        initialize_tools()

    if current_user is None:
        current_user = getattr(cache, "current_user_msg", "")

    current_user = str(current_user or "")
    cache.current_user_msg = current_user
    cache.crntTime = datetime.now().astimezone().isoformat()

    candidates = []
    if current_user:
        candidates = cache.tool_ranker.search(
            current_user,
            top_k=top_k,
            threshold=threshold,
        )

    schema_map = {
        schema["function"]["name"]: schema
        for schema in cache.tool_schemas
    }

    # Per-turn state. It is deliberately local so state never leaks into the
    # next user turn. cache.agent_state is only a debug snapshot.
    context_state: dict[str, object] = {}
    cache.agent_state = {}

    local_runner = ToolRunner(cache.tool_registry)
    completed_required: set[str] = set()

    for iteration in range(max(1, int(max_tool_iters))):
        eligible_names = get_eligible_tool_names(
            cache.tool_registry,
            context_state,
        )
        required_names = get_required_tool_names(
            cache.tool_registry,
            context_state,
        )
        missing_required = required_names - completed_required

        candidate_schemas = get_candidate_schemas(
            candidates,
            allowed_names=eligible_names,
        )

        included_names = {
            schema["function"]["name"]
            for schema in candidate_schemas
        }

        # Required tools bypass semantic ranking while their contract is active.
        for name in required_names:
            schema = schema_map.get(name)
            if (
                schema is not None
                and name in eligible_names
                and name not in included_names
            ):
                candidate_schemas.append(schema)
                included_names.add(name)

        tools = candidate_schemas or None

        request_kwargs = {
            "model": cache.model_name,
            "messages": messages,
        }

        # Zero-argument required tools can be executed deterministically by
        # the orchestrator. This removes dependence on the model honoring
        # tool_choice for dependency-only tools such as recallmemory().
        auto_required_name = None
        for required_name in sorted(missing_required):
            if required_name not in eligible_names:
                continue
            info = cache.tool_registry.tools.get(required_name)
            if info is None:
                continue
            signature = inspect.signature(info["function"])
            has_required_argument = False
            for parameter in signature.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    if parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
                        continue
                if (
                    parameter.default is inspect.Parameter.empty
                    and parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ):
                    has_required_argument = True
                    break
            if not has_required_argument:
                auto_required_name = required_name
                break

        if auto_required_name is not None:
            auto_id = f"required_auto_{iteration}_{auto_required_name}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": auto_id,
                    "type": "function",
                    "function": {
                        "name": auto_required_name,
                        "arguments": "{}",
                    },
                }],
            })

            try:
                result = local_runner.execute(
                    auto_required_name,
                    {},
                    context_state=context_state,
                )

                succeeded = result.get("success", True) is not False if isinstance(result, dict) else True

                if isinstance(result, dict):
                    provided_state = result.get("_state")
                    if isinstance(provided_state, dict):
                        declared_state = cache.tool_registry.tools[auto_required_name].get(
                            "provides_state",
                            [],
                        )
                        for state_name in declared_state:
                            if state_name in provided_state:
                                context_state[state_name] = provided_state[state_name]
                                cache.agent_state[state_name] = provided_state[state_name]

                if succeeded:
                    completed_required.add(auto_required_name)

                result_for_model = result
                if isinstance(result, dict) and "_state" in result:
                    result_for_model = dict(result)
                    result_for_model.pop("_state", None)

                result_text = (
                    result_for_model
                    if isinstance(result_for_model, str)
                    else json.dumps(result_for_model, ensure_ascii=False, default=str)
                )
            except Exception as exc:
                result_text = f"[ERROR] Tool '{auto_required_name}' failed: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": auto_id,
                "name": auto_required_name,
                "content": result_text,
            })
            continue

        if tools:
            # When required tools are missing, expose only those missing
            # required tools and use generic `required` selection. This avoids
            # relying on named tool_choice, which some local runtimes ignore.
            if missing_required:
                forced_schemas = []
                for required_name in sorted(missing_required):
                    schema = schema_map.get(required_name)
                    if schema is not None and required_name in eligible_names:
                        forced_schemas.append(schema)

                if forced_schemas:
                    request_kwargs["tools"] = forced_schemas
                    request_kwargs["tool_choice"] = "required"
                else:
                    request_kwargs["tools"] = tools
                    request_kwargs["tool_choice"] = "auto"
            else:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"

        completion = cache.client.chat.completions.create(**request_kwargs)
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            if missing_required:
                # Do not add an extra prompt message. The next iteration will
                # force the missing requirement through tool_choice.
                continue

            return message.content or ""

        messages.append({
            "role": "assistant",
            "content": message.content or "",
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

        for tc in tool_calls:
            name = tc.function.name

            try:
                result = local_runner.execute(
                    name,
                    tc.function.arguments,
                    context_state=context_state,
                )

                if isinstance(result, dict):
                    provided_state = result.get("_state")
                    if isinstance(provided_state, dict):
                        declared_state = cache.tool_registry.tools[name].get(
                            "provides_state",
                            [],
                        )
                        for state_name in declared_state:
                            if state_name in provided_state:
                                context_state[state_name] = provided_state[state_name]
                                cache.agent_state[state_name] = provided_state[state_name]

                    succeeded = result.get("success", True) is not False
                else:
                    succeeded = True

                if name in required_names and succeeded:
                    completed_required.add(name)

                result_for_model = result
                if isinstance(result, dict) and "_state" in result:
                    result_for_model = dict(result)
                    result_for_model.pop("_state", None)

                result_text = (
                    result_for_model
                    if isinstance(result_for_model, str)
                    else json.dumps(
                        result_for_model,
                        ensure_ascii=False,
                        default=str,
                    )
                )
            except Exception as e:
                result_text = f"[ERROR] Tool '{name}' failed: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": result_text,
            })

    # Never silently bypass an active mandatory tool contract.
    eligible_names = get_eligible_tool_names(
        cache.tool_registry,
        context_state,
    )
    required_names = get_required_tool_names(
        cache.tool_registry,
        context_state,
    )
    missing_required = required_names - completed_required

    if missing_required:
        return (
            "[ERROR] Required tool chain was not completed: "
            + ", ".join(sorted(missing_required))
        )

    final_completion = cache.client.chat.completions.create(
        model=cache.model_name,
        messages=messages,
    )
    return final_completion.choices[0].message.content or ""

def run_agent(user_message,model_call,registry=None,top_k=5,threshold=None,): #<-- prototype Testing
    if registry is None:
        registry = (cache.tool_registry or initialize_tools())

    if cache.tool_ranker is None:
        initialize_tools()
        registry = cache.tool_registry

    candidates = cache.tool_ranker.search(
        user_message,
        top_k=top_k,
        threshold=threshold,
    )

    candidate_schemas = (
        get_candidate_schemas(
            candidates
        )
    )

    first_response = model_call(
        user_message=user_message,
        tools=candidate_schemas,
    )

    if isinstance(first_response, str):

        try:
            decision = json.loads(
                first_response
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Model did not return valid "
                f"tool decision JSON: {first_response!r}"
            ) from exc

    else:

        decision = first_response

    if decision.get("action") == "answer":

        return decision.get(
            "content",
            ""
        )

    if decision.get("action") != "tool":

        raise ValueError(
            "Unknown model action: "
            f"{decision.get('action')!r}"
        )

    selected_tool = decision.get(
        "tool"
    )

    arguments = decision.get(
        "arguments",
        {}
    )

    if not selected_tool:

        raise ValueError(
            "Model selected a tool action "
            "without specifying a tool."
        )

    if not isinstance(arguments, dict):

        raise TypeError(
            "Tool arguments must be a dictionary."
        )

    candidate_names = {
        candidate["tool"]
        for candidate in candidates
    }

    if selected_tool not in candidate_names:

        raise ValueError(
            f"Model selected tool "
            f"'{selected_tool}' which was "
            "not present in ranked candidates."
        )

    runner = ToolRunner(
        registry
    )

    tool_result = runner.execute(
        selected_tool,
        arguments,
    )
    return model_call(
        user_message=user_message,
        tool_name=selected_tool,
        tool_result=tool_result,
    )

def run_ranked_tool(query,arguments=None,top_k=5,threshold=None):
    if cache.tool_ranker is None:
        initialize_tools()

    candidates = cache.tool_ranker.search(
        query,
        top_k=top_k,
        threshold=threshold
    )

    if not candidates:
        return None

    selected = candidates[0]

    runner = ToolRunner(cache.tool_registry)

    return runner.execute(
        selected["tool"],
        arguments or {}
    )
def get_candidate_schemas(candidates, allowed_names=None):
    schema_map = {
        schema["function"]["name"]: schema
        for schema in cache.tool_schemas
    }

    result = []

    for candidate in candidates:
        tool_name = candidate["tool"]

        if allowed_names is not None and tool_name not in allowed_names:
            continue

        schema = schema_map.get(tool_name)

        if schema is not None:
            result.append(schema)

    return result
