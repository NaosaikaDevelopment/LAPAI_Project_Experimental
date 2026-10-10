from .toolsState import *

def decsn(user_msg):
    """Legacy personal-data detector; model/storage failures are non-fatal."""
    try:
        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[{"role": "system", "content": f"just Answer Yes or No. Is there any Personal Information in this input text: {user_msg}"}]
        )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return False
        decision = parse_yesno(getattr(choices[0].message, "content", ""))
        if "Yes" not in decision:
            return False

        completion = cache.client.chat.completions.create(
            model=cache.Sum_model,
            messages=[{"role":"system", "content": f"just Answer short as possible, take the personal info from this input text: {user_msg}"}]
        )
        choices = getattr(completion, "choices", None) or []
        content = getattr(choices[0].message, "content", "") if choices else ""
        if any(q in content for q in ("No", "no", "NO")):
            return False
        append_txt(content)
        return True
    except Exception as exc:
        print(f"[WARNING] Personal-data detection failed; continuing: {exc}")
        return False

def append_txt(items):
    """Best-effort personal-memory persistence."""
    data_extract = f"[UserDataInformation: {items}]"
    try:
        with open(cache.persnoal_file, "a", encoding="utf-8") as f:
            f.write(data_extract + "\n")
    except Exception as exc:
        print(f"[WARNING] Personal memory file write failed: {exc}")
        return False

    try:
        add_personal_memory(data_extract)
    except Exception as exc:
        print(f"[WARNING] Failed to index personal memory: {exc}")
    return True

def read_txt():

    with open(cache.persnoal_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
def append_message(session_id, json_file, role, content):
    """Best-effort persistence; logging/indexing failures never stop inference."""
    history = []
    try:
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                history = loaded if isinstance(loaded, list) else []
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARNING] Chat history read failed; starting fresh log: {exc}")
        history.append({"role": role, "content": content})
        tmp_file = f"{json_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, json_file)
    except Exception as exc:
        print(f"[WARNING] Chat JSON persistence failed: {exc}")

    row_id = None
    try:
        conn = sqlite3.connect(cache.DB_FILE, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        row_id = int(cursor.lastrowid)
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[WARNING] Message DB persistence failed: {exc}")
        try:
            conn.close()
        except Exception:
            pass
        return None
    try:
        if getattr(cache, "faiss_index", None) is not None and role in {"user", "assistant", "summary"}:
            meta_type = "summary" if role == "summary" else "conversation"
            add_to_faiss(
                cache.faiss_index,
                cache.id_map,
                str(content),
                f"{session_id}:{role}:{row_id}",
                meta_type=meta_type,
                importance=compute_importance(str(content)) * (1.5 if role == "summary" else 1.0),
                session_id=session_id,
                memory_id=row_id,
            )
    except Exception as exc:
        print(f"[WARNING] Message semantic index failed; DB row kept: {exc}")

    return row_id

def recall_relevant_memory(user_input, limit=10, threshold=cache.conf.get('thresholdrrm')):
    return recall_long_term_memory(
        user_input=user_input,
        limit=limit,
        threshold=threshold,
    )

def initialize_core():
    """Initialize subsystems independently so one broken persistence layer does not stop the runtime."""
    steps = (
        ("database", init_db),
        ("memory_state", init_memory_state),
        ("memory_store", init_memory_store),
        ("conversation_memory", init_conversation_memory),
        ("legacy_summary_migration", migrate_legacy_summaries),
        ("learning_db", init_learning_db),
    )
    for name, function in steps:
        try:
            result = function()
            print(f"[INIT] {name}: OK" + (f" ({result})" if result is not None else ""))
        except Exception as exc:
            print(f"[WARNING] {name} initialization failed; continuing: {exc}")

    title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
    try:
        title_learn = "Learning" + title_hint
        cache.seid, cache.jsfile = create_session_Learning(title_learn)
    except Exception as exc:
        print(f"[WARNING] Learning session creation failed: {exc}")

    try:
        cache.faiss_index, cache.id_map = init_faiss()
    except Exception as exc:
        print(f"[WARNING] FAISS initialization failed; semantic acceleration disabled: {exc}")
        cache.faiss_index = None
        cache.id_map = {}

    try:
        cache.session_id, cache.session_file = create_session(title_hint)
    except Exception as exc:
        print(f"[WARNING] Chat session creation failed: {exc}")
        cache.session_id = None
        cache.session_file = str(Path(cache.CHAT_DIR) / f"{title_hint}.json")
        try:
            os.makedirs(cache.CHAT_DIR, exist_ok=True)
        except Exception:
            pass

    try:
        initialize_tools()
    except Exception as exc:
        print(f"[WARNING] Tool initialization failed; runtime can continue with fallbacks: {exc}")

    return cache.session_id


def build_messages(messages):
    try:
        prepared = list(get_prompt_manager().prepare_for_model(messages))
    except Exception as exc:
        print(f"[WARNING] Prompt preparation failed; using untrimmed messages: {exc}")
        prepared = [dict(message) for message in messages if isinstance(message, dict)]

    persona = getattr(cache, "persona", [])
    if not isinstance(persona, list):
        persona = []
    out = list(persona) + prepared
    if out:
        print("[DEBUG] persona_len =", len(persona), "| first_role =", out[0].get("role"))
    return out
class FailedReply(str):
    """Dummy."""

def _sanitize_final_messages(messages):
    items = [m for m in (messages or []) if isinstance(m, dict)]
    registry = getattr(cache, "tool_registry", None)
    tools = registry.tools if registry is not None else {}

    cleaned = []
    notes = []
    for message in items:
        role = message.get("role")

        if role == "tool":
            name = message.get("name") or ""
            content = str(message.get("content") or "").strip()
            if content and not tools.get(name, {}).get("required_each_turn"):
                notes.append(f"- {name or 'tool'}: {content}")
            continue

        item = dict(message)
        item.pop("tool_calls", None)
        item.pop("tool_call_id", None)
        if role == "assistant" and not (item.get("content") or "").strip():
            continue

        cleaned.append(item)

    if notes:
        for item in reversed(cleaned):
            if item.get("role") == "user":
                item["content"] = (
                    f"{item.get('content') or ''}\n\n[Hasil tool untuk pesan ini]\n" + "\n".join(notes)
                )
                break

    return cleaned

def safe_final_model_call(messages):

    final_messages = _sanitize_final_messages(messages)
    
    try:
        completion = cache.client.chat.completions.create(
            model=cache.model_name,
            messages=build_messages(final_messages),
        )
        choices = getattr(completion, "choices", None) or []
        if choices:
            return getattr(choices[0].message, "content", None) or ""
        raise RuntimeError("final model returned no choices")
    except Exception as exc:
        print(f"[WARNING] Final model call failed: {exc}")
        try:
            completion = cache.client.chat.completions.create(
                model=cache.model_name,
                messages=list(final_messages),
            )
            choices = getattr(completion, "choices", None) or []
            if choices:
                return getattr(choices[0].message, "content", None) or ""
        except Exception as retry_exc:
            print(f"[WARNING] Final model retry failed: {retry_exc}")
        return FailedReply("[ERROR] Main model unavailable; runtime kept alive.")




def run_ranked_tool(query,arguments=None,top_k=5,threshold=None):
    if cache.tool_ranker is None:
        initialize_tools()
    candidates = cache.tool_ranker.search(
        query,
        top_k=top_k,
        threshold=threshold
    )
    candidates = [{**c,"content": cache.tool_registry.tools[c["tool"]]["description"]}for c in candidates]

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