from .MainRuntime import *
#Not tested

def Main_Core_Function(user_msg,session_id,session_file,seid,jsfile):
    cache.current_user_msg = str(user_msg or "")
    cache.crntTime = datetime.now().astimezone().isoformat()

    try:
        active_memory = prepare_active_memory(
            session_id=session_id,
            user_input=cache.current_user_msg,
        )
    except Exception as exc:
        print(f"[WARNING] Active-memory preparation failed: {exc}")
        active_memory = []

    try:
        thought = thoughtm(cache.current_user_msg)
        collected_knowledge = recall_knowledge(thought)
    except Exception as exc:
        print(f"[WARNING] Knowledge/thought stage failed; continuing without it: {exc}")
        collected_knowledge = []

    try:
        prompt = build_memory_prompt(
            user_msg=cache.current_user_msg,
            summary="",
            active_memory=active_memory,
            recalled_memory=[],
            knowledge=collected_knowledge,
        )
    except Exception as exc:
        print(f"[WARNING] Memory prompt construction failed; using bare user prompt: {exc}")
        prompt = [{"role": "user", "content": cache.current_user_msg}]

    try:
        persona = load_persona()
        if persona:
            prompt.insert(0, {"role": "system", "content": persona})
    except Exception as exc:
        print(f"[WARNING] Persona loading failed; continuing without persona: {exc}")

    append_message(session_id, session_file, "user", cache.current_user_msg)

    try:
        reply = run_agent_turn(prompt, current_user=cache.current_user_msg)
    except Exception as exc:
        print(f"[ERROR] Agent turn failed unexpectedly: {exc}")
        reply = safe_final_model_call(prompt)

    if not isinstance(reply, FailedReply):
            append_message(session_id, session_file, "assistant", reply)
            try:
                record_active_turn(
                    session_id=session_id,
                    user_content=cache.current_user_msg,
                    assistant_content=reply,
                    created_at=cache.crntTime,
                )
            except Exception as exc:
                print(f"[WARNING] Failed to record active memory: {exc}")

    return reply

