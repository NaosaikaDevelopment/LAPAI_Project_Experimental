from .MainRuntime import *
#Main Function
def Main_Core_FP_Function(user_msg):
    cache.current_user_msg = str(user_msg or "")
    cache.crntTime = datetime.now().astimezone().isoformat()

    try:
        active_memory = prepare_active_memory(
            session_id=cache.session_id,
            user_input=cache.current_user_msg,
        )
    except Exception as exc:
        print(f"[WARNING] Active-memory preparation failed: {exc}")
        active_memory = []

    append_message(cache.session_id, cache.session_file, "user", cache.current_user_msg)

    try:
        cache.msgt = build_memory_prompt(
            user_msg=cache.current_user_msg,
            summary="",
            active_memory=active_memory,
            recalled_memory=[],
            knowledge=[],
        )
    except Exception as exc:
        print(f"[WARNING] Prompt construction failed: {exc}")
        cache.msgt = [{"role": "user", "content": cache.current_user_msg}]

    compacted_summary = None
    try:
        reply = run_agent_turn(cache.msgt, current_user=cache.current_user_msg)
        if not isinstance(reply, FailedReply):
            cache.msgt.append({"role": "assistant", "content": reply})
    except Exception as exc:
        reply = safe_final_model_call(cache.msgt)
        print(f"[ERROR] Core agent turn failed; fallback answer used: {exc}")
    if not isinstance(reply, FailedReply):
        append_message(cache.session_id, cache.session_file, "assistant", reply)
        try:
            record_active_turn(
                session_id=cache.session_id,
                user_content=cache.current_user_msg,
                assistant_content=reply,
                created_at=cache.crntTime,
            )
        except Exception as exc:
            print(f"[WARNING] Failed to record active memory: {exc}")

    if compacted_summary is not None:
        try:
            Learning_T = start_learning(cache.client, cache.model_name, cache.current_user_msg, cache.prompt)
            question = generate_question(cache.client, cache.Sum_model, compacted_summary)
            add_question(question)
            thought = thoughtm(cache.current_user_msg)
            append_Learning(cache.seid, cache.jsfile, "knowledge", Learning_T)
            append_Learning(cache.seid, cache.jsfile, "thought", thought)
        except Exception as exc:
            print(f"[ERROR] Learning pipeline: {exc}")

    return reply

