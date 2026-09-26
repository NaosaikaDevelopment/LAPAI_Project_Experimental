from .statecore import *
#Not tested

def Main_Core_Function(user_msg,session_id,session_file,seid,jsfile):
    cache.current_user_msg = user_msg
    cache.crntTime = datetime.now().astimezone().isoformat()

    # Active-memory management happens before the model sees the turn.
    # It removes off-topic active turns into `sum` memory and enforces the
    # active token budget.
    active_memory = prepare_active_memory(
        session_id=session_id,
        user_input=user_msg,
    )

    thought = thoughtm(user_msg)
    collected_knowledge = recall_knowledge(thought)

    # Historical conversation is recalled by the automatic memory tool chain
    # (summary -> detailed ranking) only when the current input matches a
    # stored summary. Keeping it out of the initial prompt avoids duplicate
    # context and lets the orchestrator decide when detailed recall is needed.
    prompt = build_memory_prompt(
        user_msg=user_msg,
        summary="",
        active_memory=active_memory,
        recalled_memory=[],
        knowledge=collected_knowledge,
    )

    persona = load_persona()

    if persona:
        prompt.insert(0, {
            "role": "system",
            "content": persona
        })

    append_message(
        session_id,
        session_file,
        "user",
        user_msg
    )

    try:
        cache.current_user_msg = user_msg
        cache.crntTime = datetime.now().astimezone().isoformat()
        reply = run_agent_turn(prompt, current_user=user_msg)

    except Exception as e:
        print(
            "[ERROR] When Main Model loading:",
            str(e)
        )
        return f"[ERROR] When Main Model loading: {e}"

    append_message(
        session_id,
        session_file,
        "assistant",
        reply
    )

    # Completed turns become active memory for the next turn.
    try:
        record_active_turn(
            session_id=session_id,
            user_content=user_msg,
            assistant_content=reply,
            created_at=cache.crntTime,
        )
    except Exception as e:
        print(f"[WARNING] Failed to record active memory: {e}")

    return reply
