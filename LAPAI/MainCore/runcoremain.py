from .statecore import *
def Main_Core_Function(user_msg,session_id,session_file,seid,jsfile):
    cache.current_user_msg = user_msg
    cache.crntTime = datetime.now().astimezone().isoformat()
    prompt
    compacted_summary = compact_old_memory(
        cache.client,
        cache.Sum_model,
        session_id,
        minutes=5
    )


    memory_state = get_memory_state(session_id)
    summary = memory_state["summary"]


    active_memory = recall_recent_memory(
        session_id,
        minutes=5
    )

    thought = thoughtm(user_msg)
    collected_knowledge = recall_knowledge(thought)

    if should_recall(user_msg):
        recalled = recall_relevant_memory(
            user_msg,
            limit=10
        )
    else:
        recalled = []

    prompt = build_memory_prompt(
        user_msg=user_msg,
        summary=summary,
        active_memory=active_memory,
        recalled_memory=recalled,
        knowledge=collected_knowledge
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

    if compacted_summary is not None:
        try:
            Learning_T = start_learning(
                cache.client,
                cache.model_name,
                user_msg,
                prompt
            )

            question = generate_question(
                cache.client,
                cache.Sum_model,
                compacted_summary
            )

            add_question(question)

            append_Learning(
                seid,
                jsfile,
                "knowledge",
                Learning_T
            )

            append_Learning(
                seid,
                jsfile,
                "thought",
                thought
            )

        except Exception as e:
            print(
                f"[ERROR] Learning pipeline: {e}"
            )

    return reply
