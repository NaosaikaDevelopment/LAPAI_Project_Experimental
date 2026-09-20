from .statecore import *
def Main_Core_FP_Function(user_msg):
    prompt = cache.prompt
    compacted_summary = compact_old_memory(cache.client,cache.Sum_model,cache.session_id, cache.conf.get('comMinutes'))
    memory_state = get_memory_state(cache.session_id)
    summary = memory_state["summary"]
    active_memory = recall_recent_memory(cache.session_id,minutes=5,turns=3)
    collected_knowledge = recall_knowledge(user_msg)


    recalled = recall_relevant_memory(user_msg,limit=cache.conf.get('limitrrm'))

    prompt = [
        {
            "role": "system",
            "content": get_current_time_context()
        }
    ]

    persona = load_persona()

    if persona:
        prompt.append({
            "role": "system",
            "content": persona
        })

    memory_prompt = build_memory_prompt(user_msg,summary=summary,active_memory=active_memory,recalled_memory=recalled,knowledge=collected_knowledge)

    prompt.extend(memory_prompt)

    append_message(cache.session_id,cache.session_file,"user",user_msg)

    try:
        completion = cache.client.chat.completions.create(
            model=cache.model_name,
            messages=prompt
        )

        reply = completion.choices[0].message.content

    except Exception as e:
        reply = f"[ERROR] When Main Model loading: {e}"
        print(
            f"[ERROR] When loading model in core: {e}"
        )
        return reply

    append_message(cache.session_id,cache.session_file,"assistant",reply)

    if compacted_summary is not None:
        try:
            Learning_T = start_learning(cache.client,cache.model_name,user_msg,prompt)

            question = generate_question(cache.client,cache.Sum_model,compacted_summary)

            add_question(question)

            thought = thoughtm(user_msg)

            append_Learning(cache.seid,cache.jsfile,"knowledge",Learning_T)

            append_Learning(cache.seid,cache.jsfile,"thought",thought)

        except Exception as e:
            print(
                f"[ERROR] Learning pipeline: {e}"
            )

    return reply
