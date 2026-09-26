from .statecore import *
#Main Function
def Main_Core_FP_Function(user_msg):
    cache.current_user_msg = user_msg
    cache.crntTime = datetime.now().astimezone().isoformat()

    active_memory = prepare_active_memory(
        session_id=cache.session_id,
        user_input=user_msg,
    )
    append_message(cache.session_id,cache.session_file,"user",user_msg)
    cache.msgt = build_memory_prompt(
        user_msg=user_msg,
        summary="",
        active_memory=active_memory,
        recalled_memory=[],
        knowledge=[],
    )
    compacted_summary = None
    try:
        reply = run_agent_turn(cache.msgt, current_user=user_msg)
        cache.msgt.append({"role": "assistant","content": reply})
        print(cache.msgt)
    
    except Exception as e:
        reply = f"[ERROR] When Main Model loading: {e}"
        print(f"[ERROR] When loading model in core: {e}")
        print(cache.msgt)
        return reply

    append_message(cache.session_id,cache.session_file,"assistant",reply)
    try:
        record_active_turn(
            session_id=cache.session_id,
            user_content=user_msg,
            assistant_content=reply,
            created_at=cache.crntTime,
        )
    except Exception as e:
        print(f"[WARNING] Failed to record active memory: {e}")

    if compacted_summary is not None:
        try:
            Learning_T = start_learning(cache.client,cache.model_name,user_msg,cache.prompt)
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
