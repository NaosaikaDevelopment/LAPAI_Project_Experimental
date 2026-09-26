from .statecore import *

def Main_Core_FP_Function(user_msg):
    cache.current_user_msg = user_msg
    cache.crntTime = datetime.now().astimezone().isoformat()

    append_message(cache.session_id,cache.session_file,"user",user_msg)
    compacted_summary = compact_old_memory(cache.client,cache.Sum_model,cache.session_id,cache.conf.get('comMinutes') or 5,)
    cache.msgt.append({"role": "user", "content": user_msg})
    try:
        reply = run_agent_turn(cache.msgt, current_user=user_msg)
        print(cache.msgt)
    
    except Exception as e:
        reply = f"[ERROR] When Main Model loading: {e}"
        print(
            f"[ERROR] When loading model in core: {e}"
        )
        return reply

    append_message(cache.session_id,cache.session_file,"assistant",reply)

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
