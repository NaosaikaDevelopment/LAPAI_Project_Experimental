
## How the code work?
  First of all, I build it all just in one core for easy to mod or change by yourself. Great to start for your project and handle thing.
  Alright now how exactly this gonna work? Let's start with API work.


[IN LAPAI_Core.py]
```python
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
LEMONADE_BASE_URL = "http://localhost:8000/api/v1"
LEMONADE_API_KEY = "lemonade"
```
in Base URL in here you can change any provider you want, maybe you own AI backend or anything, I have not change the name "LEMONADE/OLLAMA_BASE_URL" i still use it for my runtime development.
In Lemonade base url there you can place your own provider.

```python
#MAIN MODEL
if os.path.exists("app-Settings/dist/win-unpacked/resources/Settings/1MainNameModel.txt"):
    with open("app-Settings/dist/win-unpacked/resources/Settings/1SumNameModel.txt", "r", encoding="utf-8") as f:
        MODEL_NAME = f.read().strip()

#Sum_model
if os.path.exists("app-Settings/dist/win-unpacked/resources/Settings/1SumNameModel.txt"):
    with open("app-Settings/dist/win-unpacked/resources/Settings/1SumNameModel.txt", "r", encoding="utf-8") as f:
        Sum_model = f.read().strip()
```
In here you can change the model name you using it, and this software i made from electron to edit those file with GUI. (this tested in Lemonade backend, still not tested with own or customize backend provider).


this runtime had 2 mode, Main_Core_FP_Function; fast responses (it depend on hardware or model parameter), and Main_Core_Function; Respond with fully memory research (not optimized yet)
here i gonna telling about FP function (Fast Respond)  
```python
    try:
        client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)
            
    except:
        print("Lemonade API Not Detected")
        try:
            client = OpenAI(base_url=OLLAMA_BASE_URL)
            print("OLLAMA API Not Detected")
        except Exception as e:
            print(f"Error:Runtime Backend Not Found cannot Running program: {e}")
```
Here the double check compability to API like Ollama, Lemonade or anything before it getting something worse. this may help to debug backend compability for some reason, but im not a expert, still learning.


Here the main engine, like memory, scoring memory, filtering, all those thing build as simple as possible:
```python
    append_message(session_id, session_file, "user", user_msg)
    recent = recall_recent_memory(hours=12,limit=3)
    keywords = extract_keywords(user_msg)
    collected_knowledge = recall_knowledge(user_msg)
    collected_knowledge.extend(knowledge)
    prompt.extend(knowledge)
    recalled = recall_relevant_memory(user_msg, limit=5)
    context = collected_knowledge + recalled + recent
    context.sort(key=lambda x: x.get("score", 1.0), reverse=True)
    seen = set()
    
    
    for m in context:
        role = m.get("role", "")
        content = m.get("content", "")
        key = (role, str(content))  
        if key not in seen:
            item = {"role": role, "content": str(content)}
            if "score" in m:
                 item["score"] = m["score"]
            unique.append(item)
            seen.add(key)
    prompt.append({"role": "system", "content": format_items(unique[:5])})
    #injection
    persona = None
    if os.path.exists("app-Settings/dist/win-unpacked/resources/Settings/PersonaAI.txt"):
        with open("app-Settings/dist/win-unpacked/resources/Settings/PersonaAI.txt", "r", encoding="utf-8") as f:
            persona = f.read().strip()
            if persona == "":
                pass
            else:
                prompt.append({"role": "system", "content": persona})
    #Output
    try:  #<---Main model loading
        prompt.append({"role": "user", "content": user_msg})
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        reply = f"[ERROR] When Main Model loading: {e}"
        print(f"[ERROR]When loading model in core:{e}")
        return reply
```
Just some filtering word using loop and inject it at prompt as memory context.

and here the place when conversation is long enough, that AI start learning(self generate work as like memory) and save it in another data file:
```python
#learning Recalled
    summary = summarize_session(client, Sum_model, session_id, session_file)
    if summary:
        Learning_T = start_learning(client, MODEL_NAME, user_msg, prompt)
        question = generate_question(client, Sum_model, summary)
        add_question(question)
        thought = thoughtm(user_msg)
        append_Learning(seid, jsfile, "knowledge", Learning_T)
        append_Learning(seid, jsfile, "thought", thought)
        prompt.append({
            "role": "system",
            "content": f"Conversation summary: {summary}"
        })
        append_message(session_id, session_file, "summary", summary)
```
as you can see, after summary, it will shrink the information so that AI can more easy to follow through (Some solution i made for prompt issue)

and here some prompt trim to maintain the prompt still efficient (may still more improve, if you had idea to improve it, do it on as you like)
```python

    if estimate_tokens(prompt) >= 4056:
        prompt = trim_prompt(
            prompt,
            memory_prompt=recalled,
            chat_history=context
        )
```
may some code not completed shared in here, so you can explore it by your own.


### next in ` RunAI.py `
Here just basic run AI on console, to make sure all is work perfectly fine, you can use it as template to implement it on your project.
```python
#_____STARTUP_____
condition = True
print("ATTENTION making memory files, for first start maybe its take a little time")
init_db()
init_learning_db()
faiss_index, id_map = init_faiss()
title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
title_learn = "Learning"+title_hint
session_id, session_file = create_session(title_hint)
seid, jsfile = create_session_Learning(title_learn)
print("[INFO] Session Created", title_hint, "\n")
print("Startup The system... \n")
user_msg = "[INFO] User is Online"
append_message(session_id, session_file, "system", user_msg) #<--- gave first input for load \/ because when get first input its need a time to make save file
reply = Main_Core_FP_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)

payload = {
    "text": reply
}
try:
    r = requests.post("http://localhost:1922/tts", json=payload)
except:
    print("Xtts No Activate")
```
Basic startup and adding xtts compability. 

here the code for the core can run:
```python
#______IN MODE_____
while condition:
    try:
        try:
            done = False
            user_msg = input("\033[93m\nUser: \033[0m").strip()
            t = threading.Thread(target=animate)
            t.start()
        except (EOFError, KeyboardInterrupt):
            break
        if user_msg.lower() in {"exit", "quit", "q"}:
            append_message(session_id,session_file,"SYSTEM","Closing Program, Shutdowning system.")
            done = True
            t.join()
            break
        reply = Main_Core_FP_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)
        payload = {
            "text": reply
        }
        
        try:
            r = requests.post("http://localhost:1922/tts", json=payload)
        except:
            print("Xtts No Activate")
        done = True


        t.join()
        print("\033[92m\nLAPAI: \033[0m", end="", flush=True)
        for ch in reply:
            print(ch, end="", flush=True)
            time.sleep(0.02)
        print()

        #Here the example to use modular function
        try:
           #decsn(user_msg) #(INPROGRESS)
           pass
        except Exception as e:
            print(f"[ERROR]: When try extract personal information: {e}")
        thought = thoughtm(user_msg)
        append_Learning(seid, jsfile, "thought", thought)
        append_message(session_id, session_file, "assistant", reply)
        summary = summarize_session(client, Sum_model, session_id, session_file)
        prompt = LAPAI_Core.prompt
        if summary:
            if prompt is None:
                pass
            else:
                try:
                    Learning_T = start_learning(client, MODEL_NAME, user_msg, prompt)
                except Exception as e:
                    print(f"[ERROR] When start Learning: {e}")
            question = generate_question(client, Sum_model, summary)
            add_question(question)
            append_Learning(seid, jsfile, "knowledge", Learning_T)
            append_message(session_id, session_file, "summary", summary)
    except Exception as e:
        print(f"[ERROR] on RunAI: {e}")
        append_message(session_id, session_file, "SYSTEM", "Error Detected, System Crashed, Shutdowning system.")
        break
append_message(session_id, session_file, "SYSTEM", "[INFO] User Offline.")
append_message(session_id, session_file, "SYSTEM", "System Offline.")
```
some code is customized so that you can get the faster way the output, in here too is where trigger down the learning independently. and some progress feature that maybe i should delete it?
this is not really different with the core, just calling function. 

### To using API of this project to another program or project
you can see the template at ```LAPAI_Run_as_API-G.py```  you can use it as you like. Maybe it not advance as like ```RunAI.py```, but you can configure it by yourself:
```python
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completion(request: ChatRequest):
    user_text = " ".join([m.content for m in request.messages if m.role == "user"])
    data = repr(user_text)
    print(data)
    #Change the code bellow this as you like to improve the output \/ \/ \/ \/
    reply = Main_Core_FP_Function(user_text, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)
    #<--- Adding new content here
    
    return ChatResponse(
        id="lapai-"+str(int(time.time())),
        choices=[Choice(
            index=0,
            message={"role": "system", "content": reply},
            finish_reason="stop"
        )]
    )
    
```
and there you can get your own customize API in your own hardware locally without any connection need it
This output still pure from core and not like ```RunAI.py``` here you can configure it as you like.

to start your API, either you can use your own env of python or you can use my env in LAPAI folder.


In case you are beginner and do not know how to run it;
first in console you can "cd" to LAPAI1.4 directory and activate the env by using ```LAPAI\Scripts\activate``` then you good to use it.
to use it go "cd" to core and run the bat file or direct py file ```RunAI.py``` then there great way to start your journey.
