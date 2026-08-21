# LAPAI [Experimental]

<p align="center">
  <img src="https://img.shields.io/badge/By-NaosaikaDevelopment-red.svg">
  <img src="https://img.shields.io/badge/Version-1.4-brightgreen.svg">
  <img src="https://img.shields.io/badge/Solo-%20Developer-brightgreen.svg">
  <img src="https://img.shields.io/badge/AI-%20RuntimeLocal-blue.svg">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

<img width="1920" height="1080" alt="Start" src="https://github.com/user-attachments/assets/a836bfa6-a69e-4053-8b0b-770c1a2c09ea" />



### What is LAPAI?

> Local Agent Personal Artificial Intellegence

LAPAI Project is a project for Local AI Runtime, build Simple as possible,  great way to start or learning about AI. This is work as Runtime to give AI Feature that can have memorial and learning ability and can use any backend provider, therefore this project develop under LemonadeServer(Ryzen) and slightly with Ollama(non-Ryzen) and basiclly it can run all backend that work with OpenAI-Style API. This project for people want to have its own AI without the heavy wraper that can bleeding the pc resource for its own project. This project its for those who want to have its own AI local but didn't want to using heavy AI, at the same time want to using light AI but didn't want to make the complex system so it can peform better and the most important Work Locally or Offline.

> **Note: this is not AI platformer, this is special for AI Integrator**.
This project work for them seeking AI with API Open AI style and work for its own project sake,
and for them who want mod, create, learn, project, game to using AI locally.
Build light as possible with enhance abilty for tiny model so that can work without using too much resources, yet still powerful.
Every Memory and knowledge its save Externally, so even you change model, AI memory and knowledge will not deleted, feel free to experiment with it.

<p align="center">
  --==-- -Development- --==--
</p>

## This system project equipped with:
- Memorial System:
  - FTS5
  - FAISS integration
  - recency scoring
  - importance scoring
  - role weighting
  - semantic + keyword hybrid retrieval
- recall pipeline
- learning pipeline
- session management
- summarization
- orchestration
- local TTS runtime
- OpenAI-compatible API layer

  
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

## Contents

- [Features](#features)
- [Usage](HowToUseIt.md)
- [Preview](Changelog&Preview.md)
- [Installation](#getting-started)
- [License](#license)


# Features🧩
### 🎖️1. Hybrid memory system:
It combines SQLite (FTS5 full-text search) with FAISS vector embeddings. This means it can recall information both through keyword matching (exact recall) and semantic similarity (contextual recall). with ranking system at 1.4

### 📘2. Persistent sessions:
Conversations are saved in JSON and databases, so the assistant can resume past dialogues and maintain continuity.

### 🔧3. Embedding flexibility:
It uses an ONNX model (all-mpnet-base-v2) for efficient embeddings with GPU/DirectML support, making it lighter and portable across hardware.

### 📝4. Summarization and compression:
Long sessions are summarized automatically using a secondary model, preventing memory bloat while keeping important facts.

### 📜5. Knowledge learning mode:
A separate learning database lets the system extract insights, form new knowledge entries, and store them for reuse—giving it a "growing memory."

### ❗6. Personal data extraction & storage: <---in progress for making AI can remember more special info from user 
With simple classification, it detects if user input contains personal information, extracts it, and stores it in a personal file.

### 🗣️7. Custom persona support:
It loads personality instructions from PersonaAI.txt, so users can shape the assistant’s behavior without modifying the code.

### ♾️8. API connector:
With simple API made, to connecting two diffrent program/project or even game, make this more fun to experiment with.

# Use Cases🟢

### Personal AI assistant:
Tracks conversations, remembers context, and adapts responses over time.

### Learning system:
Extracts “thoughts” and knowledge points from conversations, building a personalized knowledge base.

### Experiment platform:
Since it integrates OpenAI-like APIs and local ONNX embeddings, it’s a good playground for experimenting with hybrid AI systems (local + remote inference).

### Privacy-aware applications:
By separating personal data into its own text file, it makes compliance with privacy rules more manageable.





# ⚠️Hardware Tested on:
NOTE: its based on your model parameter

Ram 24gigs 

RyzenAI 7 350

GPU : Radeon 860M


# 🚀Getting Started
### To install
before to installation make sure you have the Runtime Backend provider (LemonadeServer / Ollama)
1. Install Pyton3.10
2. Run ```AutoDownloadALL.bat```
3. Wait until done, and you all set

https://github.com/user-attachments/assets/a54b6656-634f-46b0-bbf0-0b579510f5da

[Back to top](#LAPAI-[Experimental])


## Contribute
Contributions to LAPAI are welcome and appreciated! If you'd like to improve this project, please consider:

  - Submitting bug reports with detailed information
  - Documenting additional configurations or solutions
  - Creating pull requests with code improvements or new features
  - Sharing your experience using LAPAI on different distributions



## License
LAPAI source code is licensed under the MIT License. - see the [LICENSE](LICENSE) file for details.


### Third-party components


LAPAI is a local AI runtime/framework that can work
with multiple backends and third-party AI providers.

Third-party components and AI models are licensed
under their own respective licenses:

- FAISS (MIT)
- ONNXRuntime (MIT)
- Coqui XTTS-v2 (CPML)
- all-mpnet-base-v2 (Apache 2.0)
- SentenceTransformers (Apache 2.0)
- HuggingFace Transformers (Apache 2.0)
- Every Component in this project with its own license

LAPAI does not redistribute third-party model weights.
Models are downloaded or installed separately by users.

Backend services, AI models, and runtimes remain
under their own respective licenses and terms.

Users are responsible for complying with the licenses
of all third-party components and models.


<p align="center">
  <img width="851" height="315" alt="Lapai-Development_20260519_024330_0000 (1)" src="https://github.com/user-attachments/assets/ed3e5918-7b24-467d-bf51-ed42e1d0431b" />
</p>


# Note
### Keep in mind this project is Experimental and Worked Alone by me (ND)
### Future plan:
- Adding Learning from Online
- Can gather information from online
### In case you want to contact me
- Discord : Naosaika#9386

### this project leading to AI integrator, for simplified project AI development 


## Info: This project will be hiatus due I who created this project, don't have time to continue developing it for a while because I am in a language course for Ausbildung

