
<img width="1000" height="200" alt="MainRdmeWRev" src="https://github.com/user-attachments/assets/88d2874c-7530-4fed-9af9-84b3794884fa" />




<h1 align="center">LAPAI Experimental Project</h1>

<p align="center">
  <img src="https://img.shields.io/badge/By-NaosaikaDevelopment-red.svg">
  <img src="https://img.shields.io/badge/Version-1.5.1-brightgreen.svg">
  <img src="https://img.shields.io/badge/Solo-%20Developer-brightgreen.svg">
  <img src="https://img.shields.io/badge/AI-%20RuntimeLocal-orange.svg">
  <img src="https://img.shields.io/badge/To-%20Framework-cyan.svg">
  <img src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center">
  LAPAI (Local Agent Personal Artificial Intelligence) aims to make AI accessible, affordable, and locally owned. By combining local model execution, memory systems, and developer-friendly APIs, LAPAI enables AI applications that work with greater privacy, lower operational costs, and reduced cloud dependence.
</p>



<p align="center">
  --==-- -Development- --==--
</p>

#

<h3 align="center">Overview:</h3>
<p align="center">
  <img width="400" height="250" alt="490056541-dc69ca07-ecb3-476d-947e-b610915ea08b" src="https://github.com/user-attachments/assets/6cfba3ea-270b-42ff-b796-d015703adf54" />
  <img width="400" height="225" alt="ezgif-44e7581b10515873" src="https://github.com/user-attachments/assets/54968f22-1395-4ca2-af30-4336dc8e50e1" />
</p>
<table>
  <tr>
    <td align="center" width="50%">
      <h4>General purpose cache system</h4>
      <img width="100%" alt="Screenshot 1" src="https://github.com/user-attachments/assets/f9946bdd-7f63-4583-bb4f-60f7c20507ea" />
    </td>
    <td align="center" width="50%">
      <h4>Import it anywhere</h4>
      <img width="735" height="222" alt="image" src="https://github.com/user-attachments/assets/dd09008c-90d8-430d-b557-c12d8aa87653" />
    </td>
  </tr>
</table>

<h3> This project is equipped with:</h3>

- Local AI execution
- OpenAI-compatible API
- Persistent memory system
- Semantic memory retrieval
- Learning-oriented memory architecture
- Session management
- Modular core design
- Dynamic module registration
- ONNX Runtime support
- Cross-platform compatibility
- Offline-first operation
- Privacy-focused data handling
- Consumer hardware optimization
- Extensible developer integration
  











<h3>Hardware Development/Tested on:</h3> 

Laptop Lenovo ideapad slim 5 gen 10

Ram 24GB

RyzenAI 7 350

GPU : Radeon 860M

OS:Windows and Linux

<h3 align="center"> 🚀Getting Started </h3>


before to installation make sure you have the Runtime Backend provider (LemonadeServer / Ollama / anything)
   1. Install Pyton3.10 
   
   2. Run ```autod.bat``` **WINDOWS**
   
   2. Run ```autod.sh``` **LINUX**

   3. Wait until done, and you all set

<h3 align="center"> How this new modular system work and how can i use it? 1.5+ </h3> 


> Please make sure you using **python 3.10** and set the settings in **folder Settings**, conf.json in 1.5.1 you can use jsonEd to edit json file, it is just simple Json Editor with simple GUI, then dont forget to set persona on settings folder too

Finally this update support linux and windows (fyi this version build on bazzite distro known as immutable distro)

On new modular System you can navigate to LAPAIv1.5.1 and see `MainCore` there you would see 
```
MainCore/
├── core/
│   ├── __init__.py
│   ├── learning.py
│   ├── memory.py
│   ├── rcore.py
│   ├── state.py
│   └── sum.py
├── runcorefp.py
├── runcoremain.py
├── statecore.py
└── ...(etc)
```
Here as you can ` runcorefp.py` and `runcoremain.py` this core had 2 different purpose.

the "FP" is work as FastResponse and "CoreMain" as full sweep search memory and AI Base Decision use as you like, in here i will explain the "FP" cause i developing focused on it. 


So when you open it you will found:
```py
from .core import *
```
that is the main core of LAPAI, and some function migrate to this fpcore while for compability reason. So now how can you use it? it is very simple actually.


first of all this project will installed with its own env or known as ` LAPAI-env `
please make sure to use the env or you can add by yourself with install the `requirement.txt`


to turn on the env you can simply run on console: 
```
nd
```
here it will automaticly turn on the *LAPAI-env* in case you had problem with directory after change directory, simply run the autod.sh/bat again, or want to delete the old shortcut by run uinsShortcut.sh/bat

### --> Directly use on LAPAI directory (1.5):
```python
from MainCore.runcorefp import *
# import module

initialize_core()
# initial the core for all memory system DB

msg = "Hello Naove!"
# Input

reply = Main_Core_FP_Function(msg)
print(reply)
#Output
```
and just like that! you can made your own project. **but** this case is integrate LAPAI system directly and muss on LAPAI directory(1.5).

how can i use it on another language prograrm or different project? with Quick API `qapi.py` in this project has OpenAI Style you can add this system almost anywhere.
to another language program you can see the template and [How to use it?]((1.4)HowToUseIt.md).


<h3 align="center"> 1.5.1 simplifier Update </h3>

this update simplified developer to using this project, after installation, no need copy LAPAI to every project, just one installation and can use it anywhere (still need LAPAI-env to useit)

Simple using this project in another directory:
```python
from MainCore import runcorefp as c #calling main core
c.initialize_core() #initialing core
# to use you need this two component

print(c.Main_Core_FP_Function('HELLOWWW'))
#to get the input

```
as you can see it is easy, no? and just that everything will run perfectly

    NOTE 

    PLEASE make sure you run the prograrm with LAPAI-env by using **'nd'** in command

### --> Explanation:

Newest core added new statecore in MainCore, to seperate def function so it can work as template.
and i added new simple cache function, you can use anywhere and any purpose so it can cache any information on **Settings/conf.json** you can edit it with GUI simple app "jsonEd" after intallation in LAPAI directory, one more after installation you will had quick shortcut to turn on LAPAI-env by run "nd" on console

as example you add new variable on conf.json= "A1" : 10

<img width="900" height="528" alt="Screenshot_20260920_234632" src="https://github.com/user-attachments/assets/6ac481c8-c30a-46b4-8187-567fd147a839" />


and the test you can another directory and use simple python:
```python
from MainCore import statecore as c
print(c.cache.conf.get('A1'))
```
the results:

<img width="766" height="244" alt="Screenshot_20260920_234806" src="https://github.com/user-attachments/assets/a8bac478-3553-4993-ad56-2cb36c7b9beb" />


easy to use, no?

<h3 align="center">to use yourown core and function (with simple cache system) 1.5.1</h3>

statecore.py added, when you need any function from MainCore in another directory (using LAPAI-env)
```python
from MainCore.statecore import * 
```
it will call any function or variable you want even it on conf.json work as cache. you can use or call simple cache function by  
```python
cache.conf.get('NameVariable')
```

### --> (**example** to use yourown runcore)

newest **runcorefp.py** work as runcorefunction and not containing another independent function , to use it, you can call the def function on the any core you had, for this example i will use default module **"MainCore/runcorefp.py"** and had its own def fucntion **"Main_Core_FP_Function"** there how the script logic working, like memorial, summary, prompt trimming, etc, if you want to make your own, here simple guide:

**YourOwnRunCore.py**
```python
from .statecore import *
def yourFunctionName(user_input):
    prompt = cache.prompt
    csum = compact_old_memory(cache.client,cache.Sum_model,cache.session_id, cache.conf.get('comMinutes'))
    ... #any function logic you want to build
```
and make sure you build it in **MainCore** directory

in case you had your own runcore as example "yourcosruncore.py" (Note make sure yourown runcore on MainCore directory), and want to use it in another directory: 
``` python 
from MainCore.yourcosruncore import * 
```
or
``` python 
from MainCore import yourcosuncore 
```
to run function just recall it by `yourFunctionName()` or `yourcosruncore.yourFunctionName()`

### --> (**to add your own def function**,)

in core if you had specific purpose you can add by yourself in `addonsfunction.py` and make your own core just copy `runcorefp.py` as template and modify by yourself, or you need somehow to make new core function, you need a little hardcoded, example:
you had your own new function core as "NewCoreFunction.py" (Make sure add it to MainCore/core) . To add it so you can use it anywhere, Register it first in ``__init__.py``(in MainCore/core) and write:
```python
from .NewCoreFunction import *
```
and make sure every module you need example "import pathlib" or something like that(Make sure too it installed on env you use or install it on LAPAI-env) is state in ``state.py`` then last step in your "NewCoreFunction.py" import ``from state import *`` then add your function as you want, to use your function for example "def YourFunction()" in another directory or want to use it as your own run core is easy

your run core case:
```python
from statecore import *
YourFunction()
```
done your new function is called

another directory case and want to use it directly
```python
from MainCore.core.NewCoreFunction import *
YourFunction()
```
Done and your new function is called

### --> API use:

Require OpenAI library to accsess the API and make sure your project or another program Language is installed OpenAI Library and know how to use it, in case you want to learn i had the template in this repo in folder `Template`

in this case i will make it simple with using python as the receiver 
```py
#PYTHON
from openai import OpenAI
#Using OpenAI

client = OpenAI(base_url="http://localhost:SEE_FROM_QAPI_GUIDER/v1", api_key="Dummy" )
#get the url localhost

reply = client.chat.completions.create(
    model="",
    messages=[{"role":"user","content": msg}]
)
#Make sure output like this client.chat.completion.create(...same as on up there) because the API litening on V1/chat/completion

print(reply.choices[0].message.content)
#make sure the replay had the "choices[0].message.content" to get the content

```



### --> Uniq API endpoint case:

Yeah maybe in rare case, or you want to pair in different project(AI backend endpoint) via OpenAI API style.
for example we had custom endpoint "http://localhost:1234/random/endpoint/chat/v1/completion/" or something like that, need to remember the endpoint where data is send "random/endpoint/chat/v1/completion".
then to use that endpoint, you can either directly hardcoded:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234", 
    api_key="dummy"
)
response = client.post(
    "/random/endpoint/chat/v1/completion", 
    body={
        "model": "Model_name",
        "messages": [{"role": "user", "content": "HOLLA!"}]
    },
    cast_to=dict 
)

print(response["choices"][0]["message"]["content"])

```
or using cache system from this project by add the variable to conf.json in LAPAI Settings folder as "
```json
{
  "openai_config": {
    "base_url": "http://localhost:1234",
    "api_key": "dummy"
  }
}

```
then using LAPAI-env to call the variable:
```python
from MainCore.statecore import *
client = OpenAI(**cache.conf["openai_config"])
response = client.post("/random/endpoint/chat/v1/completion", 
    body={
        "model": "Model_name",
        "messages": [{"role": "user", "content": "HOLLA!"}]
    },
    cast_to=dict 
)

print(response["choices"][0]["message"]["content"])
```



### Function Reference

Quick overview of all functions, grouped by module.

| # | Modul | Fungsi | Input | Output | Deskripsi |
|--:|-------|--------|-------|--------|-----------|
| 1 | Initialization & Sessions | `init_faiss()` | — | `index`, `id_map` | Initializes FAISS: loads an existing index or creates a new one, and checks embedding dimensions and map consistency. |
| 2 | Initialization & Sessions | `init_memory_state()` | — | — | Creates the `memory_state` table that stores summaries and the position of already-summarized memory. |
| 3 | Initialization & Sessions | `init_db()` | — | — | Sets up the main `Raw_Memory.db` database, including the session table, the messages FTS5 table, and `memory_state`. |
| 4 | Initialization & Sessions | `init_learning_db()` | — | — | Sets up the learning database: `knowledge`, `sessions`, and `messages`. |
| 5 | Initialization & Sessions | `initialize_core()` | — | `faiss_index`, `id_map` | Orchestrator that initializes all storage, personal data, and FAISS. |
| 6 | Initialization & Sessions | `create_session(title)` | `title` | `sid`, `json_file` | Creates a new chat session in SQLite along with a JSON history file. |
| 7 | Memory Management & Retrieval | `get_memory_state(session_id)` | `session_id` | `dict` | Fetches the summary, checkpoint timestamp, and the `rowid` of the last processed memory. |
| 8 | Memory Management & Retrieval | `save_memory_state(...)` | session + state | — | Saves or updates the memory state using `ON CONFLICT`. |
| 9 | Memory Management & Retrieval | `search_memory(query, limit=5)` | `query` | rows | Lexical memory search using SQLite FTS5 + BM25. |
| 10 | Memory Management & Retrieval | `recall_recent_memory(session_id, minutes=5)` | session, time window | `list` | Retrieves the most recent conversation within a given session. |
| 11 | Memory Management & Retrieval | `should_recall(user_msg)` | `user_msg` | `bool` | Asks a small model whether the input requires older memory. |
| 12 | Memory Management & Retrieval | `_sqlite_timestamp_to_epoch(timestamp_text)` | timestamp | epoch | Converts a SQLite timestamp to a Unix timestamp. |
| 13 | Memory Management & Retrieval | `recall_relevant_memory(...)` | input, limit, threshold | `list` | Main retrieval system: FTS + FAISS + importance + recency + role weighting. |
| 14 | Memory Management & Retrieval | `append_message(...)` | session, role, content | — | Saves a message to JSON and SQLite, then adds it to FAISS. |
| 16 | Vector Search (FAISS) | `recall_from_faiss(...)` | vector | `list` | Finds the most similar vectors using FAISS and a similarity threshold. |
| 17 | Vector Search (FAISS) | `generate_embedding(...)` | text, type | vector | Converts text into an embedding using the ONNX `multilingual-e5-small` model. |
| 18 | Vector Search (FAISS) | `add_to_faiss(...)` | index, text, metadata | updated index/map | Creates an embedding, inserts it into FAISS, and stores the metadata in the JSON map. |
| 19 | Vector Search (FAISS) | `normalize_embedding(vector)` | vector | vector | Normalizes a vector to unit length (1). |
| 20 | Learning System | `create_session_Learning(title)` | `title` | `sid`, `json_file` | Creates a dedicated learning session. |
| 21 | Learning System | `load_knowledge(json_file)` | path | `list` | Loads knowledge from a JSON file. |
| 22 | Learning System | `search_knowledge(query, limit=5)` | `query` | rows | Searches knowledge using FTS5. |
| 23 | Learning System | `recall_knowledge(...)` | user input | `list` | Extracts keywords → searches knowledge → filters results by score. |
| 24 | Learning System | `append_Learning(...)` | session, role, content | — | Saves learning results to JSON and SQLite. |
| 25 | Learning System | `start_learning(...)` | client, model, input, context | text | Asks the model to generate learnable information from the user's input. |
| 26 | Memory Compaction | `compact_old_memory(...)` | session + time | summary / `None` | Fetches old memory that has passed the time window and builds a persistent summary. |
| 27 | Memory Compaction | `summarize_session(...)` | session | summary | Compatibility wrapper that now calls `compact_old_memory()`. |
| 28 | Utilities & Scoring | `extract_keywords(text)` | text | `list[str]` | Extracts tokens/words from text. |
| 29 | Utilities & Scoring | `parse_yesno(user_it)` | text | `"Yes"` / `"No"` | Extracts a yes/no decision from model output. |
| 30 | Utilities & Scoring | `generate_question(client, model, text)` | client, model, text | question | Generates an investigative question from a piece of text. |
| 31 | Utilities & Scoring | `add_question(question)` | `question` | — | Adds a question to `questions.json`, avoiding duplicates. |
| 32 | Utilities & Scoring | `load_json(path, default=[])` | path | object | JSON reading utility. |
| 33 | Utilities & Scoring | `save_json(path, data)` | path, data | — | JSON writing utility. |
| 34 | Utilities & Scoring | `estimate_tokens(messages)` | `messages` | `int` | Counts tokens using the actual tokenizer. |
| 35 | Utilities & Scoring | `compute_importance(text)` | text | `float` | Determines importance based on text length. |
| 36 | Utilities & Scoring | `normalize_fts(bm25_score)` | BM25 score | `float` | Converts a BM25 score into a value that can be combined with other scores. |
| 37 | Utilities & Scoring | `normalize_faiss(dist)` | distance | `float` | Converts a distance into a similarity-like value. |
| 38 | Utilities & Scoring | `compute_recency(timestamp)` | timestamp | `float` | Computes a recency value with daily decay. |
| 39 | Personal Data | `init_personal_DB()` | — | — | Creates `PersonalData.txt` if it doesn't exist. |
| 40 | Personal Data | `append_txt(items)` | data | — | Appends personal information to the file. |
| 41 | Personal Data | `read_txt()` | — | `list` | Reads all personal information from the file. |
| 42 | Context & Prompt Building | `calculate_context_budget(...)` | limit, reserved, used | `int` | Calculates how many tokens remain available for context. |
| 43 | Context & Prompt Building | `trim_prompt(...)` | prompt + memory | prompt | Emergency fallback that removes context when the token limit is exceeded. |
| 44 | Context & Prompt Building | `load_persona()` | — | `str` / `None` | Loads the AI persona from `Settings/PersonaAI.txt`. |
| 45 | Context & Prompt Building | `build_memory_prompt(...)` | summary, memory, knowledge, user msg | `list` | Combines all context sources into the API message format. |
| 46 | Main Pipelines | `Main_Core_Function(...)` | user message + session info | reply | Main pipeline: memory compaction, knowledge recall, relevant memory, persona, LLM response, and learning. |
| 47 | Main Pipelines | `format_items(items)` | `list` | `str` | Formats memory results into a bullet-point string. |
| 48 | Main Pipelines | `get_current_time_context()` | — | `str` | Builds the current date, time, and day-of-week context. |
| 49 | Main Pipelines | `Main_Core_FP_Function(...)` | user message + session info | reply | Fast-processing version of the main pipeline, built on the memory + knowledge + persona principle. |
| 50 | Input Analysis | `decsn(user_msg)` | `user_msg` | — | Detects whether the input contains personal information and saves it if detected. |
| 51 | Input Analysis | `thoughtm(user_msg)` | `user_msg` | text | Asks a small model to explain the user's actual intent behind the input. |


<p align="center">
  --==-- -Hopefully this gonna be helpful- --==--
</p>







# Features
### 🎖️1. Hybrid memory system:
It combines SQLite (FTS5 full-text search) with FAISS vector embeddings. This means it can recall information both through keyword matching (exact recall) and semantic similarity (contextual recall). with ranking system at 1.4

### 📘2. Persistent sessions:
Conversations are saved in JSON and databases, so the assistant can resume past dialogues and maintain continuity.

### 🔧3. Embedding flexibility:
It uses an ONNX model (all-mpnet-base-v2) and intfloat/multilingual-e5-small for 1.5 and below for efficient embeddings with GPU/DirectML support, making it lighter and portable across hardware.

### 4. Summarization and compression:
Long sessions are summarized automatically using a secondary model, preventing memory bloat while keeping important facts.

### 5. Knowledge learning mode:
A separate learning database lets the system extract insights, form new knowledge entries, and store them for reuse—giving it a "growing memory."

### 6. Personal data extraction & storage: <---in progress for making AI can remember more special info from user 
With simple classification, it detects if user input contains personal information, extracts it, and stores it in a personal file.

### 7. Custom persona support:
It loads personality instructions from PersonaAI.txt, so users can shape the assistant’s behavior without modifying the code.

### 8. API connector:
With simple API made, to connecting two diffrent program/project or even game, make this more fun to experiment with.

# Use Cases

### Personal AI assistant:
Tracks conversations, remembers context, and adapts responses over time.

### Learning system:
Extracts “thoughts” and knowledge points from conversations, building a personalized knowledge base.

### Experiment platform:
Since it integrates OpenAI-like APIs and local ONNX embeddings, it’s a good playground for experimenting with hybrid AI systems (local + remote inference).

### Privacy-aware applications:
By separating personal data into its own text file, it makes compliance with privacy rules more manageable.







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
- intfloat/multilingual-e5-small(MIT)
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

Be advised, This project is still experimental.
### Keep in mind this project is Experimental and Worked Alone by me (ND)

### In case you want to contact me
- Discord : Naosaika#9386
