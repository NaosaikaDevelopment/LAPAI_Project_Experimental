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

LAPAI Project is a project for Local AI Runtime, this is work as Runtime to give AI Feature that can have memorial and learning ability. This local AI runtime can use any backend provider, therefore this project develop under LemonadeServer(Ryzen) and slightly with Ollama(non-Ryzen) and basiclly it can run all backend that work with OpenAI-Style API. This project for people want to have its own AI without the heavy wraper that can bleeding the pc resource for its own project. This project its for those who want to have its own AI local but didn't want to using heavy AI, at the same time want to using light AI but didn't want to make the complex system so it can peform better and the most important Work Locally or Offline.

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

### this project leading to AI integrator, for simplified project AI development 


## Info: This project will be hiatus due I who created this project, don't have time to continue developing it for a while because I am in a language course for Ausbildung

