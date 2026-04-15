# LAPAI [experimental]
## What is LAPAI?
<img width="1366" height="768" alt="L (1)" src="https://github.com/user-attachments/assets/ece679bf-463d-4021-b20e-a7c7ecbdc934" />

Local Agent Personal Artificial Intelligence, a project for AI Engine Enchantment, this is work as script to give AI Feature that can have memorial and learning ability. and completely offline, with backend LemonadeServer for Ryzen and Ollama for non Ryzen.

**Note: this is not AI platformer, this is special for AI Integrator**.
Build light as possible with enhance abilty for tiny model so that can work without using too much resources, yet still powerful

Local Agent LA-PAI is an experimental personal AI that reflects on its own conversations using self-questioning and dual-mode learning—entirely on your device.
And this project is can run entirely offline that work in your own device. dual model learning its not that special, its more like offline mode and online mode but this still in experimental.
in v1.3 self learning has been added to main core, so its actually learning from experience.

Every Memory and knowledge its save Externally, so even you change model, AI memory and knowledge will not deleted, have fun experiment with it

this script isn’t just a regular script for chatbot , it’s a self-learning, memory-based AI framework that blends keyword search, semantic recall, and summarization into one system. It’s especially strong for experiments in long-term memory AI, personalized assistants, and lightweight hybrid LLM deployments.

#### update v1.3
from here you can using self host and get API from this project, to AI development in your project without think about memory management or anything about AI management,
just paste your API you got from this project script and done, you can play or development with AI in your device without network or data go to cloud. for this have a guide check release page v1.3

#### update 1.3.1
Adding ability to command game, from this update now AI can give input for Game Development, and entirely offline, so from input user AI would decide what what user want AI todo on the game, for example here im going to debugging AI input as command:
here by using Template on Core folder, file named ```GameDevelopmentCommandAITemplate.cs``` on unity.

https://github.com/user-attachments/assets/05099062-a69d-4b0d-9cc8-1a4d58dafeba

Okay let me explain how to use: this is work as API from py i build, you can run ```GDAPIND.py``` for start API, (Note from me: Better learn how to run python automaticly when you run program from Program Language you use, tips: use and make .vbs to run without trace) 
Here how it work: 
```
[Input unity]
      ↓
Get API and send input to main API on GDAPIND.py
      ↓
AI processing on there using Llama-3.2-3B-Instruct-Hybrid (info you can change model as you want but model must be "Instruct")
      ↓
Answer return as json fill with Command, Execute, Reason. AI will decide that.
      ↓
On my program template for game development that will auto take command, execute, reason, and you can add command on there almost anythin you want
      ↓
return on unity as action.

```
#### Update Adding UI
UI its ready and has been added to new Repo, go Check [LAPAI UI](https://github.com/NaosaikaDevelopment/UI-LAPAI), im adding new feature, adding AI to get info about current time.

#### Software Release MyCompanion
This is an Side_Project for LAPAI Prototype, Its a functional software from LAPAI and LAPAI UI. this is running entirely offline, and might i say easy to install.


https://github.com/user-attachments/assets/c44a3d92-a36a-41eb-b271-80c9015d4153

Download on Release Page.


#### Update 1.3.2
Finally i have a little time to adding thing, here some new feature to use this AI.

This update adding new template program to use this AI in Arduino, and able to give command, output, and control.

Preview:




https://github.com/user-attachments/assets/cc090dc9-c6d0-49f2-b061-7fa49c607303




#### Update 1.4
fixing and improving memory, adding ui Settings, and XTTS support
make the Installation more simple

## System Advantage for tiny model running at offline mode

### 1. Hybrid memory system:
It combines SQLite (FTS5 full-text search) with FAISS vector embeddings. This means it can recall information both through keyword matching (exact recall) and semantic similarity (contextual recall). with ranking system at 1.4

### 2. Persistent sessions:
Conversations are saved in JSON and databases, so the assistant can resume past dialogues and maintain continuity.

### 3. Embedding flexibility:
It uses an ONNX model (all-mpnet-base-v2) for efficient embeddings with GPU/DirectML support, making it lighter and portable across hardware.

### 4. Summarization and compression:
Long sessions are summarized automatically using a secondary model, preventing memory bloat while keeping important facts.

### 5. Knowledge learning mode:
A separate learning database lets the system extract insights, form new knowledge entries, and store them for reuse—giving it a "growing memory."

### 6. Personal data extraction & storage: <---in progress for making AI can remember more special info from user
With simple classification, it detects if user input contains personal information, extracts it, and stores it in a personal file.

### 7. Custom persona support:
It loads personality instructions from PersonaAI.txt, so users can shape the assistant’s behavior without modifying the code.

### 8. Streaming-like output animation:
Simulates real-time typing to give a more natural conversational feel.

## Use Cases

### Personal AI assistant:
Tracks conversations, remembers context, and adapts responses over time.

### Learning system:
Extracts “thoughts” and knowledge points from conversations, building a personalized knowledge base.

### Experiment platform:
Since it integrates OpenAI-like APIs and local ONNX embeddings, it’s a good playground for experimenting with hybrid AI systems (local + remote inference).

### Privacy-aware applications:
By separating personal data into its own text file, it makes compliance with privacy rules more manageable.


## Deployment Preview:
- Minecraft using mod Touhou Little Maid by TartaricAcid:

https://github.com/user-attachments/assets/486d2def-e81b-45ea-bcca-2236af8183dd

- Unity Engine AI offline Deployment:
```
  on this case im gonna using unity engine to demonstrate how to implementation AI to Game development.
to use it on C# you can paste my program template for AI on file named ```Template_Program_AIResponsForGameEngine.cs```
and you can see the detail on ```ImplementationGameDevelopment.mp4``` 
```

https://github.com/user-attachments/assets/dc69ca07-ecb3-476d-947e-b610915ea08b

- AI Assitance (in this case im using [LAPAI UI](https://github.com/NaosaikaDevelopment/UI-LAPAI))
  


https://github.com/user-attachments/assets/0d3f242f-4518-4eec-b344-8422af5a620e



And Many more, all for support your idea to make project with AI!

## Hardware min Recommendation:
NOTE: its based on your model parameter

Ram min 24gig or more for play and run AI smoothly. 

CPU I5 gen 8 or Ryzen 5 3500U, that is mandatory or at least better above that specification.

GPU or IGPU Ryzen AI (7000 series) / Intel core ultra (Meteor Lake) / RTX 2000

but at all its based on your model, even if you dont have powerfull gpu, you at least can use CPU


## Instalation 1.4



1. Install Pyton3.10
2. Run ```AutoDownloadALL.bat```
3. Wait until done, and you all set



## How to Run it?

https://github.com/user-attachments/assets/159a85ae-cf9c-47be-8a89-c3c2cfcc27c6


1. Run ```RunTTS.bat``` to turn on XTTS
2. go to core/app-settings/dist/win-unpacked and make shortcut for ```LAPAI Settings editor.exe``` and cut it to core
3. run LAPAI Settings editor and set your model and persona (OLLAMA / LemonadeServer)
4. Run ```Run.bat``` to run it in Terminal to test it, and Run ```RunAPI-G.bat``` to run the API or ```RunAPI-G_WithoutTrace.vbs``` to run it without tracer

   

## How exactly this script run?

### LAPAI Core v1.0
```
[ User Input ]
       ↓
[ Recall Memory + Knowledge ]
       ↓
[ Main Model Response ]
       ↓
[ Summarization + Question Generation ]
       ↓
[ DB Updates: short-term memory, questions list ]
          ↘
     [Summarize → Memory DB ]
```

### LAPAI Learn Core [Experimental]
```
         Question DB
            ↓
         Prompt To main model
            ↓
      [Thought] = Factual Conclusion
            ↓
         Save tp DB Learning
            ↓
Summarize → Generate question → save to Question DB
```
If online: these questions are answered through Google Search + summarized by the model. [Experimental] <--- Still Progress

If offline: the questions remain stored waiting for internet, or can be processed manually.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party components
This project uses several open-source components:

- SentenceTransformers (Apache 2.0)
- FAISS (MIT License)
- ONNXRuntime (MIT License)
- HuggingFace Transformers (Apache 2.0)
- HuggingFace model `all-mpnet-base-v2` (Apache 2.0)

All third-party components remain under their original licenses.


# Note
### Keep in mind this project is Experimental and Worked Alone by me (ND)
in choosing model, as i reccomend is model parameter 8B for standart use, but if you have model that trained for instruct, then you can use that model.

This Project was worked on Laptop Ryzen AI 7 350 with ram 24gig using IGPU.

## How to use API to game development? v1.3





## Info: This project will be hiatus due I who created this project, don't have time to continue developing it for a while because I am in a language course for Ausbildung
