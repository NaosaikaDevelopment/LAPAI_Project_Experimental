# LAPAI [experimental]
## What is LAPAI?
<img width="1366" height="768" alt="L (1)" src="https://github.com/user-attachments/assets/ece679bf-463d-4021-b20e-a7c7ecbdc934" />

Local Agent Personal Artificial Intelligence, a project based fondation from [Lemonade Server](https://lemonade-server.ai/docs/server/),Go check their website to learn more! ```https://lemonade-server.ai```, this is work as script to give AI Feature that can have memorial and learning ability. and completely offline or we can say self-hosted

**Note: this is not AI platformer, this is special for AI Integrator**
Build light as possible with enhance abilty for tiny model so that can work without using too much resources, yer still powerful

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

## System Advantage for tiny model running at offline mode

### 1. Hybrid memory system:
It combines SQLite (FTS5 full-text search) with FAISS vector embeddings. This means it can recall information both through keyword matching (exact recall) and semantic similarity (contextual recall).

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

- Unity Engine AI offline Deployment (Detail in the end):

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


## Instalation
1. you need install [LemonadeServer](https://lemonade-server.ai/docs/server/) and download the model at least 2 [model](https://lemonade-server.ai/docs/server/server_models/) to run this project.

2. Install Python 3.10/3.11 (If you using windows, install it from Microsoft store).

3. Install Another requirment from script **InstallingRequirment.bat**.

4. Run ```EmbeddingModelAutoDownload.bat``` to download embedding model

5. Set Your model that you download from LemonadeServer, You can use ```ModelChecker.bat``` or, you can see from Terminal using this code ```lemonade-server list```, and change name model at ```core\Settings\1MainNameModel.txt``` and ```1SumNameModel.txt```

6. Set Your Lemonade-server location at ```core\Settings\YourLemonade-ServerLocation.txt```

The Installation will change according to the update and will be stated in the update description.

After all set now you are ready to start

## How to Run it?
Run bat file name ```Run.bat``` to run model and chit chat,

How to get API from self hosted based from this script project?

in here you can use ```RunAPI-G.bat``` for start, but what the different wihh D and G, 

Well G its build from main core function that focused for FP(Fast Response) but the trade off your model its not learning from experience, but this function still equipped with basic enhance ability, so it still have memorial ability and kind stuff like that, but not with learning so its base from your model with this project system memory.

D its build for Development and here script is use for you want to experiment with main core function, so on this function is totally run every function, but the trade off its take time to give output, you need wait extra 10-30s based on your device perfomance and model paramater.

but if you want to have FP and have main function does, you can make your own script by using ```RunAI.py``` script as refrence, on there im using modular function, so its run on FP with call the modular function that i make on core. 

note from me, good luck and have fun to experiment with it.

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
If online: these questions are answered through Google Search + summarized by the model. [Experimental]

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
but incase you want my recommendation, use Llama-8B-Instruct-Hybrid for main model and Llama-3B-Instruct-Hybrid for Sum model.

This Project was worked on Laptop Ryzen AI 7 350 with ram 24gig using IGPU.


## MY HOPE TO SOMEONE FIND THIS PROJECT
I really hope and wait for someone modify, make, and develop a project\software\mod\anything from this project, for they who gave up about develop or intergrate AI to their Game Development please dont give up, here i make a chance to give hope again, i once gave up to make it, but i shall make sure to not make someone gave up against their dream to make Game\software\project\mod\anything from AI and need entirely offline.

for those who really like modify thing, here you can use mine to learn, experiment, making anything from AI become real.

what i wanna say here, go make your project become real.

" This project is not the end, it’s just a start. What you build from it is the real story "

## How to use API to game development? v1.3
on this case im gonna using unity engine to demonstrate how to implementation AI to Game development.
to use it on C# you can paste my program template for AI on file named ```Template_Program_AIResponsForGameEngine.cs```
and you can see the detail on ```ImplementationGameDevelopment.mp4```
