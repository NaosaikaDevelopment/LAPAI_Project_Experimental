# LAPAI [experimental]
## What is LAPAI?
Local Agent Personal Artificial Artificial Intelligence, a project based fondation from Lemonade Server, that work as script to give AI Feature that can have memorial and learning ability.
Go check their website to learn more! ```https://lemonade-server.ai```

Local Agent LA-PAI is an experimental personal AI that reflects on its own conversations using self-questioning and dual-mode learning—entirely on your device.
And this project is can run entirely offline that work in your own device. dual model learning its not that special, its more like offline mode and online mode

Every Memory and knowledge its save Externally, so even you change model, AI memory and knowledge will not deleted, have fun experiment with it

this script isn’t just a regular chatbot , it’s a self-learning, memory-based AI framework that blends keyword search, semantic recall, and summarization into one system. It’s especially strong for experiments in long-term memory AI, personalized assistants, and lightweight hybrid LLM deployments.


## System Advantage for tiny model running at offline mode

### Hybrid memory system:
It combines SQLite (FTS5 full-text search) with FAISS vector embeddings. This means it can recall information both through keyword matching (exact recall) and semantic similarity (contextual recall).

### Persistent sessions:
Conversations are saved in JSON and databases, so the assistant can resume past dialogues and maintain continuity.

### Embedding flexibility:
It uses an ONNX model (all-mpnet-base-v2) for efficient embeddings with GPU/DirectML support, making it lighter and portable across hardware.

### Summarization and compression:
Long sessions are summarized automatically using a secondary model, preventing memory bloat while keeping important facts.

### Knowledge learning mode:
A separate learning database lets the system extract insights, form new knowledge entries, and store them for reuse—giving it a "growing memory."

### Personal data extraction & storage:
With simple classification, it detects if user input contains personal information, extracts it, and stores it in a personal file.

### Custom persona support:
It loads personality instructions from PersonaAI.txt, so users can shape the assistant’s behavior without modifying the code.

### Streaming-like output animation:
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




## Instalation
1. you need install [LemonadeServer](https://lemonade-server.ai/docs/server/) and download the model at least 2 [model](https://lemonade-server.ai/docs/server/server_models/) to run this project.

2. Install Python 3.10/3.11 (If you using windows, install it from Microsoft store).

3. Install Another requirment from script **Auto_Installing_Env.bat**.

4. Set Your model that you download from LemonadeServer, you can see from Terminal using this code ```lemonade-server list```, and change name model at **LAPAI_Core.py** and **LAPAI_Learn_Core.py**,
   ```MODEL_NAME="..."``` and ```Sum_model="..."```

5. Set Your Lemonade-server location at script ```RunAI.py``` in ```asls = r".../Lemonade-Location/Lemonade-server.vbs"``` 

The Installation will change according to the update and will be stated in the update description.

After all set now you are ready to start

## How to Run it?
Run bat file name ```RUN.bat``` to run model and chit chat, or Run bat file name ```Learn.py``` to start learning

## How exactly this script run?

### LAPAI Core
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

