# LAPAI [experimental]
## What is LAPAI?
Local Agent Personal Artificial Artificial Intelligence, a project based fondation from Lemonade Server, that work as script to give AI Feature that can have memorial and learning ability.
Go check their website to learn more! ```https://lemonade-server.ai```
## Instalation
1. you need install [LemonadeServer](https://lemonade-server.ai/docs/server/) and download the model at least 2 [model](https://lemonade-server.ai/docs/server/server_models/) to run this project.

2. Install Python 3.10/3.11 (If you using windows, install it from Microsoft store).

3. Install Another requirment from script **Auto_Installing_Env.bat**.

4. Set Your model that you download from LemonadeServer, you can see from Terminal using this code ```lemonade-server list```, and change name model at **LAPAI_Core.py** and **LAPAI_Learn_Core.py**,
   ```MODEL_NAME="..."``` and ```Sum_model="..."```

After all set now you are ready to start

## How to Run it?
Run bat file name ```RUN.bat``` to run model and chit chat, or Run bat file name ```Learn.py``` to start learning


## How exactly this script run?

### LAPAI Core
User talks → input enters.

System pulls old memory + related knowledge.

Main model answers.

System summarizes long interactions → creates new questions → stores in question list (self-learning material).

This process repeats, so the agent has short-term memory, long-term knowledge, and a reflective question list that can be used for the next learning.

### LAPAI Learn Core
```
Chat history → Extract keywords → Recall related memory
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

# License
This project using MIT license and Apache 2.0 from LemonadeServer

# Note
### Keep in mind this project is Experimental and Worked Alone by me (ND)

