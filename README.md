# LAPAI Local Agent Personal Artificial Intellegence [experimental]

<img width="1920" height="1080" alt="Start" src="https://github.com/user-attachments/assets/a836bfa6-a69e-4053-8b0b-770c1a2c09ea" />



### What is LAPAI? 📋

LAPAI Project is a project for Local AI Runtime, this is work as Runtime to give AI Feature that can have memorial and learning ability. all completely offline, with backend LemonadeServer for Ryzen and Ollama for non Ryzen. It for people want to have its own AI without the heavy wraper that can bleeding the pc resource for its own project. This project for those who want to have its own AI local but didn't want to using heavy AI, at the same time want to using light AI but didn't want to make the complex system so it can peform better and the most important Work Locally or Offline.

**Note: this is not AI platformer, this is special for AI Integrator** (ᵕ—ᴗ—).
This project work for them seeking AI with API Open AI style and work for its own project sake,
and for them who want mod, create, learn, project, game to using AI locally (  ≧ᗜ≦).
Build light as possible with enhance abilty for tiny model so that can work without using too much resources, yet still powerful.
Every Memory and knowledge its save Externally, so even you change model, AI memory and knowledge will not deleted, have fun experiment with it (˶ᵔ ᵕ ᵔ˶).


## ✅This system project equipped with:
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

# 🟢Use Cases

### Personal AI assistant:
Tracks conversations, remembers context, and adapts responses over time.

### Learning system:
Extracts “thoughts” and knowledge points from conversations, building a personalized knowledge base.

### Experiment platform:
Since it integrates OpenAI-like APIs and local ONNX embeddings, it’s a good playground for experimenting with hybrid AI systems (local + remote inference).

### Privacy-aware applications:
By separating personal data into its own text file, it makes compliance with privacy rules more manageable.


# ⚡Deployment Preview🎮:
- Minecraft using mod Touhou Little Maid by TartaricAcid:

https://github.com/user-attachments/assets/486d2def-e81b-45ea-bcca-2236af8183dd





- Unity Engine AI offline Deployment:

https://github.com/user-attachments/assets/dc69ca07-ecb3-476d-947e-b610915ea08b

- AI Assitance (in this case im using [LAPAI UI](https://github.com/NaosaikaDevelopment/UI-LAPAI))
  


https://github.com/user-attachments/assets/0d3f242f-4518-4eec-b344-8422af5a620e



And Many more, all for support your idea to make project with AI!

# ⚠️Hardware min Recommendation:
NOTE: its based on your model parameter

Ram min 24gig or more for play and run AI smoothly. 

CPU/APU AI series Ryzen AI (7000 series)/Intel Ultra or I5-12th/I7 series/Ryzen7/9 (The more Core its more recomended), 
that is mandatory or if you dont have powerfull cpu, atleast have Powerfull GPU.

GPU : RTX 2060/3000+ / IGPU  from IGPU: Ryzen AI/Intel Ultra (Vram 6-8gigs)


# 🚀Getting Started
### To install
before to installation make sure you have the Runtime Backend provider (LemonadeServer / Ollama)
1. Install Pyton3.10
2. Run ```AutoDownloadALL.bat```
3. Wait until done, and you all set

https://github.com/user-attachments/assets/a54b6656-634f-46b0-bbf0-0b579510f5da





### To Use 
#### 🍿1. Basic run test
1. Run ```RunTTS.bat``` to turn on XTTS (optional)
2. go to core/app-settings/dist/win-unpacked and make shortcut for ```LAPAI Settings editor.exe``` it for settings from model name you use to persona you want
3. run LAPAI Settings editor and set your model and persona (OLLAMA / LemonadeServer)
4. Run ```Run.bat``` to run it in Terminal to test it, then to test API from the project Run ```RunAPI-G.bat``` or ```RunAPI-G_WithoutTrace.vbs``` to run it without tracer this one for you want to use it in another project or game or development
   
https://github.com/user-attachments/assets/0a183efe-94d2-4406-9f47-676e60770dcf

#### 🔮2. API using - advance customize

make sure backend Lemonade or Ollama its activate or running

Then run LAPAI environment ```This-project-located\LAPAIv1.4\MainCore\core\LAPAI\Scripts\Activate ```:
  - Run Terminal (Search in windows and type "Terminal")
  - change directory to the project located ```cd this-project-located\LAPAIv1.4```
  - then paste ```MainCore\core\LAPAI\Scripts\Activate```

The Terminal you using now using LAPAI environment

Make your runtime here with python:
```
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import uvicorn
import subprocess
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi client
BASE_URL = "http://localhost:8000/api/v1" # make sure your its same with url backend you using
LEMONADE_API_KEY = "lemonade" # just for lemonade
client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)



class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: Optional[int] = None


class ActionMessage(BaseModel):
    command: str
    execute: bool
    reason: str


class Choice(BaseModel):
    index: int
    message: ActionMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[Choice]



@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completion(request: ChatRequest):
    user_text = " ".join([m.content for m in request.messages if m.role == "user"])

    prompt = []

    completion = client.chat.completions.create(
        model="Llama-3.2-3B-Instruct-Hybrid",
        messages=prompt,
    )

    reply = completion.choices[0].message.content.strip()

    parsed = json.loads(reply)

    return ChatResponse(
        id=str(uuid.uuid4()),
        choices=[
            Choice(
                index=0,
                message=ActionMessage(
                    command=parsed.get("command", ""),
                    execute=parsed.get("execute", False),
                    reason=parsed.get("reason", ""),
                ),
                finish_reason="stop",
            )
        ],
    )



if __name__ == "__main__":
    Main_host = "0.0.0.0"
    Main_port = 3440
    print(
        f"\033[93m\n[INFO FROM GUIDER]\033[0m: Use this url to call the api function: http://{Main_host}:{Main_port}/v1/chat/completions\n"
    )
    print(
        f"\033[93m\n[INFO FROM GUIDER]\033[0m: if error cannot connect try change the url: http://localhost:{Main_port}/v1/chat/completions\n"
    )

    uvicorn.run(app, host=Main_host, port=Main_port)


 ```

**To use in unity** 

--for action command--

1. Prompt in Runtime you make paste this:
```
prompt = [
        {
            "role": "system",
            "content": (
                "You are an AI that only returns valid JSON with keys: "
                "'command' (string), 'execute' (boolean), 'reason' (string). "
                "If the intention is to walk, set command='walk' and execute=true, "
                "else execute=false. Always explain in 'reason'."
            ),
        },
        {"role": "user", "content": user_text},
    ]
```
3. Run the API runtime you make in Terminal tab with LAPAI env in then type: ```py your-file-location/your-runtime-filename.py```

4. you can use template from ```GameDevelopmentCommandAITemplate.cs``` or copy - 
```
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;
using UnityEngine.Networking;
using System.Text;
using System.Diagnostics;

public class AIchatRespons : MonoBehaviour
{
    [Header("UI References")]
    //optional - Set as you want, modify as you want
    public TMP_InputField userInputField;   // drag from Inspector
    public TMP_Text aiResponseText;         // drag from Inspector
    public Button sendButton;               // drag from Inspector

    [Header("Server Settings")]
    private string apiUrl = "http://localhost:3440/v1/chat/completions";
    private string modelName = "Llama-3.2-3B-Instruct-Hybrid"; //Model name you use, you can change as you want


    private void Start()
    {
        sendButton.onClick.AddListener(OnSendClicked);
    }

    private void OnSendClicked()
    {
        string userInput = userInputField.text;
        if (!string.IsNullOrEmpty(userInput))
        {
            StartCoroutine(SendToLocalAI(userInput));
            userInputField.text = "";
        }
    }

    private IEnumerator SendToLocalAI(string userInput)
    {
        string jsonData = @"{
            ""model"": """ + modelName + @""",
            ""messages"": [
                { ""role"": ""user"", ""content"": """ + userInput + @""" }
            ]
        }";

        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var apiKey = "lemonade"; //its actually optional but better than nothing
            request.SetRequestHeader("Authorization", "Bearer " + apiKey);

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                UnityEngine.Debug.Log("Raw Response: " + json);

                // Parse response JSON
                LocalAIResponse parsed = JsonUtility.FromJson<LocalAIResponse>(json);

                if (parsed != null && parsed.choices != null && parsed.choices.Length > 0)
                {
                    ActionMessage msg = parsed.choices[0].message;

                    // Command AI to action
                    if (msg.execute)
                    {
                        //For example:
                        if (msg.command == "walk")
                        {
                            UnityEngine.Debug.Log("AI says: WALK");
                            // GetComponent<CharacterController>().Move(Vector3.forward * Time.deltaTime * 5f);
                        }
                        else if (msg.command == "jump")
                        {
                            UnityEngine.Debug.Log("AI says: JUMP");
                            // GetComponent<Animator>().SetTrigger("Jump");
                        }
                        // add anything you wat here, base from your command to AI, AI would automatic set as you order, if shoot AI will shoot, if walk AI will walk, anything
                    }
                    else
                    {
                        UnityEngine.Debug.Log("AI decided not to execute command: " + msg.reason);
                    }
                }
                else
                {
                    aiResponseText.text = "No response";
                }

            }
            else
            {
                aiResponseText.text = "Error: " + request.error;
            }
        }
    }

    [System.Serializable]
    public class ActionMessage
    {
        public string command;
        public bool execute;
        public string reason;
    }

    [System.Serializable]
    public class Choice
    {
        public int index;
        public ActionMessage message;
        public string finish_reason;
    }

    [System.Serializable]
    public class LocalAIResponse
    {
        public string id;
        public string @object;
        public Choice[] choices;
    }


}

```
this example for the template to using AI responses and inject it in Unity as action, so be creative ദ്ദി ˉ͈̀꒳ˉ͈́ )✧

https://github.com/user-attachments/assets/05099062-a69d-4b0d-9cc8-1a4d58dafeba

--AI Responses--

1. no need prompt, you can use from ```RunAPI-G.bat```
2. you can use template from ```Template_Program_AIResponsForGameEngine.cs``` or copy -
```
using UnityEngine;
using UnityEngine.UI;
using TMPro; 
using System.Collections;
using UnityEngine.Networking;
using System.Text;

public class AIRespons : MonoBehaviour
{
    [Header("UI References")]
    public TMP_InputField userInputField;   // drag from Inspector
    public TMP_Text aiResponseText;         // drag from Inspector
    public Button sendButton;               // drag from Inspector

    [Header("Server Settings")]
    private string apiUrl = "Place-your-Url-here";
    private string modelName = "Place-your-main-model-here"; 

    string persona = ""; // here if you wanna set persona but make sure PersonaAI on settings ist empty

    private void Start()
    {
        sendButton.onClick.AddListener(OnSendClicked);
    }

    private void OnSendClicked()
    {
        string userInput = userInputField.text;
        if (!string.IsNullOrEmpty(userInput))
        {
            StartCoroutine(SendToLocalAI(userInput));
            userInputField.text = ""; 
        }
    }

    private IEnumerator SendToLocalAI(string userInput)
    {
        string jsonData = @"{
            ""model"": """ + modelName + @""",
            ""messages"": [
                { ""role"": ""system"", ""content"": """ + persona + @""" },
                { ""role"": ""user"", ""content"": """ + userInput + @""" }
            ]
        }";

        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var apiKey = "lemonade"; //its actually optional but better than nothing
            request.SetRequestHeader("Authorization", "Bearer " + apiKey);

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                Debug.Log("Raw Response: " + json);

                // Parse response JSON
                LocalAIResponse parsed = JsonUtility.FromJson<LocalAIResponse>(json);
                if (parsed != null && parsed.choices != null && parsed.choices.Length > 0)
                {
                    aiResponseText.text = parsed.choices[0].message.content;
                }
                else
                {
                    aiResponseText.text = "No response";
                }
            }
            else
            {
                aiResponseText.text = "Error: " + request.error;
            }
        }
    }


    [System.Serializable]
    public class LocalAIResponse
    {
        public Choice[] choices;
    }

    [System.Serializable]
    public class Choice
    {
        public Message message;
    }

    [System.Serializable]
    public class Message
    {
        public string role;
        public string content;
    }
}

```
Then from here you able to talk with AI, perfect for talkable npc, because of the tiny parameter model we can use (  ≧ᗜ≦).


https://github.com/user-attachments/assets/dc69ca07-ecb3-476d-947e-b610915ea08b

**Arduino use**

1. you can add your instruction in prompt all you want, for example like this:
```
prompt = [
        {
            "role": "system",
            "content": (
                "You are an AI that only returns valid JSON with keys: "
                "'command' (string), 'execute' (boolean), 'reason' (string). "
                "Valid commands: "
                "'LED_A_ON', 'LED_A_OFF', 'LED_B_ON', 'LED_B_OFF'. "
                "else execute=false. Always explain in 'reason'."
            ),
        },
        {"role": "user", "content": user_text},
    ]
```
2. and your arduino code should be like this:
```
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_MOSI   9
#define OLED_CLK   10
#define OLED_DC    11
#define OLED_CS    12
#define OLED_RESET 13
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);


void setup()
{
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT); 
  Serial.begin(9600);
  if(!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println(F("SSD1306 allocation failed"));
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("AI STATUS: OFFLINE");
  display.display();
}
void loop() {
    if (Serial.available()) {
<img width="1920" height="1080" alt="Hijau dan Putih Modern Prompt AI Populer Presentasi" src="https://github.com/user-attachments/assets/54de4897-61fd-42e5-a922-0983ae15c295" />

        String ledcmd = Serial.readStringUntil('\n');
        ledcmd.trim();

        if (ledcmd == "LED_A_ON") digitalWrite(2, HIGH);
        else if (ledcmd == "LED_A_OFF") digitalWrite(2, LOW);
        else if (ledcmd == "LED_B_ON") digitalWrite(3, HIGH);
        else if (ledcmd == "LED_B_OFF") digitalWrite(3, LOW);

        String oledMsg = Serial.readStringUntil('\n');
        oledMsg.trim();
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(oledMsg);
        display.display();
    }
}

```

maybe you would be need some extension from ardn like Adafruit_GFX and Adafruit_SSD1306, to handle the screen.

this example using it to control LED and show output in LED or Screen OLED or what it is ( ദ്ദി ˙ᗜ˙ ).

## Changelog
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
fixing and improving memory, adding ui Settings, and XTTS support,
make the Installation more simple










## License
LAPAI source code is licensed under the MIT License. - see the [LICENSE](LICENSE) file for details.

### Third-party components

Third-party components and AI models are licensed
under their own respective licenses:

- FAISS (MIT)
- ONNXRuntime (MIT)
- Coqui XTTS-v2 (CPML)
- all-mpnet-base-v2 (Apache 2.0)
- SentenceTransformers (Apache 2.0)
- HuggingFace Transformers (Apache 2.0)

LAPAI does not redistribute third-party model weights.
Models are downloaded or installed separately by users.

Users are responsible for complying with the licenses
of all third-party components and models.


# Note
### Keep in mind this project is Experimental and Worked Alone by me (ND)
### Future plan:
- Adding Learning from Online
- Can gather information from online

### this project leading to AI assistant offline/online


## Info: This project will be hiatus due I who created this project, don't have time to continue developing it for a while because I am in a language course for Ausbildung
