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
```python
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
```python
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
```C#
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
```C#
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
```python
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
```ino
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

## _🚨Note to experiment with it_
Some code you check maybe had error because the "Location not found" or something like that.
im really sorry, sometime i make testing and forgot to clear/change it.

if you open some python file in core folder (For you wanna to experiment or learn from it) in this:

```python
clss = ["lemonade-server", "status"]
hasil = subprocess.run(clss, capture_output=True, text=True, shell=True)
if "Server is not running" in hasil.stdout:
    print("error; server Offline, start automatic!")
    condition = False
    asls = r"D:\ND\bin\lemonade_server.vbs"
    subprocess.run(asls, shell=True)
```
you can change to:
```python
try:
    clss = ["lemonade-server","status"]
    hasil = subprocess.run(clss, capture_output=True, text=True, shell=True)
    if "Server is not running" in hasil.stdout:
        print("error; server Offline, start automatic!")
        condition = False
        subprocess.run("lemonadeServer.exe", shell=True)
except Exception as e:
    print("Lemonade Server not installed")
```
and some note too, if you want to use template, you can download in this repo in "Template" folder, have fun! ٩(ˊᗜˋ*)و ♡
