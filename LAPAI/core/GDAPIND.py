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

# ====== Init FastAPI ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Cek server eksternal (lemonade) ======
clss = ["lemonade-server","status"]
print(clss)
hasil = subprocess.run(clss, capture_output=True, text=True, shell=True)
if "Server is not running" in hasil.stdout:
    print("error; server Offline, start automatic!")
    condition = False
    if os.path.exists("Settings\YourLemonade-ServerLocation.txt"):
        with open("Settings\YourLemonade-ServerLocation.txt", "r", encoding="utf-8") as f:
            asls = f.read().strip() #<--- Lset your lemonade server location
    subprocess.run(asls, shell=True)

# Inisialisasi client
LEMONADE_BASE_URL = "http://localhost:8000/api/v1"
LEMONADE_API_KEY = "lemonade"
client = OpenAI(base_url=LEMONADE_BASE_URL, api_key=LEMONADE_API_KEY)


# ====== Schema mirip OpenAI API ======
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


# ====== Endpoint utama ======
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completion(request: ChatRequest):
    user_text = " ".join([m.content for m in request.messages if m.role == "user"])

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

    completion = client.chat.completions.create(
        model="Llama-3.2-3B-Instruct-Hybrid",
        messages=prompt,
    )

    reply = completion.choices[0].message.content.strip()

    # parse JSON dari AI
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


# ====== Entry point ======
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

    subprocess.run("lemonade-server stop", shell=True)

