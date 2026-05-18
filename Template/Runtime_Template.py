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

