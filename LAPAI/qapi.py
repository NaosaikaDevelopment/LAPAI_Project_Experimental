from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from MainCore.core import *
from MainCore.runcorefp import *
initialize_core()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
print("please make sure backend is active")
print("ATTENTION making memory files, for first start maybe its take a little time")



class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: Optional[int] = None

class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[Choice]


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": cache.model_name,
                "object": "model"
            }
        ]
    }
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completion(request: ChatRequest):
    user_text = next(
        (
            m.content
            for m in reversed(request.messages)
            if m.role == "user"
        ),
        ""
    )

    print(user_text)
    reply = Main_Core_FP_Function(user_text)

    
    return ChatResponse(
        id="lapai-"+str(int(time.time())),
        choices=[Choice(
            index=0,
            message={
                "role": "system", "content": reply
            },
            finish_reason="stop"
        )]
    )
    

if __name__ == "__main__":
    Main_host="0.0.0.0"
    Main_port=2488
    print(f"\033[93m\n[INFO FROM GUIDER]\033[0m: Use this url to call the api function: http://localhost:{Main_port}/v1/chat/completions\n")

    uvicorn.run(app, host=Main_host, port=Main_port)
    
