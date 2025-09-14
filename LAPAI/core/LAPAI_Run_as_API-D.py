from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import uvicorn

from LAPAI_Core import *
import LAPAI_Core  
app = FastAPI()



if "Server is not running" in hasil.stdout:
    print("error; server Offline, start automatic!") 
    condition = False
    asls = r"..\lemonade-server\location\bin\lemonade_server.vbs" #<----- ATTENTION set your lemonade server location
    subprocess.run(asls, shell=True)
print("ATTENTION making memory files, for first start maybe its take a little time")
init_db()
init_learning_db()
faiss_index, id_map = init_faiss()
title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
title_learn = "Learning"+title_hint
session_id, session_file = create_session(title_hint)
seid, jsfile = create_session_Learning(title_learn)
print("[INFO] Session Created", title_hint, "\n")












# --- Schema mirip OpenAI API ---
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
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completion(request: ChatRequest):
    user_text = " ".join([m.content for m in request.messages if m.role == "user"])
    print(user_text)
    reply = Main_Core_Function(user_text, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)

    try:
        return ChatResponse(
            id="lapai-"+str(int(time.time())),
            choices=[Choice(
                index=0,
                message={"role": "assistant", "content": reply},
                finish_reason="stop"
            )]
        )
    except Exception as e:
        print(f"[ERROR]: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

