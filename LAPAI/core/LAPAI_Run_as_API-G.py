from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from LAPAI_Core import *
import LAPAI_Core  
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # atau spesifik "http://localhost:5500"
    allow_credentials=True,
    allow_methods=["*"],  # termasuk OPTIONS
    allow_headers=["*"],
)
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
print("ATTENTION making memory files, for first start maybe its take a little time")
init_db()
init_learning_db()
faiss_index, id_map = init_faiss()
title_hint = "FromAPI-G"+datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
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
    data = repr(user_text)
    print(data)
    reply = Main_Core_FP_Function(user_text, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)

    
    return ChatResponse(
        id="lapai-"+str(int(time.time())),
        choices=[Choice(
            index=0,
            message={"role": "assistant", "content": reply},
            finish_reason="stop"
        )]
    )
    

if __name__ == "__main__":
    Main_host="0.0.0.0"
    Main_port=8080
    print(f"\033[93m\n[INFO FROM GUIDER]\033[0m: Use this url to call the api function: http://{Main_host}:{Main_port}/v1/chat/completions\n")
    print(f"\033[93m\n[INFO FROM GUIDER]\033[0m: if error cann't connected try change the url: http://localhost:{Main_port}/v1/chat/completions\n")

    uvicorn.run(app, host=Main_host, port=Main_port)
    
    subprocess.run("lemonade-server stop", shell=True)
    append_message(session_id, session_file, "SYSTEM", "[INFO] User Offline.")
    append_message(session_id, session_file, "SYSTEM", "System Offline.")
    





