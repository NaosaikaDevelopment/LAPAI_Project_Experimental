import LAPAI_Core
from core/LAPAI_Core import *
import subprocess
condition = True


#Fungsi;
clss = ["lemonade-server","status"]
hasil = subprocess.run(clss, capture_output=True, text=True, shell=True)
if "Server is not running" in hasil.stdout:
    print("error; server Offline, start automatic!")
    condition = False
    asls = r"..\lemonade-server\location\bin\lemonade_server.vbs" #<---- Set your lemonade server location
    subprocess.run(asls, shell=True)

#_________________MAIN SYSTEM FROM MAIN CORE___________________
condition = True
print("ATTENTION making memory files, for first start maybe its take a little time")
init_db()
init_learning_db()
faiss_index, id_map = init_faiss()
title_hint = datetime.now().strftime("Sesi_%Y%m%d_%H%M%S")
title_learn = "Learning"+title_hint
session_id, session_file = create_session(title_hint)
seid, jsfile = create_session_Learning(title_learn)
print("[INFO] Session Created", title_hint, "\n")
print("Startup The system... \n")
user_msg = "[INFO] User is Online"
reply = Main_Core_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)
print("\033[92m\nLAPAI: \033[0m", end="", flush=True) #<---Animated output :D just bored spawn blend like that
for ch in reply:
    print(ch, end="", flush=True)
    time.sleep(0.02)
print()
while condition:
    try:
        user_msg = input("\033[93m\nUser: \033[0m").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKeluar.")
        break
    if user_msg.lower() in {"exit", "quit", "q"}:
        append_message(session_id,session_file,"SYSTEM","Closing Program, Shutdowning system,[INFO]User Offline")
        break
    reply = Main_Core_Function(user_msg, faiss_index, id_map, title_hint, title_learn, session_id, session_file, seid, jsfile)

    print("\033[92m\nLAPAI: \033[0m", end="", flush=True) #<---Animated output :D just bored spawn blend like that
    for ch in reply:
        print(ch, end="", flush=True)
        time.sleep(0.02)
    print()

subprocess.run("lemonade-server stop", shell=True)


