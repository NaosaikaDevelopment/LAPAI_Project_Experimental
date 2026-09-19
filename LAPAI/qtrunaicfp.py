import os
import sys
import time
import threading
import itertools
import subprocess

from MainCore.core import *
from MainCore.runcorefp import *
initialize_core()
condition = True
try:
    hasil = subprocess.run(
        ["lemonade-server", "status"],
        capture_output=True,
        text=True
    )

    if "Server is not running" in hasil.stdout:
        print("error; server Offline, start automatic!")
        subprocess.Popen("lemonadeServer.exe")
except Exception:
    print("Lemonade Server not installed")

done = False

def animate():
    global done

    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break

        sys.stdout.write('\rThinking ' + c)
        sys.stdout.flush()
        time.sleep(0.1)

    sys.stdout.write('\r' + ' ' * 20 + '\r')
    sys.stdout.flush()

print("ATTENTION making memory files, for first start maybe its take a little time")
print("Startup The system...\n")

user_msg = "[INFO] User is Online"
reply = Main_Core_FP_Function(user_msg)

print("\033[92m\nLAPAI: \033[0m", end="", flush=True)

for ch in reply:
    print(ch, end="", flush=True)
    time.sleep(0.02)

print()

while condition:

    try:
        user_msg = input(
            "\033[93m\nUser: \033[0m"
        ).strip()

    except (EOFError, KeyboardInterrupt):
        break

    if not user_msg:
        continue

    if user_msg.lower() in {"exit", "quit", "q"}:

        append_message(
            cache.session_id,
            cache.session_file,
            "system",
            "Closing Program, Shutting down system."
        )

        break

    try:
        done = False

        t = threading.Thread(
            target=animate,
            daemon=True
        )
        t.start()

        reply = Main_Core_FP_Function(user_msg)

        done = True
        t.join()

        print(
            "\033[92m\nLAPAI: \033[0m",
            end="",
            flush=True
        )

        for ch in reply:
            print(ch, end="", flush=True)
            time.sleep(0.02)

        print()

    except Exception as e:

        done = True

        try:
            t.join()
        except:
            pass

        print(f"[ERROR] on RunAI: {e}")