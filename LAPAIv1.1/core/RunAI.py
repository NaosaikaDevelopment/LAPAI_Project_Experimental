import LAPAI_Core
import subprocess



#Fungsi;
clss = ["lemonade-server","status"]
hasil = subprocess.run(clss, capture_output=True, text=True, shell=True)
if "Server is not running" in hasil.stdout:
    print("error; server Offline, start automatic!")
    asls = r"..\Lemonade-Location\bin\lemonade_server.vbs"
    subprocess.run(asls, shell=True)
    LAPAI_Core.Tm()
else:
    LAPAI_Core.Tm()