import sysconfig, pathlib
import os
import shutil
import PyInstaller.__main__

Mainfile = "maingui.py"  
argumen = [
    Mainfile,
    '--onefile',           
    '--clean',              
    #'--noconsole',       
    '--icon=logo.ico',    
]

old_w = os.path.join("dist", "maingui.exe")
old_l = os.path.join("dist", "maingui")
new_w = "jsonEd.exe"
new_l = "jsonEd"
dist = "dist"
build = "build"
cahceFile = "maingui.spec"

if os.path.exists(new_w):
    print(f"file already exists")

elif os.path.exists(new_l):
    print(f"file already exists")

else:

    try:
        print("Packing Maingui")
        PyInstaller.__main__.run(argumen)
    except Exception as e:
        print(f"\n Error while packing Maingui.py: {e}")
    
    if os.path.exists(old_w):
        shutil.move(old_w, new_w)
        print("WindowsDetected, jsonEd.exe added")
    elif os.path.exists(old_l):
        shutil.move(old_l, new_l)
        print("Linux Detected, jsonEd added!")
    else:
        print("Error added jsonEd, please when you need to change Settings conf.json, edit it manually")


if os.path.exists(dist):
    shutil.rmtree(dist)
    os.remove(cahceFile)
    shutil.rmtree(build)
    print("Deleting Temp folder&file!")

p = pathlib.Path(sysconfig.get_paths()["purelib"]) / "lapai.pth"
p.write_text(str(pathlib.Path(__file__).resolve().parent))
print("LAPAI registered:", p)
