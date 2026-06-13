
# ⚡Deployment Preview🎮:
- Minecraft using mod Touhou Little Maid by TartaricAcid:

https://github.com/user-attachments/assets/486d2def-e81b-45ea-bcca-2236af8183dd





- Unity Engine AI offline Deployment:

https://github.com/user-attachments/assets/dc69ca07-ecb3-476d-947e-b610915ea08b

- AI Assitance (in this case im using [LAPAI UI](https://github.com/NaosaikaDevelopment/UI-LAPAI))
  


https://github.com/user-attachments/assets/0d3f242f-4518-4eec-b344-8422af5a620e



And Many more, all for support your idea to make project with AI! ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧



# Changelog📰
#### update v1.3
from here you can using self host and get API from this project, to AI development in your project without think about memory management or anything about AI management,
just paste your API you got from this project script and done, you can play or development with AI in your device without network or data go to cloud. for this have a guide check release page v1.3

#### update 1.3.1
Adding ability to command game, from this update now AI can give input for Game Development, and entirely offline, so from input user AI would decide what what user want AI todo on the game, for example here im going to debugging AI input as command:
here by using Template on Core folder, file named ```GameDevelopmentCommandAITemplate.cs``` on unity.

https://github.com/user-attachments/assets/05099062-a69d-4b0d-9cc8-1a4d58dafeba

Okay let me explain how to use: this is work as API from py i build, you can run ```GDAPIND.py``` for start API, (Note from me: Better learn how to run python automaticly when you run program from Program Language you use, tips: use and make .vbs to run without trace) 
Here how it work: 
```
[Input unity]
      ↓
Get API and send input to main API on GDAPIND.py
      ↓
AI processing on there using Llama-3.2-3B-Instruct-Hybrid (info you can change model as you want but model must be "Instruct")
      ↓
Answer return as json fill with Command, Execute, Reason. AI will decide that.
      ↓
On my program template for game development that will auto take command, execute, reason, and you can add command on there almost anythin you want
      ↓
return on unity as action.

```
#### Update Adding UI
UI its ready and has been added to new Repo, go Check [LAPAI UI](https://github.com/NaosaikaDevelopment/UI-LAPAI), im adding new feature, adding AI to get info about current time.

#### Software Release MyCompanion
This is an Side_Project for LAPAI Prototype, Its a functional software from LAPAI and LAPAI UI. this is running entirely offline, and might i say easy to install.


https://github.com/user-attachments/assets/c44a3d92-a36a-41eb-b271-80c9015d4153

Download on Release Page.


#### Update 1.3.2
Finally i have a little time to adding thing, here some new feature to use this AI.

This update adding new template program to use this AI in Arduino, and able to give command, output, and control.

Preview:




https://github.com/user-attachments/assets/cc090dc9-c6d0-49f2-b061-7fa49c607303




#### Update 1.4
fixing and improving memory, adding ui Settings, and XTTS support,
make the Installation more simple
