from MainCore.runcorefp import *
initialize_core()

questions = [
    "What is the name of Rina's pet cat, and what is its favorite food?",
    "When and where was Sandi born?",
    "Who are the founders of XTech company, and in what year was it founded?",
    "What are Siska's favorite colors for her room decoration?",
    "When is the marketing team's routine evaluation meeting scheduled?",
    "Where does Budi keep his black notebook?",
    "What is the Wi-Fi password for the guest room area?",
    "How much does one VIP ticket cost for this weekend's concert?",
    "What is the distance between the head office and the nearest train station?",
    "What is the name of Andi's family's regular dentist?",
    "From which terminal and gate will the flight to Tokyo board?",
    "What vegetables can Budi not eat?",
    "What is the access code required to enter the logistics warehouse?",
    "How many and what types of fruit did Maya buy at the market?",
    "What is the ideal temperature for storing this red wine?",
    "What version of the 'SuperApp' application was just released last week?",
    "What drink does the VIP customer named Mr. Joko always order?",
    "What is the height of Mount Fiktif Peak in meters?",
    "What is Doni's position or role in the Alpha project team?",
    "At what time will the lights in the main meeting room turn off automatically?",
    "What color are the frames of the glasses Mrs. Susi wears when reading?",
    "What is the maximum battery capacity for the Z series laptop?",
    "In what year was the mango tree in the backyard planted?",
    "What caused the fire alarm on the 4th floor to go off yesterday afternoon?",
    "What is Fajar's sports shoe size?",
    "Where does the neighbor's cat named Belang often fall asleep?",
    "What date is the maximum deadline for submitting the monthly financial report?",
    "How many floors and basements does the city library building have?",
    "Where does Uncle Scrooge keep his first gold coin?",
    "Which streets does the route of bus number 45 pass through?",
    "What is the last queue number at the eye clinic today?",
    "What is the make and color of the CEO's official vehicle?",
    "At which restaurant is the HR Manager's farewell party being held?",
    "What is the name of Aunt Ratna's husband, and what is his profession?",
    "What is the minimum sales target this month for the Surabaya branch?",
    "Where is the third-quarter presentation file located?",
    "How many eggs and how much wheat flour are needed to make a pandan sponge cake?",
    "What time does the night shift for employees start and end?",
    "How many Oscar awards did the film 'The Time Explorer' win?",
    "What is the codename of the secret project being worked on by the R&D team?",
    "What foods trigger Andi's allergies?",
    "Which artist or band will perform at the company's 15th-anniversary celebration?",
    "What is the color combination of the new uniforms for the restaurant's operational staff?",
    "At which university is the first child of the Wijaya family studying?",
    "By what percentage is the budget for next month's digital marketing campaign increasing?",
    "Who holds the spare keys to the company's main safe?",
    "How often does the decorative cactus near the window need to be watered?",
    "What is the license plate number of the new official vehicle used by the Sub-district Head?",
    "Who will facilitate the orientation training session for new employees?",
    "What is the amount of the monthly internet quota provided to field team members?"
]

canswers = [
    ["miko", "tuna"],
    ["bandung", "august 14", "1995"],
    ["anton", "budi", "2018"],
    ["navy blue", "white"],
    ["tuesday", "10:00"],
    ["refrigerator", "kitchen"],
    ["tamu1234!"],
    ["1,500,000"],
    ["3.5"],
    ["sarah", "sehat"],
    ["terminal 3", "gate 5"],
    ["spinach", "water spinach"],
    ["889922"],
    ["three", "green apples", "two", "oranges"],
    ["15"],
    ["2.4"],
    ["iced americano", "without sugar"],
    ["4,230"],
    ["lead programmer"],
    ["9:00", "pm"],
    ["red"],
    ["8500"],
    ["2010"],
    ["smoke", "pantry"],
    ["42"],
    ["hood", "sedan"],
    ["5th"],
    ["5 floors", "2 basements"],
    ["wooden box", "under the mattress"],
    ["sudirman", "thamrin", "blok m"],
    ["120"],
    ["white", "toyota alphard"],
    ["bunga rampai"],
    ["hendra", "architect"],
    ["500"],
    ["q3_report", "drive d"],
    ["4", "eggs", "200", "wheat flour"],
    ["10:00 pm", "6:00 am"],
    ["5"],
    ["project phoenix"],
    ["peanuts", "seafood"],
    ["sheila on 7"],
    ["maroon", "black"],
    ["university of indonesia"],
    ["20"],
    ["finance manager", "operations director"],
    ["two weeks"],
    ["b 1234 kaa"],
    ["hendro", "hr"],
    ["50"]
]

def evaluate_answer(generated_answer, expected_keywords):
    gen_ans_lower = str(generated_answer).lower()
    for keyword in expected_keywords:
        if keyword.lower() not in gen_ans_lower:
            return 0 
    return 1
Nilai = 0
scores = []

print("=== STARTING MEMORY TEST ===")

for i, (question, keywords) in enumerate(zip(questions, canswers), 1):
    generated_answer = Main_Core_FP_Function(question)
    
    score = evaluate_answer(generated_answer, keywords)
    scores.append(score)
    Nilai += score
    
    status = "SUCCESS (1)" if score == 1 else "FAILED  (0)"
    print(f"Test {i:02d}: {status}")

print("============================")
print("Test Complete")
print(f"Total Score: {Nilai} / {len(questions)}")