from pathlib import Path
import json
import random
import os

if __name__ == "__main__":
    with open("test_cases - filtered.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    simple_cases, moderate_cases, challenging_cases = [], [], []
    random.shuffle(questions)
    counts = [0, 0, 0]
    for q in questions:
        if q["difficulty"].lower() == "simple":
            simple_cases.append(q)
        elif q["difficulty"].lower() == "moderate":
            moderate_cases.append(q)
        elif q["difficulty"].lower() == "challenging":
            challenging_cases.append(q)
        else:
            print("weird test case: \n")
            print(q["question_id"])
    
    print(len(simple_cases), len(moderate_cases), len(challenging_cases))

    simple_cases = random.sample(simple_cases, 100)
    moderate_cases = random.sample(moderate_cases, 100)
    challenging_cases = random.sample(challenging_cases, 100)
    
    print(len(simple_cases), len(moderate_cases), len(challenging_cases))

    with open("./sampled_cases.json", "w", encoding="utf-8") as f:
        json.dump(simple_cases+moderate_cases+challenging_cases, f, indent=2, ensure_ascii=False)

