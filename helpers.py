import json
import csv
from pathlib import Path
import pandas as pd

def convert(input_file):
    COLUMNS_TO_DROP = {
        "model_sql", "golden_sql", "retrieved_data",
        "golden_data", "listed_tables", "question"
    }

    def clean_result(value):
        if value is None:
            return ""
        return str(value).split("-")[0].strip()
    
    rows = []
    output_file = input_file+".csv"

    with open(input_file+".jsonl", "r", encoding="utf-8") as f:
        # Handle both .jsonl (one object per line) and .json (array)
        content = f.read().strip()
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    for record in records:
        row = {}
        for key, value in record.items():
            if key in COLUMNS_TO_DROP:
                continue
            if key == "result":
                row[key] = clean_result(value)
            else:
                row[key] = value
        rows.append(row)

    if not rows:
        print("No records found.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done — {len(rows)} rows written to {output_file}")

def rq1(input_file):
    DIFFICULTIES = ["simple", "moderate", "challenging"]

    df = pd.read_csv(input_file)
    stem = Path(input_file).stem.split("-")[-1]
    suffix = Path(input_file).suffix

    for difficulty in DIFFICULTIES:
        subset = df[df["difficulty"].str.lower().str.strip() == difficulty]
        if subset.empty:
            print(f"No rows found for difficulty: {difficulty}")
            continue
        output_file = Path(input_file).parent / "RQ1" / f"{difficulty}_{stem}{suffix}"
        subset.to_csv(output_file, index=False)
        print(f"Saved {len(subset)} rows to {output_file}")

if __name__ == "__main__":
    pass
    #convert("results_Qwen-3-235B")
    #convert("results_Qwen-3-Coder-30B")
    #convert("results_Qwen-3-Coder-480B")
    #rq1("results_Qwen-3-235B.csv")
    #rq1("results_Qwen-3-Coder-30B.csv")
    #rq1("results_Qwen-3-Coder-480B.csv")
