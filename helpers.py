import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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


def scatter_plot(input_file):
    input_path = Path(f"./base_csv/"+input_file)

    # Load data
    df = pd.read_csv(input_path)

    # Keep only rows where efficiency_score is defined
    df = df[df["result"] == "correct"].copy()
    df = df[df["efficiency_score"].notna()].copy()

    # Clean difficulty labels
    df["difficulty_clean"] = df["difficulty"].astype(str).str.strip().str.lower()

    plt.figure(figsize=(8, 5))

    # One scatter layer per difficulty
    for difficulty in ["simple", "moderate", "challenging"]:
        subset = df[df["difficulty_clean"] == difficulty]
        plt.scatter(
            subset["question_id"],
            subset["efficiency_score"],
            s=12,
            alpha=0.7,
            label=difficulty
        )

    plt.xlabel("question_id")
    plt.ylabel("Runtime ratio (model SQL / golden SQL)")
    plt.title("Runtime ratio by question and difficulty")
    plt.legend(title="difficulty")
    plt.tight_layout()

    output_file = Path(input_file).parent / "RQ1" / f"scatter_by_difficulty_{input_path.stem.split('-')[-1]}.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return output_file

if __name__ == "__main__":
    pass
    #convert("results_Qwen-3-235B")
    #convert("results_Qwen-3-Coder-30B")
    #convert("results_Qwen-3-Coder-480B")
    #rq1("results_Qwen-3-235B.csv")
    #rq1("results_Qwen-3-Coder-30B.csv")
    #rq1("results_Qwen-3-Coder-480B.csv")
    scatter_plot("results_Qwen-3-235B.csv")
    scatter_plot("results_Qwen-3-Coder-30B.csv")
    scatter_plot("results_Qwen-3-Coder-480B.csv")
