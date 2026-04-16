import json
import random
import time
from unittest import result
from openai import OpenAI
import boto3
import sqlite3
import re
import os

# =========================
# CONFIG & Models
# =========================
region = "eu-north-1"
api_key = ""

models = {
    "qwen.qwen3-coder-30b-a3b-v1:0": "Qwen 3 Coder 30B",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen 3 235B",
    "qwen.qwen3-coder-480b-a35b-v1:0": "Qwen 3 Coder 480B"
}

# =========================
# Bedrock Client
# =========================

def execute_sql(db_id, query):
    db_path = f"./databases/{db_id}/{db_id}.sqlite"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        conn.close()

        return rows

    except Exception as e:
        return f"ERROR: {str(e)}"

def call_model(messages, max_tokens=512):
    """
    OpenAI-style wrapper for Bedrock (Claude-style models).
    """
    client = OpenAI(
        api_key=api_key,
        base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
    )

    response = client.chat.completions.create(
        model="qwen.qwen3-235b-a22b-2507-v1:0",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens: {total_tokens}")

    return response.choices[0].message.content


# =========================
# Prompt 1
# =========================
def build_prompt1(case):
    schema_lines = []
    for key, value in case.items():
        if key not in ["question", "difficulty", "SQL", "evidence", "question_id", "db_id"]:
            table_name = key
            # extract column names roughly
            cols = []
            for line in value.split("\n"):
                line = line.strip()
                if line.startswith('"'):
                    col = line.split('"')[1]
                    cols.append(col)
            schema_lines.append(f"{table_name}({', '.join(cols)})")

    schema_text = "\n".join(schema_lines)

    return f"""
You are a natural-language-to-SQL schema selector.

Given the database schema below and the natural language question, identify the smallest set of tables that are likely needed to answer the question. Include any join/bridge tables that may be needed. Be conservative: if a table might be relevant, include it.

Return only a comma-separated list of table names.

Natural Language Question:
{case["question"]}

Database Schema:
{schema_text}
"""

def clean_sql(response_text):
    # Remove ```sql ... ``` or ``` ... ```
    cleaned = re.sub(r"```sql|```", "", response_text, flags=re.IGNORECASE)
    return cleaned.strip()

# =========================
# Prompt 2
# =========================
def build_prompt2(case, selected_tables):
    schema_blocks = []
    for table in selected_tables:
        table = table.strip()
        if table in case:
            schema_blocks.append(case[table])

    schema_text = "\n\n".join(schema_blocks)

    return f"""
You are a text-to-SQL expert.

Using the selected tables below, the database clarifications, and the natural language question, write one SQLite SQL query that answers the question.

Selected Tables:
{', '.join(selected_tables)}

Database Schema for Selected Tables:
{schema_text}

Clarifications:
{case.get("evidence", "")}

Natural Language Question:
{case["question"]}

Return only the SQL query.
"""

def normalize(value):
    """Normalize values for comparison."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


def compare_numbers(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except:
        return False


def compare_lists(a, b):
    """ Compare lists """
    if len(a) != len(b):
        return False

    return a == b


def evaluate_result(retrieved_data, golden_data):
    # 1. Exact match
    if retrieved_data == golden_data:
        return True, "correct"

    # 2. Handle None
    if retrieved_data is None or golden_data is None:
        return retrieved_data == golden_data, "handling none"

    # 3. Numbers (with tolerance)
    if isinstance(retrieved_data, (int, float)) or isinstance(golden_data, (int, float)):
        return compare_numbers(retrieved_data, golden_data), "numeric comparison"

    # 4. Strings
    if isinstance(retrieved_data, str) and isinstance(golden_data, str):
        return normalize(retrieved_data) == normalize(golden_data), "string comparison"

    # 5. Lists (e.g., SQL rows)
    if isinstance(retrieved_data, list) and isinstance(golden_data, list):
        return compare_lists(retrieved_data, golden_data), "list comparison"

    # 7. Fallback - Can create FN
    return False, "other"

def save_record(case, model, sql, selected_tables, model_result, golden_result, efficiency_score, result, j):
    output_path = f"results_{models[model]}.jsonl"
    record = {
        "model_name": models[model],
        "question_id": case["question_id"],
        "question": case["question"],
        "difficulty": case["difficulty"],
        "golden_sql": case["SQL"],
        "model_sql": sql,
        "listed_tables": selected_tables,
        "retrieved_data": model_result,
        "golden_data": golden_result,
        "result": result,
        "efficiency_score": efficiency_score,
        "repetition_number": j + 1
    }
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    
def main():
    with open("sampled_cases.json", "r") as f:
        data = json.load(f)

    for model in models.keys():

        for i, case in enumerate(data):
            for j in range(1):
                print(f"\n====================")
                print(f"CASE {i+1} ({case['difficulty']})")
                print(f"QUESTION: {case['question']}\n")

                # ---- Prompt 1 ----
                prompt1 = build_prompt1(case)
                messages1 = [{"role": "user", "content": prompt1}]
                tables_output = call_model(messages1)

                print("Selected Tables:")
                print(tables_output)

                selected_tables = [t.strip() for t in tables_output.split(",")]

                print(selected_tables)

                # ---- Prompt 2 ----
                prompt2 = build_prompt2(case, selected_tables)
                messages2 = [{"role": "user", "content": prompt2}]
                sql_output = call_model(messages2, max_tokens=1024)

                print("\nGenerated SQL:")
                print(sql_output)

                sql = clean_sql(sql_output)

                print("\nExecuting SQL...")
                model_start = time.time()
                try:
                    model_result = execute_sql(case["db_id"], sql)
                except sqlite3.Error as e:
                    msg = str(e).lower()

                    if "no such table" in msg or "no such column" in msg:
                        res = f"hallucination - {msg}"
                    elif "syntax error" in msg or "near" in msg:
                        res = f"syntactic - {msg}"
                    else:
                        res = f"other - {msg}"

                    save_record(case, model, sql, selected_tables, None, None, None, res, j)
                    continue
                model_end = time.time()

                model_runtime = model_end - model_start
                
                golden_start = time.time()
                try:
                    golden_result = execute_sql(case["db_id"], case["SQL"])
                except sqlite3.Error as e:
                    golden_result = f"ERROR: {str(e)}"
                    print("Unexpected error in golden query execution, questions:", case["question_id"])
                golden_end = time.time()
                
                golden_runtime = golden_end - golden_start

                evaluation, message = evaluate_result(model_result, golden_result)
                if evaluation:
                    efficiency_score = model_runtime / golden_runtime if golden_runtime > 0 else None
                    save_record(case, model, sql, selected_tables, model_result, golden_result, efficiency_score, message, j)
                else:
                    efficiency_score = None
                    save_record(case, model, sql, selected_tables, model_result, golden_result, efficiency_score, f"semantic - {message}", j)
                

"""
Answer object:

model_name: str
question_id: int
question: str
difficulty: str
golden_sql: str
model_sql: str
listed_tables: list[str]
result = syntactic, hallucination, semantic, or correct
retrieved_data = list of rows retrieved by model_sql or error
efficiency_score (if correct) = runtime in seconds or none
repetition_number = int

"""