import json
import time
from openai import OpenAI
import sqlite3
import re
import os

# =========================
# CONFIG & Models
# =========================
region = "eu-north-1"
api_key = ""
attempts = 100

models = {
    "qwen.qwen3-coder-30b-a3b-v1:0": "Qwen-3-Coder-30B",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen-3-235B",
    "qwen.qwen3-coder-480b-a35b-v1:0": "Qwen-3-Coder-480B"
}

# =========================
# Bedrock Client
# =========================

def execute_sql(db_id, query):
    db_path = f"./databases/{db_id}/{db_id}.sqlite"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()
    return rows

def call_model(messages, model, max_tokens=512):
    """
    OpenAI-style wrapper for Bedrock (Claude-style models).
    """
    client = OpenAI(
        api_key=api_key,
        base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        timeout=300
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    return response.choices[0].message.content

def get_columns(case):
    table_names = []
    schema_lines = []
    db_id = case["db_id"]
    db_path = f"./databases/{db_id}/{db_id}.sqlite"

    for key, _ in case.items():
        if key not in ["question", "difficulty", "SQL", "evidence", "question_id", "db_id"]:
            table_name = key
            table_names.append(table_name)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{table_name}")')
            cols = [row[1] for row in cur.fetchall()]
            conn.close()
            schema_lines.append(f"{table_name}({', '.join(cols)})")
    return table_names, "\n".join(schema_lines)

# =========================
# Prompt 1
# =========================
def build_prompt1(question, allowed_tables, schema_text):
    return f"""
You are a schema selection assistant for text-to-SQL.

Select the smallest set of tables needed to answer the question. Include JOIN table names if needed. If unsure, include the table name.

Rules:
- Output ONLY valid JSON.
- Format: {{"tables": ["table1", "table2"]}}
- Use ONLY table names from the allowed list.
- Do NOT include explanations or extra text.
- Do NOT include parentheses or anything besides the JSON.

Valid examples:
{{"tables": ["students"]}}
{{"tables": ["orders", "customers"]}}

Allowed tables:
{allowed_tables}

Question:
{question}

Schema:
{schema_text}
"""

def clean_sql(response_text):
    # Remove ```sql ... ``` or ``` ... ```
    cleaned = re.sub(r"```sql|```|```sqlite", "", response_text, flags=re.IGNORECASE)
    return cleaned.strip()

def clean_json_tables(text):
    try:
        data = json.loads(text)
        return data.get("tables", [])
    except:
        return []

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
        return False, f"list comparison: {a}, {b}"

    for a_element, b_element in zip(a, b):
        if isinstance(a_element, (int, float)) or isinstance(b_element, (int, float)):
            if not compare_numbers(a_element, b_element):
                return False, f"numeric comparison: {a_element}, {b_element}"

        elif isinstance(a_element, str) and isinstance(b_element, str):
            if normalize(a_element) != normalize(b_element):
                return False, f"string comparison: {a_element}, {b_element}"
            
        else:
            if a_element != b_element:
                return False, f"other: {a_element}, {b_element}"

    return True, f"list comparison: {a_element}, {b_element}"


def evaluate_result(retrieved_data, golden_data):
    # 1. Exact match
    if retrieved_data == golden_data:
        return True, "correct"

    # 2. Handle None
    if retrieved_data is None or golden_data is None:
        return retrieved_data == golden_data, f"handling none: {retrieved_data}, {golden_data}"

    # 3. Numbers (with tolerance)
    if isinstance(retrieved_data, (int, float)) or isinstance(golden_data, (int, float)):
        return compare_numbers(retrieved_data, golden_data), f"numeric comparison: {retrieved_data}, {golden_data}"

    # 4. Strings
    if isinstance(retrieved_data, str) and isinstance(golden_data, str):
        return normalize(retrieved_data) == normalize(golden_data), f"string comparison: {retrieved_data}, {golden_data}"

    # 5. Lists (e.g., SQL rows)
    if isinstance(retrieved_data, list) and isinstance(golden_data, list):
        return compare_lists(retrieved_data, golden_data)

    # 7. Fallback - Can create FN
    return False, f"other: {retrieved_data}, {golden_data}"

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
    with open("sampled_cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for model in models.keys():

        for i, case in enumerate(data):
            if case["question_id"] not in [82, 799, 202]:
                continue
            print("Question ID:", case["question_id"])
            tables, schemas = get_columns(case)
            messages1 = [{"role": "user", "content": build_prompt1(case["question"], tables, schemas)}]

            for j in range(attempts):
                # ---- Prompt 1 ----
                tables_output = call_model(messages1, model)

                selected_tables = clean_json_tables(tables_output)

                # ---- Prompt 2 ----
                messages2 = [{"role": "user", "content": build_prompt2(case, selected_tables)}]
                sql_output = call_model(messages2, model, max_tokens=1024)

                sql = clean_sql(sql_output)

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


if __name__ == "__main__":
    main()

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