import os
import json
import time
import glob
import sqlite3
import traceback
import pandas as pd
from ollama import chat
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# ==============================
# CONFIG
# ==============================

MODEL_NAME = "deepseek-coder-v2:16b"
QUESTIONS_FILE = "dev-questions-1.json"
OUTPUT_FILE = f"results_{MODEL_NAME.replace(":", "_")}.json"
TIMEOUT_SECONDS = 30 * 60  # 60 minutes


# ==============================
# HELPER FUNCTIONS
# ==============================

'''
def load_schema_jsons(db_id):
    """
    Convert all CSV files in ./db_id/database_description/*.csv
    into JSON objects (in-memory).
    """
    schema_path = os.path.join(db_id, "database_description", "*.csv")
    csv_files = glob.glob(schema_path)

    schema_json = {}

    for csv_file in csv_files:
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        schema_json[table_name] = df.to_dict(orient="records")

    return schema_json
'''
def load_schema_jsons(db_id: str) -> dict:
    """
    Loads the schema for a specific db_id from dev_tables.json.
    Assumes dev_tables.json is located two directories above this file.
    """

    # Go two levels up from this file's directory
    base_path = Path(__file__).resolve().parents[2]
    dev_tables_path = base_path / "dev_tables.json"

    if not dev_tables_path.exists():
        raise FileNotFoundError(f"dev_tables.json not found at {dev_tables_path}")

    with open(dev_tables_path, "r", encoding="utf-8") as f:
        all_schemas = json.load(f)

    for schema in all_schemas:
        if schema.get("db_id") == db_id:
            return schema

    raise ValueError(f"Schema for db_id '{db_id}' not found in dev_tables.json")

def format_schema_for_prompt(schema_json: dict) -> str:
    table_names = schema_json["table_names_original"]
    column_names = schema_json["column_names_original"]
    column_types = schema_json["column_types"]
    primary_keys = set(schema_json["primary_keys"])

    tables = {i: [] for i in range(len(table_names))}

    for idx, (table_id, col_name) in enumerate(column_names):
        if table_id == -1:
            continue  # skip *
        col_type = column_types[idx]
        pk_marker = " [PK]" if idx in primary_keys else ""
        tables[table_id].append(f"  - {col_name} ({col_type}){pk_marker}")

    output = []
    for table_id, cols in tables.items():
        output.append(f"Table: {table_names[table_id]}")
        output.append("Columns:")
        output.extend(cols)
        output.append("")

    return "\n".join(output)

"""
def format_schema_for_prompt(schema_json: dict) -> str:
    table_names = schema_json["table_names_original"]
    column_names = schema_json["column_names_original"]
    column_types = schema_json["column_types"]
    raw_primary_keys = schema_json["primary_keys"]

    # Separate single and composite primary keys
    single_pks = set()
    composite_pks = []

    for pk in raw_primary_keys:
        if isinstance(pk, list):
            composite_pks.append(pk)  # e.g. [3, 4]
        else:
            single_pks.add(pk)

    tables = {i: [] for i in range(len(table_names))}

    for idx, (table_id, col_name) in enumerate(column_names):
        if table_id == -1:
            continue  # skip *

        col_type = column_types[idx]
        # Check if part of a composite PK
        composite_marker = ""
        for comp in composite_pks:
            if idx in comp:
                composite_marker = " [PK (composite)]"
                break
        # Otherwise check single PK
        pk_marker = " [PK]" if idx in single_pks else ""
        marker = composite_marker if composite_marker else pk_marker
        tables[table_id].append(f"  - {col_name} ({col_type}){marker}")

    output = []
    for table_id, cols in tables.items():
        output.append(f"Table: {table_names[table_id]}")
        output.append("Columns:")
        output.extend(cols)
        output.append("")

    return "\n".join(output)
"""

def build_prompt(question, evidence, schema_json):
    """
    Construct LLM prompt with schema and instructions.
    """
    return f"""
You are an expert SQLite SQL generator.

Carefully study the provided database schema (JSON format) and generate a valid SQLite SQL query that answers the question.

Use ONLY the provided tables and columns.
Do NOT hallucinate tables or columns.
Return ONLY the SQL query. No explanation.

DATABASE SCHEMA:
{json.dumps(schema_json, indent=2)}

QUESTION:
{question}

CONTEXT:
{evidence}

Produce a syntactically correct SQLite query.
"""


def execute_sql(sqlite_path, query):
    """
    Execute SQL query and return (results, runtime).
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    start = time.time()
    cursor.execute(query)
    rows = cursor.fetchall()
    runtime = time.time() - start

    conn.close()
    return rows, runtime


def compare_results(model_rows, golden_rows):
    """
    Compare query outputs.
    """
    return sorted(model_rows) == sorted(golden_rows)


def map_difficulty(difficulty):
    """
    Map question difficulty to required enum.
    """
    mapping = {
        "simple": "easy",
        "moderate": "medium",
        "challenging": "difficult"
    }
    return mapping.get(difficulty.lower(), "medium")

def extract_sql(raw_output: str) -> str:
    import re

    if not raw_output:
        return ""

    text = raw_output.strip()

    # Remove markdown fences
    text = re.sub(r"```sql|```sqlite|```", "", text, flags=re.IGNORECASE).strip()

    # Remove garbage before SELECT
    match = re.search(r"(SELECT\b.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)

    # Remove trailing backticks
    text = text.strip("`").strip()

    # Replace literal "\n" with space (if model produced them)
    text = text.replace("\\n", " ")

    # Replace real newlines with spaces
    text = text.replace("\n", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove trailing semicolon
    text = text.rstrip(";").strip()

    return text

# ==============================
# MAIN EVALUATION LOOP
# ==============================

def main():

    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)

    # Group questions by db_id
    questions_by_db = {}
    for q in questions:
        questions_by_db.setdefault(q["db_id"], []).append(q)

    results = []

    output_f = open(OUTPUT_FILE, "w", encoding="utf-8")

    for db_id, db_questions in questions_by_db.items():
        print(f"Processing DB: {db_id}")

        # Load schema JSONs once per DB
        relevant_schema = load_schema_jsons(db_id)
        schema_formatted = format_schema_for_prompt(relevant_schema)
        sqlite_path = os.path.join(db_id, f"{db_id}.sqlite")

        for q in db_questions:
            print(f"  Question {q['question_id']}")

            question_id = q["question_id"]
            question_text = q["question"]
            evidence = q.get("evidence", "")
            golden_query = q["SQL"]

            result_entry = {
                "question_id": question_id,
                "db_id": db_id,
                "model_name": MODEL_NAME,
                "model_output": None,
                "golden_query": golden_query,
                #"explanation": evidence,
                "model_output_runtime": None,
                "efficiency_score": None,
                "query_complexity": map_difficulty(q["difficulty"]),
                "result": None
            }

            try:
                prompt = build_prompt(question_text, evidence, schema_formatted)

                def call_model():
                    return chat(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}]
                    )
                start_model = time.time()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_model)
                    response = future.result(timeout=TIMEOUT_SECONDS)
                model_runtime = time.time() - start_model

                model_sql = extract_sql(response["message"]["content"]).strip()
                golden_sql_stripped = golden_query.strip()

                result_entry["model_output"] = model_sql
                result_entry["model_output_runtime"] = model_runtime

            except FuturesTimeoutError as e:
                result_entry["result"] = f"skipped - {str(e).lower()}"
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            except Exception as e:
                result_entry["result"] = f"skipped - {str(e).lower()}"
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            # Execute model SQL first (always required)
            try:
                model_rows, model_query_runtime = execute_sql(sqlite_path, model_sql)

            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()

                if "no such table" in error_msg or "no such column" in error_msg:
                    result_entry["result"] = f"hallucination - {error_msg}"
                else:
                    result_entry["result"] = f"syntactic - {error_msg}"

                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            except Exception as e:
                result_entry["result"] = f"syntactic - {str(e).lower()}"
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            # --------------------------------------------------
            # NEW LOGIC: Compare SQL strings before running golden
            # --------------------------------------------------
            if model_sql == golden_sql_stripped:
                result_entry["result"] = "correct"
                result_entry["efficiency_score"] = 1.0
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            # Only run golden query if SQL strings differ
            try:
                golden_rows, golden_runtime = execute_sql(sqlite_path, golden_query)
            except Exception as e:
                result_entry["result"] = f"syntactic - {str(e).lower()}"
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
                continue

            # Compare execution results
            try:
                if compare_results(model_rows, golden_rows):
                    result_entry["result"] = "correct"

                    if golden_runtime > 0:
                        result_entry["efficiency_score"] = (
                            model_query_runtime / golden_runtime
                        )
                else:
                    result_entry["result"] = "semantic"

                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()
            except Exception as e:
                print(f"Compare failed for Question {question_id}: {e}")
                result_entry["result"] = f"failed - {str(e)}"
                output_f.write(json.dumps(result_entry) + "\n")
                output_f.flush()

    #with open(OUTPUT_FILE, "w") as f:
    #    json.dump(results, f, indent=2)

    output_f.close()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()