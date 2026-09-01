# Error and Reliability Analysis of Open-Source LLMs for Text-to-SQL Generation Across Query Complexities

[![Research](https://img.shields.io/badge/research-Text--to--SQL-blue)](#)
[![Models](https://img.shields.io/badge/models-Qwen3-orange)](#)
[![Year](https://img.shields.io/badge/year-2026-lightgrey)](#)

This repository accompanies the master's thesis **"Error and Reliability
Analysis of Open-Source LLMs for Text-to-SQL Generation Across Query
Complexities"** by **Mojtaba Alizade and Omar Younes**.

## Overview

This work evaluates three Qwen3 models under a common
experimental setup:

| Model      | Active parameters |
| --------   | -------           |
| Qwen3 A3B  | 3B                |
| Qwen3 A22B | 22B               |
| Qwen3 A35B | 35B               |

The experiment uses **300 read-only Text-to-SQL test cases**, sampled
evenly across three complexity levels:

-   **Simple:** 100 test cases
-   **Moderate:** 100 test cases
-   **Challenging:** 100 test cases

Each test case is executed **50 times per model** using a two-step
least-to-most prompting strategy. Generated SQL is executed against the
corresponding database and compared with the golden query result.

## Research Questions

### RQ1 --- Failure modes

> How do the most commonly occurring subcategories of failure modes
> differ between LLMs?

### RQ2 --- Model size and accuracy

> For each query complexity, how do Text-to-SQL accuracies differ
> between LLMs of larger and smaller sizes?

### RQ3 --- Failure consistency

> For each analyzed LLM, how are the consistencies of test-case failures
> affected as query complexity increases?

## Key Findings

-   **Schema linking is the most common failure subcategory.**
-   **Performance decreases as query complexity increases.**
-   **Larger models are not consistently more accurate.**
-   **Failures are often systematic rather than random.**
-   **Accuracy alone is insufficient for evaluation.**
-   **Semantic failures dominate for all models.**

## Acknowledgements

Master's thesis in Computer Science and Engineering at **Chalmers
University of Technology** and the **University of Gothenburg**.

**Supervisors:** Bengt Haraldsson and Miroslaw Staron\
**Examiner:** Francisco Gomes de Oliveira Neto
