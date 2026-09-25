"""
Integrated execution runner for the ML Challenge 2026 Candidate Blocking pipeline.
Integrates:
- Person 1: Data Engineering, Preprocessing, Schema Validation, and Submission Validator
- Person 2: Candidate Generation / Blocking, Multi-strategy Inverted Indexing, and Recall Evaluation
"""
import subprocess
import sys
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75 + "\n")

def run_step(description: str, cmd: list[str]):
    print_header(description)
    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] {description} completed successfully.")

def main():
    print_header("STARTING INTEGRATED CANDIDATE BLOCKING PIPELINE EXECUTION")

    # Step 1: Run Person 1's data integrity audit
    run_step("1. Person 1: Data Integrity & Schema Audit", [
        sys.executable, "scripts/audit_data_integrity.py"
    ])

    # Step 2: Run all unit and integration tests (Person 1 Preprocessing + Person 2 Blocking)
    run_step("2. Running Full Test Suites (Preprocessing & Candidate Blocking)", [
        sys.executable, "-m", "pytest", "tests", "-v"
    ])

    # Step 3: Run candidate generation for test split with integrated preprocessing & schema validation
    out_file = Path("output/candidate_pairs.tsv")
    run_step("3. Person 2: Running Candidate Generation (Test Split)", [
        sys.executable, "run_blocking.py",
        "--split", "test",
        "--out", str(out_file)
    ])

    # Show preview of generated output
    if out_file.exists():
        print("\n--- Output Preview (output/candidate_pairs.tsv) ---")
        with open(out_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(line.rstrip())
        print("---------------------------------------------------\n")

    # Step 4: Validate candidate output format using Person 1's validator
    run_step("4. Person 1: Validating Candidate Output Format & ID Consistency", [
        sys.executable, "utils/validate_submission.py",
        "--candidate", str(out_file),
        "--candidate-only",
        "--test-dir", "dataset/test",
        "--check-ids"
    ])

    # Step 5: Run recall evaluation and per-strategy ablation on training split
    run_step("5. Evaluating Recall & Strategy Ablation (Train Split)", [
        sys.executable, "evaluate_blocking.py", "--ablation"
    ])

    print_header("ALL PIPELINE CODES EXECUTED AND VERIFIED SUCCESSFULLY")

if __name__ == "__main__":
    main()
