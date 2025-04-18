"""
Created on April 17, 2025

@author: samlevy

Main script to run pH-Gated GPT and Standard GPT experiments.
This file serves as the centralized entry point for:
- Data tokenization
- Training (Standard + pH-Gated)
- Evaluation (Macro + Micro)
- Gating diagnostic plots

Usage:
$ python main.py

Note:
Ensure that 'tokenized_data.json' and required paths in the scripts point to valid data locations.
Update any hardcoded base paths like BASE_OUTPUT_DIR inside scripts before running.
"""

import os
import subprocess

# ---------------------------------------------
# Optional: Adjust working directory if needed
# ---------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------
# Optional: Environment prep or configuration
# ---------------------------------------------
DATA_FILE = "tokenized_data.json"
REQUIRED_FILES = [
    "Dataset_Construct.py",
    "Train_StandardGPT.py",
    "Train_PHGateGPT.py",
    "Eval1_Macro.py",
    "Eval2_Micro.py",
    "gating_plot.py"
]

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------
def check_dependencies():
    """Ensure that all required Python scripts exist."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing files: {missing}")
        exit(1)

def check_data_file():
    """Ensure that tokenized data exists before training."""
    if not os.path.exists(DATA_FILE):
        print(f"[INFO] Tokenized data not found. Running tokenizer...")
        subprocess.run(["python", "Dataset_Construct.py"])

def run_script(script_name):
    """Utility to run a script with subprocess and log its output."""
    print(f"\n[INFO] Running {script_name}...")
    log_file_name = f"{os.path.splitext(script_name)[0]}.log"
    with open(log_file_name, "w") as log_file:
        subprocess.run(["python", script_name], check=True, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"[INFO] Logs for {script_name} saved to {log_file_name}")

# ---------------------------------------------
# Main Experiment Pipeline
# ---------------------------------------------
if __name__ == "__main__":
    print("========================================")
    print(" pH-Gated GPT - Full Experiment Pipeline ")
    print("========================================")

    check_dependencies()
    check_data_file()

    # Step-by-step pipeline
    run_script("Train_StandardGPT.py")
    run_script("Train_PHGateGPT.py")
    run_script("Eval1_Macro.py")
    run_script("Eval2_Micro.py")
    run_script("gating_plot.py")

    print("\n[INFO] All steps completed successfully!")
    print("[INFO] Review logs, plots, and JSON output for results.")
