"""
Created on April 16, 2025

@author: samlevy

Main script to run pHGate GPT and Standard GPT experiments.
This is the centralized entry point for:
- Data tokenization
- Training (Standard + pHGate GPT)
- Evaluation (Macro + Micro)
- Gating diagnostic plots

"""

import os
import subprocess

# Optional: Adjust working directory if needed

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Optional: Environment prep or configuration
DATA_FILE = "tokenized_data.json"
REQUIRED_FILES = [
    "Dataset_Construct.py",
    "StandardGPT_Training.py",
    "pHGate_Training.py",
    "Eval1_Macro.py",
    "Eval2_Micro.pyy",
]

# Utility Functions
def check_dependencies():
    """Ensure that all scripts exist."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing files: {missing}")
        exit(1)

def check_data_file():
    """Ensure that tokenized data exists."""
    if not os.path.exists(DATA_FILE):
        print(f"[INFO] Tokenized data not found. Running tokenizer...")
        subprocess.run(["python", "Dataset_Construct.py"])

def run_script(script_name):
    """Display logs in real time."""
    print(f"\n[INFO] Running {script_name}...")
    log_file_name = f"{os.path.splitext(script_name)[0]}.log"

    with open(log_file_name, "w") as log_file:
        process = subprocess.Popen(
            ["python", "-u", script_name],  # '-u' forces unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="")  # Print to console
            log_file.write(line)  # Save to log file
        process.wait()
    
    if process.returncode == 0:
        print(f"\n[INFO] {script_name} completed successfully!")
    else:
        print(f"\n[ERROR] {script_name} failed with exit code {process.returncode}")

# Main Experiment
if __name__ == "__main__":
    print(" Executing pHGate GPT pipeline...  ")

    check_dependencies()
    check_data_file()

    # Step-by-step pipeline
    run_script("StandardGPT_Training.py")
    run_script("pHGate_Training.py")
    run_script("Eval1_Macro.py")
    run_script("Eval2_Micro.py).py")

    print("\nAll steps completed successfully!")
    print("Review the logs, plots, and JSON output for results.")
