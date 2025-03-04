"""
Complete Data Processing Pipeline

This script runs the full data processing pipeline:
1. Average the exchange data
2. Create the combined dataset
3. Filter the dataset to include only specified columns with proper naming

Usage:
    python pipeline.py
"""

import os
import subprocess
import sys
import time

def run_script(script_path, description):
    """Run a Python script and return its exit code"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    
    # Get the path to the current Python interpreter
    python_exe = sys.executable
    
    # Build the command
    cmd = [python_exe, script_path]
    
    # Run the script
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        exit_code = result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        exit_code = e.returncode
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds with exit code: {exit_code}")
    
    return exit_code

def main():
    """Run the complete data processing pipeline"""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print(f"Starting complete data processing pipeline...")
    print(f"Project root: {project_root}")
    
    # Step 1: Average exchange data
    avg_script = os.path.join(project_root, "AI_Gen_Code", "data_processing", "combining", "average_exchanges.py")
    if run_script(avg_script, "Average Exchange Data") != 0:
        print("Error in averaging exchanges step. Pipeline stopped.")
        return 1
    
    # Step 2: Create combined dataset
    combined_script = os.path.join(project_root, "AI_Gen_Code", "data_collection", "create_combined_dataset.py")
    if run_script(combined_script, "Create Combined Dataset") != 0:
        print("Error in combined dataset creation step. Pipeline stopped.")
        return 1
    
    # Step 3: Filter combined dataset
    filter_script = os.path.join(project_root, "AI_Gen_Code", "data_processing", "combining", "filter_combined_dataset.py")
    if run_script(filter_script, "Filter Combined Dataset") != 0:
        print("Error in filtering dataset step. Pipeline stopped.")
        return 1
    
    print("\n\nComplete pipeline executed successfully!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 