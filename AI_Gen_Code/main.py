"""
Main Data Processing Pipeline

This script runs the entire data processing pipeline:
1. Data collection from various sources
2. Data validation and cleaning
3. Data processing and transformation
4. Optional: Renaming validated files to replace originals

Usage:
    python main.py [--rename-validated]
    
    Options:
        --rename-validated: Replace original files with validated versions

Author: AI Assistant
Created: March 2025
"""

import os
import sys
import argparse
import subprocess
import logging
import datetime
import time

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
log_filename = os.path.join(SCRIPT_DIR, f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_pipeline')

# Define script paths with absolute paths
COLLECTION_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/data_collection_workflow.py")
VALIDATION_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/validation.py")
RENAME_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/rename_validated_files.py")

# Optional processing scripts
PROCESSING_SCRIPTS = [
    # Add any additional data processing scripts here
]

def ensure_directories_exist():
    """Create required directories if they don't exist"""
    directories = [
        os.path.join(SCRIPT_DIR, 'datasets'),
        os.path.join(SCRIPT_DIR, 'datasets/processed_exchanges'),
        os.path.join(SCRIPT_DIR, 'datasets/additional_features'),
        os.path.join(SCRIPT_DIR, 'datasets/processed_validated'),
        os.path.join(SCRIPT_DIR, 'visualization_output')
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
            
    return True

def run_script(script_path, args=None, cwd=None):
    """Run a Python script and log its output"""
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Construct the command
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    # Log the command
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the process
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        # Log the output
        if process.stdout:
            for line in process.stdout.splitlines():
                logger.info(f"OUTPUT: {line}")
                
        # Log any errors
        if process.stderr:
            for line in process.stderr.splitlines():
                logger.warning(f"ERROR: {line}")
        
        # Check return code
        if process.returncode != 0:
            logger.error(f"Script {script_path} failed with exit code {process.returncode}")
            return False
            
        logger.info(f"Script {script_path} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Exception running {script_path}: {e}")
        return False

def run_pipeline(rename_validated=False):
    """Run the complete data processing pipeline"""
    start_time = time.time()
    
    # Step 1: Ensure directories exist
    logger.info("Step 1: Ensuring directories exist")
    ensure_directories_exist()
    
    # Step 2: Run data collection workflow
    logger.info("Step 2: Running data collection workflow")
    collection_success = run_script(COLLECTION_SCRIPT, ["all"])
    if not collection_success:
        logger.warning("Data collection had issues, but continuing with pipeline")
    
    # Step 3: Run additional data processing scripts
    if PROCESSING_SCRIPTS:
        logger.info("Step 3: Running additional data processing scripts")
        for script in PROCESSING_SCRIPTS:
            success = run_script(script)
            if not success:
                logger.warning(f"Script {script} had issues, but continuing with pipeline")
    
    # Step 4: Rename validated files if requested
    if rename_validated:
        logger.info("Step 4: Renaming validated files to replace originals")
        run_script(RENAME_SCRIPT)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    
    logger.info("Data processing pipeline complete!")

def main():
    """Parse arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument("--rename-validated", action="store_true", 
                        help="Replace original files with validated versions")
    
    args = parser.parse_args()
    
    logger.info(f"Starting data processing pipeline")
    run_pipeline(rename_validated=args.rename_validated)

if __name__ == "__main__":
    main() 