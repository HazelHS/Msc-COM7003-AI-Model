"""
Main Data Processing Pipeline

This script runs the entire data processing pipeline:
1. Data collection from various sources (crypto, exchange, features)
2. Data validation and cleaning
3. Data processing and transformation
4. Update currency metrics with BTC/USD prices and Gold/BTC ratio
5. Creating the combined dataset
6. Optional: Renaming validated files to replace originals

All datasets are stored in the 'datasets' directory at the root of the workspace,
with combined datasets specifically in 'datasets/combined_dataset'.

Usage:
    python main.py [--data-type TYPE] [--rename-validated]
    
    Options:
        --data-type TYPE: Specify data type to collect (crypto, exchange, features, all)
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
import importlib.util
import pandas as pd

# Get the workspace root directory (parent of AI_Gen_Code)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define datasets directories with absolute paths to workspace root
DATASETS_DIR = os.path.join(WORKSPACE_ROOT, "datasets")
PROCESSED_EXCHANGES_DIR = os.path.join(DATASETS_DIR, "processed_exchanges")
ADDITIONAL_FEATURES_DIR = os.path.join(DATASETS_DIR, "additional_features")
COMBINED_DATASET_DIR = os.path.join(DATASETS_DIR, "combined_dataset")

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
COLLECTION_SCRIPTS = {
    'crypto': [
        os.path.join(SCRIPT_DIR, 'data_collection/crypto_features_collector.py')
    ],
    'exchange': [
        os.path.join(SCRIPT_DIR, 'data_collection/get_exchange_data_simple.py')
    ],
    'features': [
        os.path.join(SCRIPT_DIR, 'data_collection/crypto_features_collector.py'),
        os.path.join(SCRIPT_DIR, 'data_collection/update_currency_metrics.py')
    ],
    'validation': [
        os.path.join(SCRIPT_DIR, 'data_collection/validation.py')
    ]
}

# Add specific script paths for other operations
VALIDATION_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/validation.py")
RENAME_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/rename_validated_files.py")
UPDATE_CURRENCY_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/update_currency_metrics.py")
COMBINED_DATASET_SCRIPT = os.path.join(SCRIPT_DIR, "data_collection/create_combined_dataset.py")

# Optional processing scripts
PROCESSING_SCRIPTS = [
    UPDATE_CURRENCY_SCRIPT,
    COMBINED_DATASET_SCRIPT,
    # Add any additional data processing scripts here
]

def ensure_directories_exist():
    """Create required directories if they don't exist"""
    directories = [
        DATASETS_DIR,
        PROCESSED_EXCHANGES_DIR,
        ADDITIONAL_FEATURES_DIR,
        COMBINED_DATASET_DIR,
        os.path.join(WORKSPACE_ROOT, 'visualization_output')
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
            
    return True

def update_script_env_vars():
    """Update environment variables for scripts to use the correct paths"""
    # Set environment variables for child processes
    os.environ['DATASETS_DIR'] = DATASETS_DIR
    os.environ['PROCESSED_EXCHANGES_DIR'] = PROCESSED_EXCHANGES_DIR 
    os.environ['ADDITIONAL_FEATURES_DIR'] = ADDITIONAL_FEATURES_DIR
    os.environ['COMBINED_DATASET_DIR'] = COMBINED_DATASET_DIR
    logger.info(f"Set environment variables for datasets paths")

def check_script_exists(script_path):
    """Check if a script exists at the given path"""
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    return True

def run_script(script_path, args=None, cwd=None):
    """Run a Python script and log its output"""
    if not check_script_exists(script_path):
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
            cwd=cwd,
            env=os.environ.copy()  # Pass updated environment variables
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

def import_module_from_file(filepath):
    """Import a Python module from a file path"""
    if not check_script_exists(filepath):
        return None
        
    try:
        module_name = os.path.basename(filepath).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing module from {filepath}: {e}")
        return None

def run_validation():
    """Run the validation script on the collected data"""
    logger.info("Running data validation...")
    
    # Import the validation module
    validation_module = import_module_from_file(VALIDATION_SCRIPT)
    if validation_module is None:
        logger.error("Could not import validation module. Running as subprocess instead.")
        return run_script(VALIDATION_SCRIPT, ['--all'])
    
    try:
        # Run the validation directly
        logger.info("Validating all files...")
        validation_result = validation_module.validate_all_files()
        return validation_result
    except Exception as e:
        logger.error(f"Error running validation: {e}")
        return False

def collect_data(data_types):
    """Collect data for the specified types"""
    results = {}
    
    for data_type in data_types:
        if data_type not in COLLECTION_SCRIPTS:
            logger.warning(f"Unknown data type: {data_type}")
            continue
            
        logger.info(f"Collecting {data_type} data...")
        scripts = COLLECTION_SCRIPTS[data_type]
        
        type_results = []
        for script in scripts:
            success = run_script(script)
            type_results.append(success)
        
        results[data_type] = all(type_results)
    
    return results

def print_summary(collection_results, validation_result):
    """Print a summary of the collection and validation results"""
    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION AND VALIDATION SUMMARY")
    logger.info("="*80)
    
    # Collection summary
    logger.info("\nData Collection Results:")
    for data_type, success in collection_results.items():
        status = "Success" if success else "Failed"
        logger.info(f"  {data_type}: {status}")
    
    # Validation summary
    validation_status = "All files valid" if validation_result else "Some files have issues"
    logger.info(f"\nData Validation: {validation_status}")
    
    # Overall status
    all_collection_success = all(collection_results.values())
    overall_status = "Success" if all_collection_success and validation_result else "Issues detected"
    logger.info(f"\nOverall Status: {overall_status}")
    
    logger.info("\nNext Steps:")
    if not all_collection_success or not validation_result:
        logger.info("  1. Check the logs for errors")
        logger.info("  2. Fix any issues in the data collection or validation scripts")
        logger.info("  3. Re-run the workflow for failed components")
    else:
        logger.info("  1. Use the validated data for analysis")
        logger.info("  2. Consider running rename_validated_files.py to replace original files")
    
    logger.info("="*80)

def run_pipeline(data_types=None, rename_validated=False):
    """Run the complete data processing pipeline"""
    start_time = time.time()
    
    # Step 1: Ensure directories exist
    logger.info("Step 1: Ensuring directories exist")
    ensure_directories_exist()
    
    # Step 2: Update environment variables for child processes
    update_script_env_vars()
    
    # Step 3: Run data collection workflow
    if data_types is None:
        data_types = ['crypto', 'exchange', 'features', 'validation']
    
    logger.info(f"Step 3: Collecting data for: {', '.join(data_types)}")
    collection_results = collect_data(data_types)
    
    # Step 4: Validate the collected data
    logger.info("Step 4: Validating collected data")
    validation_result = run_validation()
    
    # Step 5: Run additional data processing scripts
    if PROCESSING_SCRIPTS:
        logger.info("Step 5: Running additional data processing scripts")
        for script in PROCESSING_SCRIPTS:
            script_name = os.path.basename(script)
            logger.info(f"Running script: {script_name}")
            success = run_script(script)
            if not success:
                logger.warning(f"Script {script_name} had issues, but continuing with pipeline")
    
    # Step 6: Rename validated files if requested
    if rename_validated:
        logger.info("Step 6: Renaming validated files to replace originals")
        run_script(RENAME_SCRIPT)
    
    # Step 7: Print summary
    print_summary(collection_results, validation_result)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    
    logger.info("Data processing pipeline complete!")

def main():
    """Parse arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument("--data-type", type=str, default="all",
                       choices=["crypto", "exchange", "features", "validation", "all"],
                       help="Specify data type to collect")
    parser.add_argument("--rename-validated", action="store_true", 
                        help="Replace original files with validated versions")
    
    args = parser.parse_args()
    
    # Determine which data types to process
    if args.data_type == 'all':
        data_types = ['crypto', 'exchange', 'features', 'validation']
    else:
        data_types = [args.data_type]
    
    logger.info(f"Starting data processing pipeline")
    run_pipeline(data_types=data_types, rename_validated=args.rename_validated)

if __name__ == "__main__":
    main() 