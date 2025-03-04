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

# Add specific script paths for data processing pipeline
AVERAGE_EXCHANGES_SCRIPT = os.path.join(SCRIPT_DIR, "data_processing", "combining", "average_exchanges.py")
FILTER_DATASET_SCRIPT = os.path.join(SCRIPT_DIR, "data_processing", "combining", "filter_combined_dataset.py")

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

def run_data_processing_pipeline():
    """
    Run the complete data processing pipeline:
    1. Average exchange data
    2. Create the combined dataset
    3. Filter the dataset to include only specified columns with proper naming
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Average exchange data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Average Exchange Data")
    logger.info("="*80)
    if not run_script(AVERAGE_EXCHANGES_SCRIPT):
        logger.error("Error in averaging exchanges step. Pipeline stopped.")
        return False
    
    # Step 2: Create combined dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Create Combined Dataset")
    logger.info("="*80)
    if not run_script(COMBINED_DATASET_SCRIPT):
        logger.error("Error in combined dataset creation step. Pipeline stopped.")
        return False
    
    # Step 3: Filter combined dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Filter Combined Dataset")
    logger.info("="*80)
    if not run_script(FILTER_DATASET_SCRIPT):
        logger.error("Error in filtering dataset step. Pipeline stopped.")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE DATA PROCESSING PIPELINE EXECUTED SUCCESSFULLY!")
    logger.info("="*80)
    
    return True

def run_pipeline(data_types=None, rename_validated=False, run_processing=False):
    """Run the complete data pipeline"""
    ensure_directories_exist()
    update_script_env_vars()
    
    # Default to all data types if none specified
    if not data_types:
        data_types = ['crypto', 'exchange', 'features']
    
    # Run data collection
    collection_results = collect_data(data_types)
    
    # Run validation
    validation_result = run_validation()
    
    # Print summary of collection and validation
    print_summary(collection_results, validation_result)
    
    # Rename validated files if requested
    if rename_validated and validation_result:
        logger.info("\nRenaming validated files to replace originals...")
        run_script(RENAME_SCRIPT)
    
    # Run processing scripts if requested
    if run_processing:
        logger.info("\nRunning data processing pipeline...")
        return run_data_processing_pipeline()
    
    return True

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Run the complete data processing pipeline')
    parser.add_argument('--data-type', 
                        choices=['crypto', 'exchange', 'features', 'all'],
                        help='Specify data type to collect (crypto, exchange, features, all)')
    parser.add_argument('--rename-validated', 
                        action='store_true',
                        help='Replace original files with validated versions')
    parser.add_argument('--process-data', 
                        action='store_true',
                        help='Run the data processing pipeline after collection')
    parser.add_argument('--pipeline-only', 
                        action='store_true',
                        help='Run only the data processing pipeline (skip collection)')
    
    args = parser.parse_args()
    
    # Set up data types to process
    data_types = []
    if args.data_type:
        if args.data_type == 'all':
            data_types = ['crypto', 'exchange', 'features']
        else:
            data_types = [args.data_type]
    
    # If pipeline-only is specified, just run the processing pipeline
    if args.pipeline_only:
        logger.info("Running only the data processing pipeline...")
        return 0 if run_data_processing_pipeline() else 1
    
    # Otherwise run the complete pipeline including collection if requested
    success = run_pipeline(
        data_types=data_types, 
        rename_validated=args.rename_validated,
        run_processing=args.process_data
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 