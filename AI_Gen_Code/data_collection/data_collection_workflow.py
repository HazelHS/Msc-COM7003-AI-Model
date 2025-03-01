"""
Integrated Data Collection Workflow

This script provides a complete workflow for data collection that includes:
1. Collecting data from various sources
2. Validating the collected data for quality and consistency
3. Standardizing date formats and structure
4. Generating reports on data quality

Usage:
    python data_collection_workflow.py [data_type]
    
    data_type options:
        crypto       - Collect cryptocurrency data
        exchange     - Collect stock exchange data
        features     - Collect additional features
        all          - Collect all data types

Author: AI Assistant
Created: March 1, 2025
"""

import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime
import importlib.util
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_collection')

# Define paths to collection scripts
COLLECTION_SCRIPTS = {
    'crypto': [
        '../get_crypto_data.py',
        '../get_crypto_features.py'
    ],
    'exchange': [
        '../get_exchange_data_simple.py'
    ],
    'features': [
        '../collect_individual_features.py'
    ]
}

def check_script_exists(script_path):
    """Check if a script exists at the given path"""
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    return True

def run_script(script_path, args=None):
    """Run a Python script and capture its output"""
    if not check_script_exists(script_path):
        return False

    cmd = ['python', script_path]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Output: {process.stdout}")
        if process.stderr:
            logger.warning(f"Warnings/Errors: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}:")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
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
    validation_module = import_module_from_file('validation.py')
    if validation_module is None:
        logger.error("Could not import validation module. Running as subprocess instead.")
        return run_script('validation.py', ['--all'])
    
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
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {data_type}: {status}")
    
    # Validation summary
    validation_status = "✓ All files valid" if validation_result else "✗ Some files have issues"
    logger.info(f"\nData Validation: {validation_status}")
    
    # Overall status
    all_collection_success = all(collection_results.values())
    overall_status = "✓ Success" if all_collection_success and validation_result else "✗ Issues detected"
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

def main():
    """Main function"""
    logger.info("Starting Integrated Data Collection Workflow")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_type = sys.argv[1].lower()
        if data_type == 'all':
            data_types = list(COLLECTION_SCRIPTS.keys())
        elif data_type in COLLECTION_SCRIPTS:
            data_types = [data_type]
        else:
            logger.error(f"Unknown data type: {data_type}")
            logger.info("Available data types: crypto, exchange, features, all")
            return
    else:
        logger.info("No data type specified. Defaulting to 'all'.")
        data_types = list(COLLECTION_SCRIPTS.keys())
    
    # Step 1: Collect the data
    logger.info(f"Step 1: Collecting data for: {', '.join(data_types)}")
    collection_results = collect_data(data_types)
    
    # Step 2: Validate the collected data
    logger.info("Step 2: Validating collected data")
    validation_result = run_validation()
    
    # Step 3: Print summary
    print_summary(collection_results, validation_result)
    
    logger.info("Data collection workflow completed")

if __name__ == "__main__":
    main() 