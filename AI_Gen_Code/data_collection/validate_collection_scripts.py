"""
Collection Script Validator

This script analyzes data collection scripts to ensure they:
1. Create properly formatted CSV files with consistent date columns
2. Follow best practices for data quality
3. Include proper error handling and validation

This helps prevent date formatting and structure issues in the future.

Usage:
    python validate_collection_scripts.py [script_path]  # Validate a specific script
    python validate_collection_scripts.py --all          # Validate all collection scripts

Author: AI Assistant
Created: March 1, 2025
"""

import os
import sys
import ast
import re
import glob
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("script_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('script_validator')

# Define patterns to check for in scripts
PATTERNS = {
    'date_format': re.compile(r"to_datetime|strftime\(['\"](.*?)['\"]"),
    'date_column': re.compile(r"['\"]Date['\"]|df\[['\"](Date|date)['\"]"),
    'csv_write': re.compile(r"to_csv\("),
    'error_handling': re.compile(r"try:|except|Exception"),
    'pandas_import': re.compile(r"import\s+pandas|from\s+pandas\s+import"),
}

# Define paths to collection scripts
COLLECTION_SCRIPT_PATTERNS = [
    "../get_*.py",
    "../collect_*.py",
    "fix_*.py",
    "combine_*.py",
]

class ScriptValidator:
    def __init__(self, script_path):
        self.script_path = script_path
        self.filename = os.path.basename(script_path)
        self.source_code = ""
        self.ast_tree = None
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def read_script(self):
        """Read the script file"""
        try:
            with open(self.script_path, 'r') as f:
                self.source_code = f.read()
            return True
        except Exception as e:
            logger.error(f"Error reading {self.script_path}: {e}")
            return False
    
    def parse_ast(self):
        """Parse the script into an AST"""
        try:
            self.ast_tree = ast.parse(self.source_code)
            return True
        except SyntaxError as e:
            self.issues.append(f"Syntax error: {e}")
            return False
        except Exception as e:
            self.issues.append(f"Error parsing AST: {e}")
            return False
    
    def check_patterns(self):
        """Check for various patterns in the source code"""
        # Check for date formatting patterns
        date_format_matches = PATTERNS['date_format'].findall(self.source_code)
        if not date_format_matches:
            self.issues.append("No date formatting found. Script may not handle date formats properly.")
        else:
            # Check for consistent date formats
            date_formats = set(date_format_matches)
            if len(date_formats) > 1:
                self.warnings.append(f"Multiple date formats found: {date_formats}. Consider standardizing.")
            
            # Check if YYYY-MM-DD format is used
            std_format_found = any('%Y-%m-%d' in fmt for fmt in date_formats)
            if not std_format_found:
                self.recommendations.append("Recommend using '%Y-%m-%d' format for dates.")
        
        # Check for Date column usage
        if not PATTERNS['date_column'].search(self.source_code):
            self.issues.append("No 'Date' column found. Script may not create proper date-indexed data.")
        
        # Check for CSV writing
        if not PATTERNS['csv_write'].search(self.source_code):
            self.warnings.append("No CSV writing detected. Script may not output CSV files.")
        
        # Check for error handling
        if not PATTERNS['error_handling'].search(self.source_code):
            self.warnings.append("No error handling detected. Script may fail without proper error messages.")
        
        # Check for pandas import
        if not PATTERNS['pandas_import'].search(self.source_code):
            self.warnings.append("No pandas import detected. Script may not handle dataframes properly.")
    
    def check_ast_patterns(self):
        """Check for patterns using the AST"""
        if not self.ast_tree:
            return
        
        # Look for date validation in the AST
        date_validation_found = False
        
        for node in ast.walk(self.ast_tree):
            # Check for to_datetime calls
            if isinstance(node, ast.Call) and hasattr(node, 'func'):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'to_datetime':
                    date_validation_found = True
                    
                    # Check if errors parameter is used
                    has_errors_param = False
                    for keyword in node.keywords:
                        if keyword.arg == 'errors':
                            has_errors_param = True
                            break
                    
                    if not has_errors_param:
                        self.recommendations.append(
                            "Add error handling to pd.to_datetime: pd.to_datetime(df['Date'], errors='coerce')"
                        )
        
        if not date_validation_found:
            self.warnings.append("No date validation found. Script may not handle invalid dates properly.")
    
    def generate_fix_recommendations(self):
        """Generate specific recommendations to fix issues"""
        if "No 'Date' column found" in self.issues:
            self.recommendations.append(
                "Add a Date column: df['Date'] = pd.to_datetime(date_column, errors='coerce')"
            )
            
        if "No date formatting found" in self.issues:
            self.recommendations.append(
                "Add proper date formatting: df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')"
            )
            
        if "No error handling detected" in self.warnings:
            self.recommendations.append(
                "Add basic error handling:\n"
                "try:\n"
                "    # Your code here\n"
                "except Exception as e:\n"
                "    print(f\"Error: {e}\")"
            )
            
        if "No pandas import detected" in self.warnings:
            self.recommendations.append(
                "Add pandas import: import pandas as pd"
            )
            
        if "No CSV writing detected" in self.warnings:
            self.recommendations.append(
                "Add CSV output: df.to_csv('output_file.csv', index=False)"
            )
    
    def validate(self):
        """Run all validation checks on the script"""
        if not self.read_script():
            return False
            
        valid_syntax = self.parse_ast()
        self.check_patterns()
        
        if valid_syntax:
            self.check_ast_patterns()
            
        self.generate_fix_recommendations()
        return len(self.issues) == 0
    
    def print_report(self):
        """Print validation report"""
        logger.info(f"\n{'='*80}")
        logger.info(f"SCRIPT VALIDATION REPORT FOR: {self.filename}")
        logger.info(f"{'='*80}")
        
        if not self.issues and not self.warnings:
            logger.info("\n✓ Script passed validation with no issues")
        else:
            if self.issues:
                logger.info("\nIssues:")
                for issue in self.issues:
                    logger.info(f"  ✗ {issue}")
            
            if self.warnings:
                logger.info("\nWarnings:")
                for warning in self.warnings:
                    logger.info(f"  ⚠ {warning}")
        
        if self.recommendations:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info(f"{'='*80}\n")

def find_collection_scripts():
    """Find all collection scripts in the project"""
    scripts = []
    for pattern in COLLECTION_SCRIPT_PATTERNS:
        scripts.extend(glob.glob(pattern))
    return scripts

def validate_script(script_path):
    """Validate a single script"""
    logger.info(f"Validating script: {script_path}")
    
    validator = ScriptValidator(script_path)
    validator.validate()
    validator.print_report()
    return len(validator.issues) == 0

def validate_all_scripts():
    """Validate all collection scripts"""
    all_valid = True
    scripts = find_collection_scripts()
    
    logger.info(f"Found {len(scripts)} collection scripts to validate")
    
    for script in scripts:
        script_valid = validate_script(script)
        all_valid = all_valid and script_valid
    
    logger.info(f"\n{'='*80}")
    logger.info("SCRIPT VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total scripts validated: {len(scripts)}")
    logger.info(f"Overall status: {'✓ All scripts passed' if all_valid else '✗ Some scripts have issues'}")
    logger.info(f"{'='*80}\n")
    
    return all_valid

def main():
    """Main function"""
    logger.info("Collection Script Validator")
    logger.info("==========================")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            validate_all_scripts()
        else:
            script_path = sys.argv[1]
            if os.path.exists(script_path):
                validate_script(script_path)
            else:
                logger.error(f"Script not found: {script_path}")
    else:
        logger.info("No script specified. Use --all to validate all collection scripts.")
        logger.info("Or specify a script path to validate a single script.")

if __name__ == "__main__":
    main() 