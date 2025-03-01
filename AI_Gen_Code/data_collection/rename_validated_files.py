"""
File Renamer for Validated Files

This script:
1. Backs up original CSV files to a backup directory
2. Renames the validated CSV files to replace the originals
3. Provides a summary of the renamed files

Usage:
    python rename_validated_files.py           # Rename all validated files
    python rename_validated_files.py --no-backup  # Skip backup step
    python rename_validated_files.py --dry-run    # Show what would happen without making changes

Author: AI Assistant
Created: March 1, 2025
"""

import os
import sys
import shutil
from datetime import datetime

# Define constants
DATA_DIRS = [
    '../datasets/additional_features',
    '../datasets/processed_exchanges'
]
BACKUP_DIR = '../datasets/backup_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

def backup_original_files():
    """Create backups of all original files"""
    print(f"\nCreating backup directory: {BACKUP_DIR}")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    backed_up_count = 0
    for dir_path in DATA_DIRS:
        if not os.path.exists(dir_path):
            continue
            
        # Create the same directory structure in the backup dir
        backup_subdir = os.path.join(BACKUP_DIR, os.path.basename(dir_path))
        os.makedirs(backup_subdir, exist_ok=True)
        
        for file in os.listdir(dir_path):
            if file.endswith('.csv') and not file.endswith('_validated.csv') and not file.endswith('_fixed.csv'):
                src_path = os.path.join(dir_path, file)
                dst_path = os.path.join(backup_subdir, file)
                shutil.copy2(src_path, dst_path)
                backed_up_count += 1
                print(f"  Backed up: {src_path} → {dst_path}")
    
    print(f"\nBackup complete: {backed_up_count} files backed up to {BACKUP_DIR}")
    return backed_up_count

def rename_validated_files(dry_run=False):
    """Rename the validated files to replace the originals"""
    renamed_count = 0
    skipped_count = 0
    
    print("\nRenaming validated files to replace originals:")
    for dir_path in DATA_DIRS:
        if not os.path.exists(dir_path):
            continue
            
        for file in os.listdir(dir_path):
            if file.endswith('_validated.csv'):
                original_file = file.replace('_validated.csv', '.csv')
                validated_path = os.path.join(dir_path, file)
                original_path = os.path.join(dir_path, original_file)
                
                # Check if original file exists
                if not os.path.exists(original_path):
                    print(f"  Skipped: {validated_path} (original file not found)")
                    skipped_count += 1
                    continue
                
                if dry_run:
                    print(f"  Would rename: {validated_path} → {original_path}")
                else:
                    try:
                        os.remove(original_path)  # Remove the original file
                        os.rename(validated_path, original_path)  # Rename validated to original
                        print(f"  ✓ Renamed: {validated_path} → {original_path}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"  ✗ Error renaming {validated_path}: {str(e)}")
                        skipped_count += 1
    
    print(f"\nRenaming complete: {renamed_count} files renamed, {skipped_count} files skipped")
    return renamed_count, skipped_count

def main():
    """Main function"""
    print("Validated File Renamer")
    print("=====================")
    
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    skip_backup = '--no-backup' in sys.argv
    
    if dry_run:
        print("\nDRY RUN MODE: No changes will be made")
    
    # Backup original files if not skipped
    if not skip_backup and not dry_run:
        backed_up = backup_original_files()
        if backed_up == 0:
            print("No files backed up. Exiting.")
            return
    elif skip_backup:
        print("\nSkipping backup step (--no-backup specified)")
    
    # Rename validated files
    renamed, skipped = rename_validated_files(dry_run)
    
    print("\nSummary:")
    if not skip_backup and not dry_run:
        print(f"  - {backed_up} files backed up to {BACKUP_DIR}")
    print(f"  - {renamed} validated files renamed to replace originals")
    print(f"  - {skipped} files skipped")
    
    if not dry_run:
        print("\nNext steps:")
        print("  1. Verify the renamed files work correctly in your analysis")
        print("  2. Delete the backup directory if everything works as expected")
        print(f"     rm -rf {BACKUP_DIR}")
    else:
        print("\nTo perform the actual renaming, run the script without --dry-run")

if __name__ == "__main__":
    main() 