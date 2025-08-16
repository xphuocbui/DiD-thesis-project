"""
Apply Gender Mapping to Master Dataset

This script applies the gender value mappings from the mapping template CSV
back to the master dataset, creating a cleaned version with standardized gender values.

Author: Data Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

def load_gender_mapping():
    """
    Load the gender mapping from the CSV template.
    
    Returns:
        dict: Mapping dictionary from original to mapped values
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    mapping_file = project_root / "data" / "data summary" / "gender column" / "gender_mapping_template.csv"
    
    try:
        mapping_df = pd.read_csv(mapping_file)
        print(f"Loading mapping from: {mapping_file}")
        
        # Create mapping dictionary, excluding empty mapped values
        mapping_dict = {}
        for _, row in mapping_df.iterrows():
            original = str(row['original_value']).strip()
            mapped = str(row['mapped_value']).strip()
            
            if mapped and mapped != 'nan' and mapped != '':
                mapping_dict[original] = mapped
                print(f"  '{original}' -> '{mapped}'")
            else:
                print(f"  '{original}' -> (no mapping specified)")
        
        print(f"\nLoaded {len(mapping_dict)} valid mappings")
        return mapping_dict
    
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_file}")
        return None
    except Exception as e:
        print(f"Error loading mapping file: {str(e)}")
        return None

def apply_gender_mapping(backup=True, dry_run=False):
    """
    Apply gender mapping to the master dataset.
    
    Args:
        backup (bool): Create backup of original file before modification
        dry_run (bool): Show what would be changed without actually modifying the file
    
    Returns:
        tuple: (success, stats_dict)
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    
    # File paths
    master_file = project_root / "data" / "all_years_merged_dataset_final_corrected.csv"
    backup_file = project_root / "data" / "backup" / f"all_years_merged_dataset_before_gender_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_dir = project_root / "data" / "data summary" / "gender column"
    
    try:
        # Load mapping
        mapping_dict = load_gender_mapping()
        if not mapping_dict:
            print("No valid mappings found. Please check your mapping template.")
            return False, {}
        
        # Load master dataset
        print(f"\nLoading master dataset from: {master_file}")
        df = pd.read_csv(master_file, low_memory=False)
        print(f"Dataset loaded. Shape: {df.shape}")
        
        # Check if gender column exists
        if 'gender' not in df.columns:
            print("Error: 'gender' column not found in dataset!")
            return False, {}
        
        # Analyze current gender values
        print("\nCurrent gender column statistics:")
        original_counts = df['gender'].value_counts(dropna=False)
        print(original_counts)
        
        # Apply mapping
        print(f"\nApplying gender mapping{'(DRY RUN)' if dry_run else ''}...")
        
        # Convert gender column to string for consistent mapping
        df['gender'] = df['gender'].astype(str)
        
        # Track changes
        changes_made = 0
        mapping_stats = {}
        
        for original, mapped in mapping_dict.items():
            mask = df['gender'] == original
            count = mask.sum()
            
            if count > 0:
                print(f"  Mapping '{original}' -> '{mapped}': {count:,} records")
                mapping_stats[original] = {'mapped_to': mapped, 'count': count}
                
                if not dry_run:
                    df.loc[mask, 'gender'] = mapped
                
                changes_made += count
        
        # Handle unmapped values
        mapped_values = set(mapping_dict.keys())
        current_values = set(df['gender'].unique())
        unmapped_values = current_values - mapped_values
        
        if unmapped_values:
            print(f"\nWarning: Found {len(unmapped_values)} unmapped values:")
            for value in unmapped_values:
                if value != 'nan':
                    count = (df['gender'] == value).sum()
                    print(f"  '{value}': {count:,} records (not mapped)")
        
        print(f"\nTotal records that would be changed: {changes_made:,}")
        
        if dry_run:
            print("\n--- DRY RUN COMPLETE ---")
            print("No changes were made to the actual file.")
            return True, mapping_stats
        
        # Create backup if requested
        if backup:
            print(f"\nCreating backup at: {backup_file}")
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(master_file, backup_file)
            print("Backup created successfully.")
        
        # Save updated dataset
        print(f"\nSaving updated dataset to: {master_file}")
        df.to_csv(master_file, index=False, encoding='utf-8')
        print("Dataset updated successfully!")
        
        # Generate updated statistics
        print("\nUpdated gender column statistics:")
        updated_counts = df['gender'].value_counts(dropna=False)
        print(updated_counts)
        
        # Create mapping report
        report_file = output_dir / f"gender_mapping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GENDER MAPPING APPLICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Applied on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source file: {master_file}\n")
            f.write(f"Backup file: {backup_file if backup else 'No backup created'}\n")
            f.write(f"Total records changed: {changes_made:,}\n\n")
            
            f.write("MAPPINGS APPLIED:\n")
            f.write("-" * 20 + "\n")
            for original, stats in mapping_stats.items():
                f.write(f"'{original}' -> '{stats['mapped_to']}': {stats['count']:,} records\n")
            
            f.write(f"\nORIGINAL COUNTS:\n")
            f.write("-" * 15 + "\n")
            for value, count in original_counts.items():
                f.write(f"'{value}': {count:,}\n")
            
            f.write(f"\nUPDATED COUNTS:\n")
            f.write("-" * 14 + "\n")
            for value, count in updated_counts.items():
                f.write(f"'{value}': {count:,}\n")
            
            if unmapped_values:
                f.write(f"\nUNMAPPED VALUES:\n")
                f.write("-" * 15 + "\n")
                for value in unmapped_values:
                    if value != 'nan':
                        count = (df['gender'] == value).sum()
                        f.write(f"'{value}': {count:,} records\n")
        
        print(f"\nMapping report saved to: {report_file}")
        
        return True, mapping_stats
        
    except Exception as e:
        print(f"Error applying gender mapping: {str(e)}")
        return False, {}

def main():
    """Main function with user interaction."""
    print("GENDER MAPPING APPLICATION TOOL")
    print("=" * 50)
    
    # Load and display current mapping
    mapping_dict = load_gender_mapping()
    if not mapping_dict:
        print("Cannot proceed without valid mapping. Please check your mapping template.")
        return
    
    print(f"\nFound {len(mapping_dict)} valid mappings in template.")
    print("\nOptions:")
    print("1. Dry run (preview changes without applying)")
    print("2. Apply mapping with backup")
    print("3. Apply mapping without backup")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\n--- RUNNING DRY RUN ---")
            success, stats = apply_gender_mapping(backup=False, dry_run=True)
            break
        elif choice == '2':
            print("\n--- APPLYING MAPPING WITH BACKUP ---")
            success, stats = apply_gender_mapping(backup=True, dry_run=False)
            break
        elif choice == '3':
            confirm = input("Are you sure you want to proceed without backup? (yes/no): ").strip().lower()
            if confirm == 'yes':
                print("\n--- APPLYING MAPPING WITHOUT BACKUP ---")
                success, stats = apply_gender_mapping(backup=False, dry_run=False)
                break
            else:
                print("Operation cancelled.")
                return
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    if success:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
