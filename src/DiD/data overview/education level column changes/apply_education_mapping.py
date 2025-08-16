#!/usr/bin/env python3
"""
Efficient script to apply education level mappings to the master dataset.
This version processes data more efficiently for large datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

def apply_education_mapping_efficient():
    """
    Apply education level mappings efficiently to the master dataset
    """
    # File paths
    dataset_path = Path("data/all_years_merged_dataset_final_corrected.csv")
    mapping_path = Path("data/education_level_mapping_template.csv")
    backup_path = Path("data/all_years_merged_dataset_final_corrected_backup.csv")
    
    print("EFFICIENT EDUCATION LEVEL MAPPING APPLICATION")
    print("=" * 60)
    
    # Step 1: Load and validate mapping file
    print(f"1. Loading mapping file: {mapping_path}")
    try:
        mapping_df = pd.read_csv(mapping_path, encoding='utf-8-sig')
        print(f"   ✓ Mapping file loaded: {len(mapping_df)} rules")
    except Exception as e:
        print(f"   ✗ Error loading mapping file: {e}")
        return
    
    # Create mapping dictionary
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        original = str(row['original_education_level'])
        new_value = row['new_education_level_mapping']
        
        if pd.notna(new_value) and str(new_value).strip():
            mapping_dict[original] = str(new_value).strip()
    
    print(f"   ✓ Created {len(mapping_dict)} valid mappings")
    
    # Display the mappings that will be applied
    print(f"\n   MAPPINGS TO APPLY:")
    for original, new_val in mapping_dict.items():
        print(f"     '{original}' → '{new_val}'")
    
    # Step 2: Load dataset
    print(f"\n2. Loading dataset: {dataset_path}")
    try:
        # Use chunked reading for efficiency with large files
        chunk_size = 50000
        chunks = []
        total_rows = 0
        
        print(f"   Reading in chunks of {chunk_size:,} rows...")
        for chunk in pd.read_csv(dataset_path, low_memory=False, chunksize=chunk_size):
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"   Processed {total_rows:,} rows...", end='\r')
        
        print(f"\n   ✓ Dataset loaded: {total_rows:,} rows")
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    education_col = 'education level'
    if education_col not in df.columns:
        print(f"   ✗ Column '{education_col}' not found")
        return
    
    # Step 3: Create backup if it doesn't exist
    if not backup_path.exists():
        print(f"\n3. Creating backup: {backup_path}")
        try:
            shutil.copy2(dataset_path, backup_path)
            print(f"   ✓ Backup created")
        except Exception as e:
            print(f"   ⚠ Warning: Could not create backup: {e}")
    else:
        print(f"\n3. Backup already exists: {backup_path}")
    
    # Step 4: Apply mappings efficiently using pandas map
    print(f"\n4. Applying mappings to '{education_col}' column")
    
    # Store original values for comparison
    original_counts = df[education_col].value_counts(dropna=False)
    
    # Convert to string for consistent mapping
    df[education_col] = df[education_col].astype(str)
    
    # Apply mappings using pandas map (very efficient)
    df[education_col] = df[education_col].map(mapping_dict).fillna(df[education_col])
    
    # Convert 'nan' strings back to actual NaN for missing values
    df.loc[df[education_col] == 'nan', education_col] = np.nan
    
    print(f"   ✓ Mappings applied successfully")
    
    # Step 5: Analyze results
    print(f"\n5. Analyzing results")
    new_counts = df[education_col].value_counts(dropna=False)
    
    print(f"   NEW VALUE DISTRIBUTION:")
    for value, count in new_counts.head(10).items():
        percentage = (count / len(df)) * 100
        if pd.isna(value):
            print(f"     [MISSING]: {count:,} ({percentage:.2f}%)")
        else:
            print(f"     '{value}': {count:,} ({percentage:.2f}%)")
    
    # Calculate changes made
    changes_made = 0
    change_details = []
    
    for original_val, new_val in mapping_dict.items():
        original_count = original_counts.get(original_val, 0)
        if original_count > 0:
            changes_made += original_count
            change_details.append({
                'original': original_val,
                'new': new_val,
                'count': original_count,
                'percentage': (original_count / len(df)) * 100
            })
    
    print(f"\n   ✓ Total changes applied: {changes_made:,}")
    
    # Step 6: Save updated dataset
    print(f"\n6. Saving updated dataset")
    try:
        df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
        print(f"   ✓ Dataset saved to: {dataset_path}")
    except Exception as e:
        print(f"   ✗ Error saving dataset: {e}")
        return
    
    # Step 7: Generate change report
    report_path = Path("data/data summary/education_mapping_change_report.txt")
    print(f"\n7. Generating change report: {report_path}")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EDUCATION LEVEL MAPPING CHANGE REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total changes: {changes_made:,}\n\n")
        
        f.write("CHANGES APPLIED:\n")
        f.write("-" * 40 + "\n")
        for change in sorted(change_details, key=lambda x: x['count'], reverse=True):
            f.write(f"'{change['original']}' → '{change['new']}': "
                   f"{change['count']:,} records ({change['percentage']:.2f}%)\n")
        
        f.write(f"\nFINAL DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for value, count in new_counts.items():
            percentage = (count / len(df)) * 100
            if pd.isna(value):
                f.write(f"[MISSING]: {count:,} ({percentage:.2f}%)\n")
            else:
                f.write(f"'{value}': {count:,} ({percentage:.2f}%)\n")
    
    print(f"   ✓ Change report saved")
    
    print(f"\n" + "=" * 60)
    print("MAPPING APPLICATION COMPLETE!")
    print("=" * 60)
    print(f"✓ {changes_made:,} values successfully mapped")
    print(f"✓ Dataset updated: {dataset_path}")
    print(f"✓ Backup available: {backup_path}")
    print(f"✓ Report saved: {report_path}")

def main():
    """Main function"""
    apply_education_mapping_efficient()

if __name__ == "__main__":
    main()
