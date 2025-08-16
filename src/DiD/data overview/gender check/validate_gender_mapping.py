"""
Validate Gender Mapping Template

This script validates the gender mapping template and shows what changes would be applied.
Use this before running the actual mapping application.

Author: Data Analysis Team
Date: 2024
"""

import pandas as pd
from pathlib import Path

def validate_mapping():
    """Validate the gender mapping template and show preview."""
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    
    # File paths
    mapping_file = project_root / "data" / "data summary" / "gender column" / "gender_mapping_template.csv"
    master_file = project_root / "data" / "all_years_merged_dataset_final_corrected.csv"
    
    print("GENDER MAPPING VALIDATION")
    print("=" * 50)
    
    try:
        # Load mapping
        print(f"Loading mapping from: {mapping_file}")
        mapping_df = pd.read_csv(mapping_file)
        print("Mapping template loaded successfully.\n")
        
        # Display current mappings
        print("CURRENT MAPPINGS:")
        print("-" * 20)
        valid_mappings = 0
        empty_mappings = 0
        
        for _, row in mapping_df.iterrows():
            original = str(row['original_value']).strip()
            mapped = str(row['mapped_value']).strip()
            
            if mapped and mapped != 'nan' and mapped != '':
                print(f"✅ '{original}' -> '{mapped}'")
                valid_mappings += 1
            else:
                print(f"⚠️  '{original}' -> (no mapping specified)")
                empty_mappings += 1
        
        print(f"\nSummary: {valid_mappings} valid mappings, {empty_mappings} empty mappings")
        
        # Load master dataset to show current distribution
        print(f"\nLoading master dataset from: {master_file}")
        df = pd.read_csv(master_file, low_memory=False)
        
        if 'gender' not in df.columns:
            print("Error: 'gender' column not found!")
            return
        
        print(f"Dataset loaded. Total records: {len(df):,}")
        
        # Show current gender distribution
        print(f"\nCURRENT GENDER DISTRIBUTION:")
        print("-" * 30)
        gender_counts = df['gender'].value_counts(dropna=False)
        
        total_affected = 0
        for value, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            
            # Check if this value has a mapping
            mapping_status = ""
            for _, map_row in mapping_df.iterrows():
                if str(map_row['original_value']).strip() == str(value):
                    mapped_val = str(map_row['mapped_value']).strip()
                    if mapped_val and mapped_val != 'nan' and mapped_val != '':
                        mapping_status = f" -> will become '{mapped_val}'"
                        total_affected += count
                    else:
                        mapping_status = " -> no mapping (will remain unchanged)"
                    break
            
            print(f"'{value}': {count:,} ({percentage:.2f}%){mapping_status}")
        
        print(f"\nTotal records that will be changed: {total_affected:,} ({(total_affected/len(df)*100):.2f}%)")
        
        # Show what the final distribution would look like
        if valid_mappings > 0:
            print(f"\nPREDICTED FINAL DISTRIBUTION:")
            print("-" * 32)
            
            # Create a copy for simulation
            df_sim = df.copy()
            df_sim['gender'] = df_sim['gender'].astype(str)
            
            # Apply mappings
            for _, row in mapping_df.iterrows():
                original = str(row['original_value']).strip()
                mapped = str(row['mapped_value']).strip()
                
                if mapped and mapped != 'nan' and mapped != '':
                    mask = df_sim['gender'] == original
                    df_sim.loc[mask, 'gender'] = mapped
            
            final_counts = df_sim['gender'].value_counts(dropna=False)
            for value, count in final_counts.items():
                percentage = (count / len(df_sim)) * 100
                print(f"'{value}': {count:,} ({percentage:.2f}%)")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 15)
        
        if empty_mappings > 0:
            print(f"⚠️  You have {empty_mappings} unmapped values. Consider:")
            for _, row in mapping_df.iterrows():
                original = str(row['original_value']).strip()
                mapped = str(row['mapped_value']).strip()
                
                if not mapped or mapped == 'nan' or mapped == '':
                    count = (df['gender'].astype(str) == original).sum()
                    print(f"   - Map '{original}' ({count:,} records) to appropriate value")
        
        # Check for potential issues
        if 'Nam' in [str(row['mapped_value']).strip() for _, row in mapping_df.iterrows()]:
            print("⚠️  Warning: 'Nam' appears as a mapped value. Did you mean 'Male'?")
        
        if 'Nữ' in [str(row['mapped_value']).strip() for _, row in mapping_df.iterrows()]:
            print("⚠️  Warning: 'Nữ' appears as a mapped value. Did you mean 'Female'?")
        
        print(f"\n✅ Validation complete! Review the mappings above before applying.")
        
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {mapping_file}")
        print("Please run gender_unique_values_analysis.py first to create the template.")
    
    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == "__main__":
    validate_mapping()
