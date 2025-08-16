"""
Gender Column Unique Values Analysis

This script extracts unique values from the gender column in the master dataset
and creates a mapping template CSV file for data cleaning purposes.

Author: Data Analysis Team
Date: 2024
"""

import pandas as pd
import os
from pathlib import Path

def extract_gender_unique_values():
    """
    Extract unique values from the gender column and create a mapping template.
    
    Returns:
        tuple: (unique_values_list, output_file_path)
    """
    
    # Define file paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    
    # Input file path
    input_file = project_root / "data" / "all_years_merged_dataset_final_corrected.csv"
    
    # Output directory and file
    output_dir = project_root / "data" / "data summary" / "gender column"
    output_file = output_dir / "gender_mapping_template.csv"
    
    print(f"Reading data from: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    try:
        # Read the dataset
        print("Loading dataset...")
        df = pd.read_csv(input_file, low_memory=False)
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Total rows: {len(df):,}")
        
        # Check if gender column exists
        if 'gender' not in df.columns:
            print("Error: 'gender' column not found in the dataset!")
            print("Available columns:", list(df.columns))
            return None, None
        
        # Extract unique values from gender column
        print("\nExtracting unique values from 'gender' column...")
        unique_genders = df['gender'].dropna().unique()
        unique_genders = sorted([str(val) for val in unique_genders])
        
        print(f"Found {len(unique_genders)} unique gender values:")
        for i, value in enumerate(unique_genders, 1):
            print(f"  {i}. '{value}'")
        
        # Count occurrences of each unique value
        print("\nCounting occurrences of each value:")
        gender_counts = df['gender'].value_counts(dropna=False)
        for value in unique_genders:
            count = gender_counts.get(value, 0)
            percentage = (count / len(df)) * 100
            print(f"  '{value}': {count:,} ({percentage:.2f}%)")
        
        # Check for missing values
        missing_count = df['gender'].isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            print(f"  Missing/NaN values: {missing_count:,} ({missing_percentage:.2f}%)")
        
        # Create mapping template DataFrame
        mapping_data = {
            'original_value': unique_genders,
            'mapped_value': [''] * len(unique_genders)  # Empty column for manual mapping
        }
        
        mapping_df = pd.DataFrame(mapping_data)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        mapping_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nMapping template saved successfully to: {output_file}")
        print("\nMapping template preview:")
        print(mapping_df.to_string(index=False))
        
        # Generate analysis summary
        summary_file = output_dir / "gender_analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("GENDER COLUMN ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source File: {input_file}\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Unique Gender Values: {len(unique_genders)}\n\n")
            
            f.write("UNIQUE VALUES AND COUNTS:\n")
            f.write("-" * 30 + "\n")
            for value in unique_genders:
                count = gender_counts.get(value, 0)
                percentage = (count / len(df)) * 100
                f.write(f"'{value}': {count:,} ({percentage:.2f}%)\n")
            
            if missing_count > 0:
                f.write(f"Missing/NaN: {missing_count:,} ({missing_percentage:.2f}%)\n")
            
            f.write(f"\nMapping template created: {output_file}\n")
            f.write("\nNext Steps:\n")
            f.write("1. Review the unique values in the mapping template\n")
            f.write("2. Fill in the 'mapped_value' column with standardized values\n")
            f.write("3. Use the completed mapping to clean the gender column\n")
        
        print(f"\nAnalysis summary saved to: {summary_file}")
        
        return unique_genders, output_file
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file: {input_file}")
        print("Please ensure the file exists and the path is correct.")
        return None, None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def main():
    """Main function to run the gender analysis."""
    print("GENDER COLUMN UNIQUE VALUES ANALYSIS")
    print("=" * 50)
    
    unique_values, output_file = extract_gender_unique_values()
    
    if unique_values is not None:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Files created in: data/data summary/gender column/")
        print(f"📄 Mapping template: gender_mapping_template.csv")
        print(f"📄 Analysis summary: gender_analysis_summary.txt")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
