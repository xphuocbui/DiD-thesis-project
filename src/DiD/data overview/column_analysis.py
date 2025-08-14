#!/usr/bin/env python3
"""
Script to analyze CSV dataset columns:
- Count rows (non-missing values) for all columns
- Count missing/empty values for all columns
- Provide summary statistics
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_csv_columns(file_path):
    """
    Analyze CSV file columns for row counts and missing values
    
    Args:
        file_path (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print("=" * 80)
        
        # Analyze each column
        results = []
        
        for col in df.columns:
            col_data = df[col]
            
            # Count missing values (NaN, None, empty strings)
            missing_count = col_data.isnull().sum() + (col_data == '').sum()
            total_rows = len(col_data)
            missing_percentage = (missing_count / total_rows) * 100
            
            # Count non-missing values (total rows with data)
            non_missing_count = total_rows - missing_count
            total_value = f"{non_missing_count:,}"
            data_type = str(col_data.dtype)
            
            results.append({
                'Column': col,
                'Data Type': data_type,
                'Total Rows': f"{total_rows:,}",
                'Missing Rows': f"{missing_count:,} ({missing_percentage:.1f}%)",
                'Non-missing Rows': f"{non_missing_count:,}"
            })
        
        # Create results DataFrame for better formatting
        results_df = pd.DataFrame(results)
        
        # Print detailed results
        print("\nCOLUMN ANALYSIS RESULTS")
        print("=" * 80)
        
        for i, row in results_df.iterrows():
            print(f"\nColumn: {row['Column']}")
            print(f"  Data Type: {row['Data Type']}")
            print(f"  Total Rows: {row['Total Rows']}")
            print(f"  Missing Rows: {row['Missing Rows']}")
            print(f"  Non-missing Rows: {row['Non-missing Rows']}")
        
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        
        # Print summary table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        print(results_df.to_string(index=False))
        
        # Additional summary statistics
        print(f"\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df):,}")
        
        # Count numeric vs non-numeric columns
        numeric_cols = []
        non_numeric_cols = []
        
        for col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if not numeric_col.isnull().all():
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
        
        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Non-numeric columns: {len(non_numeric_cols)}")
        
        # Overall missing data statistics
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum() + (df == '').sum().sum()
        missing_percentage = (total_missing / total_cells) * 100
        
        print(f"Total cells: {total_cells:,}")
        print(f"Total missing cells: {total_missing:,} ({missing_percentage:.2f}%)")
        print(f"Total non-missing cells: {total_cells - total_missing:,}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)

def main():
    """Main function"""
    # Default file path - calculate relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..", "..")
    default_file = os.path.join(project_root, "data", "all_years_merged_dataset_final_corrected.csv")
    default_file = os.path.normpath(default_file)  # Clean up the path
    
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        print(f"Usage: python {sys.argv[0]} [path_to_csv_file]")
        print(f"Default file: {default_file}")
        sys.exit(1)
    
    analyze_csv_columns(file_path)

if __name__ == "__main__":
    main()
