"""
Script to analyze unique values in the 'tinh' column and generate a summary CSV.
This script will create a 4-column CSV file with:
1. Unique tinh values
2. Correct name (placeholder for manual input)
3. Years where that tinh value appears
4. Count of records for each tinh value
"""

import pandas as pd
import os
from pathlib import Path

def analyze_tinh_values():
    """
    Analyze unique values in the tinh column and create a summary CSV.
    """
    # Define file paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_file = project_root / "data" / "all_years_merged_dataset_final_corrected.csv"
    output_dir = project_root / "data" / "data summary"
    output_file = output_dir / "tinh_summary.csv"
    
    print(f"Reading data from: {data_file}")
    
    # Read the dataset
    try:
        df = pd.read_csv(data_file, low_memory=False)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
    
    # Get unique tinh values with their corresponding years and counts
    print("Analyzing unique tinh values, their years, and counts...")
    
    # Group by tinh and get unique years and count for each
    tinh_summary = df.groupby('tinh').agg({
        'survey_year': [lambda x: ', '.join(map(str, sorted(x.unique()))), 'count']
    }).reset_index()
    
    # Flatten column names and rename for clarity
    tinh_summary.columns = ['tinh', 'years', 'count']
    
    # Create the summary dataframe
    summary_df = pd.DataFrame({
        'unique_value': tinh_summary['tinh'],
        'correct_name': '',  # Empty column for manual input
        'years': tinh_summary['years'],
        'count': tinh_summary['count']
    })
    
    # Sort by unique_value for better organization
    summary_df = summary_df.sort_values('unique_value').reset_index(drop=True)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nSummary statistics:")
    print(f"Total unique tinh values: {len(summary_df)}")
    print(f"Total records in dataset: {summary_df['count'].sum()}")
    print(f"Output saved to: {output_file}")
    
    # Display first few rows
    print(f"\nFirst 10 rows of the summary:")
    print(summary_df.head(10).to_string(index=False))
    
    # Display some statistics about counts
    print(f"\nCount statistics:")
    print(f"Average records per tinh: {summary_df['count'].mean():.1f}")
    print(f"Median records per tinh: {summary_df['count'].median():.1f}")
    print(f"Min records per tinh: {summary_df['count'].min()}")
    print(f"Max records per tinh: {summary_df['count'].max()}")
    
    # Display some statistics about years
    print(f"\nYear distribution analysis:")
    all_years = df['survey_year'].unique()
    print(f"All years in dataset: {sorted(all_years)}")
    
    # Count how many tinh values appear in each number of years
    year_counts = summary_df['years'].apply(lambda x: len(x.split(', ')))
    print(f"\nDistribution of tinh values by number of years they appear:")
    print(year_counts.value_counts().sort_index().to_string())

if __name__ == "__main__":
    analyze_tinh_values()
