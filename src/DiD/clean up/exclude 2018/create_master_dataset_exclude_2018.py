#!/usr/bin/env python3
"""
Master Dataset Creation - Exclude 2018

This script creates a new master dataset excluding the problematic 2018 survey data.
Based on the comprehensive 2018 data quality investigation, the 2018 data has:
- 88.34% missing school attendance data
- 42.01% missing education level data  
- Multiple variables with 0% completeness
- Evidence of different survey methodology

The new master dataset will include only high-quality survey years:
2008, 2010, 2012, 2014, 2016, 2020

Author: DiD Analysis Project
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_original_dataset():
    """Load the original master dataset"""
    print("Loading original master dataset...")
    try:
        df = pd.read_csv("data/all_years_merged_dataset_final_corrected.csv", low_memory=False)
        print(f"Original dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Original dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return None

def analyze_original_dataset(df):
    """Analyze the original dataset before exclusion"""
    print("\n" + "="*70)
    print("ORIGINAL DATASET ANALYSIS")
    print("="*70)
    
    if 'survey_year' not in df.columns:
        print("Error: survey_year column not found")
        return None
    
    # Survey year distribution
    year_counts = df['survey_year'].value_counts().sort_index()
    print("\nSurvey year distribution:")
    total_obs = len(df)
    
    for year, count in year_counts.items():
        pct = count / total_obs * 100
        print(f"  {year}: {count:,} observations ({pct:.2f}%)")
    
    print(f"\nTotal observations: {total_obs:,}")
    print(f"Total survey years: {len(year_counts)}")
    
    # 2018 specific analysis
    df_2018 = df[df['survey_year'] == 2018]
    obs_2018 = len(df_2018)
    pct_2018 = obs_2018 / total_obs * 100
    
    print(f"\n2018 Data Summary:")
    print(f"  Observations: {obs_2018:,} ({pct_2018:.2f}% of total)")
    print(f"  Will be excluded due to data quality issues")
    
    return year_counts

def exclude_2018_data(df):
    """Exclude 2018 data and create clean dataset"""
    print("\n" + "="*70)
    print("EXCLUDING 2018 DATA")
    print("="*70)
    
    # Filter out 2018
    df_clean = df[df['survey_year'] != 2018].copy()
    
    original_count = len(df)
    clean_count = len(df_clean)
    excluded_count = original_count - clean_count
    
    print(f"Original dataset: {original_count:,} observations")
    print(f"Excluded (2018): {excluded_count:,} observations")
    print(f"Clean dataset: {clean_count:,} observations")
    print(f"Retention rate: {clean_count/original_count*100:.2f}%")
    
    # Verify no 2018 data remains
    remaining_years = df_clean['survey_year'].unique()
    print(f"\nRemaining survey years: {sorted(remaining_years)}")
    
    if 2018 in remaining_years:
        print("ERROR: 2018 data still present!")
        return None
    else:
        print("✓ 2018 data successfully excluded")
    
    return df_clean

def analyze_clean_dataset(df_clean):
    """Analyze the clean dataset after 2018 exclusion"""
    print("\n" + "="*70)
    print("CLEAN DATASET ANALYSIS")
    print("="*70)
    
    # Survey year distribution
    year_counts = df_clean['survey_year'].value_counts().sort_index()
    print("\nClean dataset survey year distribution:")
    total_obs = len(df_clean)
    
    for year, count in year_counts.items():
        pct = count / total_obs * 100
        print(f"  {year}: {count:,} observations ({pct:.2f}%)")
    
    print(f"\nTotal clean observations: {total_obs:,}")
    print(f"Survey years: {len(year_counts)}")
    
    # Data quality improvement check
    key_variables = ['is going to school?', 'Current edu. level', 'education level', 'Is employed?']
    
    print(f"\nData quality improvement (missing rates):")
    for var in key_variables:
        if var in df_clean.columns:
            missing_count = df_clean[var].isnull().sum()
            missing_pct = missing_count / len(df_clean) * 100
            print(f"  {var}: {missing_pct:.2f}% missing ({missing_count:,} observations)")
    
    return year_counts

def save_clean_dataset(df_clean, output_path):
    """Save the clean dataset to the specified location"""
    print("\n" + "="*70)
    print("SAVING CLEAN MASTER DATASET")
    print("="*70)
    
    try:
        df_clean.to_csv(output_path, index=False)
        print(f"Clean master dataset saved to: {output_path}")
        print(f"Dataset shape: {df_clean.shape}")
        
        # File size information
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False

def create_comparison_report(original_year_counts, clean_year_counts, output_dir):
    """Create a comparison report between original and clean datasets"""
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "dataset_exclusion_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("MASTER DATASET 2018 EXCLUSION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RATIONALE FOR 2018 EXCLUSION\n")
        f.write("-" * 30 + "\n")
        f.write("Based on comprehensive data quality investigation:\n")
        f.write("• School attendance: 88.34% missing (vs 9.0% in other years)\n")
        f.write("• Education level: 42.01% missing (vs 12.4% in other years)\n")
        f.write("• Employment status: 41.66% missing (vs 9.5% in other years)\n")
        f.write("• Multiple variables with 0% completeness\n")
        f.write("• Evidence of different survey methodology\n\n")
        
        f.write("DATASET COMPARISON\n")
        f.write("-" * 20 + "\n")
        f.write("Original Dataset:\n")
        original_total = original_year_counts.sum()
        for year, count in original_year_counts.items():
            pct = count / original_total * 100
            f.write(f"  {year}: {count:,} observations ({pct:.2f}%)\n")
        f.write(f"  Total: {original_total:,} observations\n\n")
        
        f.write("Clean Dataset (2018 excluded):\n")
        clean_total = clean_year_counts.sum()
        for year, count in clean_year_counts.items():
            pct = count / clean_total * 100
            f.write(f"  {year}: {count:,} observations ({pct:.2f}%)\n")
        f.write(f"  Total: {clean_total:,} observations\n\n")
        
        excluded_count = original_total - clean_total
        f.write("EXCLUSION SUMMARY\n")
        f.write("-" * 18 + "\n")
        f.write(f"Observations excluded: {excluded_count:,}\n")
        f.write(f"Retention rate: {clean_total/original_total*100:.2f}%\n")
        f.write(f"Data quality improvement: Significant\n\n")
        
        f.write("IMPACT ON ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Benefits of exclusion:\n")
        f.write("• Eliminates 88% missing data problem\n")
        f.write("• Creates consistent longitudinal trends\n")
        f.write("• Focuses on reliable survey methodology\n")
        f.write("• Improves statistical power for DiD analysis\n")
        f.write("• Maintains 6 years of high-quality data\n\n")
        
        f.write("RECOMMENDED USAGE\n")
        f.write("-" * 18 + "\n")
        f.write("• Use this clean dataset for primary DiD analysis\n")
        f.write("• Document 2018 exclusion in methodology\n")
        f.write("• Consider sensitivity analysis with/without 2018\n")
        f.write("• Reference investigation report for justification\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 18 + "\n")
        f.write("• master_dataset_exclude_2018.csv (main clean dataset)\n")
        f.write("• dataset_exclusion_report.txt (this report)\n")
        f.write("• exclusion_summary.csv (statistical summary)\n")
    
    print(f"Comparison report saved to: {report_file}")
    
    # Create summary CSV
    summary_data = []
    summary_data.append({
        'metric': 'original_observations',
        'value': original_year_counts.sum()
    })
    summary_data.append({
        'metric': 'clean_observations', 
        'value': clean_year_counts.sum()
    })
    summary_data.append({
        'metric': 'excluded_observations',
        'value': original_year_counts.sum() - clean_year_counts.sum()
    })
    summary_data.append({
        'metric': 'retention_rate_percent',
        'value': clean_year_counts.sum() / original_year_counts.sum() * 100
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "exclusion_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary statistics saved to: {summary_file}")

def main():
    """Main execution function"""
    print("MASTER DATASET CREATION - EXCLUDE 2018")
    print("="*50)
    print(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define output paths
    output_dataset_path = "data/master_dataset_exclude_2018.csv"
    output_report_dir = "data/data summary/dataset_exclusion"
    
    print(f"Output dataset: {output_dataset_path}")
    print(f"Output reports: {output_report_dir}")
    
    # Load original dataset
    df_original = load_original_dataset()
    if df_original is None:
        return
    
    # Analyze original dataset
    original_year_counts = analyze_original_dataset(df_original)
    if original_year_counts is None:
        return
    
    # Exclude 2018 data
    df_clean = exclude_2018_data(df_original)
    if df_clean is None:
        return
    
    # Analyze clean dataset
    clean_year_counts = analyze_clean_dataset(df_clean)
    
    # Save clean dataset
    success = save_clean_dataset(df_clean, output_dataset_path)
    if not success:
        return
    
    # Create comparison report
    create_comparison_report(original_year_counts, clean_year_counts, output_report_dir)
    
    print("\n" + "="*70)
    print("MASTER DATASET CREATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Clean dataset: {output_dataset_path}")
    print(f"Reports: {output_report_dir}")
    print(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final validation
    print(f"\nFINAL VALIDATION:")
    print(f"✓ 2018 data excluded")
    print(f"✓ Clean dataset saved")
    print(f"✓ Documentation generated")
    print(f"✓ Ready for DiD analysis")

if __name__ == "__main__":
    main()
