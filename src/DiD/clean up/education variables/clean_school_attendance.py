#!/usr/bin/env python3
"""
School Attendance Variable Cleaning Script

This script cleans the 'is going to school?' variable to create a standardized 
'currently_enrolled' binary indicator for education analysis.

Mapping Logic:
- 'Có' (Vietnamese: Yes) → 1 (attending school)
- '1' (numeric code: Yes) → 1 (attending school)
- 'Không' (Vietnamese: No) → 0 (not attending school)
- '3' (numeric code: No) → 0 (not attending school)
- 'Nghỉ hè' (Vietnamese: Summer break) → 1 (still enrolled)
- 'Ngh? hè' (Vietnamese: Summer break, encoding issue) → 1 (still enrolled)
- '2' (numeric code: Summer break) → 1 (still enrolled)
- NaN and other values → NaN (missing)

Author: DiD Analysis Project
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "data/data summary/education variables/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data():
    """Load the main dataset"""
    print("Loading main dataset...")
    try:
        df = pd.read_csv("data/all_years_merged_dataset_final_corrected.csv")
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def analyze_original_variable(df):
    """Analyze the original 'is going to school?' variable before cleaning"""
    print("\n" + "="*70)
    print("ANALYZING ORIGINAL 'is going to school?' VARIABLE")
    print("="*70)
    
    school_var = 'is going to school?'
    
    if school_var not in df.columns:
        print(f"Error: Variable '{school_var}' not found in dataset")
        return None
    
    # Value counts
    print("\nOriginal Value Counts:")
    value_counts = df[school_var].value_counts(dropna=False)
    print(value_counts)
    
    # Percentages
    print("\nOriginal Percentages:")
    percentages = df[school_var].value_counts(normalize=True, dropna=False) * 100
    print(percentages.round(2))
    
    return value_counts

def create_currently_enrolled_variable(df):
    """Create the clean currently_enrolled variable"""
    print("\n" + "="*70)
    print("CREATING CLEAN 'currently_enrolled' VARIABLE")
    print("="*70)
    
    school_var = 'is going to school?'
    
    if school_var not in df.columns:
        print(f"Error: Variable '{school_var}' not found in dataset")
        return df
    
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Initialize the new variable with NaN
    df['currently_enrolled'] = np.nan
    
    # Apply mapping logic
    print("\nApplying mapping rules:")
    
    # Attending school (Yes) = 1
    yes_conditions = (
        (df[school_var] == 'Có') |
        (df[school_var] == '1') |
        (df[school_var] == 1)
    )
    df.loc[yes_conditions, 'currently_enrolled'] = 1
    yes_count = yes_conditions.sum()
    print(f"- 'Có' and '1' (Yes) → 1: {yes_count:,} observations")
    
    # Not attending school (No) = 0  
    no_conditions = (
        (df[school_var] == 'Không') |
        (df[school_var] == '3') |
        (df[school_var] == 3)
    )
    df.loc[no_conditions, 'currently_enrolled'] = 0
    no_count = no_conditions.sum()
    print(f"- 'Không' and '3' (No) → 0: {no_count:,} observations")
    
    # Summer break (still enrolled) = 1
    summer_conditions = (
        (df[school_var] == 'Nghỉ hè') |
        (df[school_var] == 'Ngh? hè') |
        (df[school_var] == '2') |
        (df[school_var] == 2)
    )
    df.loc[summer_conditions, 'currently_enrolled'] = 1
    summer_count = summer_conditions.sum()
    print(f"- 'Nghỉ hè' and '2' (Summer break) → 1: {summer_count:,} observations")
    
    # Check for unmapped values
    mapped_conditions = yes_conditions | no_conditions | summer_conditions | df[school_var].isnull()
    unmapped_count = (~mapped_conditions).sum()
    if unmapped_count > 0:
        print(f"\nWARNING: {unmapped_count:,} observations have unmapped values:")
        unmapped_values = df.loc[~mapped_conditions, school_var].value_counts()
        print(unmapped_values)
    
    # Summary of new variable
    print(f"\nSummary of 'currently_enrolled' variable:")
    enrolled_counts = df['currently_enrolled'].value_counts(dropna=False)
    print(enrolled_counts)
    
    print(f"\nPercentages:")
    enrolled_pct = df['currently_enrolled'].value_counts(normalize=True, dropna=False) * 100
    print(enrolled_pct.round(2))
    
    return df

def analyze_distribution_by_year(df, output_dir):
    """Show distribution of currently_enrolled by survey year"""
    print("\n" + "="*70)
    print("DISTRIBUTION OF 'currently_enrolled' BY SURVEY YEAR")
    print("="*70)
    
    if 'survey_year' not in df.columns or 'currently_enrolled' not in df.columns:
        print("Error: Required variables not found")
        return
    
    # Create cross-tabulation
    year_crosstab = pd.crosstab(df['survey_year'], df['currently_enrolled'], 
                               margins=True, dropna=False)
    print("\nEnrollment by Survey Year (counts):")
    print(year_crosstab)
    
    # Calculate percentages (excluding missing values for cleaner interpretation)
    year_pct = pd.crosstab(df['survey_year'], df['currently_enrolled'], 
                          normalize='index', dropna=True) * 100
    print("\nEnrollment by Survey Year (percentages, excluding missing):")
    print(year_pct.round(2))
    
    # Save results
    crosstab_file = os.path.join(output_dir, "enrollment_by_year_counts.csv")
    year_crosstab.to_csv(crosstab_file)
    
    pct_file = os.path.join(output_dir, "enrollment_by_year_percentages.csv")
    year_pct.to_csv(pct_file)
    
    print(f"\nResults saved to:")
    print(f"- Counts: {crosstab_file}")
    print(f"- Percentages: {pct_file}")
    
    return year_crosstab, year_pct

def create_age_group_crosstab(df, output_dir):
    """Create age group cross-tabulation with original values"""
    print("\n" + "="*70)
    print("AGE GROUP CROSS-TABULATION WITH ORIGINAL VALUES")
    print("="*70)
    
    school_var = 'is going to school?'
    age_var = 'tuoi'
    
    if age_var not in df.columns or school_var not in df.columns:
        print("Error: Required variables not found")
        return
    
    # Create age groups - first convert age to numeric
    df = df.copy()
    df[age_var] = pd.to_numeric(df[age_var], errors='coerce')
    
    df['age_group'] = pd.cut(df[age_var], 
                            bins=[0, 5, 11, 17, float('inf')], 
                            labels=['0-5 (pre-school)', '6-11 (primary)', 
                                   '12-17 (secondary)', '18+ (post-secondary)'],
                            right=True, include_lowest=True)
    
    # Convert school variable to string for consistent sorting
    df[school_var] = df[school_var].astype(str)
    
    # Create cross-tabulation
    age_crosstab = pd.crosstab(df['age_group'], df[school_var], 
                              margins=True, dropna=False)
    
    print("\nAge Group vs Original 'is going to school?' Values:")
    print(age_crosstab)
    
    # Calculate percentages by age group
    age_pct = pd.crosstab(df['age_group'], df[school_var], 
                         normalize='index', dropna=False) * 100
    
    print("\nPercentages by Age Group:")
    print(age_pct.round(2))
    
    # Focus on school-age population (6-17) for detailed analysis
    school_age_df = df[(df[age_var] >= 6) & (df[age_var] <= 17)].copy()
    
    if len(school_age_df) > 0:
        print(f"\nDetailed analysis for school-age population (6-17 years): {len(school_age_df):,} observations")
        
        school_age_crosstab = pd.crosstab(school_age_df['age_group'], 
                                         school_age_df[school_var], 
                                         margins=True, dropna=False)
        print("\nSchool-age Cross-tabulation:")
        print(school_age_crosstab)
    
    # Save results
    age_crosstab_file = os.path.join(output_dir, "age_group_original_values_crosstab.csv")
    age_crosstab.to_csv(age_crosstab_file)
    
    age_pct_file = os.path.join(output_dir, "age_group_original_values_percentages.csv")
    age_pct.to_csv(age_pct_file)
    
    print(f"\nResults saved to:")
    print(f"- Counts: {age_crosstab_file}")
    print(f"- Percentages: {age_pct_file}")
    
    return age_crosstab, age_pct

def generate_cleaning_report(df, original_counts, output_dir):
    """Generate comprehensive cleaning report"""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE CLEANING REPORT")
    print("="*70)
    
    report_file = os.path.join(output_dir, "school_attendance_cleaning_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SCHOOL ATTENDANCE VARIABLE CLEANING REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MAPPING LOGIC APPLIED:\n")
        f.write("-" * 25 + "\n")
        f.write("• 'Có' (Vietnamese: Yes) → 1 (attending school)\n")
        f.write("• '1' (numeric code: Yes) → 1 (attending school)\n")
        f.write("• 'Không' (Vietnamese: No) → 0 (not attending school)\n")
        f.write("• '3' (numeric code: No) → 0 (not attending school)\n")
        f.write("• 'Nghỉ hè' (Vietnamese: Summer break) → 1 (still enrolled)\n")
        f.write("• 'Ngh? hè' (encoding issue: Summer break) → 1 (still enrolled)\n")
        f.write("• '2' (numeric code: Summer break) → 1 (still enrolled)\n")
        f.write("• NaN and other values → NaN (missing)\n\n")
        
        f.write("BEFORE CLEANING:\n")
        f.write("-" * 20 + "\n")
        f.write("Original 'is going to school?' variable:\n")
        f.write(str(original_counts) + "\n\n")
        
        f.write("AFTER CLEANING:\n")
        f.write("-" * 18 + "\n")
        f.write("New 'currently_enrolled' variable:\n")
        clean_counts = df['currently_enrolled'].value_counts(dropna=False)
        f.write(str(clean_counts) + "\n\n")
        
        # Calculate cleaning effectiveness
        original_valid = len(df) - original_counts.get(np.nan, 0)
        clean_valid = df['currently_enrolled'].notna().sum()
        
        f.write("CLEANING EFFECTIVENESS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total observations: {len(df):,}\n")
        f.write(f"Original valid values: {original_valid:,}\n")
        f.write(f"Clean valid values: {clean_valid:,}\n")
        f.write(f"Values successfully mapped: {clean_valid:,}\n")
        f.write(f"Mapping success rate: {(clean_valid/original_valid)*100:.2f}%\n\n")
        
        # Summary statistics
        if clean_valid > 0:
            enrollment_rate = (df['currently_enrolled'] == 1).sum() / clean_valid * 100
            f.write(f"Overall enrollment rate: {enrollment_rate:.2f}%\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-" * 18 + "\n")
        f.write("• enrollment_by_year_counts.csv\n")
        f.write("• enrollment_by_year_percentages.csv\n")
        f.write("• age_group_original_values_crosstab.csv\n")
        f.write("• age_group_original_values_percentages.csv\n")
        f.write("• cleaned_dataset_with_enrollment.csv\n")
    
    print(f"Comprehensive cleaning report saved to: {report_file}")

def save_cleaned_dataset(df, output_dir):
    """Save the dataset with the new currently_enrolled variable"""
    print("\n" + "="*70)
    print("SAVING CLEANED DATASET")
    print("="*70)
    
    output_file = os.path.join(output_dir, "cleaned_dataset_with_enrollment.csv")
    
    # Save the full dataset with new variable
    df.to_csv(output_file, index=False)
    
    print(f"Cleaned dataset saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"New variable 'currently_enrolled' added successfully")
    
    # Quick verification
    if 'currently_enrolled' in df.columns:
        valid_count = df['currently_enrolled'].notna().sum()
        total_count = len(df)
        print(f"Valid enrollment data: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.2f}%)")

def main():
    """Main execution function"""
    print("SCHOOL ATTENDANCE VARIABLE CLEANING")
    print("="*50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Analyze original variable
    original_counts = analyze_original_variable(df)
    if original_counts is None:
        return
    
    # Create clean variable
    df = create_currently_enrolled_variable(df)
    
    # Analyze distributions and patterns
    analyze_distribution_by_year(df, output_dir)
    create_age_group_crosstab(df, output_dir)
    
    # Generate comprehensive report
    generate_cleaning_report(df, original_counts, output_dir)
    
    # Save cleaned dataset
    save_cleaned_dataset(df, output_dir)
    
    print("\n" + "="*70)
    print("SCHOOL ATTENDANCE CLEANING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
