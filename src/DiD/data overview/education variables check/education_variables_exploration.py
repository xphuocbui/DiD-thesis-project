#!/usr/bin/env python3
"""
Education Variables Exploration Script

This script explores education-related variables in the Vietnam Household Living Standards Survey (VHLSS) 
dataset to understand their coding, patterns, and data quality before conducting DiD analysis.

Author: DiD Analysis Project
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "data/data summary/education variables"
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

def analyze_school_attendance(df, output_dir):
    """Analyze 'is going to school?' variable"""
    print("\n" + "="*60)
    print("ANALYZING 'is going to school?' VARIABLE")
    print("="*60)
    
    school_var = 'is going to school?'
    
    # Check if variable exists
    if school_var not in df.columns:
        print(f"Variable '{school_var}' not found in dataset")
        return
    
    # Value counts
    print("\nValue Counts for 'is going to school?':")
    value_counts = df[school_var].value_counts(dropna=False)
    print(value_counts)
    
    # Percentages
    print("\nPercentages:")
    percentages = df[school_var].value_counts(normalize=True, dropna=False) * 100
    print(percentages.round(2))
    
    # Missing data
    missing_count = df[school_var].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"\nMissing values: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Save results
    results_file = os.path.join(output_dir, "school_attendance_analysis.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSIS OF 'is going to school?' VARIABLE\n")
        f.write("="*50 + "\n\n")
        f.write("Value Counts:\n")
        f.write(str(value_counts) + "\n\n")
        f.write("Percentages:\n")
        f.write(str(percentages.round(2)) + "\n\n")
        f.write(f"Missing values: {missing_count:,} ({missing_pct:.2f}%)\n")
    
    print(f"\nResults saved to: {results_file}")

def analyze_current_education_level(df, output_dir):
    """Analyze 'Current edu. level' variable"""
    print("\n" + "="*60)
    print("ANALYZING 'Current edu. level' VARIABLE")
    print("="*60)
    
    current_edu_var = 'Current edu. level'
    
    # Check if variable exists
    if current_edu_var not in df.columns:
        print(f"Variable '{current_edu_var}' not found in dataset")
        return
    
    # Value counts (top 20)
    print("\nTop 20 Value Counts for 'Current edu. level':")
    value_counts = df[current_edu_var].value_counts(dropna=False)
    print(value_counts.head(20))
    
    # Total unique values
    unique_count = df[current_edu_var].nunique()
    print(f"\nTotal unique values: {unique_count}")
    
    # Missing data
    missing_count = df[current_edu_var].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"Missing values: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "current_edu_level_analysis.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSIS OF 'Current edu. level' VARIABLE\n")
        f.write("="*50 + "\n\n")
        f.write("All Value Counts:\n")
        f.write(str(value_counts) + "\n\n")
        f.write(f"Total unique values: {unique_count}\n")
        f.write(f"Missing values: {missing_count:,} ({missing_pct:.2f}%)\n")
    
    # Save value counts as CSV
    value_counts_df = pd.DataFrame({
        'value': value_counts.index,
        'count': value_counts.values
    })
    csv_file = os.path.join(output_dir, "current_edu_level_values.csv")
    value_counts_df.to_csv(csv_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Value counts CSV saved to: {csv_file}")

def analyze_education_level(df, output_dir):
    """Analyze 'education level' (highest completed) variable"""
    print("\n" + "="*60)
    print("ANALYZING 'education level' VARIABLE")
    print("="*60)
    
    edu_level_var = 'education level'
    
    # Check if variable exists
    if edu_level_var not in df.columns:
        print(f"Variable '{edu_level_var}' not found in dataset")
        return
    
    # Value counts
    print("\nValue Counts for 'education level':")
    value_counts = df[edu_level_var].value_counts(dropna=False)
    print(value_counts)
    
    # Percentages
    print("\nPercentages:")
    percentages = df[edu_level_var].value_counts(normalize=True, dropna=False) * 100
    print(percentages.round(2))
    
    # Total unique values
    unique_count = df[edu_level_var].nunique()
    print(f"\nTotal unique values: {unique_count}")
    
    # Missing data
    missing_count = df[edu_level_var].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"Missing values: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Save results
    results_file = os.path.join(output_dir, "education_level_analysis.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ANALYSIS OF 'education level' VARIABLE\n")
        f.write("="*50 + "\n\n")
        f.write("Value Counts:\n")
        f.write(str(value_counts) + "\n\n")
        f.write("Percentages:\n")
        f.write(str(percentages.round(2)) + "\n\n")
        f.write(f"Total unique values: {unique_count}\n")
        f.write(f"Missing values: {missing_count:,} ({missing_pct:.2f}%)\n")
    
    print(f"\nResults saved to: {results_file}")

def create_age_grade_crosstab(df, output_dir):
    """Create cross-tabulation of age vs Current edu. level"""
    print("\n" + "="*60)
    print("CREATING AGE vs CURRENT EDUCATION LEVEL CROSS-TABULATION")
    print("="*60)
    
    age_var = 'tuoi'
    current_edu_var = 'Current edu. level'
    
    # Check if variables exist
    if age_var not in df.columns or current_edu_var not in df.columns:
        missing_vars = []
        if age_var not in df.columns:
            missing_vars.append(age_var)
        if current_edu_var not in df.columns:
            missing_vars.append(current_edu_var)
        print(f"Variables not found: {missing_vars}")
        return
    
    # Filter to school-age population for meaningful analysis (ages 5-25)
    school_age_df = df[(df[age_var] >= 5) & (df[age_var] <= 25)].copy()
    print(f"Analyzing school-age population (ages 5-25): {len(school_age_df):,} observations")
    
    # Create crosstab
    crosstab = pd.crosstab(school_age_df[age_var], 
                          school_age_df[current_edu_var], 
                          margins=True, 
                          dropna=False)
    
    print("\nAge vs Current Education Level Cross-tabulation (showing first 10 rows and columns):")
    print(crosstab.iloc[:10, :10])
    
    # Save full crosstab
    crosstab_file = os.path.join(output_dir, "age_grade_crosstab.csv")
    crosstab.to_csv(crosstab_file)
    
    # Create summary statistics
    summary_stats = school_age_df.groupby(age_var)[current_edu_var].agg([
        'count', 
        lambda x: x.isnull().sum(),  # missing count
        lambda x: (x.isnull().sum() / len(x)) * 100  # missing percentage
    ]).round(2)
    summary_stats.columns = ['total_count', 'missing_count', 'missing_percentage']
    
    summary_file = os.path.join(output_dir, "age_education_summary.csv")
    summary_stats.to_csv(summary_file)
    
    print(f"\nFull cross-tabulation saved to: {crosstab_file}")
    print(f"Summary statistics saved to: {summary_file}")

def analyze_missing_data_by_year(df, output_dir):
    """Analyze missing data patterns by survey year"""
    print("\n" + "="*60)
    print("ANALYZING MISSING DATA PATTERNS BY SURVEY YEAR")
    print("="*60)
    
    education_vars = ['is going to school?', 'Current edu. level', 'education level']
    survey_year_var = 'survey_year'
    
    # Check if survey year variable exists
    if survey_year_var not in df.columns:
        print(f"Variable '{survey_year_var}' not found in dataset")
        return
    
    missing_analysis = []
    
    for var in education_vars:
        if var in df.columns:
            print(f"\nAnalyzing missing data for: {var}")
            
            # Missing data by year
            yearly_missing = df.groupby(survey_year_var).agg({
                var: [
                    'count',  # non-missing count
                    lambda x: x.isnull().sum(),  # missing count
                    lambda x: (x.isnull().sum() / len(x)) * 100  # missing percentage
                ]
            }).round(2)
            
            yearly_missing.columns = ['non_missing_count', 'missing_count', 'missing_percentage']
            yearly_missing['variable'] = var
            yearly_missing['survey_year'] = yearly_missing.index
            
            missing_analysis.append(yearly_missing.reset_index(drop=True))
            
            print(f"Missing data summary for {var}:")
            print(yearly_missing[['missing_count', 'missing_percentage']])
        else:
            print(f"Variable '{var}' not found in dataset")
    
    # Combine all missing data analysis
    if missing_analysis:
        combined_missing = pd.concat(missing_analysis, ignore_index=True)
        
        # Save results
        missing_file = os.path.join(output_dir, "missing_data_by_year.csv")
        combined_missing.to_csv(missing_file, index=False)
        
        # Create summary report
        summary_file = os.path.join(output_dir, "missing_data_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("MISSING DATA ANALYSIS BY SURVEY YEAR\n")
            f.write("="*50 + "\n\n")
            
            for var in education_vars:
                if var in df.columns:
                    var_data = combined_missing[combined_missing['variable'] == var]
                    f.write(f"Variable: {var}\n")
                    f.write("-" * (len(var) + 10) + "\n")
                    f.write(var_data[['survey_year', 'missing_count', 'missing_percentage']].to_string(index=False))
                    f.write("\n\n")
        
        print(f"\nMissing data analysis saved to: {missing_file}")
        print(f"Summary report saved to: {summary_file}")

def check_data_patterns(df, output_dir):
    """Check for obvious patterns in the education data"""
    print("\n" + "="*60)
    print("CHECKING FOR DATA PATTERNS")
    print("="*60)
    
    patterns_found = []
    
    # Check age distribution
    if 'tuoi' in df.columns:
        age_stats = df['tuoi'].describe()
        print("\nAge distribution:")
        print(age_stats)
        patterns_found.append(f"Age range: {age_stats['min']:.0f} to {age_stats['max']:.0f} years")
    
    # Check survey year distribution
    if 'survey_year' in df.columns:
        year_counts = df['survey_year'].value_counts().sort_index()
        print("\nSurvey year distribution:")
        print(year_counts)
        patterns_found.append(f"Survey years: {list(year_counts.index)}")
    
    # Check relationship between age and school attendance
    if 'tuoi' in df.columns and 'is going to school?' in df.columns:
        school_by_age = df.groupby('tuoi')['is going to school?'].value_counts(normalize=True).unstack(fill_value=0)
        if 'Yes' in school_by_age.columns:
            print("\nSchool attendance rates by age (ages 5-20):")
            age_range = school_by_age.loc[5:20, 'Yes'] if len(school_by_age.loc[5:20]) > 0 else school_by_age['Yes']
            print(age_range.round(3))
    
    # Save patterns summary
    patterns_file = os.path.join(output_dir, "data_patterns_summary.txt")
    with open(patterns_file, 'w', encoding='utf-8') as f:
        f.write("DATA PATTERNS SUMMARY\n")
        f.write("="*30 + "\n\n")
        f.write("Key Patterns Found:\n")
        for i, pattern in enumerate(patterns_found, 1):
            f.write(f"{i}. {pattern}\n")
        
        if 'tuoi' in df.columns:
            f.write(f"\nAge Statistics:\n")
            f.write(str(df['tuoi'].describe()))
        
        if 'survey_year' in df.columns:
            f.write(f"\n\nSurvey Year Distribution:\n")
            f.write(str(df['survey_year'].value_counts().sort_index()))
    
    print(f"\nData patterns summary saved to: {patterns_file}")

def main():
    """Main execution function"""
    print("VIETNAM EDUCATION VARIABLES EXPLORATION")
    print("="*50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Run all analyses
    analyze_school_attendance(df, output_dir)
    analyze_current_education_level(df, output_dir)
    analyze_education_level(df, output_dir)
    create_age_grade_crosstab(df, output_dir)
    analyze_missing_data_by_year(df, output_dir)
    check_data_patterns(df, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
