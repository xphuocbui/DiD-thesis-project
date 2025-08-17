#!/usr/bin/env python3
"""
2018 Data Quality Investigation Script

This script investigates the severe missing data issues in 2018 survey data,
particularly the 88.34% missing rate in school attendance variable.

Objectives:
1. Compare 2018 data structure with other years
2. Analyze missing data patterns across all variables  
3. Identify potential survey design changes
4. Understand if 2018 data is usable for analysis

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
    output_dir = "data/data summary/education variables/2018_investigation"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data():
    """Load the main dataset"""
    print("Loading main dataset...")
    try:
        df = pd.read_csv("data/all_years_merged_dataset_final_corrected.csv", low_memory=False)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_2018_overview(df):
    """Get basic overview of 2018 data"""
    print("\n" + "="*70)
    print("2018 DATA BASIC OVERVIEW")
    print("="*70)
    
    if 'survey_year' not in df.columns:
        print("Error: survey_year column not found")
        return None
    
    # Filter 2018 data
    df_2018 = df[df['survey_year'] == 2018].copy()
    
    print(f"2018 observations: {len(df_2018):,}")
    print(f"2018 percentage of total: {len(df_2018)/len(df)*100:.2f}%")
    
    # Compare with other years
    year_counts = df['survey_year'].value_counts().sort_index()
    print(f"\nAll survey years:")
    for year, count in year_counts.items():
        pct = count / len(df) * 100
        print(f"  {year}: {count:,} observations ({pct:.2f}%)")
    
    return df_2018

def analyze_missing_patterns_2018(df, df_2018, output_dir):
    """Analyze missing data patterns in 2018 vs other years"""
    print("\n" + "="*70)
    print("MISSING DATA PATTERNS ANALYSIS")
    print("="*70)
    
    # Key variables to analyze
    key_variables = [
        'is going to school?',
        'Current edu. level', 
        'education level',
        'tuoi',  # age
        'gender',
        'Is employed?',
        'thu nhap',  # income
        'hhsize'  # household size
    ]
    
    missing_analysis = []
    
    for var in key_variables:
        if var in df.columns:
            # Overall missing rate
            overall_missing = df[var].isnull().sum() / len(df) * 100
            
            # 2018 missing rate
            missing_2018 = df_2018[var].isnull().sum() / len(df_2018) * 100
            
            # Other years missing rate
            df_other_years = df[df['survey_year'] != 2018]
            missing_other = df_other_years[var].isnull().sum() / len(df_other_years) * 100
            
            missing_analysis.append({
                'variable': var,
                'overall_missing_pct': overall_missing,
                'missing_2018_pct': missing_2018,
                'missing_other_years_pct': missing_other,
                'difference_2018_vs_others': missing_2018 - missing_other
            })
            
            print(f"\n{var}:")
            print(f"  Overall: {overall_missing:.2f}% missing")
            print(f"  2018: {missing_2018:.2f}% missing")
            print(f"  Other years: {missing_other:.2f}% missing")
            print(f"  Difference: {missing_2018 - missing_other:+.2f} percentage points")
    
    # Create DataFrame and save results
    missing_df = pd.DataFrame(missing_analysis)
    missing_file = os.path.join(output_dir, "missing_data_comparison_2018.csv")
    missing_df.to_csv(missing_file, index=False)
    
    print(f"\nMissing data analysis saved to: {missing_file}")
    return missing_df

def analyze_2018_data_structure(df_2018, output_dir):
    """Analyze the structure and completeness of 2018 data"""
    print("\n" + "="*70)
    print("2018 DATA STRUCTURE ANALYSIS")
    print("="*70)
    
    # Data completeness by column
    completeness = []
    
    for col in df_2018.columns:
        total_count = len(df_2018)
        non_null_count = df_2018[col].notna().sum()
        null_count = df_2018[col].isnull().sum()
        completeness_pct = (non_null_count / total_count) * 100
        
        completeness.append({
            'column': col,
            'total_records': total_count,
            'non_null_count': non_null_count,
            'null_count': null_count,
            'completeness_pct': completeness_pct
        })
    
    completeness_df = pd.DataFrame(completeness)
    completeness_df = completeness_df.sort_values('completeness_pct')
    
    print("Data completeness by column (worst to best):")
    print(completeness_df[['column', 'completeness_pct', 'null_count']].head(15))
    
    # Save detailed results
    completeness_file = os.path.join(output_dir, "2018_data_completeness.csv")
    completeness_df.to_csv(completeness_file, index=False)
    
    print(f"\nComplete analysis saved to: {completeness_file}")
    return completeness_df

def analyze_education_variables_2018(df_2018, output_dir):
    """Deep dive into education variables in 2018"""
    print("\n" + "="*70)
    print("2018 EDUCATION VARIABLES DEEP DIVE")
    print("="*70)
    
    education_vars = ['is going to school?', 'Current edu. level', 'education level']
    
    for var in education_vars:
        if var in df_2018.columns:
            print(f"\n--- {var} ---")
            
            # Value counts
            value_counts = df_2018[var].value_counts(dropna=False)
            print(f"Value counts (top 10):")
            print(value_counts.head(10))
            
            # Missing statistics
            total = len(df_2018)
            missing = df_2018[var].isnull().sum()
            valid = total - missing
            
            print(f"\nStatistics:")
            print(f"  Total records: {total:,}")
            print(f"  Valid values: {valid:,} ({valid/total*100:.2f}%)")
            print(f"  Missing values: {missing:,} ({missing/total*100:.2f}%)")
            
            # Check if there are any patterns in the valid data
            if valid > 0:
                print(f"  Unique values: {df_2018[var].nunique()}")
                
                # Age distribution of those with valid data
                if 'tuoi' in df_2018.columns:
                    valid_data = df_2018[df_2018[var].notna()]
                    if len(valid_data) > 0:
                        age_stats = valid_data['tuoi'].describe()
                        print(f"  Age stats for valid records: min={age_stats['min']:.0f}, mean={age_stats['mean']:.1f}, max={age_stats['max']:.0f}")

def compare_sample_characteristics(df, output_dir):
    """Compare demographic characteristics between 2018 and other years"""
    print("\n" + "="*70)
    print("SAMPLE CHARACTERISTICS COMPARISON")
    print("="*70)
    
    df_2018 = df[df['survey_year'] == 2018].copy()
    df_others = df[df['survey_year'] != 2018].copy()
    
    # Key demographic variables
    demo_vars = ['tuoi', 'gender', 'hhsize']
    
    comparison_results = []
    
    for var in demo_vars:
        if var in df.columns:
            print(f"\n--- {var} ---")
            
            if var in ['tuoi', 'hhsize']:  # Numeric variables
                # 2018 statistics
                stats_2018 = df_2018[var].describe()
                stats_others = df_others[var].describe()
                
                print(f"2018 - Mean: {stats_2018['mean']:.2f}, Median: {stats_2018['50%']:.2f}")
                print(f"Others - Mean: {stats_others['mean']:.2f}, Median: {stats_others['50%']:.2f}")
                
                comparison_results.append({
                    'variable': var,
                    'type': 'numeric',
                    '2018_mean': stats_2018['mean'],
                    '2018_median': stats_2018['50%'],
                    'others_mean': stats_others['mean'],
                    'others_median': stats_others['50%'],
                    'mean_difference': stats_2018['mean'] - stats_others['mean']
                })
                
            else:  # Categorical variables
                # 2018 distribution
                dist_2018 = df_2018[var].value_counts(normalize=True, dropna=False)
                dist_others = df_others[var].value_counts(normalize=True, dropna=False)
                
                print(f"2018 distribution:")
                print(dist_2018.head())
                print(f"Other years distribution:")
                print(dist_others.head())
    
    # Save comparison results
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_file = os.path.join(output_dir, "sample_characteristics_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nComparison results saved to: {comparison_file}")

def investigate_survey_design_changes(df, output_dir):
    """Investigate potential survey design changes in 2018"""
    print("\n" + "="*70)
    print("SURVEY DESIGN CHANGES INVESTIGATION")
    print("="*70)
    
    # Check variable availability by year
    print("Variable availability by survey year:")
    
    key_vars = ['is going to school?', 'Current edu. level', 'education level', 
               'Is employed?', 'thu nhap']
    
    availability_matrix = []
    
    for year in sorted(df['survey_year'].unique()):
        year_data = df[df['survey_year'] == year]
        year_availability = {'survey_year': year}
        
        for var in key_vars:
            if var in df.columns:
                # Check if variable has any non-null values
                has_data = year_data[var].notna().any()
                non_null_count = year_data[var].notna().sum()
                total_count = len(year_data)
                availability_pct = (non_null_count / total_count) * 100
                
                year_availability[var] = availability_pct
            else:
                year_availability[var] = 0
        
        availability_matrix.append(year_availability)
    
    availability_df = pd.DataFrame(availability_matrix)
    
    print("\nVariable availability (% non-missing) by year:")
    print(availability_df.round(2))
    
    # Save results
    availability_file = os.path.join(output_dir, "variable_availability_by_year.csv")
    availability_df.to_csv(availability_file, index=False)
    
    print(f"\nVariable availability analysis saved to: {availability_file}")
    
    # Look for patterns that might indicate survey design changes
    print("\nPotential survey design change indicators:")
    
    for var in key_vars:
        if var in availability_df.columns:
            availability_2018 = availability_df[availability_df['survey_year'] == 2018][var].iloc[0]
            avg_others = availability_df[availability_df['survey_year'] != 2018][var].mean()
            
            difference = availability_2018 - avg_others
            if abs(difference) > 20:  # More than 20 percentage point difference
                print(f"  {var}: 2018 has {difference:+.1f}% difference vs other years")

def generate_comprehensive_report(df, df_2018, missing_df, completeness_df, output_dir):
    """Generate comprehensive 2018 investigation report"""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE 2018 INVESTIGATION REPORT")
    print("="*70)
    
    report_file = os.path.join(output_dir, "2018_data_quality_investigation_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("2018 DATA QUALITY INVESTIGATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"2018 survey data shows severe quality issues:\n")
        f.write(f"• 2018 sample size: {len(df_2018):,} observations ({len(df_2018)/len(df)*100:.1f}% of total)\n")
        f.write(f"• School attendance missing: {df_2018['is going to school?'].isnull().sum()/len(df_2018)*100:.1f}%\n")
        
        # Most problematic variables
        worst_vars = completeness_df.nsmallest(5, 'completeness_pct')
        f.write(f"\nMost problematic variables in 2018:\n")
        for _, row in worst_vars.iterrows():
            f.write(f"• {row['column']}: {row['completeness_pct']:.1f}% complete\n")
        
        f.write(f"\nKEY FINDINGS\n")
        f.write("-" * 15 + "\n")
        
        # School attendance specific issue
        school_missing_2018 = missing_df[missing_df['variable'] == 'is going to school?']
        if not school_missing_2018.empty:
            row = school_missing_2018.iloc[0]
            f.write(f"1. School Attendance Variable:\n")
            f.write(f"   - 2018: {row['missing_2018_pct']:.1f}% missing\n")
            f.write(f"   - Other years: {row['missing_other_years_pct']:.1f}% missing\n")
            f.write(f"   - Difference: +{row['difference_2018_vs_others']:.1f} percentage points\n\n")
        
        # Education variables pattern
        edu_vars = ['Current edu. level', 'education level']
        f.write(f"2. Education Variables Pattern:\n")
        for var in edu_vars:
            var_data = missing_df[missing_df['variable'] == var]
            if not var_data.empty:
                row = var_data.iloc[0]
                f.write(f"   {var}: 2018 = {row['missing_2018_pct']:.1f}% missing, Others = {row['missing_other_years_pct']:.1f}% missing\n")
        
        f.write(f"\nRECOMMendations\n")
        f.write("-" * 18 + "\n")
        f.write("1. EXCLUDE 2018 from school attendance analysis\n")
        f.write("2. Investigate if 2018 used different survey instruments\n")
        f.write("3. Check if 2018 had different target populations\n")
        f.write("4. Consider 2018 as sensitivity analysis exclusion\n")
        f.write("5. Focus DiD analysis on 2008-2016, 2020 data\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 18 + "\n")
        f.write("• missing_data_comparison_2018.csv\n")
        f.write("• 2018_data_completeness.csv\n") 
        f.write("• sample_characteristics_comparison.csv\n")
        f.write("• variable_availability_by_year.csv\n")
        f.write("• 2018_data_quality_investigation_report.txt\n")
    
    print(f"Comprehensive investigation report saved to: {report_file}")

def main():
    """Main execution function"""
    print("2018 DATA QUALITY INVESTIGATION")
    print("="*50)
    print(f"Investigation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic 2018 overview
    df_2018 = basic_2018_overview(df)
    if df_2018 is None:
        return
    
    # Run all investigations
    missing_df = analyze_missing_patterns_2018(df, df_2018, output_dir)
    completeness_df = analyze_2018_data_structure(df_2018, output_dir)
    analyze_education_variables_2018(df_2018, output_dir)
    compare_sample_characteristics(df, output_dir)
    investigate_survey_design_changes(df, output_dir)
    
    # Generate comprehensive report
    generate_comprehensive_report(df, df_2018, missing_df, completeness_df, output_dir)
    
    print("\n" + "="*70)
    print("2018 INVESTIGATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print(f"Investigation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
