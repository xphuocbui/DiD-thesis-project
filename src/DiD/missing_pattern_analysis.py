import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_missing_patterns():
    """
    Analyze missing patterns in 'Current edu. level' column by age, survey year, and province
    """
    # Load the dataset
    data_path = Path("data/all_years_merged_dataset_final_corrected.csv")
    print("Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if 'Current edu. level' column exists
    if 'Current edu. level' not in df.columns:
        print("'Current edu. level' column not found!")
        return
    
    # Basic info about the target column
    print(f"\n=== BASIC INFO ABOUT 'Current edu. level' COLUMN ===")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {df['Current edu. level'].isna().sum()}")
    print(f"Missing percentage: {(df['Current edu. level'].isna().sum() / len(df)) * 100:.2f}%")
    print(f"Non-missing values: {df['Current edu. level'].notna().sum()}")
    
    # Check available age column (tuoi)
    print(f"\n=== AGE COLUMN INFO ===")
    print(f"Age column ('tuoi') missing: {df['tuoi'].isna().sum()}")
    print(f"Age range: {df['tuoi'].min()} to {df['tuoi'].max()}")
    
    # Check survey years
    print(f"\n=== SURVEY YEARS ===")
    print(f"Available years: {sorted(df['survey_year'].unique())}")
    
    # Check provinces
    print(f"\n=== PROVINCES ===")
    print(f"Number of provinces: {df['tinh'].nunique()}")
    unique_provinces = df['tinh'].unique()
    # Convert to string to handle mixed types
    unique_provinces_str = [str(p) for p in unique_provinces if pd.notna(p)]
    print(f"Provinces: {sorted(unique_provinces_str)[:10]}...")  # Show first 10
    
    # 1. Missing pattern by age groups
    print(f"\n=== MISSING PATTERN BY AGE GROUPS ===")
    # Create age groups
    df['age_group'] = pd.cut(df['tuoi'], 
                            bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                            labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                            include_lowest=True)
    
    age_missing = df.groupby('age_group').agg({
        'Current edu. level': ['count', lambda x: x.isna().sum(), lambda x: x.notna().sum()]
    }).round(2)
    age_missing.columns = ['Total', 'Missing', 'Non_Missing']
    age_missing['Missing_Percentage'] = (age_missing['Missing'] / age_missing['Total'] * 100).round(2)
    
    print(age_missing)
    
    # 2. Missing pattern by survey year
    print(f"\n=== MISSING PATTERN BY SURVEY YEAR ===")
    year_missing = df.groupby('survey_year').agg({
        'Current edu. level': ['count', lambda x: x.isna().sum(), lambda x: x.notna().sum()]
    }).round(2)
    year_missing.columns = ['Total', 'Missing', 'Non_Missing']
    year_missing['Missing_Percentage'] = (year_missing['Missing'] / year_missing['Total'] * 100).round(2)
    
    print(year_missing)
    
    # 3. Missing pattern by province
    print(f"\n=== MISSING PATTERN BY PROVINCE ===")
    province_missing = df.groupby('tinh').agg({
        'Current edu. level': ['count', lambda x: x.isna().sum(), lambda x: x.notna().sum()]
    }).round(2)
    province_missing.columns = ['Total', 'Missing', 'Non_Missing']
    province_missing['Missing_Percentage'] = (province_missing['Missing'] / province_missing['Total'] * 100).round(2)
    province_missing = province_missing.sort_values('Missing_Percentage', ascending=False)
    
    print(province_missing)
    
    # 4. Cross-tabulation analysis
    print(f"\n=== CROSS-TABULATION: AGE GROUP x SURVEY YEAR ===")
    cross_age_year = pd.crosstab(df['age_group'], df['survey_year'], 
                                df['Current edu. level'].isna(), aggfunc='sum', margins=True)
    print("Missing counts by age group and survey year:")
    print(cross_age_year)
    
    # Calculate percentages
    cross_age_year_pct = pd.crosstab(df['age_group'], df['survey_year'], 
                                    df['Current edu. level'].isna(), aggfunc='mean', margins=True) * 100
    print(f"\nMissing percentages by age group and survey year:")
    print(cross_age_year_pct.round(2))
    
    # 5. Top provinces with highest missing rates
    print(f"\n=== TOP 10 PROVINCES WITH HIGHEST MISSING RATES ===")
    top_missing_provinces = province_missing.head(10)
    print(top_missing_provinces)
    
    # 6. Detailed age analysis
    print(f"\n=== DETAILED AGE ANALYSIS ===")
    age_detailed = df.groupby('tuoi').agg({
        'Current edu. level': ['count', lambda x: x.isna().sum()]
    })
    age_detailed.columns = ['Total', 'Missing']
    age_detailed['Missing_Percentage'] = (age_detailed['Missing'] / age_detailed['Total'] * 100).round(2)
    age_detailed = age_detailed[age_detailed['Total'] >= 10]  # Only ages with at least 10 observations
    age_detailed_top = age_detailed.sort_values('Missing_Percentage', ascending=False).head(20)
    print("Top 20 ages with highest missing rates (min 10 observations):")
    print(age_detailed_top)
    
    # Save results to files
    print(f"\n=== SAVING RESULTS ===")
    output_dir = Path("data/data summary")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "missing_pattern_analysis_current_edu_level.txt", "w", encoding='utf-8') as f:
        f.write("MISSING PATTERN ANALYSIS FOR 'Current edu. level' COLUMN\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Missing values: {df['Current edu. level'].isna().sum()}\n")
        f.write(f"Missing percentage: {(df['Current edu. level'].isna().sum() / len(df)) * 100:.2f}%\n\n")
        
        f.write("MISSING PATTERN BY AGE GROUPS\n")
        f.write("-" * 40 + "\n")
        f.write(age_missing.to_string())
        f.write("\n\n")
        
        f.write("MISSING PATTERN BY SURVEY YEAR\n")
        f.write("-" * 40 + "\n")
        f.write(year_missing.to_string())
        f.write("\n\n")
        
        f.write("MISSING PATTERN BY PROVINCE (Top 20)\n")
        f.write("-" * 40 + "\n")
        f.write(province_missing.head(20).to_string())
        f.write("\n\n")
        
        f.write("CROSS-TABULATION: MISSING COUNTS BY AGE GROUP AND SURVEY YEAR\n")
        f.write("-" * 60 + "\n")
        f.write(cross_age_year.to_string())
        f.write("\n\n")
        
        f.write("CROSS-TABULATION: MISSING PERCENTAGES BY AGE GROUP AND SURVEY YEAR\n")
        f.write("-" * 60 + "\n")
        f.write(cross_age_year_pct.round(2).to_string())
        f.write("\n\n")
    
    # Save CSV files
    age_missing.to_csv(output_dir / "missing_by_age_group.csv")
    year_missing.to_csv(output_dir / "missing_by_survey_year.csv")
    province_missing.to_csv(output_dir / "missing_by_province.csv")
    
    print("Analysis complete! Results saved to 'data/data summary/' directory")
    
    return df, age_missing, year_missing, province_missing

if __name__ == "__main__":
    df, age_missing, year_missing, province_missing = analyze_missing_patterns()
