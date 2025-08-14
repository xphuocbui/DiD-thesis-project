import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def final_missing_analysis():
    """
    Final corrected analysis of missing patterns in 'Current edu. level' column
    """
    # Load the dataset
    data_path = Path("data/all_years_merged_dataset_final_corrected.csv")
    print("Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    
    target_col = 'Current edu. level'
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: '{target_col}'")
    
    # Basic statistics
    total_records = len(df)
    missing_count = df[target_col].isna().sum()
    non_missing_count = df[target_col].notna().sum()
    missing_percentage = (missing_count / total_records) * 100
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total records: {total_records:,}")
    print(f"Missing values: {missing_count:,}")
    print(f"Non-missing values: {non_missing_count:,}")
    print(f"Missing percentage: {missing_percentage:.2f}%")
    
    # Create age groups for analysis
    df_clean = df.dropna(subset=['tuoi']).copy()
    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
    age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df_clean['age_group'] = pd.cut(df_clean['tuoi'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # 1. ANALYSIS BY AGE GROUPS
    print(f"\n{'='*60}")
    print("1. MISSING PATTERN BY AGE GROUPS")
    print(f"{'='*60}")
    
    age_stats = []
    for age_group in age_labels:
        group_data = df_clean[df_clean['age_group'] == age_group]
        total = len(group_data)
        missing = group_data[target_col].isna().sum()
        non_missing = group_data[target_col].notna().sum()
        missing_pct = (missing / total * 100) if total > 0 else 0
        
        age_stats.append({
            'Age_Group': age_group,
            'Total': total,
            'Missing': missing,
            'Non_Missing': non_missing,
            'Missing_Percentage': round(missing_pct, 2)
        })
    
    age_df = pd.DataFrame(age_stats)
    age_df.set_index('Age_Group', inplace=True)
    print(age_df)
    
    # 2. ANALYSIS BY SURVEY YEAR
    print(f"\n{'='*60}")
    print("2. MISSING PATTERN BY SURVEY YEAR")
    print(f"{'='*60}")
    
    year_stats = []
    for year in sorted(df['survey_year'].unique()):
        year_data = df[df['survey_year'] == year]
        total = len(year_data)
        missing = year_data[target_col].isna().sum()
        non_missing = year_data[target_col].notna().sum()
        missing_pct = (missing / total * 100) if total > 0 else 0
        
        year_stats.append({
            'Survey_Year': year,
            'Total': total,
            'Missing': missing,
            'Non_Missing': non_missing,
            'Missing_Percentage': round(missing_pct, 2)
        })
    
    year_df = pd.DataFrame(year_stats)
    year_df.set_index('Survey_Year', inplace=True)
    print(year_df)
    
    # 3. ANALYSIS BY PROVINCE
    print(f"\n{'='*60}")
    print("3. MISSING PATTERN BY PROVINCE")
    print(f"{'='*60}")
    
    province_stats = []
    for province in df['tinh'].unique():
        if pd.isna(province):
            continue
        prov_data = df[df['tinh'] == province]
        total = len(prov_data)
        missing = prov_data[target_col].isna().sum()
        non_missing = prov_data[target_col].notna().sum()
        missing_pct = (missing / total * 100) if total > 0 else 0
        
        province_stats.append({
            'Province': str(province),
            'Total': total,
            'Missing': missing,
            'Non_Missing': non_missing,
            'Missing_Percentage': round(missing_pct, 2)
        })
    
    province_df = pd.DataFrame(province_stats)
    province_df = province_df.sort_values('Missing_Percentage', ascending=False)
    
    print("Top 20 provinces with highest missing rates:")
    print(province_df.head(20))
    
    print(f"\nTop 10 provinces with lowest missing rates:")
    print(province_df.tail(10))
    
    # 4. CROSS-TABULATION
    print(f"\n{'='*60}")
    print("4. CROSS-TABULATION: AGE GROUP vs SURVEY YEAR")
    print(f"{'='*60}")
    
    # Create cross-tabulation of missing percentages
    crosstab_data = []
    for age_group in age_labels:
        row = {'Age_Group': age_group}
        for year in sorted(df_clean['survey_year'].unique()):
            subset = df_clean[(df_clean['age_group'] == age_group) & (df_clean['survey_year'] == year)]
            if len(subset) > 0:
                missing_pct = (subset[target_col].isna().sum() / len(subset)) * 100
                row[f'Year_{year}'] = round(missing_pct, 1)
            else:
                row[f'Year_{year}'] = 0.0
        crosstab_data.append(row)
    
    crosstab_df = pd.DataFrame(crosstab_data)
    crosstab_df.set_index('Age_Group', inplace=True)
    print("Missing percentages by age group and survey year:")
    print(crosstab_df)
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("5. CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Missing Pattern Analysis for Current Education Level', fontsize=16, fontweight='bold')
    
    # Plot 1: Missing percentage by age group
    ax1 = axes[0, 0]
    age_df['Missing_Percentage'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Missing Percentage by Age Group')
    ax1.set_ylabel('Missing Percentage (%)')
    ax1.set_xlabel('Age Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Missing percentage by survey year
    ax2 = axes[0, 1]
    year_df['Missing_Percentage'].plot(kind='line', ax=ax2, marker='o', color='red', linewidth=2)
    ax2.set_title('Missing Percentage by Survey Year')
    ax2.set_ylabel('Missing Percentage (%)')
    ax2.set_xlabel('Survey Year')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top 15 provinces with highest missing rates
    ax3 = axes[1, 0]
    top_15_provinces = province_df.head(15)
    top_15_provinces['Missing_Percentage'].plot(kind='barh', ax=ax3, color='lightcoral')
    ax3.set_title('Top 15 Provinces with Highest Missing Rates')
    ax3.set_xlabel('Missing Percentage (%)')
    ax3.set_ylabel('Province')
    
    # Plot 4: Heatmap of age group vs survey year
    ax4 = axes[1, 1]
    # Prepare data for heatmap
    heatmap_data = crosstab_df.values
    years = [col.replace('Year_', '') for col in crosstab_df.columns]
    
    im = ax4.imshow(heatmap_data, cmap='Reds', aspect='auto')
    ax4.set_xticks(range(len(years)))
    ax4.set_xticklabels(years)
    ax4.set_yticks(range(len(age_labels)))
    ax4.set_yticklabels(age_labels)
    ax4.set_title('Missing Percentage Heatmap\\n(Age Group vs Survey Year)')
    ax4.set_xlabel('Survey Year')
    ax4.set_ylabel('Age Group')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Missing Percentage (%)')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("data/data summary")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "missing_pattern_visualizations.png", dpi=300, bbox_inches='tight')
    print("Visualization saved as 'missing_pattern_visualizations.png'")
    
    # Save detailed results
    print(f"\n{'='*60}")
    print("6. SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save comprehensive text report
    with open(output_dir / "final_missing_analysis_report.txt", "w", encoding='utf-8') as f:
        f.write("COMPREHENSIVE MISSING PATTERN ANALYSIS\\n")
        f.write("Current Education Level Column\\n")
        f.write("="*80 + "\\n\\n")
        
        f.write("OVERALL STATISTICS\\n")
        f.write("-"*50 + "\\n")
        f.write(f"Total records: {total_records:,}\\n")
        f.write(f"Missing values: {missing_count:,}\\n")
        f.write(f"Non-missing values: {non_missing_count:,}\\n")
        f.write(f"Missing percentage: {missing_percentage:.2f}%\\n\\n")
        
        f.write("MISSING PATTERN BY AGE GROUPS\\n")
        f.write("-"*50 + "\\n")
        f.write(age_df.to_string())
        f.write("\\n\\n")
        
        f.write("MISSING PATTERN BY SURVEY YEAR\\n")
        f.write("-"*50 + "\\n")
        f.write(year_df.to_string())
        f.write("\\n\\n")
        
        f.write("MISSING PATTERN BY PROVINCE (Top 30)\\n")
        f.write("-"*50 + "\\n")
        f.write(province_df.head(30).to_string())
        f.write("\\n\\n")
        
        f.write("CROSS-TABULATION: AGE GROUP vs SURVEY YEAR\\n")
        f.write("-"*50 + "\\n")
        f.write(crosstab_df.to_string())
        f.write("\\n\\n")
    
    # Save CSV files
    age_df.to_csv(output_dir / "missing_by_age_final.csv")
    year_df.to_csv(output_dir / "missing_by_year_final.csv")
    province_df.to_csv(output_dir / "missing_by_province_final.csv")
    crosstab_df.to_csv(output_dir / "missing_crosstab_final.csv")
    
    print("Files saved:")
    print("- final_missing_analysis_report.txt")
    print("- missing_by_age_final.csv")
    print("- missing_by_year_final.csv")
    print("- missing_by_province_final.csv")
    print("- missing_crosstab_final.csv")
    print("- missing_pattern_visualizations.png")
    
    # Key insights
    print(f"\\n{'='*60}")
    print("7. KEY INSIGHTS")
    print(f"{'='*60}")
    
    print(f"1. Overall missing rate: {missing_percentage:.1f}% - Very high!")
    print(f"2. Age pattern: Older age groups have higher missing rates")
    print(f"3. Worst age group: {age_df.loc[age_df['Missing_Percentage'].idxmax()].name} ({age_df['Missing_Percentage'].max():.1f}%)")
    print(f"4. Best age group: {age_df.loc[age_df['Missing_Percentage'].idxmin()].name} ({age_df['Missing_Percentage'].min():.1f}%)")
    print(f"5. Year trend: {year_df['Missing_Percentage'].min():.1f}% to {year_df['Missing_Percentage'].max():.1f}%")
    print(f"6. Province variation: {province_df['Missing_Percentage'].min():.1f}% to {province_df['Missing_Percentage'].max():.1f}%")
    
    return age_df, year_df, province_df, crosstab_df

if __name__ == "__main__":
    age_results, year_results, province_results, crosstab_results = final_missing_analysis()
