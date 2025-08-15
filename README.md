# Measuring the Impact of Vietnam's 2013 Reform on Education Outcomes: A Difference-in-Differences Approach

This project implements a cross-sectional Difference-in-Differences (DiD) model to analyze the impact of Vietnam's 2013 education reform on educational outcomes.

## Project Overview

This thesis project uses a DiD methodology to measure the causal effect of Vietnam's 2013 education policy reform on various education-related outcomes. The analysis leverages longitudinal household survey data to compare outcomes before and after the reform implementation.

## Data Source

The analysis uses data from the **Vietnam Household Living Standards Survey (VHLSS)** spanning from **2008 to 2020**. The VHLSS is conducted biennially, providing data points for every 2 years within this timeframe.

### Survey Years Available:
- 2008, 2010, 2012, 2014, 2016, 2018, 2020

### Current Dataset
- **Main Dataset**: `all_years_merged_dataset_final_corrected.csv` 
- **Total Records**: 622,575 observations
- **Backup**: `all_years_merged_dataset_final_corrected_backup.csv`

## Dataset Variables

The cleaned and standardized dataset contains the following variables:

### Geographic Identifiers
- `tinh` - Province code
- `huyen` - District code  
- `xa` - Commune code
- `diaban` - Area classification

### Household & Individual Identifiers
- `hoso` - Household ID
- `matv` - Individual member ID
- `survey_year` - Survey year

### Income Variables
- `Tong thu nhap` - Total income
- `Thu binh quan` - Average income
- `thu nhap` - Income
- `thu nhap binh quan` - Average income per capita

### Demographic Variables
- `household head?` - Household head indicator
- `nam sinh` - Birth year
- `tuoi` - Age
- `noi dk ho khau` - Household registration location
- `hhsize` - Household size
- `gender` - Gender
- `chu ho` - Head of household

### Education Variables
- `education level` - Education level
- `Current edu. level` - Current education level
- `Current edu. level_original` - Original current education level
- `education level (vocational)` - Vocational education level
- `is going to school?` - School attendance indicator

### Employment Variables
- `Is employed?` - Employment status
- `Is self-employed? (Agri)` - Self-employment in agriculture
- `Is self-employed? (non-Agri)` - Self-employment in non-agriculture
- `Is really working?` - Actual working status
- `Why not working?` - Reason for not working

## Data Quality and Analysis Progress

### 🔍 **Comprehensive Data Analysis Completed**

#### Education Data Standardization
- **Education Level Mapping**: Standardized 466,788+ Vietnamese education values to English
- **Mapping Applied**: Vietnamese terms (`THCS`, `THPT`, `Tiểu học`) → English terms (`Secondary school`, `High school`, `Primary school`)
- **Change Report**: Generated detailed transformation log with statistics

#### Missing Data Pattern Analysis  
- **Critical Finding**: 77.90% missing rate in "Current edu. level" column (484,998/622,575 records)
- **Age Pattern**: Children (0-18) have best coverage (35.9% missing), adults 26+ have 98-100% missing
- **Temporal Pattern**: Data quality deteriorated over time, 2018 worst year (83.7% missing)
- **Geographic Pattern**: Mekong Delta provinces have highest missing rates (82-84%)
- **Recommendation**: Focus analysis on 0-18 age group for education outcomes

#### Cohort Filtering for DiD Analysis
- **Original Records**: 622,575 → **Filtered Records**: 23,986 (3.85% retention)
- **Treated Cohort**: 9,231 observations (born 2007-2009, experienced 2013 reform during school age)
- **Control Cohort**: 14,755 observations (born 1997-1999, completed primary school before reform)
- **Data Coverage**: 5 survey years (2012, 2014, 2016, 2018, 2020)

## Project Structure

```
DiD-thesis-project/
├── data/
│   ├── all_years_merged_dataset_final_corrected.csv        # Main dataset (622,575 records)
│   ├── all_years_merged_dataset_final_corrected_backup.csv # Backup copy
│   └── data summary/                                       # Analysis reports
│       ├── current edu. level/                            # Missing pattern analysis
│       │   ├── EXECUTIVE_SUMMARY_Missing_Patterns.md      # Key findings summary
│       │   ├── final_missing_analysis_report.txt          # Detailed statistical report
│       │   ├── missing_by_age_final.csv                   # Age group patterns
│       │   ├── missing_by_year_final.csv                  # Temporal patterns
│       │   ├── missing_by_province_final.csv              # Geographic patterns
│       │   └── missing_pattern_visualizations.png         # Charts and heatmaps
│       ├── education_mapping_change_report.txt            # Education standardization log
│       └── column_analysis_results.txt                    # Full dataset analysis
├── results/
│   ├── did_cohorts_filtered.csv                          # Filtered cohorts for DiD
│   └── cohort_filtering_summary.txt                      # Filtering results summary
├── src/
│   ├── DiD/
│   │   ├── main_did_analysis.py                          # Main orchestration script
│   │   ├── data overview/                                # Data analysis scripts
│   │   │   ├── column_analysis.py                       # Dataset column analysis
│   │   │   ├── current edu. level_missing_pattern_analysis.py # Missing data analysis
│   │   │   ├── tinh_unique_values_analysis.py           # Province analysis
│   │   │   └── education level column changes/          # Education standardization
│   │   │       ├── education_mapping_master.py         # Master orchestration
│   │   │       ├── apply_education_mapping.py          # Apply mappings
│   │   │       ├── education_unique_values_analysis.py # Values analysis
│   │   │       └── README.md                           # Process documentation
│   │   └── regression run/                             # DiD analysis pipeline
│   │       ├── cohort_filtering.py                    # Cohort preparation
│   │       └── results/                               # Analysis outputs
│   └── notebook/
│       └── data testing.ipynb                          # Exploratory analysis
└── README.md
```

## Current Analysis Status

### ✅ **Completed Steps**
1. **Data Import & Cleaning** - 622,575 records processed
2. **Education Data Standardization** - 466,788+ values mapped Vietnamese → English  
3. **Missing Data Analysis** - Comprehensive pattern analysis completed
4. **Cohort Definition & Filtering** - Treatment/control groups identified and filtered
5. **Data Quality Assessment** - Critical issues identified and documented

### 🔄 **In Progress**
- DiD regression analysis pipeline development
- Parallel trends assumption testing
- Robustness checks implementation

### 📋 **Next Steps**
1. **Parallel Trends Test** - Verify pre-treatment trends between cohorts
2. **Main DiD Estimation** - Estimate treatment effects  
3. **Robustness Checks** - Alternative specifications and sensitivity analysis
4. **Results Export** - Final tables and visualizations

## Getting Started

### Quick Start
The main analysis orchestrator is in `src/DiD/main_did_analysis.py`. This script coordinates the complete DiD analysis pipeline.

### Key Entry Points
- **Main Analysis**: `python src/DiD/main_did_analysis.py`
- **Cohort Filtering**: `python "src/DiD/regression run/cohort_filtering.py"`
- **Data Analysis**: Scripts in `src/DiD/data overview/`

## Research Questions

This project aims to answer:
- What was the impact of Vietnam's 2013 education reform on educational outcomes?
- How did the reform affect school enrollment and educational attainment across different demographic groups?
- What are the differential effects of the reform across geographic regions and socioeconomic groups?

## Methodology

The project employs a **cross-sectional Difference-in-Differences** approach to:

1. Compare education outcomes before (2008-2012) and after (2014-2020) the 2013 reform
2. Identify treatment and control groups based on reform exposure
3. Estimate the causal impact of the policy reform on educational attainment and school participation

### Treatment Design
- **Treated Cohort**: Students born 2007-2009 (ages 4-6 in 2013, experienced reform during primary school)
- **Control Cohort**: Students born 1997-1999 (ages 14-16 in 2013, completed primary school before reform)
- **Identification Strategy**: Birth year cohorts with differential exposure to 2013 education reform

## Data Period

- **Pre-treatment period**: 2008, 2010, 2012
- **Post-treatment period**: 2014, 2016, 2018, 2020
- **Treatment year**: 2013 (Vietnam education reform implementation)

## Key Findings from Data Analysis

### 🚨 **Critical Data Quality Issues Identified**
1. **High Missing Rates**: 77.9% missing in primary education outcome variable
2. **Age Bias**: Data primarily available for children/students (0-18 age group)
3. **Temporal Decline**: Data quality worsened over time, especially 2018
4. **Geographic Bias**: Southern provinces (Mekong Delta) have poorest coverage

### 📊 **Successful Cohort Construction**
- **23,986 observations** retained after filtering (3.85% of original data)
- **Balanced treatment design**: 9,231 treated vs 14,755 control observations
- **Clear DiD cells**: Proper pre/post treatment distribution for causal identification

### 🎯 **Analysis Recommendations**
1. Focus on children/students for education outcome analysis
2. Consider excluding 2018 survey year due to data quality issues  
3. Account for geographic bias in result interpretation
4. Use robust estimation techniques given data limitations
