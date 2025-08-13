# Measuring the Impact of Vietnam's 2013 Reform on Education Outcomes: A Difference-in-Differences Approach

This project implements a cross-sectional Difference-in-Differences (DiD) model to analyze the impact of Vietnam's 2013 education reform on educational outcomes.

## Project Overview

This thesis project uses a DiD methodology to measure the causal effect of Vietnam's 2013 education policy reform on various education-related outcomes. The analysis leverages longitudinal household survey data to compare outcomes before and after the reform implementation.

## Data Source

The analysis uses data from the **Vietnam Household Living Standards Survey (VHLSS)** spanning from **2008 to 2020**. The VHLSS is conducted biennially, providing data points for every 2 years within this timeframe.

### Survey Years Available:
- 2008, 2010, 2012, 2014, 2016, 2018, 2020

## Dataset Variables

The cleaned dataset (`all_years_merged_dataset_mapped_final.csv`) contains the following variables:

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

## Methodology

The project employs a **cross-sectional Difference-in-Differences** approach to:

1. Compare education outcomes before (2008-2012) and after (2014-2020) the 2013 reform
2. Identify treatment and control groups based on reform exposure
3. Estimate the causal impact of the policy reform on educational attainment and school participation

## Project Structure

```
DiD-thesis-project/
├── data/
│   └── all_years_merged_dataset_mapped_final.csv
├── src/
│   └── DiD/
│       └── main_did_analysis.py
└── README.md
```

## Getting Started

The main analysis is contained in `src/DiD/main_did_analysis.py`. This script implements the DiD model and generates results for the thesis analysis.

## Research Questions

This project aims to answer:
- What was the impact of Vietnam's 2013 education reform on educational outcomes?
- How did the reform affect school enrollment and educational attainment across different demographic groups?
- What are the differential effects of the reform across geographic regions and socioeconomic groups?

## Data Period

- **Pre-treatment period**: 2008, 2010, 2012
- **Post-treatment period**: 2014, 2016, 2018, 2020
- **Treatment year**: 2013 (Vietnam education reform implementation)
