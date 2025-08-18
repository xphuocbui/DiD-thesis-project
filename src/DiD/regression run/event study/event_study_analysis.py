#!/usr/bin/env python3
"""
Event Study Analysis for Vietnam Education Reform
===============================================

This script implements an event study analysis to examine dynamic treatment effects 
of the 2013 education reform in Vietnam using PyFixest.

Event Study Specification:
currently_enrolled ~ i(treatment_group, i(survey_year, ref=2012)) + male | birth_year + survey_year + tinh + age_group

Dynamic Effects:
- 2008 vs 2012: Pre-treatment effect (should ≈ 0 if parallel trends hold)
- 2010 vs 2012: Pre-treatment effect (should ≈ 0 if parallel trends hold)  
- 2014 vs 2012: Immediate reform effect (+2 years)
- 2016 vs 2012: Medium-term effect (+4 years)
- 2020 vs 2012: Long-term effect (+8 years)

Sample:
- Treated cohorts: Born 2002-2007 (ages 6-11 in 2013)
- Control cohorts: Born 1995-1998 (ages 15-18 in 2013)
- Analysis years: 2008, 2010, 2012, 2014, 2016, 2020
- Outcome: General school enrollment (currently_enrolled)

Author: DiD Analysis Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyfixest as pf
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, Tuple, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EventStudyAnalysis:
    """
    Class to implement event study analysis for Vietnam education reform.
    Examines dynamic treatment effects over time.
    """
    
    def __init__(self, data_path: str = "data/master_dataset_exclude_2018.csv", 
                 output_path: str = "results/event study/"):
        """
        Initialize the event study analysis.
        
        Args:
            data_path: Path to the master dataset with currently_enrolled column
            output_path: Path to save results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define analysis parameters (same as basic DiD)
        self.pre_reform_years = [2008, 2010, 2012]  # Pre-reform years
        self.post_reform_years = [2014, 2016, 2020]  # Post-reform analysis years
        self.all_analysis_years = self.pre_reform_years + self.post_reform_years
        self.treated_birth_years = [2002, 2003, 2004, 2005, 2006, 2007]  # Ages 6-11 in 2013
        self.control_birth_years = [1995, 1996, 1997, 1998]  # Ages 15-18 in 2013  
        self.analysis_birth_years = list(range(1995, 2008))  # Birth years for relevant age groups
        self.reform_year = 2013
        self.reference_year = 2012  # Reference year for event study
        
        # Results storage
        self.data = None
        self.analysis_sample = None
        self.event_model = None
        self.results = {}
        self.dynamic_effects = {}
        
        logger.info("Event Study Analysis initialized")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Reference year: {self.reference_year}")
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data for event study analysis.
        
        Returns:
            Prepared DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Verify required columns exist
            required_cols = ['survey_year', 'nam sinh', 'tuoi', 'currently_enrolled', 'tinh', 'gender']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info("✓ All required columns present")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_analysis_sample(self) -> pd.DataFrame:
        """
        Create the analysis sample for event study (same filtering as basic DiD).
        
        Returns:
            Filtered DataFrame with analysis sample
        """
        logger.info("Creating analysis sample for event study analysis")
        
        # Start with the full dataset
        sample = self.data.copy()
        logger.info(f"Starting with {len(sample):,} observations")
        
        # Filter to all analysis years
        sample = sample[sample['survey_year'].isin(self.all_analysis_years)]
        logger.info(f"After filtering to analysis years {self.all_analysis_years}: {len(sample):,} observations")
        
        # Check year distribution
        year_counts = sample['survey_year'].value_counts().sort_index()
        logger.info("Observations by year after year filtering:")
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count:,}")
        
        # Verify reference year is present
        if self.reference_year not in year_counts.index:
            raise ValueError(f"Reference year {self.reference_year} not found in data")
        logger.info(f"✓ Reference year {self.reference_year} present with {year_counts[self.reference_year]:,} observations")
        
        # Filter to relevant birth years (ages 6-18 in 2013)
        sample = sample[sample['nam sinh'].isin(self.analysis_birth_years)]
        logger.info(f"After filtering to birth years {min(self.analysis_birth_years)}-{max(self.analysis_birth_years)}: {len(sample):,} observations")
        
        # Remove observations with missing enrollment data
        sample = sample.dropna(subset=['currently_enrolled'])
        logger.info(f"After removing missing enrollment: {len(sample):,} observations")
        
        # Remove observations with missing control variables
        sample = sample.dropna(subset=['tuoi', 'gender', 'tinh', 'nam sinh'])
        logger.info(f"After removing missing controls: {len(sample):,} observations")
        
        self.analysis_sample = sample
        logger.info("✓ Analysis sample created successfully")
        
        return self.analysis_sample
    
    def create_treatment_variables(self) -> pd.DataFrame:
        """
        Create treatment variables for event study analysis.
        
        Returns:
            DataFrame with treatment variables added
        """
        logger.info("Creating treatment variables for event study")
        
        sample = self.analysis_sample.copy()
        
        # Create treatment group indicator (same as basic DiD)
        sample['treatment_group'] = sample['nam sinh'].apply(
            lambda x: 1 if x in self.treated_birth_years else 
                     (0 if x in self.control_birth_years else np.nan)
        )
        
        # Remove observations not in treatment or control groups
        sample = sample.dropna(subset=['treatment_group'])
        sample['treatment_group'] = sample['treatment_group'].astype(int)
        
        # Create demographic controls
        sample['age'] = sample['tuoi']
        sample['male'] = (sample['gender'] == 'Male').astype(int)
        sample['birth_year'] = sample['nam sinh']
        
        # Create age groups for more flexible age control
        sample['age_group'] = pd.cut(sample['age'], 
                                   bins=[0, 6, 9, 12, 15, 18, 25], 
                                   labels=['0-5', '6-8', '9-11', '12-14', '15-17', '18+'])
        
        # Convert age_group to string to avoid issues with categorical in PyFixest
        sample['age_group'] = sample['age_group'].astype(str)
        
        logger.info("Age group distribution:")
        age_group_dist = sample['age_group'].value_counts().sort_index()
        for age_group, count in age_group_dist.items():
            logger.info(f"  {age_group}: {count:,} observations")
        
        # Remove observations with missing age groups (from pd.cut with NaN ages)
        initial_count = len(sample)
        sample = sample.dropna(subset=['age_group'])
        final_count = len(sample)
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count:,} observations with missing age groups")
        
        # Ensure survey_year is categorical for interactions
        sample['survey_year'] = sample['survey_year'].astype(int)
        
        logger.info(f"Treatment group distribution:")
        logger.info(f"  Control (0): {sum(sample['treatment_group'] == 0):,} observations")
        logger.info(f"  Treated (1): {sum(sample['treatment_group'] == 1):,} observations")
        
        # Final year breakdown
        final_year_counts = sample['survey_year'].value_counts().sort_index()
        logger.info("Final sample by year:")
        for year, count in final_year_counts.items():
            logger.info(f"  {year}: {count:,}")
        
        # Check DiD cells
        logger.info("Event study cells (treatment_group × survey_year):")
        cell_counts = sample.groupby(['treatment_group', 'survey_year']).size().unstack(fill_value=0)
        logger.info(f"\n{cell_counts}")
        
        self.analysis_sample = sample
        return self.analysis_sample
    
    def log_sample_summary(self):
        """Log summary statistics of the analysis sample"""
        logger.info("="*70)
        logger.info("EVENT STUDY ANALYSIS SAMPLE SUMMARY")
        logger.info("="*70)
        
        df = self.analysis_sample
        
        # Overall sample
        logger.info(f"Total observations: {len(df):,}")
        logger.info(f"Reference year (baseline): {self.reference_year}")
        
        # By survey year
        logger.info("\nBy Survey Year:")
        for year in sorted(df['survey_year'].unique()):
            year_data = df[df['survey_year'] == year]
            n = len(year_data)
            enroll_rate = year_data['currently_enrolled'].mean()
            treat_pct = year_data['treatment_group'].mean()
            year_type = "Reference" if year == self.reference_year else ("Pre-reform" if year < self.reform_year else "Post-reform")
            logger.info(f"  {year} ({year_type}): {n:,} obs, {enroll_rate:.1%} enrolled, {treat_pct:.1%} treated")
        
        # By treatment group
        logger.info("\nBy Treatment Group:")
        for treat in [0, 1]:
            treat_data = df[df['treatment_group'] == treat]
            n = len(treat_data)
            enroll_rate = treat_data['currently_enrolled'].mean()
            group_name = "Treated" if treat == 1 else "Control"
            logger.info(f"  {group_name}: {n:,} obs, {enroll_rate:.1%} enrolled")
        
        # By treatment group and year (event study cells)
        logger.info("\nEvent Study Cells (Treatment × Year):")
        for treat in [0, 1]:
            group_name = "Treated" if treat == 1 else "Control"
            logger.info(f"  {group_name}:")
            treat_data = df[df['treatment_group'] == treat]
            for year in sorted(treat_data['survey_year'].unique()):
                year_data = treat_data[treat_data['survey_year'] == year]
                n = len(year_data)
                enroll_rate = year_data['currently_enrolled'].mean() if n > 0 else 0
                relative_year = year - self.reference_year
                logger.info(f"    {year} (t{relative_year:+d}): {n:,} obs, {enroll_rate:.1%} enrolled")
        
        # Age group distribution by treatment
        logger.info("\nAge Group Distribution by Treatment:")
        for treat in [0, 1]:
            group_name = "Treated" if treat == 1 else "Control"
            logger.info(f"  {group_name}:")
            treat_data = df[df['treatment_group'] == treat]
            age_dist = treat_data['age_group'].value_counts().sort_index()
            for age_group, count in age_dist.items():
                pct = count / len(treat_data) * 100
                logger.info(f"    {age_group}: {count:,} ({pct:.1f}%)")
    
    def create_interaction_variables(self) -> pd.DataFrame:
        """
        Create explicit interaction variables for event study since PyFixest i() function is not available.
        
        Returns:
            DataFrame with interaction variables added
        """
        logger.info("Creating interaction variables for event study")
        
        df = self.analysis_sample.copy()
        
        # Create interaction terms for each year (excluding reference year)
        # treatment_group_year_YYYY = treatment_group * (survey_year == YYYY)
        interaction_vars = []
        
        for year in self.all_analysis_years:
            if year != self.reference_year:
                var_name = f'treatment_group_year_{year}'
                df[var_name] = df['treatment_group'] * (df['survey_year'] == year).astype(int)
                interaction_vars.append(var_name)
                logger.info(f"Created interaction: {var_name} (sum: {df[var_name].sum()})")
        
        self.interaction_variables = interaction_vars
        self.analysis_sample = df
        
        logger.info(f"Created {len(interaction_vars)} interaction variables")
        return df
    
    def run_event_study_regression(self) -> Dict[str, Any]:
        """
        Run the event study regression with year × treatment interactions.
        
        Returns:
            Dictionary with regression results
        """
        logger.info("Running event study regression")
        
        df = self.analysis_sample
        
        # Create the formula with explicit interaction terms and age group fixed effects
        interaction_terms = " + ".join(self.interaction_variables)
        formula = f'currently_enrolled ~ treatment_group + {interaction_terms} + male | birth_year + survey_year + tinh + age_group'
        
        logger.info(f"Event study formula: {formula}")
        logger.info(f"Sample size: {len(df):,} observations")
        logger.info(f"Reference year: {self.reference_year}")
        logger.info(f"Interaction variables: {self.interaction_variables}")
        
        try:
            # Run regression with clustered standard errors by birth year
            self.event_model = pf.feols(
                formula,
                data=df,
                vcov={'CRV1': 'birth_year'}  # Cluster by birth year as in main analysis
            )
            
            logger.info("✓ Event study regression completed successfully")
            
            # Extract key results
            results = {
                'model': self.event_model,
                'formula': formula,
                'n_observations': len(df),
                'reference_year': self.reference_year,
                'summary': str(self.event_model.summary()),
                'coefficients': self.event_model.coef().to_dict(),
                'pvalues': self.event_model.pvalue().to_dict(),
                'se': self.event_model.se().to_dict(),
                'sample_years': sorted(df['survey_year'].unique().tolist()),
                'interaction_variables': self.interaction_variables
            }
            
            logger.info("Event study coefficients extracted successfully")
            
            self.results['regression'] = results
            return results
            
        except Exception as e:
            logger.error(f"Error running event study regression: {str(e)}")
            raise
    
    def extract_dynamic_effects(self) -> Dict[str, Any]:
        """
        Extract dynamic treatment effects from event study regression.
        
        Returns:
            Dictionary with dynamic effects by year
        """
        logger.info("Extracting dynamic treatment effects")
        
        try:
            regression_results = self.results['regression']
            coefficients = regression_results['coefficients']
            standard_errors = regression_results['se']
            pvalues = regression_results['pvalues']
            interaction_variables = regression_results['interaction_variables']
            
            dynamic_effects = {}
            
            # Process each year in our analysis
            for year in self.all_analysis_years:
                if year == self.reference_year:
                    # Reference year effect is 0 by construction
                    dynamic_effects[year] = {
                        'year': year,
                        'relative_year': 0,
                        'coefficient': 0.0,
                        'standard_error': 0.0,
                        'p_value': np.nan,
                        'confidence_interval': {'lower': 0.0, 'upper': 0.0},
                        'significant_5pct': False,
                        'is_reference': True,
                        'period_type': 'Reference'
                    }
                else:
                    # Look for interaction variable
                    interaction_var = f'treatment_group_year_{year}'
                    
                    if interaction_var in coefficients:
                        coef = coefficients[interaction_var]
                        se = standard_errors[interaction_var]
                        pval = pvalues[interaction_var]
                        
                        # Calculate confidence interval
                        ci_lower = coef - 1.96 * se
                        ci_upper = coef + 1.96 * se
                        
                        # Determine period type
                        if year < self.reform_year:
                            period_type = 'Pre-reform'
                        else:
                            period_type = 'Post-reform'
                        
                        dynamic_effects[year] = {
                            'year': year,
                            'relative_year': year - self.reference_year,
                            'coefficient': float(coef),
                            'standard_error': float(se),
                            'p_value': float(pval),
                            'confidence_interval': {'lower': float(ci_lower), 'upper': float(ci_upper)},
                            'significant_5pct': float(pval) < 0.05,
                            'is_reference': False,
                            'period_type': period_type,
                            'interaction_variable': interaction_var
                        }
                        
                        logger.info(f"Year {year} (t{year - self.reference_year:+d}): {coef:.4f} (SE: {se:.4f}, p: {pval:.4f})")
                    else:
                        logger.warning(f"Interaction variable not found for year {year}: {interaction_var}")
                        logger.info(f"Available interaction variables: {interaction_variables}")
                        logger.info(f"Available coefficients: {list(coefficients.keys())}")
                        
                        # Create placeholder entry
                        dynamic_effects[year] = {
                            'year': year,
                            'relative_year': year - self.reference_year,
                            'coefficient': np.nan,
                            'standard_error': np.nan,
                            'p_value': np.nan,
                            'confidence_interval': {'lower': np.nan, 'upper': np.nan},
                            'significant_5pct': False,
                            'is_reference': False,
                            'period_type': 'Pre-reform' if year < self.reform_year else 'Post-reform',
                            'interaction_variable': interaction_var,
                            'error': 'Variable not found'
                        }
            
            # Store results
            self.dynamic_effects = dynamic_effects
            self.results['dynamic_effects'] = dynamic_effects
            
            # Log summary of dynamic effects
            logger.info("\nDynamic Treatment Effects Summary:")
            logger.info("=" * 50)
            for year in sorted(dynamic_effects.keys()):
                effect = dynamic_effects[year]
                if not effect['is_reference'] and not np.isnan(effect['coefficient']):
                    signif = " *" if effect['significant_5pct'] else ""
                    logger.info(f"{year} vs {self.reference_year} (t{effect['relative_year']:+d}): "
                              f"{effect['coefficient']:.4f} (p={effect['p_value']:.4f}){signif}")
            
            # Test parallel trends (pre-reform effects should be ≈ 0)
            pre_reform_effects = {year: effect for year, effect in dynamic_effects.items() 
                                if effect['period_type'] == 'Pre-reform' and not np.isnan(effect['coefficient'])}
            
            if pre_reform_effects:
                parallel_trends_violated = any(effect['significant_5pct'] for effect in pre_reform_effects.values())
                logger.info(f"\nParallel Trends Test: {'VIOLATED' if parallel_trends_violated else 'PASSED'}")
                
                self.results['parallel_trends_test'] = {
                    'pre_reform_effects': pre_reform_effects,
                    'parallel_trends_violated': parallel_trends_violated,
                    'parallel_trends_pass': not parallel_trends_violated
                }
            
            return dynamic_effects
            
        except Exception as e:
            logger.error(f"Error extracting dynamic effects: {str(e)}")
            raise
    
    def create_event_study_plot(self):
        """Create event study visualization plot"""
        logger.info("Creating event study visualization")
        
        # Prepare data for plotting
        plot_data = []
        for year, effect in self.dynamic_effects.items():
            if not np.isnan(effect['coefficient']):
                plot_data.append({
                    'year': year,
                    'relative_year': effect['relative_year'],
                    'coefficient': effect['coefficient'],
                    'ci_lower': effect['confidence_interval']['lower'],
                    'ci_upper': effect['confidence_interval']['upper'],
                    'period_type': effect['period_type'],
                    'significant': effect['significant_5pct']
                })
        
        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df.sort_values('relative_year')
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Set style
        sns.set_style("whitegrid")
        
        # Define colors
        colors = {
            'Reference': '#666666',
            'Pre-reform': '#1f77b4', 
            'Post-reform': '#ff7f0e'
        }
        
        # Plot points and confidence intervals
        for _, row in plot_df.iterrows():
            color = colors[row['period_type']]
            
            # Plot point
            marker = 'D' if row['period_type'] == 'Reference' else 'o'
            markersize = 10 if row['significant'] else 8
            alpha = 1.0 if row['significant'] else 0.7
            
            plt.plot(row['relative_year'], row['coefficient'], 
                    marker=marker, markersize=markersize, color=color, alpha=alpha)
            
            # Plot confidence interval
            if row['period_type'] != 'Reference':
                plt.plot([row['relative_year'], row['relative_year']], 
                        [row['ci_lower'], row['ci_upper']], 
                        color=color, alpha=0.7, linewidth=2)
        
        # Connect points with line
        plt.plot(plot_df['relative_year'], plot_df['coefficient'], 
                color='black', alpha=0.5, linewidth=1, linestyle='-')
        
        # Add reference line at zero
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add vertical line at reform year
        plt.axvline(x=0, color='red', linestyle=':', alpha=0.5, linewidth=1, 
                   label=f'Reform Year ({self.reform_year})')
        
        # Customize plot
        plt.xlabel('Years Relative to 2012 (Reference Year)', fontsize=14, fontweight='bold')
        plt.ylabel('Treatment Effect on School Enrollment', fontsize=14, fontweight='bold')
        plt.title('Event Study: Dynamic Treatment Effects of 2013 Education Reform\n' +
                 f'(Reference Year: {self.reference_year})', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='D', color=colors['Reference'], linestyle='None',
                      markersize=10, label=f'Reference ({self.reference_year})'),
            plt.Line2D([0], [0], marker='o', color=colors['Pre-reform'], linestyle='None',
                      markersize=8, label='Pre-reform'),
            plt.Line2D([0], [0], marker='o', color=colors['Post-reform'], linestyle='None',
                      markersize=8, label='Post-reform'),
            plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5,
                      label='Zero Effect Line'),
            plt.Line2D([0], [0], color='red', linestyle=':', alpha=0.5,
                      label=f'Reform Year ({self.reform_year})')
        ]
        
        plt.legend(handles=legend_elements, fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Format axes
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add annotations for significant effects
        for _, row in plot_df.iterrows():
            if row['significant'] and row['period_type'] != 'Reference':
                plt.annotate('*', 
                           xy=(row['relative_year'], row['coefficient']),
                           xytext=(0, 10), textcoords='offset points',
                           fontsize=16, fontweight='bold', ha='center',
                           color='red')
        
        # Add text box with key information
        textstr = '\n'.join([
            f"Sample: {self.results['regression']['n_observations']:,} observations",
            f"Treatment: Ages 6-11 in 2013 (born {min(self.treated_birth_years)}-{max(self.treated_birth_years)})",
            f"Control: Ages 15-18 in 2013 (born {min(self.control_birth_years)}-{max(self.control_birth_years)})",
            f"Controls: Age group FE + Male + Birth year + Survey year + Province FE",
            f"* = Significant at 5% level"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_path / "event_study_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Event study plot saved to: {plot_path}")
        
        # Don't show plot to avoid hanging
        # plt.show()
        plt.close()
        
        return plot_df
    
    def generate_summary_report(self):
        """Generate comprehensive event study summary report"""
        logger.info("Generating event study summary report")
        
        report_path = self.output_path / "event_study_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EVENT STUDY ANALYSIS REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS SPECIFICATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Formula: {self.results['regression']['formula']}\n")
            f.write(f"Sample size: {self.results['regression']['n_observations']:,} observations\n")
            f.write(f"Reference year: {self.reference_year}\n")
            f.write(f"Analysis years: {', '.join(map(str, self.all_analysis_years))}\n")
            f.write(f"Clustering: CRV1 by birth_year\n")
            f.write(f"Outcome: General school enrollment (currently_enrolled)\n\n")
            
            f.write("SAMPLE DEFINITION:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total observations: {len(self.analysis_sample):,}\n")
            f.write(f"Treated cohorts: Born {min(self.treated_birth_years)}-{max(self.treated_birth_years)} (ages 6-11 in 2013)\n")
            f.write(f"Control cohorts: Born {min(self.control_birth_years)}-{max(self.control_birth_years)} (ages 15-18 in 2013)\n")
            
            # Sample by treatment group
            for treat in [0, 1]:
                group_name = "Control" if treat == 0 else "Treated"
                treat_data = self.analysis_sample[self.analysis_sample['treatment_group'] == treat]
                n = len(treat_data)
                enroll_rate = treat_data['currently_enrolled'].mean()
                f.write(f"{group_name} group: {n:,} obs ({enroll_rate:.1%} enrolled)\n")
            
            f.write("\nDYNAMIC TREATMENT EFFECTS:\n")
            f.write("-" * 30 + "\n")
            f.write("Year vs 2012 | Relative Year | Coefficient | Std Error | P-value | 95% CI | Significant\n")
            f.write("-" * 85 + "\n")
            
            for year in sorted(self.dynamic_effects.keys()):
                effect = self.dynamic_effects[year]
                if effect['is_reference']:
                    f.write(f"{year} vs 2012  |      0        |    0.0000   |   0.0000  |   ---   | [0.0000, 0.0000] | Reference\n")
                elif not np.isnan(effect['coefficient']):
                    signif = "Yes" if effect['significant_5pct'] else "No"
                    f.write(f"{year} vs 2012  |   {effect['relative_year']:+3d}        | "
                           f"{effect['coefficient']:8.4f}   | {effect['standard_error']:8.4f}  | "
                           f"{effect['p_value']:7.4f} | [{effect['confidence_interval']['lower']:7.4f}, "
                           f"{effect['confidence_interval']['upper']:7.4f}] | {signif}\n")
                else:
                    f.write(f"{year} vs 2012  |   {effect['relative_year']:+3d}        |     ---     |    ---    |   ---   |       ---       | Error\n")
            
            # Parallel trends test
            pt_test = self.results.get('parallel_trends_test', {})
            if pt_test:
                f.write("\nPARALLEL TRENDS TEST:\n")
                f.write("-" * 25 + "\n")
                f.write("Pre-reform effects (should be ≈ 0):\n")
                
                for year, effect in pt_test['pre_reform_effects'].items():
                    signif = " *" if effect['significant_5pct'] else ""
                    f.write(f"  {year} vs 2012: {effect['coefficient']:.4f} (p={effect['p_value']:.4f}){signif}\n")
                
                result = "VIOLATED" if pt_test.get('parallel_trends_violated', True) else "PASSED"
                f.write(f"\nParallel Trends Assumption: {result}\n")
                
                if pt_test.get('parallel_trends_pass', False):
                    f.write("✓ No significant differential pre-trends detected\n")
                    f.write("✓ Event study design is valid\n")
                else:
                    f.write("⚠ Significant differential pre-trends detected\n")
                    f.write("⚠ Consider robustness checks\n")
            
            f.write("\nTREATMENT EFFECT INTERPRETATION:\n")
            f.write("-" * 35 + "\n")
            
            # Analyze pre-reform effects
            pre_effects = [effect for year, effect in self.dynamic_effects.items() 
                          if effect['period_type'] == 'Pre-reform' and not np.isnan(effect['coefficient'])]
            if pre_effects:
                f.write("Pre-reform effects (2008, 2010 vs 2012):\n")
                pre_significant = any(effect['significant_5pct'] for effect in pre_effects)
                if not pre_significant:
                    f.write("✓ No significant pre-trends, supporting parallel trends assumption\n")
                else:
                    f.write("⚠ Some significant pre-trends detected\n")
            
            # Analyze post-reform effects
            post_effects = [effect for year, effect in self.dynamic_effects.items() 
                           if effect['period_type'] == 'Post-reform' and not np.isnan(effect['coefficient'])]
            if post_effects:
                f.write("\nPost-reform effects (2014, 2016, 2020 vs 2012):\n")
                
                for effect in sorted(post_effects, key=lambda x: x['year']):
                    year = effect['year']
                    years_since = year - self.reform_year
                    if effect['significant_5pct']:
                        direction = "increased" if effect['coefficient'] > 0 else "decreased"
                        f.write(f"• {year} (+{years_since} years): {direction} enrollment by {abs(effect['coefficient']):.1%} (significant)\n")
                    else:
                        f.write(f"• {year} (+{years_since} years): {effect['coefficient']:+.1%} change (not significant)\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 18 + "\n")
            f.write("• event_study_plot.png\n")
            f.write("• event_study_report.txt\n")
            f.write("• event_study_results.json\n")
        
        logger.info(f"Summary report saved to: {report_path}")
    
    def save_results(self):
        """Save detailed results to JSON file"""
        logger.info("Saving event study results")
        
        # Prepare results for JSON serialization
        json_results = {
            'analysis_parameters': {
                'reference_year': self.reference_year,
                'all_analysis_years': self.all_analysis_years,
                'treated_birth_years': self.treated_birth_years,
                'control_birth_years': self.control_birth_years,
                'reform_year': self.reform_year
            },
            'sample_summary': {
                'total_observations': len(self.analysis_sample),
                'by_treatment_group': {},
                'by_year': {}
            },
            'regression_results': {},
            'dynamic_effects': {},
            'parallel_trends_test': self.results.get('parallel_trends_test', {})
        }
        
        # Add sample summary by treatment group
        for treat in [0, 1]:
            group_name = "control" if treat == 0 else "treated"
            treat_data = self.analysis_sample[self.analysis_sample['treatment_group'] == treat]
            json_results['sample_summary']['by_treatment_group'][group_name] = {
                'n_observations': len(treat_data),
                'enrollment_rate': float(treat_data['currently_enrolled'].mean())
            }
        
        # Add sample summary by year
        for year in self.all_analysis_years:
            year_data = self.analysis_sample[self.analysis_sample['survey_year'] == year]
            json_results['sample_summary']['by_year'][str(year)] = {
                'n_observations': len(year_data),
                'enrollment_rate': float(year_data['currently_enrolled'].mean()) if len(year_data) > 0 else 0.0
            }
        
        # Add regression results (excluding non-serializable model object)
        reg_results = self.results.get('regression', {})
        json_results['regression_results'] = {
            k: v for k, v in reg_results.items() 
            if k not in ['model', 'summary']  # Exclude non-serializable objects
        }
        
        # Add dynamic effects (convert any numpy types)
        for year, effect in self.dynamic_effects.items():
            json_results['dynamic_effects'][str(year)] = {
                k: (float(v) if isinstance(v, (np.integer, np.floating)) else 
                    v if not isinstance(v, dict) else
                    {k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 for k2, v2 in v.items()})
                for k, v in effect.items()
            }
        
        # Save to file
        results_path = self.output_path / "event_study_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Detailed results saved to: {results_path}")
        
        return {
            'event_study_results': str(results_path),
            'event_study_report': str(self.output_path / "event_study_report.txt"),
            'event_study_plot': str(self.output_path / "event_study_plot.png")
        }
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete event study analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting event study analysis pipeline")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Create analysis sample
            self.create_analysis_sample()
            
            # Step 3: Create treatment variables
            self.create_treatment_variables()
            
            # Step 4: Create interaction variables
            self.create_interaction_variables()
            
            # Step 5: Log sample summary
            self.log_sample_summary()
            
            # Step 6: Run event study regression
            regression_results = self.run_event_study_regression()
            
            # Step 7: Extract dynamic effects
            self.extract_dynamic_effects()
            
            # Step 8: Create visualization
            self.create_event_study_plot()
            
            # Step 9: Generate reports
            self.generate_summary_report()
            saved_files = self.save_results()
            
            logger.info("Event study analysis completed successfully")
            
            # Return summary
            pt_test = self.results.get('parallel_trends_test', {})
            summary = {
                'status': 'completed',
                'sample_size': len(self.analysis_sample),
                'reference_year': self.reference_year,
                'dynamic_effects_extracted': len(self.dynamic_effects),
                'parallel_trends_pass': pt_test.get('parallel_trends_pass', False),
                'significant_post_effects': sum(1 for effect in self.dynamic_effects.values() 
                                              if effect['period_type'] == 'Post-reform' and effect['significant_5pct']),
                'files_generated': list(saved_files.values())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in event study analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("EVENT STUDY ANALYSIS FOR VIETNAM EDUCATION REFORM")
    print("="*60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run analysis
    event_study = EventStudyAnalysis()
    
    try:
        results = event_study.run_full_analysis()
        
        print("\n" + "="*60)
        print("EVENT STUDY ANALYSIS COMPLETED")
        print("="*60)
        print(f"Sample size: {results['sample_size']:,} observations")
        print(f"Reference year: {results['reference_year']}")
        print(f"Dynamic effects extracted: {results['dynamic_effects_extracted']}")
        print(f"Parallel trends assumption: {'PASSED' if results['parallel_trends_pass'] else 'VIOLATED'}")
        print(f"Significant post-reform effects: {results['significant_post_effects']}")
        print(f"Files generated: {len(results['files_generated'])}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results['parallel_trends_pass']:
            print("\n✅ EVENT STUDY VALIDATES PARALLEL TRENDS ASSUMPTION")
        else:
            print("\n⚠️  CAUTION: PRE-TRENDS DETECTED - CONSIDER ROBUSTNESS")
            
        if results['significant_post_effects'] > 0:
            print(f"📈 SIGNIFICANT DYNAMIC EFFECTS DETECTED ({results['significant_post_effects']} periods)")
        else:
            print("📊 NO SIGNIFICANT DYNAMIC EFFECTS DETECTED")
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
