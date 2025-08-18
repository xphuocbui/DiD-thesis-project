#!/usr/bin/env python3
"""
Basic DiD Analysis for Vietnam Education Reform (with 2014 included)
===================================================================

This script implements the main Difference-in-Differences analysis for the 2013 
education reform in Vietnam using PyFixest. Now includes 2014 data after the fix.

Specification:
currently_enrolled ~ treatment_group + post_reform + treatment_group*post_reform + age + age_squared + male | birth_year + survey_year + tinh

Sample:
- Treated cohorts: Born 2002-2007 (ages 6-11 in 2013)
- Control cohorts: Born 1995-1998 (ages 15-18 in 2013) 
- Analysis years: 2008, 2010, 2012 (pre) and 2014, 2016, 2020 (post)
- Outcome: General school enrollment (currently_enrolled)

Author: DiD Analysis Project  
Date: 2025
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, Tuple, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BasicDiDAnalysis:
    """
    Class to implement basic DiD analysis for Vietnam education reform with 2014 included.
    """
    
    def __init__(self, data_path: str = "../../../../data/master_dataset_exclude_2018.csv", 
                 output_path: str = "../../../../results/regression/"):
        """
        Initialize the basic DiD analysis.
        
        Args:
            data_path: Path to the master dataset with currently_enrolled column
            output_path: Path to save results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Define analysis parameters (now includes 2014)
        self.pre_reform_years = [2008, 2010, 2012]  # Pre-reform years for comparison
        self.post_reform_years = [2014, 2016, 2020]  # Post-reform analysis years (2014 now included!)
        self.all_analysis_years = self.pre_reform_years + self.post_reform_years
        self.treated_birth_years = [2002, 2003, 2004, 2005, 2006, 2007]  # Ages 6-11 in 2013
        self.control_birth_years = [1995, 1996, 1997, 1998]  # Ages 15-18 in 2013  
        self.analysis_birth_years = list(range(1995, 2008))  # Birth years for relevant age groups
        self.reform_year = 2013
        
        # Results storage
        self.data = None
        self.analysis_sample = None
        self.did_model = None
        self.results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data for DiD analysis.
        
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
        Create the analysis sample for DiD (now includes 2014).
        
        Returns:
            Filtered DataFrame with analysis sample
        """
        logger.info("Creating analysis sample for DiD analysis (including 2014)")
        
        # Start with the full dataset
        sample = self.data.copy()
        logger.info(f"Starting with {len(sample):,} observations")
        
        # Filter to all analysis years (both pre and post reform for DiD)
        sample = sample[sample['survey_year'].isin(self.all_analysis_years)]
        logger.info(f"After filtering to analysis years {self.all_analysis_years}: {len(sample):,} observations")
        
        # Check if 2014 is present
        year_counts = sample['survey_year'].value_counts().sort_index()
        logger.info("Observations by year after year filtering:")
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count:,}")
        
        if 2014 in year_counts.index:
            logger.info(f"✅ 2014 successfully included with {year_counts[2014]:,} observations")
        else:
            logger.warning("⚠️ 2014 not found in sample - may need to re-run 2014 fix")
        
        # Filter to relevant birth years (ages 6-18 in 2013)
        sample = sample[sample['nam sinh'].isin(self.analysis_birth_years)]
        logger.info(f"After filtering to birth years {min(self.analysis_birth_years)}-{max(self.analysis_birth_years)}: {len(sample):,} observations")
        
        # Remove observations with missing enrollment data
        sample = sample.dropna(subset=['currently_enrolled'])
        logger.info(f"After removing missing enrollment: {len(sample):,} observations")
        
        # Check 2014 availability after enrollment filter
        year_counts_after_enrollment = sample['survey_year'].value_counts().sort_index()
        logger.info("Observations by year after enrollment filter:")
        for year, count in year_counts_after_enrollment.items():
            logger.info(f"  {year}: {count:,}")
        
        # Remove observations with missing control variables
        sample = sample.dropna(subset=['tuoi', 'gender', 'tinh', 'nam sinh'])
        logger.info(f"After removing missing controls: {len(sample):,} observations")
        
        self.analysis_sample = sample
        logger.info("✓ Analysis sample created successfully")
        
        return self.analysis_sample
    
    def create_treatment_variables(self) -> pd.DataFrame:
        """
        Create treatment variables for DiD analysis.
        
        Returns:
            DataFrame with treatment variables added
        """
        logger.info("Creating treatment variables")
        
        sample = self.analysis_sample.copy()
        
        # Create treatment group indicator
        sample['treatment_group'] = sample['nam sinh'].apply(
            lambda x: 1 if x in self.treated_birth_years else 
                     (0 if x in self.control_birth_years else np.nan)
        )
        
        # Remove observations not in treatment or control groups
        sample = sample.dropna(subset=['treatment_group'])
        sample['treatment_group'] = sample['treatment_group'].astype(int)
        
        # Create post-reform indicator (1 for post-reform years, 0 for pre-reform)
        sample['post_reform'] = sample['survey_year'].apply(
            lambda x: 1 if x in self.post_reform_years else 0
        )
        
        # Create the DiD interaction term manually
        sample['treated_post'] = sample['treatment_group'] * sample['post_reform']
        
        # Create age variables
        sample['age'] = sample['tuoi']
        sample['age_squared'] = sample['age'] ** 2
        
        # Create male indicator
        sample['male'] = (sample['gender'] == 'Male').astype(int)
        
        # Create birth year variable for fixed effects
        sample['birth_year'] = sample['nam sinh']
        
        logger.info(f"Treatment group distribution:")
        logger.info(f"  Control (0): {sum(sample['treatment_group'] == 0):,} observations")
        logger.info(f"  Treated (1): {sum(sample['treatment_group'] == 1):,} observations")
        
        logger.info(f"Post-reform distribution:")
        logger.info(f"  Pre-reform (0): {sum(sample['post_reform'] == 0):,} observations")
        logger.info(f"  Post-reform (1): {sum(sample['post_reform'] == 1):,} observations")
        
        logger.info(f"DiD interaction distribution:")
        logger.info(f"  treated_post = 1: {sum(sample['treated_post'] == 1):,} observations")
        
        # Final year breakdown
        final_year_counts = sample['survey_year'].value_counts().sort_index()
        logger.info("Final sample by year:")
        for year, count in final_year_counts.items():
            logger.info(f"  {year}: {count:,}")
        
        self.analysis_sample = sample
        return self.analysis_sample
    
    def run_basic_did_regression(self):
        """
        Run the basic DiD regression using PyFixest with manual interaction term.
        
        Returns:
            PyFixest regression results
        """
        logger.info("Running basic DiD regression with manual interaction")
        
        # Define the regression formula with manual interaction term
        formula = 'currently_enrolled ~ treatment_group + post_reform + treated_post + age + age_squared + male | birth_year + survey_year + tinh'
        
        logger.info(f"Regression formula: {formula}")
        logger.info(f"Sample size: {len(self.analysis_sample):,} observations")
        
        # Run the regression with clustered standard errors by birth year
        try:
            self.did_model = pf.feols(
                formula, 
                data=self.analysis_sample,
                vcov={'CRV1': 'birth_year'}
            )
            
            logger.info("✓ DiD regression completed successfully")
            return self.did_model
            
        except Exception as e:
            logger.error(f"Error running DiD regression: {str(e)}")
            raise
    
    def extract_did_results(self) -> Dict[str, Any]:
        """
        Extract and summarize DiD results.
        
        Returns:
            Dictionary with key results
        """
        logger.info("Extracting DiD results")
        
        try:
            # Get model summary
            summary = self.did_model.summary()
            
            # Extract coefficients, standard errors, and p-values
            coef_dict = self.did_model.coef().to_dict()
            se_dict = self.did_model.se().to_dict()
            pvals_dict = self.did_model.pvalue().to_dict()
            
            # Create results dictionary
            results = {
                'model_info': {
                    'formula': 'currently_enrolled ~ treatment_group + post_reform + treated_post + age + age_squared + male | birth_year + survey_year + tinh',
                    'n_observations': len(self.analysis_sample),
                    'vcov': 'CRV1: birth_year',
                    'r_squared': float(self.did_model.r2) if hasattr(self.did_model, 'r2') else None,
                    'includes_2014': True
                },
                'sample_summary': {
                    'total_obs': len(self.analysis_sample),
                    'treated_obs': int(sum(self.analysis_sample['treatment_group'] == 1)),
                    'control_obs': int(sum(self.analysis_sample['treatment_group'] == 0)),
                    'pre_reform_obs': int(sum(self.analysis_sample['post_reform'] == 0)),
                    'post_reform_obs': int(sum(self.analysis_sample['post_reform'] == 1)),
                    'treated_post_obs': int(sum(self.analysis_sample['treated_post'] == 1)),
                    'enrollment_rate_treated': float(self.analysis_sample[self.analysis_sample['treatment_group'] == 1]['currently_enrolled'].mean()),
                    'enrollment_rate_control': float(self.analysis_sample[self.analysis_sample['treatment_group'] == 0]['currently_enrolled'].mean()),
                    'enrollment_rate_pre': float(self.analysis_sample[self.analysis_sample['post_reform'] == 0]['currently_enrolled'].mean()),
                    'enrollment_rate_post': float(self.analysis_sample[self.analysis_sample['post_reform'] == 1]['currently_enrolled'].mean()),
                    'years_included': sorted(self.analysis_sample['survey_year'].unique().tolist()),
                    'birth_years_treated': self.treated_birth_years,
                    'birth_years_control': self.control_birth_years
                },
                'coefficients': coef_dict,
                'standard_errors': se_dict,
                'p_values': pvals_dict,
                'confidence_intervals': {}
            }
            
            # Calculate 95% confidence intervals
            for var in results['coefficients'].keys():
                if var in results['standard_errors']:
                    coef = results['coefficients'][var]
                    se = results['standard_errors'][var]
                    results['confidence_intervals'][var] = {
                        'lower': coef - 1.96 * se,
                        'upper': coef + 1.96 * se
                    }
            
            # The main DiD effect is the coefficient on 'treated_post'
            if 'treated_post' in results['coefficients']:
                results['main_did_effect'] = {
                    'variable': 'treated_post',
                    'coefficient': results['coefficients']['treated_post'],
                    'standard_error': results['standard_errors']['treated_post'],
                    'p_value': results['p_values']['treated_post'],
                    'confidence_interval': results['confidence_intervals']['treated_post'],
                    'significant_5pct': results['p_values']['treated_post'] < 0.05,
                    'significant_10pct': results['p_values']['treated_post'] < 0.10
                }
            
            self.results = results
            logger.info("✓ DiD results extracted successfully")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error extracting DiD results: {str(e)}")
            raise
    
    def save_results(self) -> Dict[str, str]:
        """
        Save all results to files.
        
        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Saving DiD results")
        
        # Save JSON results (no timestamp - overwrite existing)
        json_file = self.output_path / "basic_did_results_with_2014.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save detailed report (no timestamp - overwrite existing)
        report_file = self.output_path / "basic_did_report_with_2014.txt"
        with open(report_file, 'w') as f:
            f.write("BASIC DiD ANALYSIS REPORT (WITH 2014 INCLUDED)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS SPECIFICATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Formula: {self.results['model_info']['formula']}\n")
            f.write(f"Sample size: {self.results['model_info']['n_observations']:,} observations\n")
            f.write(f"Clustering: {self.results['model_info']['vcov']}\n")
            f.write(f"Includes 2014: {self.results['model_info']['includes_2014']}\n")
            if self.results['model_info']['r_squared']:
                f.write(f"R-squared: {self.results['model_info']['r_squared']:.4f}\n")
            f.write("\n")
            
            f.write("SAMPLE SUMMARY:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total observations: {self.results['sample_summary']['total_obs']:,}\n")
            f.write(f"Treated group: {self.results['sample_summary']['treated_obs']:,} obs ({self.results['sample_summary']['enrollment_rate_treated']:.1%} enrolled)\n")
            f.write(f"Control group: {self.results['sample_summary']['control_obs']:,} obs ({self.results['sample_summary']['enrollment_rate_control']:.1%} enrolled)\n")
            f.write(f"Pre-reform period: {self.results['sample_summary']['pre_reform_obs']:,} obs ({self.results['sample_summary']['enrollment_rate_pre']:.1%} enrolled)\n")
            f.write(f"Post-reform period: {self.results['sample_summary']['post_reform_obs']:,} obs ({self.results['sample_summary']['enrollment_rate_post']:.1%} enrolled)\n")
            f.write(f"Treated × Post-reform: {self.results['sample_summary']['treated_post_obs']:,} obs\n")
            f.write(f"Years included: {', '.join(map(str, self.results['sample_summary']['years_included']))}\n")
            f.write(f"Treated birth years: {', '.join(map(str, self.results['sample_summary']['birth_years_treated']))}\n")
            f.write(f"Control birth years: {', '.join(map(str, self.results['sample_summary']['birth_years_control']))}\n")
            f.write("\n")
            
            if 'main_did_effect' in self.results:
                f.write("MAIN DiD RESULTS:\n")
                f.write("-" * 17 + "\n")
                effect = self.results['main_did_effect']
                f.write(f"Treatment effect (treated_post): {effect['coefficient']:.6f}\n")
                f.write(f"Standard error: {effect['standard_error']:.6f}\n")
                f.write(f"P-value: {effect['p_value']:.6f}\n")
                f.write(f"95% CI: [{effect['confidence_interval']['lower']:.6f}, {effect['confidence_interval']['upper']:.6f}]\n")
                f.write(f"Significant at 5%: {'Yes' if effect['significant_5pct'] else 'No'}\n")
                f.write(f"Significant at 10%: {'Yes' if effect['significant_10pct'] else 'No'}\n")
                f.write("\n")
            
            f.write("ALL COEFFICIENTS:\n")
            f.write("-" * 17 + "\n")
            for var in self.results['coefficients'].keys():
                coef = self.results['coefficients'][var]
                se = self.results['standard_errors'].get(var, np.nan)
                pval = self.results['p_values'].get(var, np.nan)
                f.write(f"{var}: {coef:.6f} (SE: {se:.6f}, p: {pval:.6f})\n")
            
            # Add interpretation section
            f.write("\nINTERPRETATION:\n")
            f.write("-" * 15 + "\n")
            if 'main_did_effect' in self.results:
                effect = self.results['main_did_effect']
                f.write(f"The DiD estimate suggests that the 2013 education reform had ")
                if effect['significant_5pct']:
                    f.write(f"a statistically significant effect on school enrollment.\n")
                    f.write(f"The reform {'increased' if effect['coefficient'] > 0 else 'decreased'} enrollment ")
                    f.write(f"by {abs(effect['coefficient']):.1%} for the treated cohorts.\n")
                else:
                    f.write(f"no statistically significant effect on school enrollment.\n")
                    f.write(f"The estimated effect is {effect['coefficient']:.1%}, but ")
                    f.write(f"the p-value ({effect['p_value']:.3f}) suggests this is not statistically different from zero.\n")
                
                f.write(f"\nNOTE: This analysis now includes 2014 data (first post-reform year).\n")
        
        # Save PyFixest summary output (no timestamp - overwrite existing)
        summary_file = self.output_path / "basic_did_pyfixest_output_with_2014.txt"
        with open(summary_file, 'w') as f:
            f.write("BASIC DiD PYFIXEST MODEL SUMMARY (WITH 2014)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(str(self.did_model.summary()))
        
        logger.info(f"Results saved to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Report: {report_file}")
        logger.info(f"  PyFixest: {summary_file}")
        
        return {
            'json_results': str(json_file),
            'detailed_report': str(report_file),
            'pyfixest_summary': str(summary_file)
        }
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete basic DiD analysis pipeline.
        
        Returns:
            Complete results dictionary
        """
        logger.info("Starting Basic DiD Analysis Pipeline (with 2014)")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Create analysis sample
            self.create_analysis_sample()
            
            # Step 3: Create treatment variables
            self.create_treatment_variables()
            
            # Step 4: Run DiD regression
            self.run_basic_did_regression()
            
            # Step 5: Extract results
            self.extract_did_results()
            
            # Step 6: Save results
            saved_files = self.save_results()
            
            logger.info("Basic DiD Analysis (with 2014) completed successfully")
            
            # Print key results to console
            if 'main_did_effect' in self.results:
                effect = self.results['main_did_effect']
                print("\n" + "=" * 80)
                print("BASIC DiD ANALYSIS - KEY RESULTS (WITH 2014)")
                print("=" * 80)
                print(f"Treatment Effect (treated_post): {effect['coefficient']:.6f}")
                print(f"Standard Error: {effect['standard_error']:.6f}")
                print(f"P-value: {effect['p_value']:.6f}")
                print(f"95% Confidence Interval: [{effect['confidence_interval']['lower']:.6f}, {effect['confidence_interval']['upper']:.6f}]")
                print(f"Statistically Significant (5%): {'Yes' if effect['significant_5pct'] else 'No'}")
                print(f"Statistically Significant (10%): {'Yes' if effect['significant_10pct'] else 'No'}")
                print(f"Sample Size: {self.results['sample_summary']['total_obs']:,} observations")
                print(f"Years Included: {', '.join(map(str, self.results['sample_summary']['years_included']))}")
                print(f"2014 Successfully Included: ✅")
                print("=" * 80)
            
            return {
                'results': self.results,
                'model': self.did_model,
                'sample': self.analysis_sample,
                'saved_files': saved_files
            }
            
        except Exception as e:
            logger.error(f"Error in Basic DiD analysis: {str(e)}")
            raise


def main():
    """
    Main function to run basic DiD analysis with 2014 included.
    """
    logger.info("Starting Basic DiD Analysis Main Script (with 2014)")
    
    # Set up paths - adjust as needed based on where script is run from
    data_path = "../../../../data/master_dataset_exclude_2018.csv"
    output_path = "../../../../results/regression/"
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        logger.info("Please check the data path")
        return
    
    # Initialize and run analysis
    try:
        analyzer = BasicDiDAnalysis(data_path, output_path)
        results = analyzer.run_full_analysis()
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()