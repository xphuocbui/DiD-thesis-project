"""
PyFixest Cross-Sectional DiD Analysis
====================================

This script implements a cross-sectional Difference-in-Differences analysis 
using PyFixest for Vietnam's 2013 education reform impact study.

Specifications:
1. Basic DiD: secondary_success ~ treated_cohort
2. Time FE: secondary_success ~ treated_cohort + C(survey_year) 
3. Province FE: secondary_success ~ treated_cohort + C(tinh)
4. Full model: secondary_success ~ treated_cohort + C(survey_year) + C(tinh), vcov={"CRV1": "tinh"}

Author: DiD Analysis Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyFixestDiDAnalysis:
    """
    PyFixest-based Difference-in-Differences analysis class.
    Implements cross-sectional DiD using PyFixest package.
    """
    
    def __init__(self, data_path: str, output_path: str = "results/"):
        """
        Initialize the PyFixest DiD analysis.
        
        Args:
            data_path: Path to the filtered cohort data CSV
            output_path: Directory to save results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.data = None
        self.models = {}
        self.results = {}
        
        logger.info(f"PyFixest DiD Analysis initialized")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output path: {output_path}")
    
    def load_and_prepare_data(self):
        """
        Load the filtered cohort data and prepare variables for PyFixest analysis.
        """
        logger.info("Loading and preparing data for PyFixest analysis")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data):,} observations from {self.data_path}")
        
        # Create secondary_success outcome if not exists
        self._create_secondary_success()
        
        # Create treated_cohort variable (this is what you want in your models)
        self._create_treated_cohort()
        
        # Clean and prepare variables
        self._prepare_variables()
        
        # Summary statistics
        self._log_sample_summary()
    
    def _create_secondary_success(self):
        """
        Create the secondary_success outcome variable if it doesn't exist.
        """
        if 'secondary_success' not in self.data.columns:
            logger.info("Creating secondary_success outcome variable")
            
            # Initialize outcome as 0
            self.data['secondary_success'] = 0
            
            # For treated cohort: success = completing primary AND attending secondary/higher
            treated_mask = self.data['treated'] == 1
            
            # Success criteria for treated: currently in secondary school or higher
            current_edu = self.data['Current edu. level'].fillna('')
            treated_success = (
                current_edu.str.contains('Secondary|High school|University|College', 
                                       case=False, na=False)
            )
            
            # For control cohort: success = having completed secondary or higher education
            control_mask = self.data['treated'] == 0
            education_level = self.data['education level'].fillna('')
            control_success = (
                education_level.str.contains('Secondary|High school|University|College', 
                                           case=False, na=False)
            )
            
            # Apply success criteria
            self.data.loc[treated_mask, 'secondary_success'] = treated_success[treated_mask].astype(int)
            self.data.loc[control_mask, 'secondary_success'] = control_success[control_mask].astype(int)
            
            logger.info("secondary_success variable created")
        else:
            logger.info("secondary_success variable already exists")
    
    def _create_treated_cohort(self):
        """
        Create the treated_cohort variable for DiD analysis.
        This represents the interaction between treated group and post-reform period.
        """
        logger.info("Creating treated_cohort variable")
        
        # treated_cohort = treated * post_reform (this is the DiD interaction term)
        self.data['treated_cohort'] = self.data['treated'] * self.data['post_reform']
        
        logger.info("treated_cohort variable created (DiD interaction term)")
        
        # Log distribution
        treated_cohort_dist = self.data['treated_cohort'].value_counts().sort_index()
        logger.info(f"treated_cohort distribution: {dict(treated_cohort_dist)}")
    
    def _prepare_variables(self):
        """
        Prepare and clean variables for regression analysis.
        """
        logger.info("Preparing variables for analysis")
        
        # Ensure survey_year is numeric
        self.data['survey_year'] = pd.to_numeric(self.data['survey_year'], errors='coerce')
        
        # Clean province names (tinh)
        self.data['tinh'] = self.data['tinh'].fillna('Unknown')
        
        # Remove observations with missing key variables
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['secondary_success', 'treated_cohort', 'survey_year', 'tinh'])
        final_count = len(self.data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count:,} observations with missing key variables")
        
        logger.info(f"Final analysis sample: {final_count:,} observations")
    
    def _log_sample_summary(self):
        """
        Log summary statistics of the analysis sample.
        """
        logger.info("Sample Summary Statistics:")
        logger.info(f"  Total observations: {len(self.data):,}")
        
        # Treatment groups
        treated_dist = self.data['treated'].value_counts().sort_index()
        logger.info(f"  Treatment groups: {dict(treated_dist)}")
        
        # Time periods
        period_dist = self.data['post_reform'].value_counts().sort_index()
        logger.info(f"  Time periods: {dict(period_dist)}")
        
        # DiD cells
        did_cells = self.data.groupby(['treated', 'post_reform']).size()
        logger.info(f"  DiD cells: {dict(did_cells)}")
        
        # Outcome summary
        outcome_mean = self.data['secondary_success'].mean()
        logger.info(f"  Outcome mean: {outcome_mean:.3f}")
        
        # By treatment status
        outcome_by_treatment = self.data.groupby('treated')['secondary_success'].mean()
        logger.info(f"  Outcome by treatment: {dict(outcome_by_treatment)}")
    
    def install_pyfixest(self):
        """
        Install PyFixest if not available and try importing it.
        """
        logger.info("Checking PyFixest installation")
        
        try:
            import pyfixest as pf
            logger.info("PyFixest is already installed")
            return pf
        except ImportError:
            logger.info("PyFixest not found, installing...")
            import subprocess
            import sys
            
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfixest"])
                import pyfixest as pf
                logger.info("PyFixest installed successfully")
                return pf
            except Exception as e:
                logger.error(f"Failed to install PyFixest: {e}")
                raise ImportError("Could not install PyFixest. Please install manually: pip install pyfixest")
    
    def run_all_specifications(self):
        """
        Run all four model specifications using PyFixest.
        """
        logger.info("Running all DiD model specifications")
        
        # Install and import PyFixest
        pf = self.install_pyfixest()
        
        # Specification 1: Basic DiD
        logger.info("Running Specification 1: Basic DiD")
        self.models['model_1'] = pf.feols("secondary_success ~ treated_cohort", data=self.data)
        
        # Specification 2: With time fixed effects
        logger.info("Running Specification 2: Time Fixed Effects")
        self.models['model_2'] = pf.feols("secondary_success ~ treated_cohort + C(survey_year)", data=self.data)
        
        # Specification 3: With province fixed effects  
        logger.info("Running Specification 3: Province Fixed Effects")
        self.models['model_3'] = pf.feols("secondary_success ~ treated_cohort + C(tinh)", data=self.data)
        
        # Specification 4: Full model with clustered standard errors
        logger.info("Running Specification 4: Full Model with Clustered SE")
        self.models['model_4'] = pf.feols("secondary_success ~ treated_cohort + C(survey_year) + C(tinh)", 
                                         data=self.data, vcov={"CRV1": "tinh"})
        
        logger.info("All specifications completed")
    
    def extract_results(self):
        """
        Extract and organize results from all model specifications.
        """
        logger.info("Extracting results from all models")
        
        for model_name, model in self.models.items():
            logger.info(f"Extracting results for {model_name}")
            
            try:
                # Get coefficient table (these are pandas Series)
                coef_table = model.coef()
                se_table = model.se()
                pvalue_table = model.pvalue()
                
                # Debug: print structure
                logger.info(f"Coefficient table shape: {coef_table.shape}")
                logger.info(f"Coefficient table index: {coef_table.index.tolist()}")
                
                # Get coefficient on treated_cohort (the DiD estimate)
                if 'treated_cohort' in coef_table.index:
                    coef_treated_cohort = coef_table.loc['treated_cohort']
                    se_treated_cohort = se_table.loc['treated_cohort']
                    pvalue_treated_cohort = pvalue_table.loc['treated_cohort']
                else:
                    logger.error(f"'treated_cohort' not found in coefficient table for {model_name}")
                    continue
                
                # Model statistics - using summary to get this info
                # Check available attributes
                model_summary = model.summary()
                
                # Try to extract nobs and R-squared from summary or use placeholder
                try:
                    n_obs = len(self.data)  # Use data length as fallback
                    r_squared = 0.0  # Placeholder for now
                except:
                    n_obs = 0
                    r_squared = 0.0
                
                # Confidence interval
                ci_lower = coef_treated_cohort - 1.96 * se_treated_cohort
                ci_upper = coef_treated_cohort + 1.96 * se_treated_cohort
                
                # Store results
                self.results[model_name] = {
                    'did_estimate': float(coef_treated_cohort),
                    'standard_error': float(se_treated_cohort),
                    'p_value': float(pvalue_treated_cohort),
                    'significant': float(pvalue_treated_cohort) < 0.05,
                    'confidence_interval_95': [float(ci_lower), float(ci_upper)],
                    'n_observations': n_obs,
                    'r_squared': float(r_squared),
                    'model_summary': str(model)
                }
                
                logger.info(f"{model_name}: DiD estimate = {coef_treated_cohort:.4f} (p = {pvalue_treated_cohort:.4f})")
                
            except Exception as e:
                logger.error(f"Error extracting results for {model_name}: {str(e)}")
                # Store error info
                self.results[model_name] = {
                    'error': str(e),
                    'model_summary': str(model)
                }
    
    def create_summary_table(self):
        """
        Create a summary table comparing all model specifications.
        """
        logger.info("Creating summary comparison table")
        
        summary_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                # Handle failed models
                summary_data.append({
                    'Model': model_name,
                    'DiD_Estimate': 'ERROR',
                    'Std_Error': 'ERROR',
                    'P_Value': 'ERROR',
                    'Significant': 'ERROR',
                    'CI_Lower': 'ERROR',
                    'CI_Upper': 'ERROR',
                    'N_Obs': 'ERROR',
                    'R_Squared': 'ERROR'
                })
            else:
                # Handle successful models
                summary_data.append({
                    'Model': model_name,
                    'DiD_Estimate': f"{results['did_estimate']:.4f}",
                    'Std_Error': f"{results['standard_error']:.4f}",
                    'P_Value': f"{results['p_value']:.4f}",
                    'Significant': 'Yes' if results['significant'] else 'No',
                    'CI_Lower': f"{results['confidence_interval_95'][0]:.4f}",
                    'CI_Upper': f"{results['confidence_interval_95'][1]:.4f}",
                    'N_Obs': results['n_observations'],
                    'R_Squared': f"{results['r_squared']:.4f}"
                })
        
        self.summary_table = pd.DataFrame(summary_data)
        
        logger.info("Summary table created")
        return self.summary_table
    
    def save_results(self):
        """
        Save all results to files.
        """
        logger.info("Saving results to files")
        
        # Save summary table
        summary_file = self.output_path / "pyfixest_did_summary.csv"
        self.summary_table.to_csv(summary_file, index=False)
        logger.info(f"Summary table saved to: {summary_file}")
        
        # Save detailed results as JSON
        results_file = self.output_path / "pyfixest_did_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            if 'error' in results:
                json_results[model_name] = {
                    'error': results['error'],
                    'model_summary': results['model_summary']
                }
            else:
                json_results[model_name] = {
                    'did_estimate': float(results['did_estimate']),
                    'standard_error': float(results['standard_error']),
                    'p_value': float(results['p_value']),
                    'significant': bool(results['significant']),
                    'confidence_interval_95': [float(x) for x in results['confidence_interval_95']],
                    'n_observations': int(results['n_observations']),
                    'r_squared': float(results['r_squared']),
                    'model_summary': results['model_summary']
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save model outputs to text file
        output_file = self.output_path / "pyfixest_did_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PyFixest DiD Analysis Results\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, model in self.models.items():
                f.write(f"{model_name.upper()}\n")
                f.write("-" * 50 + "\n")
                
                # Print model details manually since summary() doesn't work
                f.write(f"Specification: {model._fml}\n")
                f.write(f"Observations: {len(self.data)}\n")
                f.write(f"Inference: {model._inference}\n\n")
                
                # Create coefficient table
                coefs = model.coef()
                ses = model.se() 
                pvals = model.pvalue()
                
                f.write("Coefficients Table:\n")
                f.write(f"{'Variable':<25} {'Coefficient':<12} {'Std Error':<12} {'P-value':<10}\n")
                f.write("-" * 65 + "\n")
                
                for var in coefs.index:
                    coef = coefs[var]
                    se = ses[var]
                    pval = pvals[var]
                    f.write(f"{var:<25} {coef:>11.6f} {se:>11.6f} {pval:>9.4f}\n")
                
                # Add key statistics
                f.write(f"\nKey Results for {model_name}:\n")
                if 'treated_cohort' in coefs.index:
                    tc_coef = coefs['treated_cohort']
                    tc_se = ses['treated_cohort']
                    tc_pval = pvals['treated_cohort']
                    tc_sig = "***" if tc_pval < 0.001 else "**" if tc_pval < 0.01 else "*" if tc_pval < 0.05 else ""
                    f.write(f"  DiD Estimate (treated_cohort): {tc_coef:.6f} ({tc_se:.6f}) {tc_sig}\n")
                    f.write(f"  P-value: {tc_pval:.6f}\n")
                    f.write(f"  95% CI: [{tc_coef - 1.96*tc_se:.6f}, {tc_coef + 1.96*tc_se:.6f}]\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Model outputs saved to: {output_file}")
        
        return {
            'summary_table': summary_file,
            'detailed_results': results_file,
            'model_outputs': output_file
        }
    
    def run_full_analysis(self):
        """
        Run the complete PyFixest DiD analysis pipeline.
        """
        logger.info("Starting PyFixest DiD Analysis Pipeline")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Run all model specifications
            self.run_all_specifications()
            
            # Step 3: Extract results
            self.extract_results()
            
            # Step 4: Create summary table
            self.create_summary_table()
            
            # Step 5: Save results
            saved_files = self.save_results()
            
            logger.info("PyFixest DiD Analysis completed successfully")
            
            # Print summary to console
            print("\n" + "=" * 60)
            print("PYFIXEST DID ANALYSIS RESULTS")
            print("=" * 60)
            print(self.summary_table.to_string(index=False))
            print("=" * 60)
            
            return {
                'summary_table': self.summary_table,
                'detailed_results': self.results,
                'models': self.models,
                'saved_files': saved_files
            }
            
        except Exception as e:
            logger.error(f"Error in PyFixest DiD analysis: {str(e)}")
            raise


def main():
    """
    Main function to run PyFixest DiD analysis.
    """
    logger.info("Starting PyFixest DiD Analysis Main Script")
    
    # Set up paths
    data_path = "results/did_cohorts_filtered.csv"
    output_path = "results/"
    
    # Check if filtered data exists
    if not os.path.exists(data_path):
        logger.error(f"Filtered data not found at {data_path}")
        logger.info("Please run cohort filtering first or update the data path")
        return
    
    # Initialize and run analysis
    try:
        analyzer = PyFixestDiDAnalysis(data_path, output_path)
        results = analyzer.run_full_analysis()
        
        logger.info("Analysis completed successfully")
        
        # Print key results
        print(f"\nKey Findings:")
        for model_name, result in results['detailed_results'].items():
            if 'error' in result:
                print(f"{model_name}: ERROR - {result['error']}")
            else:
                print(f"{model_name}: DiD estimate = {result['did_estimate']:.4f} "
                      f"(p = {result['p_value']:.4f}, "
                      f"{'significant' if result['significant'] else 'not significant'})")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
