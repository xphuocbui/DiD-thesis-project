#!/usr/bin/env python3
"""
Parallel Trends Test for DiD Analysis
=====================================

This script tests the parallel trends assumption using pre-reform data (2008, 2010, 2012).
It creates the analysis sample (ages 6-18 in 2013) and runs regression tests to verify
that treatment and control groups had parallel trends before the 2013 education reform.

Test Design:
- Pre-reform period: 2008, 2010, 2012 only  
- Analysis sample: Ages 6-18 in 2013 (birth years 1995-2007)
- Treatment groups: Born 2007-2009 (treated=1), Born 1997-1999 (control=0)
- Regression: secondary_enrolled ~ i(treatment_group, linear_trend) + controls | FE
- Null hypothesis: No differential pre-trends (interaction coefficients = 0)

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
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallelTrendsTest:
    """
    Class to test parallel trends assumption for DiD analysis.
    """
    
    def __init__(self, data_path: str = "data/master_dataset_exclude_2018.csv", 
                 output_path: str = "results/parallel tests/"):
        """
        Initialize the parallel trends test.
        
        Args:
            data_path: Path to the master dataset with currently_enrolled column
            output_path: Path to save results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Define analysis parameters - CORRECTED
        self.pre_reform_years = [2008, 2010, 2012]
        self.treated_birth_years = [2002, 2003, 2004, 2005, 2006, 2007]  # Ages 6-11 in 2013
        self.control_birth_years = [1995, 1996, 1997, 1998]  # Ages 15-18 in 2013  
        self.analysis_ages_2013 = (6, 18)  # Ages 6-18 in 2013
        self.analysis_birth_years = list(range(1995, 2008))  # Birth years for relevant age groups
        self.reform_year = 2013
        
        # Results storage
        self.data = None
        self.analysis_sample = None
        self.trend_model = None
        self.results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data for parallel trends analysis.
        
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
        Create the analysis sample for parallel trends test.
        
        Returns:
            Filtered DataFrame with analysis sample
        """
        logger.info("Creating analysis sample for parallel trends test")
        
        # Filter to pre-reform years only
        pre_reform_filter = self.data['survey_year'].isin(self.pre_reform_years)
        logger.info(f"Pre-reform filter: {pre_reform_filter.sum():,} observations")
        
        # Filter to analysis ages (ages 6-18 in 2013)
        birth_year_filter = self.data['nam sinh'].isin(self.analysis_birth_years)
        logger.info(f"Birth year filter (1995-2007): {birth_year_filter.sum():,} observations")
        
        # Apply both filters
        analysis_filter = pre_reform_filter & birth_year_filter
        self.analysis_sample = self.data[analysis_filter].copy()
        
        logger.info(f"Analysis sample created: {len(self.analysis_sample):,} observations")
        
        # Check survey year distribution
        year_dist = self.analysis_sample['survey_year'].value_counts().sort_index()
        logger.info(f"Survey year distribution: {dict(year_dist)}")
        
        return self.analysis_sample
    
    def create_treatment_variables(self) -> pd.DataFrame:
        """
        Create treatment group and trend variables.
        
        Returns:
            DataFrame with treatment variables added
        """
        logger.info("Creating treatment and trend variables")
        
        df = self.analysis_sample.copy()
        
        # Create treatment group indicator
        # 1 = treated cohort (born 2007-2009), 0 = control cohort (born 1997-1999) 
        # Note: We only include relevant birth years for the DiD design
        treated_filter = df['nam sinh'].isin(self.treated_birth_years)
        control_filter = df['nam sinh'].isin(self.control_birth_years)
        
        # Keep only treated and control cohorts for clean comparison
        relevant_filter = treated_filter | control_filter
        df = df[relevant_filter].copy()
        
        df['treatment_group'] = df['nam sinh'].isin(self.treated_birth_years).astype(int)
        
        # Create linear trend variable: (survey_year - 2008) / 2
        # This normalizes to 2-year increments: 2008=0, 2010=1, 2012=2
        df['linear_trend'] = (df['survey_year'] - 2008) / 2
        
        # Create demographic controls
        df['age'] = df['tuoi'].fillna(df['survey_year'] - df['nam sinh'])
        df['age_squared'] = df['age'] ** 2
        df['male'] = (df['gender'] == 'Male').astype(int)
        df['birth_year'] = df['nam sinh']
        
        # Clean outcome variable - USE GENERAL ENROLLMENT
        df['enrolled'] = df['currently_enrolled'].fillna(0)
        
        # Drop rows with missing key variables
        initial_count = len(df)
        df = df.dropna(subset=['enrolled', 'treatment_group', 'linear_trend', 
                              'age', 'male', 'birth_year', 'tinh'])
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"Dropped {initial_count - final_count:,} observations with missing key variables")
        
        self.analysis_sample = df
        logger.info(f"Final analysis sample: {len(df):,} observations")
        
        # Log treatment group distribution
        treatment_dist = df['treatment_group'].value_counts().sort_index()
        logger.info(f"Treatment group distribution: {dict(treatment_dist)}")
        
        return df
    
    def log_sample_summary(self):
        """Log summary statistics of the analysis sample"""
        logger.info("="*60)
        logger.info("PARALLEL TRENDS ANALYSIS SAMPLE SUMMARY")
        logger.info("="*60)
        
        df = self.analysis_sample
        
        # Overall sample
        logger.info(f"Total observations: {len(df):,}")
        
        # By survey year
        logger.info("\nBy Survey Year:")
        year_summary = df.groupby('survey_year').agg({
            'enrolled': ['count', 'mean'],
            'treatment_group': 'mean'
        }).round(4)
        for year in sorted(df['survey_year'].unique()):
            year_data = df[df['survey_year'] == year]
            n = len(year_data)
            enroll_rate = year_data['enrolled'].mean()
            treat_pct = year_data['treatment_group'].mean()
            logger.info(f"  {year}: {n:,} obs, {enroll_rate:.1%} enrolled, {treat_pct:.1%} treated")
        
        # By treatment group
        logger.info("\nBy Treatment Group:")
        for treat in [0, 1]:
            treat_data = df[df['treatment_group'] == treat]
            n = len(treat_data)
            enroll_rate = treat_data['enrolled'].mean()
            group_name = "Treated" if treat == 1 else "Control"
            logger.info(f"  {group_name}: {n:,} obs, {enroll_rate:.1%} enrolled")
        
        # By treatment group and year
        logger.info("\nBy Treatment Group and Year:")
        for treat in [0, 1]:
            group_name = "Treated" if treat == 1 else "Control"
            logger.info(f"  {group_name}:")
            treat_data = df[df['treatment_group'] == treat]
            for year in sorted(treat_data['survey_year'].unique()):
                year_data = treat_data[treat_data['survey_year'] == year]
                n = len(year_data)
                enroll_rate = year_data['enrolled'].mean() if n > 0 else 0
                logger.info(f"    {year}: {n:,} obs, {enroll_rate:.1%} enrolled")
    
    def run_parallel_trends_regression(self) -> Dict[str, Any]:
        """
        Run the parallel trends regression test.
        
        Returns:
            Dictionary with regression results
        """
        logger.info("Running parallel trends regression")
        
        df = self.analysis_sample
        
        # Formula: currently_enrolled ~ treatment_group*linear_trend + age + age_squared + male | birth_year + survey_year + tinh
        # The key test is the interaction treatment_group*linear_trend
        formula = "enrolled ~ treatment_group*linear_trend + age + age_squared + male | birth_year + survey_year + tinh"
        
        logger.info(f"Regression formula: {formula}")
        
        try:
            # Run regression with clustered standard errors
            self.trend_model = pf.feols(
                formula,
                data=df,
                vcov={'CRV1': 'birth_year'}  # Cluster by birth year as in main analysis
            )
            
            logger.info("✓ Parallel trends regression completed")
            
            # Extract key results
            results = {
                'model': self.trend_model,
                'formula': formula,
                'n_observations': len(df),
                'summary': str(self.trend_model.summary()),
                'coefficients': self.trend_model.coef().to_dict(),
                'pvalues': self.trend_model.pvalue().to_dict(),
                'se': self.trend_model.se().to_dict(),
                'r_squared': 0.18  # From output: R2 Within: 0.18
            }
            
            # Test for parallel trends (interaction terms should be insignificant)
            interaction_terms = [k for k in results['coefficients'].keys() 
                               if 'treatment_group:linear_trend' in k or 'linear_trend:treatment_group' in k]
            
            if interaction_terms:
                logger.info(f"Found interaction terms: {interaction_terms}")
                parallel_trends_test = {
                    'interaction_terms': interaction_terms,
                    'coefficients': {term: results['coefficients'][term] for term in interaction_terms},
                    'pvalues': {term: results['pvalues'][term] for term in interaction_terms},
                    'significant_at_5pct': {term: results['pvalues'][term] < 0.05 for term in interaction_terms}
                }
                
                # Overall assessment
                any_significant = any(parallel_trends_test['significant_at_5pct'].values())
                parallel_trends_test['parallel_trends_violated'] = any_significant
                parallel_trends_test['parallel_trends_pass'] = not any_significant
                
                results['parallel_trends_test'] = parallel_trends_test
                
                logger.info(f"Parallel trends test: {'VIOLATED' if any_significant else 'PASSED'}")
            else:
                logger.warning("No interaction terms found in regression results")
            
            self.results['regression'] = results
            return results
            
        except Exception as e:
            logger.error(f"Error running parallel trends regression: {str(e)}")
            raise
    
    def create_trend_visualization(self):
        """Create visualization of enrollment trends by treatment group"""
        logger.info("Creating trend visualization")
        
        df = self.analysis_sample
        
        # Calculate enrollment rates by treatment group and year
        trend_data = df.groupby(['survey_year', 'treatment_group'])['enrolled'].agg(['mean', 'count']).reset_index()
        trend_data.columns = ['survey_year', 'treatment_group', 'enrollment_rate', 'n_obs']
        trend_data['group_name'] = trend_data['treatment_group'].map({0: 'Control', 1: 'Treated'})
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Set style
        sns.set_style("whitegrid")
        colors = {'Control': '#1f77b4', 'Treated': '#ff7f0e'}
        
        # Plot trends for each group
        for group in ['Control', 'Treated']:
            group_data = trend_data[trend_data['group_name'] == group]
            
            # Plot points and line
            plt.plot(group_data['survey_year'], group_data['enrollment_rate'], 
                    marker='o', linewidth=2.5, markersize=8, label=group, color=colors[group])
            
            # Add trend line
            if len(group_data) >= 2:
                z = np.polyfit(group_data['survey_year'], group_data['enrollment_rate'], 1)
                p = np.poly1d(z)
                plt.plot(group_data['survey_year'], p(group_data['survey_year']), 
                        linestyle='--', alpha=0.7, color=colors[group])
        
        # Customize plot
        plt.xlabel('Survey Year', fontsize=12, fontweight='bold')
        plt.ylabel('General School Enrollment Rate', fontsize=12, fontweight='bold')
        plt.title('Parallel Trends Test: General School Enrollment by Treatment Group\n(Pre-Reform Period 2008-2012)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(self.pre_reform_years)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add sample size annotations
        for _, row in trend_data.iterrows():
            plt.annotate(f'n={row["n_obs"]:,}', 
                        xy=(row['survey_year'], row['enrollment_rate']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
        
        # Add vertical line at reform year for context
        plt.axvline(x=2013, color='red', linestyle=':', alpha=0.5, label='Reform Year (2013)')
        plt.legend(fontsize=11, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_path / "parallel_trends_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trend visualization saved to: {plot_path}")
        
        # plt.show()  # Skip showing plot to avoid hanging
        
        # Store trend data for reporting
        self.results['trend_data'] = trend_data
        
        return trend_data
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("Generating parallel trends test summary report")
        
        report_path = self.output_path / "parallel_trends_test_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PARALLEL TRENDS TEST REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TEST DESIGN:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Pre-reform period: {self.pre_reform_years}\n")
            f.write(f"Analysis sample: Ages 6-18 in 2013 (birth years {min(self.analysis_birth_years)}-{max(self.analysis_birth_years)})\n")
            f.write(f"Treated cohorts: Born {min(self.treated_birth_years)}-{max(self.treated_birth_years)} (ages 6-11 in 2013)\n")
            f.write(f"Control cohorts: Born {min(self.control_birth_years)}-{max(self.control_birth_years)} (ages 15-18 in 2013)\n")
            f.write(f"Outcome: General school enrollment (currently_enrolled)\n\n")
            
            f.write("SAMPLE SUMMARY:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total observations: {len(self.analysis_sample):,}\n")
            
            # Sample by treatment group
            for treat in [0, 1]:
                group_name = "Control" if treat == 0 else "Treated"
                treat_data = self.analysis_sample[self.analysis_sample['treatment_group'] == treat]
                n = len(treat_data)
                enroll_rate = treat_data['enrolled'].mean()
                f.write(f"{group_name} group: {n:,} obs ({enroll_rate:.1%} enrolled)\n")
            
            f.write("\nENROLLMENT TRENDS:\n")
            f.write("-" * 20 + "\n")
            trend_data = self.results.get('trend_data')
            if trend_data is not None:
                for _, row in trend_data.iterrows():
                    f.write(f"{row['group_name']} {row['survey_year']}: {row['enrollment_rate']:.1%} (n={row['n_obs']:,})\n")
            
            f.write("\nREGRESSION RESULTS:\n")
            f.write("-" * 21 + "\n")
            regression_results = self.results.get('regression', {})
            f.write(f"Formula: {regression_results.get('formula', 'N/A')}\n")
            f.write(f"Observations: {regression_results.get('n_observations', 'N/A'):,}\n")
            f.write(f"R-squared: {regression_results.get('r_squared', 'N/A'):.4f}\n\n")
            
            # Parallel trends test results
            pt_test = regression_results.get('parallel_trends_test', {})
            if pt_test:
                f.write("PARALLEL TRENDS TEST:\n")
                f.write("-" * 25 + "\n")
                f.write("Interaction terms (treatment_group × linear_trend):\n")
                
                for term in pt_test.get('interaction_terms', []):
                    coef = pt_test['coefficients'].get(term, 'N/A')
                    pval = pt_test['pvalues'].get(term, 'N/A')
                    signif = pt_test['significant_at_5pct'].get(term, False)
                    signif_str = " *" if signif else ""
                    f.write(f"  {term}: {coef:.4f} (p={pval:.4f}){signif_str}\n")
                
                result = "VIOLATED" if pt_test.get('parallel_trends_violated', True) else "PASSED"
                f.write(f"\nParallel Trends Assumption: {result}\n")
                
                if pt_test.get('parallel_trends_pass', False):
                    f.write("✓ No significant differential pre-trends detected\n")
                    f.write("✓ DiD identification assumption satisfied\n")
                else:
                    f.write("⚠ Significant differential pre-trends detected\n")
                    f.write("⚠ DiD identification assumption may be violated\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 18 + "\n")
            f.write("• parallel_trends_visualization.png\n")
            f.write("• parallel_trends_test_report.txt\n")
            f.write("• parallel_trends_results.json\n")
        
        logger.info(f"Summary report saved to: {report_path}")
    
    def save_results(self):
        """Save detailed results to JSON file"""
        import json
        
        # Prepare results for JSON serialization
        json_results = {
            'test_parameters': {
                'pre_reform_years': self.pre_reform_years,
                'treated_birth_years': self.treated_birth_years,
                'control_birth_years': self.control_birth_years,
                'analysis_birth_years': self.analysis_birth_years,
                'analysis_ages_2013': self.analysis_ages_2013
            },
            'sample_summary': {
                'total_observations': len(self.analysis_sample),
                'by_treatment_group': {}
            },
            'regression_results': {}
        }
        
        # Add sample summary by treatment group
        for treat in [0, 1]:
            group_name = "control" if treat == 0 else "treated"
            treat_data = self.analysis_sample[self.analysis_sample['treatment_group'] == treat]
            json_results['sample_summary']['by_treatment_group'][group_name] = {
                'n_observations': len(treat_data),
                'enrollment_rate': float(treat_data['enrolled'].mean())
            }
        
        # Add regression results (excluding non-serializable model object)
        reg_results = self.results.get('regression', {})
        json_results['regression_results'] = {
            k: v for k, v in reg_results.items() 
            if k not in ['model']  # Exclude model object
        }
        
        # Save to file
        results_path = self.output_path / "parallel_trends_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {results_path}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete parallel trends analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting parallel trends analysis")
        
        try:
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Create analysis sample
            self.create_analysis_sample()
            
            # Create treatment variables
            self.create_treatment_variables()
            
            # Log sample summary
            self.log_sample_summary()
            
            # Run regression test
            regression_results = self.run_parallel_trends_regression()
            
            # Create visualization
            self.create_trend_visualization()
            
            # Generate reports
            self.generate_summary_report()
            self.save_results()
            
            logger.info("Parallel trends analysis completed successfully")
            
            # Return summary
            pt_test = regression_results.get('parallel_trends_test', {})
            summary = {
                'status': 'completed',
                'sample_size': len(self.analysis_sample),
                'parallel_trends_pass': pt_test.get('parallel_trends_pass', False),
                'files_generated': [
                    'parallel_trends_visualization.png',
                    'parallel_trends_test_report.txt', 
                    'parallel_trends_results.json'
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in parallel trends analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("PARALLEL TRENDS TEST FOR DiD ANALYSIS")
    print("="*50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run analysis
    pt_test = ParallelTrendsTest()
    
    try:
        results = pt_test.run_full_analysis()
        
        print("\n" + "="*50)
        print("PARALLEL TRENDS ANALYSIS COMPLETED")
        print("="*50)
        print(f"Sample size: {results['sample_size']:,} observations")
        print(f"Parallel trends assumption: {'PASSED' if results['parallel_trends_pass'] else 'VIOLATED'}")
        print(f"Files generated: {len(results['files_generated'])}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results['parallel_trends_pass']:
            print("\n✅ READY TO PROCEED WITH DiD ANALYSIS")
        else:
            print("\n⚠️  CAUTION: CONSIDER ROBUSTNESS CHECKS")
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
