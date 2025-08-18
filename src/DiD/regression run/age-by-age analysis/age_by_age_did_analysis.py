"""
Age-by-Age Difference-in-Differences Analysis with Interaction Terms
==================================================================

This script implements an age-by-age DiD analysis using age-specific treatment 
indicators and their interactions with post-reform periods to examine 
heterogeneous effects of Vietnam's 2013 education reform.

Model Specification:
currently_enrolled ~ age_6_post + age_7_post + ... + age_18_post + male | birth_year + survey_year + tinh

Where:
- age_X_post = age_X_2013 * post_reform (DiD interaction for each age)
- age_X_2013 = 1 if individual was age X in 2013, 0 otherwise
- post_reform = 1 if survey year is after 2013, 0 otherwise

Author: DiD Analysis Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import os
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgeByAgeInteractionDiD:
    """
    Age-by-Age DiD analysis using interaction terms between age indicators and post-reform period.
    """
    
    def __init__(self, data_path: str = "data/master_dataset_exclude_2018.csv",
                 output_path: str = "results/age-by-age analysis/"):
        """
        Initialize the Age-by-Age DiD analysis.
        
        Args:
            data_path: Path to the master dataset
            output_path: Directory to save results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Define analysis parameters
        self.ages_to_analyze = list(range(6, 19))  # Ages 6-18 in 2013
        self.reform_year = 2013
        self.pre_reform_years = [2008, 2010, 2012]
        self.post_reform_years = [2014, 2016, 2020]
        self.all_analysis_years = self.pre_reform_years + self.post_reform_years
        
        # Results storage
        self.data = None
        self.analysis_sample = None
        self.model = None
        self.age_effects = {}
        self.results = {}
        
        logger.info(f"Age-by-Age Interaction DiD Analysis initialized")
        logger.info(f"Ages to analyze: {self.ages_to_analyze}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output path: {output_path}")
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for age-by-age DiD analysis.
        """
        logger.info("Loading and preparing data")
        
        try:
            # Load data
            self.data = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Loaded {len(self.data):,} observations from {self.data_path}")
            
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
    
    def create_analysis_sample(self):
        """
        Create the analysis sample for age-by-age DiD.
        """
        logger.info("Creating analysis sample for age-by-age DiD")
        
        # Start with the full dataset
        sample = self.data.copy()
        logger.info(f"Starting with {len(sample):,} observations")
        
        # Filter to analysis years
        sample = sample[sample['survey_year'].isin(self.all_analysis_years)]
        logger.info(f"After filtering to analysis years {self.all_analysis_years}: {len(sample):,} observations")
        
        # Create age_2013 variable (age that person was in 2013)
        sample['age_2013'] = sample['tuoi'] - (sample['survey_year'] - self.reform_year)
        logger.info("Created age_2013 variable")
        
        # Filter to relevant ages (6-18 in 2013)
        sample = sample[sample['age_2013'].isin(self.ages_to_analyze)]
        logger.info(f"After filtering to ages {min(self.ages_to_analyze)}-{max(self.ages_to_analyze)} in 2013: {len(sample):,} observations")
        
        # Remove observations with missing enrollment data
        sample = sample.dropna(subset=['currently_enrolled'])
        logger.info(f"After removing missing enrollment: {len(sample):,} observations")
        
        # Remove observations with missing control variables
        sample = sample.dropna(subset=['tuoi', 'gender', 'tinh', 'nam sinh'])
        logger.info(f"After removing missing controls: {len(sample):,} observations")
        
        # Check age distribution in 2013
        age_2013_dist = sample['age_2013'].value_counts().sort_index()
        logger.info("Age distribution in 2013:")
        for age, count in age_2013_dist.items():
            logger.info(f"  Age {age}: {count:,} observations")
        
        self.analysis_sample = sample
        logger.info("✓ Analysis sample created successfully")
        
        return self.analysis_sample
    
    def create_age_treatment_variables(self):
        """
        Create age-specific treatment indicators and interaction terms.
        """
        logger.info("Creating age-specific treatment indicators and interactions")
        
        sample = self.analysis_sample.copy()
        
        # Create post-reform indicator
        sample['post_reform'] = sample['survey_year'].apply(
            lambda x: 1 if x in self.post_reform_years else 0
        )
        
        logger.info("Created post_reform indicator")
        logger.info(f"Post-reform distribution: {sample['post_reform'].value_counts().to_dict()}")
        
        # Create age-specific indicators for 2013
        for age in self.ages_to_analyze:
            age_indicator = f'age_{age}_2013'
            sample[age_indicator] = (sample['age_2013'] == age).astype(int)
            logger.info(f"Created {age_indicator}: {sample[age_indicator].sum():,} observations")
        
        # Create age-specific interaction terms (age_X_2013 * post_reform)
        for age in self.ages_to_analyze:
            age_post_var = f'age_{age}_post'
            age_indicator = f'age_{age}_2013'
            sample[age_post_var] = sample[age_indicator] * sample['post_reform']
            logger.info(f"Created {age_post_var}: {sample[age_post_var].sum():,} observations")
        
        # Create additional control variables
        sample['male'] = (sample['gender'] == 'Male').astype(int)
        sample['birth_year'] = sample['nam sinh']
        
        # Log final sample summary
        logger.info(f"Final sample summary:")
        logger.info(f"  Total observations: {len(sample):,}")
        logger.info(f"  Pre-reform: {sum(sample['post_reform'] == 0):,}")
        logger.info(f"  Post-reform: {sum(sample['post_reform'] == 1):,}")
        logger.info(f"  Male: {sample['male'].sum():,}")
        logger.info(f"  Survey years: {sorted(sample['survey_year'].unique())}")
        
        self.analysis_sample = sample
        return self.analysis_sample
    
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
    
    def run_age_by_age_regression(self):
        """
        Run the age-by-age DiD regression using interaction terms.
        """
        logger.info("Running age-by-age DiD regression with interaction terms")
        
        # Install and import PyFixest
        pf = self.install_pyfixest()
        
        # Build the formula with age interaction terms
        age_interaction_terms = [f'age_{age}_post' for age in self.ages_to_analyze]
        age_terms_str = " + ".join(age_interaction_terms)
        
        # Formula: currently_enrolled ~ age_6_post + age_7_post + ... + age_18_post + male | birth_year + survey_year + tinh
        formula = f"currently_enrolled ~ {age_terms_str} + male | birth_year + survey_year + tinh"
        
        logger.info(f"Regression formula: {formula}")
        logger.info(f"Sample size: {len(self.analysis_sample):,} observations")
        
        # Run regression with clustered standard errors by province
        try:
            self.model = pf.feols(
                formula,
                data=self.analysis_sample,
                vcov={'CRV1': 'tinh'}  # Cluster by province
            )
            
            logger.info("✓ Age-by-age DiD regression completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error running regression: {str(e)}")
            raise
    
    def extract_age_effects(self):
        """
        Extract age-specific treatment effects from the regression.
        """
        logger.info("Extracting age-specific treatment effects")
        
        try:
            # Get coefficient table
            coef_table = self.model.coef()
            se_table = self.model.se()
            pvalue_table = self.model.pvalue()
            
            # Extract effects for each age
            for age in self.ages_to_analyze:
                age_post_var = f'age_{age}_post'
                
                if age_post_var in coef_table.index:
                    coef = coef_table.loc[age_post_var]
                    se = se_table.loc[age_post_var]
                    pval = pvalue_table.loc[age_post_var]
                    
                    # Calculate confidence intervals
                    ci_lower = coef - 1.96 * se
                    ci_upper = coef + 1.96 * se
                    
                    self.age_effects[age] = {
                        'coefficient': float(coef),
                        'standard_error': float(se),
                        'p_value': float(pval),
                        'significant_5pct': float(pval) < 0.05,
                        'significant_10pct': float(pval) < 0.10,
                        'ci_lower': float(ci_lower),
                        'ci_upper': float(ci_upper),
                        'percentage_effect': float(coef * 100)  # Convert to percentage points
                    }
                    
                    logger.info(f"Age {age}: coef = {coef:.4f} (p = {pval:.4f}, {coef*100:.2f} p.p.)")
                else:
                    logger.warning(f"Age {age} interaction term not found in regression results")
            
            logger.info(f"Extracted effects for {len(self.age_effects)} ages")
            return self.age_effects
            
        except Exception as e:
            logger.error(f"Error extracting age effects: {str(e)}")
            raise
    
    def create_age_effects_dataframe(self):
        """
        Create a DataFrame with age effects for analysis and visualization.
        """
        logger.info("Creating age effects DataFrame")
        
        age_effects_data = []
        for age in sorted(self.age_effects.keys()):
            effects = self.age_effects[age]
            age_effects_data.append({
                'age_in_2013': age,
                'coefficient': effects['coefficient'],
                'standard_error': effects['standard_error'],
                'p_value': effects['p_value'],
                'significant_5pct': effects['significant_5pct'],
                'significant_10pct': effects['significant_10pct'],
                'ci_lower': effects['ci_lower'],
                'ci_upper': effects['ci_upper'],
                'percentage_effect': effects['percentage_effect']
            })
        
        self.age_effects_df = pd.DataFrame(age_effects_data)
        logger.info("Age effects DataFrame created")
        
        return self.age_effects_df
    
    def create_visualization(self):
        """
        Create visualization of age-specific treatment effects.
        """
        logger.info("Creating age-specific treatment effects visualization")
        
        # Set up the plot
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract data for plotting
        ages = self.age_effects_df['age_in_2013']
        coefficients = self.age_effects_df['coefficient']
        ci_lower = self.age_effects_df['ci_lower']
        ci_upper = self.age_effects_df['ci_upper']
        significant_5pct = self.age_effects_df['significant_5pct']
        
        # Create colors based on significance
        colors = ['#1f77b4' if sig else '#ff7f0e' for sig in significant_5pct]
        
        # Plot coefficients with different markers for significance
        for i, (age, coef, sig, color) in enumerate(zip(ages, coefficients, significant_5pct, colors)):
            marker = 'o' if sig else 's'
            markersize = 8 if sig else 6
            alpha = 1.0 if sig else 0.6
            
            ax.scatter(age, coef, color=color, marker=marker, s=markersize**2,
                      alpha=alpha, edgecolor='black', linewidth=0.5, zorder=3)
        
        # Plot confidence intervals
        for age, coef, ci_l, ci_u, color in zip(ages, coefficients, ci_lower, ci_upper, colors):
            ax.plot([age, age], [ci_l, ci_u], color=color, linewidth=2, alpha=0.7, zorder=2)
            ax.plot([age-0.1, age+0.1], [ci_l, ci_l], color=color, linewidth=2, alpha=0.7, zorder=2)
            ax.plot([age-0.1, age+0.1], [ci_u, ci_u], color=color, linewidth=2, alpha=0.7, zorder=2)
        
        # Connect points with a line
        ax.plot(ages, coefficients, color='gray', linewidth=1, alpha=0.5, zorder=1)
        
        # Add horizontal line at zero (no effect)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Age in 2013 (Reform Year)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reform Effect on School Enrollment\n(Percentage Points)', fontsize=12, fontweight='bold')
        ax.set_title('Age-Specific Effects of Vietnam\'s 2013 Education Reform\n'
                    'on School Enrollment Rates', fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis ticks
        ax.set_xticks(ages)
        ax.set_xticklabels([f'{int(age)}' for age in ages])
        
        # Add grid
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None',
                   markersize=8, label='Significant (p < 0.05)', markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='s', color='#ff7f0e', linestyle='None',
                   markersize=6, label='Not Significant', markeredgecolor='black', markeredgewidth=0.5, alpha=0.6),
            Line2D([0], [0], color='black', linestyle='--', label='No Effect (Reference)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_path / "age_by_age_effects_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Age effects plot saved to: {plot_file}")
        
        # Show plot
        plt.show()
        
        return fig, ax
    
    def create_summary_statistics(self):
        """
        Create summary statistics and interpretation.
        """
        logger.info("Creating summary statistics")
        
        # Basic statistics
        stats = {
            'total_ages_analyzed': len(self.age_effects_df),
            'ages_analyzed': self.ages_to_analyze,
            'significant_ages_5pct': self.age_effects_df['significant_5pct'].sum(),
            'significant_ages_10pct': self.age_effects_df['significant_10pct'].sum(),
            'sample_size': len(self.analysis_sample),
            'pre_reform_obs': sum(self.analysis_sample['post_reform'] == 0),
            'post_reform_obs': sum(self.analysis_sample['post_reform'] == 1)
        }
        
        # Effect statistics
        stats['mean_effect'] = self.age_effects_df['coefficient'].mean()
        stats['mean_percentage_effect'] = self.age_effects_df['percentage_effect'].mean()
        stats['max_positive_effect'] = self.age_effects_df['coefficient'].max()
        stats['max_negative_effect'] = self.age_effects_df['coefficient'].min()
        
        # Most/least affected ages
        most_positive_idx = self.age_effects_df['coefficient'].idxmax()
        most_negative_idx = self.age_effects_df['coefficient'].idxmin()
        
        stats['most_positive_age'] = int(self.age_effects_df.loc[most_positive_idx, 'age_in_2013'])
        stats['most_positive_effect'] = self.age_effects_df.loc[most_positive_idx, 'coefficient']
        stats['most_negative_age'] = int(self.age_effects_df.loc[most_negative_idx, 'age_in_2013'])
        stats['most_negative_effect'] = self.age_effects_df.loc[most_negative_idx, 'coefficient']
        
        self.summary_stats = stats
        logger.info("Summary statistics created")
        
        return stats
    
    def save_results(self):
        """
        Save all results to files.
        """
        logger.info("Saving results to files")
        
        # Save age effects DataFrame
        age_effects_file = self.output_path / "age_by_age_effects.csv"
        self.age_effects_df.to_csv(age_effects_file, index=False)
        logger.info(f"Age effects saved to: {age_effects_file}")
        
        # Save detailed results as JSON
        results_file = self.output_path / "age_by_age_results.json"
        json_results = {
            'age_effects': self.age_effects,
            'summary_statistics': self.summary_stats,
            'model_info': {
                'formula': str(self.model._fml),
                'n_observations': len(self.analysis_sample),
                'ages_analyzed': self.ages_to_analyze,
                'reform_year': self.reform_year,
                'pre_reform_years': self.pre_reform_years,
                'post_reform_years': self.post_reform_years
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save comprehensive text report
        report_file = self.output_path / "age_by_age_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Age-by-Age DiD Analysis Results (Interaction Terms Method)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL SPECIFICATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Formula: {self.model._fml}\n")
            f.write(f"Method: Age-specific DiD interaction terms (age_X_post = age_X_2013 * post_reform)\n")
            f.write(f"Sample Size: {len(self.analysis_sample):,} observations\n")
            f.write(f"Clustered Standard Errors: By Province (tinh)\n")
            f.write(f"Ages Analyzed: {', '.join(map(str, self.ages_to_analyze))}\n")
            f.write(f"Reform Year: {self.reform_year}\n")
            f.write(f"Pre-reform Years: {', '.join(map(str, self.pre_reform_years))}\n")
            f.write(f"Post-reform Years: {', '.join(map(str, self.post_reform_years))}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Ages Analyzed: {self.summary_stats['total_ages_analyzed']}\n")
            f.write(f"Significant Effects (5%): {self.summary_stats['significant_ages_5pct']}\n")
            f.write(f"Significant Effects (10%): {self.summary_stats['significant_ages_10pct']}\n")
            f.write(f"Mean Effect: {self.summary_stats['mean_effect']:.4f} ({self.summary_stats['mean_percentage_effect']:.2f} p.p.)\n")
            f.write(f"Largest Positive Effect: Age {self.summary_stats['most_positive_age']} ({self.summary_stats['most_positive_effect']:.4f})\n")
            f.write(f"Largest Negative Effect: Age {self.summary_stats['most_negative_age']} ({self.summary_stats['most_negative_effect']:.4f})\n\n")
            
            f.write("DETAILED AGE EFFECTS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Age':<4} {'Coefficient':<12} {'Std Error':<10} {'P-Value':<10} {'Sig (5%)':<9} {'Sig (10%)':<10} {'95% CI':<20} {'Effect (p.p.)':<12}\n")
            f.write("-" * 95 + "\n")
            
            for _, row in self.age_effects_df.iterrows():
                age = int(row['age_in_2013'])
                coef = row['coefficient']
                se = row['standard_error']
                pval = row['p_value']
                sig5 = "Yes" if row['significant_5pct'] else "No"
                sig10 = "Yes" if row['significant_10pct'] else "No"
                ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                perc_effect = row['percentage_effect']
                
                f.write(f"{age:<4} {coef:<12.4f} {se:<10.4f} {pval:<10.4f} {sig5:<9} {sig10:<10} {ci:<20} {perc_effect:<12.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 15 + "\n")
            f.write("Each coefficient represents the DiD effect of the 2013 reform on school\n")
            f.write("enrollment for individuals who were that specific age in 2013.\n\n")
            f.write("The interaction terms (age_X_post) capture:\n")
            f.write("- age_X_post = age_X_2013 * post_reform\n")
            f.write("- The differential effect of the reform for each age group\n")
            f.write("- Positive coefficients = reform increased enrollment for that age\n")
            f.write("- Negative coefficients = reform decreased enrollment for that age\n\n")
            
            f.write("METHODOLOGICAL NOTES:\n")
            f.write("-" * 22 + "\n")
            f.write("- This analysis uses interaction terms between age indicators and post-reform period\n")
            f.write("- Fixed effects control for unobserved heterogeneity across birth years, survey years, and provinces\n")
            f.write("- Standard errors are clustered at the province level\n")
            f.write("- The outcome is currently_enrolled (general school enrollment)\n")
        
        logger.info(f"Comprehensive report saved to: {report_file}")
        
        return {
            'age_effects_csv': age_effects_file,
            'detailed_results_json': results_file,
            'analysis_report': report_file
        }
    
    def run_full_analysis(self):
        """
        Run the complete age-by-age DiD analysis pipeline.
        """
        logger.info("Starting Age-by-Age DiD Analysis Pipeline (Interaction Terms Method)")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Create analysis sample
            self.create_analysis_sample()
            
            # Step 3: Create age treatment variables and interactions
            self.create_age_treatment_variables()
            
            # Step 4: Run age-by-age regression
            self.run_age_by_age_regression()
            
            # Step 5: Extract age-specific effects
            self.extract_age_effects()
            
            # Step 6: Create age effects DataFrame
            self.create_age_effects_dataframe()
            
            # Step 7: Create summary statistics
            self.create_summary_statistics()
            
            # Step 8: Create visualization
            self.create_visualization()
            
            # Step 9: Save results
            saved_files = self.save_results()
            
            logger.info("Age-by-Age DiD Analysis completed successfully")
            
            # Print summary to console
            print("\n" + "=" * 80)
            print("AGE-BY-AGE DID ANALYSIS RESULTS (INTERACTION TERMS METHOD)")
            print("=" * 80)
            print(f"Sample Size: {len(self.analysis_sample):,} observations")
            print(f"Ages Analyzed: {', '.join(map(str, self.ages_to_analyze))}")
            print(f"Significant Effects (5%): {self.summary_stats['significant_ages_5pct']}/{self.summary_stats['total_ages_analyzed']}")
            print(f"Mean Effect: {self.summary_stats['mean_percentage_effect']:.2f} percentage points")
            print("=" * 80)
            print(self.age_effects_df.to_string(index=False, float_format='%.4f'))
            print("=" * 80)
            
            return {
                'age_effects_df': self.age_effects_df,
                'age_effects': self.age_effects,
                'summary_stats': self.summary_stats,
                'model': self.model,
                'sample': self.analysis_sample,
                'saved_files': saved_files
            }
            
        except Exception as e:
            logger.error(f"Error in Age-by-Age DiD analysis: {str(e)}")
            raise


def main():
    """
    Main function to run age-by-age DiD analysis.
    """
    logger.info("Starting Age-by-Age DiD Analysis Main Script (Interaction Terms Method)")
    
    # Set up paths
    data_path = "data/master_dataset_exclude_2018.csv"
    output_path = "results/age-by-age analysis/"
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        logger.info("Please check the data path")
        return
    
    # Initialize and run analysis
    try:
        analyzer = AgeByAgeInteractionDiD(data_path, output_path)
        results = analyzer.run_full_analysis()
        
        logger.info("Analysis completed successfully!")
        
        # Print key findings
        print(f"\nKEY FINDINGS:")
        print(f"Most positive effect: Age {results['summary_stats']['most_positive_age']} "
              f"({results['summary_stats']['most_positive_effect']:.4f})")
        print(f"Most negative effect: Age {results['summary_stats']['most_negative_age']} "
              f"({results['summary_stats']['most_negative_effect']:.4f})")
        print(f"Significant effects: {results['summary_stats']['significant_ages_5pct']} out of {results['summary_stats']['total_ages_analyzed']} ages")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
