"""
DiD Cohort Filtering Script
===========================

This script filters the VHLSS dataset to create treated and control cohorts for 
cross-section Difference-in-Differences analysis of Vietnam's Education Reform 29.

Treatment Definition:
- Reform passed: November 2013
- Treated cohorts: Born 2007-2009 (ages 4-6 in 2013) - experienced reformed primary education
- Control cohorts: Born 1997-1999 (ages 14-16 in 2013) - experienced old primary education

Author: DiD Analysis Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CohortFilter:
    """
    Class to handle cohort filtering for DiD analysis.
    """
    
    def __init__(self, data_path: str, output_path: str = "results/"):
        """
        Initialize the cohort filter.
        
        Args:
            data_path: Path to the input dataset
            output_path: Path to save filtered results
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Define cohort parameters
        self.treated_birth_years = [2007, 2008, 2009]  # Ages 4-6 in 2013
        self.control_birth_years = [1997, 1998, 1999]  # Ages 14-16 in 2013
        self.treated_age_range = (11, 16)  # Secondary transition period
        self.control_age_range = (15, 25)  # Post-secondary transition period
        self.reform_year = 2013
        
        # Results storage
        self.original_data = None
        self.filtered_data = None
        self.filtering_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the original dataset.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.original_data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.original_data.shape}")
            
            # Store initial statistics
            self.filtering_stats['original_records'] = len(self.original_data)
            self.filtering_stats['original_columns'] = len(self.original_data.columns)
            
            return self.original_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def filter_birth_cohorts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to include only target birth cohorts.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame with only target birth cohorts
        """
        logger.info("Step 1: Filtering birth cohorts")
        
        # Get birth year column
        birth_year_col = 'nam sinh'
        
        if birth_year_col not in df.columns:
            raise ValueError(f"Birth year column '{birth_year_col}' not found in dataset")
        
        # Create cohort filter
        treated_filter = df[birth_year_col].isin(self.treated_birth_years)
        control_filter = df[birth_year_col].isin(self.control_birth_years)
        cohort_filter = treated_filter | control_filter
        
        # Apply filter
        filtered_df = df[cohort_filter].copy()
        
        # Log statistics
        treated_count = treated_filter.sum()
        control_count = control_filter.sum()
        total_filtered = len(filtered_df)
        
        logger.info(f"Birth cohort filtering results:")
        logger.info(f"  - Treated cohort (2007-2009): {treated_count:,} records")
        logger.info(f"  - Control cohort (1997-1999): {control_count:,} records")
        logger.info(f"  - Total after filtering: {total_filtered:,} records")
        logger.info(f"  - Excluded: {len(df) - total_filtered:,} records")
        
        # Store statistics
        self.filtering_stats['after_birth_filter'] = total_filtered
        self.filtering_stats['treated_birth_count'] = treated_count
        self.filtering_stats['control_birth_count'] = control_count
        self.filtering_stats['excluded_birth'] = len(df) - total_filtered
        
        return filtered_df
    
    def apply_age_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply age constraints for each cohort.
        
        Args:
            df: DataFrame with birth cohort filtering applied
            
        Returns:
            DataFrame with age constraints applied
        """
        logger.info("Step 2: Applying age constraints")
        
        birth_year_col = 'nam sinh'
        age_col = 'tuoi'
        
        if age_col not in df.columns:
            raise ValueError(f"Age column '{age_col}' not found in dataset")
        
        # Create age filters for each cohort
        treated_birth_filter = df[birth_year_col].isin(self.treated_birth_years)
        control_birth_filter = df[birth_year_col].isin(self.control_birth_years)
        
        # Age constraints
        treated_age_filter = (df[age_col] >= self.treated_age_range[0]) & (df[age_col] <= self.treated_age_range[1])
        control_age_filter = (df[age_col] >= self.control_age_range[0]) & (df[age_col] <= self.control_age_range[1])
        
        # Combined filters
        treated_filter = treated_birth_filter & treated_age_filter
        control_filter = control_birth_filter & control_age_filter
        final_filter = treated_filter | control_filter
        
        # Apply filter
        filtered_df = df[final_filter].copy()
        
        # Log statistics
        treated_count = treated_filter.sum()
        control_count = control_filter.sum()
        total_filtered = len(filtered_df)
        
        logger.info(f"Age constraint filtering results:")
        logger.info(f"  - Treated cohort (ages 11-16): {treated_count:,} records")
        logger.info(f"  - Control cohort (ages 15-25): {control_count:,} records")
        logger.info(f"  - Total after age filtering: {total_filtered:,} records")
        logger.info(f"  - Excluded by age constraints: {len(df) - total_filtered:,} records")
        
        # Store statistics
        self.filtering_stats['after_age_filter'] = total_filtered
        self.filtering_stats['treated_age_count'] = treated_count
        self.filtering_stats['control_age_count'] = control_count
        self.filtering_stats['excluded_age'] = len(df) - total_filtered
        
        return filtered_df
    
    def handle_missing_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove observations with missing outcome variables.
        
        Args:
            df: DataFrame with cohort and age filtering applied
            
        Returns:
            DataFrame with missing outcomes removed
        """
        logger.info("Step 3: Handling missing outcome variables")
        
        birth_year_col = 'nam sinh'
        current_edu_col = 'Current edu. level'
        education_level_col = 'education level'
        
        # Create cohort filters
        treated_filter = df[birth_year_col].isin(self.treated_birth_years)
        control_filter = df[birth_year_col].isin(self.control_birth_years)
        
        # Handle missing outcomes differently for each cohort
        treated_complete = treated_filter & df[current_edu_col].notna()
        control_complete = control_filter & df[education_level_col].notna()
        
        # Combined filter for complete cases
        complete_filter = treated_complete | control_complete
        filtered_df = df[complete_filter].copy()
        
        # Log statistics
        treated_complete_count = treated_complete.sum()
        control_complete_count = control_complete.sum()
        total_complete = len(filtered_df)
        
        treated_missing = treated_filter.sum() - treated_complete_count
        control_missing = control_filter.sum() - control_complete_count
        
        logger.info(f"Missing outcome filtering results:")
        logger.info(f"  - Treated with complete 'Current edu. level': {treated_complete_count:,} records")
        logger.info(f"  - Control with complete 'education level': {control_complete_count:,} records")
        logger.info(f"  - Total with complete outcomes: {total_complete:,} records")
        logger.info(f"  - Treated excluded (missing outcome): {treated_missing:,} records")
        logger.info(f"  - Control excluded (missing outcome): {control_missing:,} records")
        
        # Store statistics
        self.filtering_stats['final_records'] = total_complete
        self.filtering_stats['treated_final_count'] = treated_complete_count
        self.filtering_stats['control_final_count'] = control_complete_count
        self.filtering_stats['treated_missing_outcome'] = treated_missing
        self.filtering_stats['control_missing_outcome'] = control_missing
        
        return filtered_df
    
    def create_treatment_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create treatment and post-reform indicator variables.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            DataFrame with treatment variables added
        """
        logger.info("Step 4: Creating treatment variables")
        
        df = df.copy()
        birth_year_col = 'nam sinh'
        survey_year_col = 'survey_year'
        
        # Treatment indicator (1 = treated cohort, 0 = control cohort)
        df['treated'] = df[birth_year_col].isin(self.treated_birth_years).astype(int)
        
        # Post-reform indicator (1 = post-2013, 0 = pre-2013)
        df['post_reform'] = (df[survey_year_col] >= 2014).astype(int)
        
        # DiD interaction term
        df['treated_x_post'] = df['treated'] * df['post_reform']
        
        # Birth cohort indicators for robustness
        df['birth_cohort'] = df[birth_year_col].apply(
            lambda x: f"born_{x}" if pd.notna(x) else "unknown"
        )
        
        # Age at treatment (age in 2013)
        df['age_at_treatment'] = 2013 - df[birth_year_col]
        
        logger.info("Treatment variables created:")
        logger.info(f"  - 'treated': Treatment group indicator")
        logger.info(f"  - 'post_reform': Post-2013 time indicator")
        logger.info(f"  - 'treated_x_post': DiD interaction term")
        logger.info(f"  - 'birth_cohort': Birth cohort categories")
        logger.info(f"  - 'age_at_treatment': Age in 2013")
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the filtered dataset.
        
        Args:
            df: Final filtered DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary statistics")
        
        summary = {}
        
        # Overall statistics
        summary['total_observations'] = len(df)
        summary['survey_years'] = sorted(df['survey_year'].unique())
        summary['birth_years'] = sorted(df['nam sinh'].unique())
        
        # Cohort breakdown
        treated_mask = df['treated'] == 1
        control_mask = df['treated'] == 0
        
        summary['treated_cohort'] = {
            'total_obs': treated_mask.sum(),
            'birth_years': sorted(df[treated_mask]['nam sinh'].unique()),
            'age_range': (df[treated_mask]['tuoi'].min(), df[treated_mask]['tuoi'].max()),
            'survey_years': sorted(df[treated_mask]['survey_year'].unique()),
            'outcome_variable': 'Current edu. level'
        }
        
        summary['control_cohort'] = {
            'total_obs': control_mask.sum(),
            'birth_years': sorted(df[control_mask]['nam sinh'].unique()),
            'age_range': (df[control_mask]['tuoi'].min(), df[control_mask]['tuoi'].max()),
            'survey_years': sorted(df[control_mask]['survey_year'].unique()),
            'outcome_variable': 'education level'
        }
        
        # Treatment timing breakdown
        summary['treatment_timing'] = {
            'pre_reform_treated': ((df['treated'] == 1) & (df['post_reform'] == 0)).sum(),
            'post_reform_treated': ((df['treated'] == 1) & (df['post_reform'] == 1)).sum(),
            'pre_reform_control': ((df['treated'] == 0) & (df['post_reform'] == 0)).sum(),
            'post_reform_control': ((df['treated'] == 0) & (df['post_reform'] == 1)).sum()
        }
        
        # DiD cells
        summary['did_cells'] = df.groupby(['treated', 'post_reform']).size().to_dict()
        
        # Outcome variable coverage
        treated_outcomes = df[treated_mask]['Current edu. level'].value_counts()
        control_outcomes = df[control_mask]['education level'].value_counts()
        
        summary['outcome_distributions'] = {
            'treated_current_edu_level': treated_outcomes.head(10).to_dict(),
            'control_education_level': control_outcomes.head(10).to_dict()
        }
        
        return summary
    
    def save_results(self, df: pd.DataFrame, summary: Dict[str, Any]) -> None:
        """
        Save filtered dataset and summary statistics.
        
        Args:
            df: Filtered DataFrame
            summary: Summary statistics dictionary
        """
        logger.info("Saving results")
        
        # Save filtered dataset
        dataset_path = self.output_path / "did_cohorts_filtered.csv"
        df.to_csv(dataset_path, index=False)
        logger.info(f"Filtered dataset saved to: {dataset_path}")
        
        # Save summary statistics
        summary_path = self.output_path / "cohort_filtering_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DiD COHORT FILTERING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Filtering pipeline summary
            f.write("FILTERING PIPELINE RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original records: {self.filtering_stats['original_records']:,}\n")
            f.write(f"After birth cohort filter: {self.filtering_stats['after_birth_filter']:,}\n")
            f.write(f"After age constraints: {self.filtering_stats['after_age_filter']:,}\n")
            f.write(f"Final records (complete outcomes): {self.filtering_stats['final_records']:,}\n")
            f.write(f"Overall retention rate: {(self.filtering_stats['final_records']/self.filtering_stats['original_records']*100):.2f}%\n\n")
            
            # Cohort summary
            f.write("COHORT COMPOSITION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Treated cohort (2007-2009): {summary['treated_cohort']['total_obs']:,} observations\n")
            f.write(f"Control cohort (1997-1999): {summary['control_cohort']['total_obs']:,} observations\n")
            f.write(f"Total observations: {summary['total_observations']:,}\n\n")
            
            # Treatment timing
            f.write("DiD TREATMENT CELLS:\n")
            f.write("-" * 20 + "\n")
            timing = summary['treatment_timing']
            f.write(f"Treated × Pre-reform: {timing['pre_reform_treated']:,}\n")
            f.write(f"Treated × Post-reform: {timing['post_reform_treated']:,}\n")
            f.write(f"Control × Pre-reform: {timing['pre_reform_control']:,}\n")
            f.write(f"Control × Post-reform: {timing['post_reform_control']:,}\n\n")
            
            # Data coverage
            f.write("DATA COVERAGE:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Survey years: {', '.join(map(str, summary['survey_years']))}\n")
            f.write(f"Birth years: {', '.join(map(str, summary['birth_years']))}\n")
            f.write(f"Treated age range: {summary['treated_cohort']['age_range'][0]}-{summary['treated_cohort']['age_range'][1]}\n")
            f.write(f"Control age range: {summary['control_cohort']['age_range'][0]}-{summary['control_cohort']['age_range'][1]}\n\n")
            
            # Outcome distributions
            f.write("OUTCOME VARIABLE DISTRIBUTIONS:\n")
            f.write("-" * 35 + "\n")
            f.write("Treated cohort (Current edu. level):\n")
            for outcome, count in summary['outcome_distributions']['treated_current_edu_level'].items():
                f.write(f"  {outcome}: {count:,}\n")
            
            f.write("\nControl cohort (education level):\n")
            for outcome, count in summary['outcome_distributions']['control_education_level'].items():
                f.write(f"  {outcome}: {count:,}\n")
        
        logger.info(f"Summary statistics saved to: {summary_path}")
    
    def run_full_filtering(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete cohort filtering pipeline.
        
        Returns:
            Tuple of (filtered_dataframe, summary_statistics)
        """
        logger.info("Starting DiD cohort filtering pipeline")
        
        try:
            # Load data
            df = self.load_data()
            
            # Apply filters step by step
            df = self.filter_birth_cohorts(df)
            df = self.apply_age_constraints(df)
            df = self.handle_missing_outcomes(df)
            df = self.create_treatment_variables(df)
            
            # Generate summary
            summary = self.generate_summary_statistics(df)
            
            # Save results
            self.save_results(df, summary)
            
            # Store results
            self.filtered_data = df
            
            logger.info("DiD cohort filtering pipeline completed successfully")
            logger.info(f"Final dataset shape: {df.shape}")
            
            return df, summary
            
        except Exception as e:
            logger.error(f"Error in cohort filtering pipeline: {str(e)}")
            raise


def main():
    """
    Main function to run cohort filtering.
    """
    # Set up paths (relative to project root)
    project_root = Path(__file__).parents[3]  # Go up from src/DiD/regression run/ to project root
    data_path = project_root / "data" / "all_years_merged_dataset_final_corrected.csv"
    output_path = project_root / "results"
    
    # Initialize filter
    cohort_filter = CohortFilter(data_path, output_path)
    
    # Run filtering
    filtered_data, summary = cohort_filter.run_full_filtering()
    
    print("\n" + "="*60)
    print("DiD COHORT FILTERING COMPLETED")
    print("="*60)
    print(f"Final dataset: {len(filtered_data):,} observations")
    print(f"Treated cohort: {summary['treated_cohort']['total_obs']:,} observations")
    print(f"Control cohort: {summary['control_cohort']['total_obs']:,} observations")
    print("\nFiles saved:")
    print("- results/did_cohorts_filtered.csv")
    print("- results/cohort_filtering_summary.txt")
    print("="*60)


if __name__ == "__main__":
    main()
