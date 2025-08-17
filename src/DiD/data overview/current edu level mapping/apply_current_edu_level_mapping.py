#!/usr/bin/env python3
"""
CURRENT EDUCATION LEVEL MAPPING SCRIPT
======================================
This script applies Vietnamese-to-English mappings for the 'Current edu. level' column
using a CSV mapping file provided by the user.

The script will:
1. Load the main dataset
2. Load the mapping CSV file 
3. Apply mappings to standardize Vietnamese values to English
4. Create backup and generate detailed reports
5. Update the main dataset with mapped values

Author: DiD Analysis Project
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import shutil

class CurrentEduLevelMapper:
    def __init__(self):
        self.dataset_path = "data/all_years_merged_dataset_final_corrected.csv"
        self.backup_path = "data/all_years_merged_dataset_final_corrected_backup.csv"
        self.column_name = "Current edu. level"
        self.report_dir = "data/data summary/current edu. level"
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
        
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)

    def print_step(self, step_num, description):
        """Print formatted step"""
        print(f"\n{'='*15} STEP {step_num}: {description} {'='*15}")

    def load_dataset(self):
        """Load the main dataset"""
        print("📂 Loading main dataset...")
        try:
            df = pd.read_csv(self.dataset_path, low_memory=False)
            print(f"✅ Dataset loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"❌ ERROR: Dataset not found at {self.dataset_path}")
            return None
        except Exception as e:
            print(f"❌ ERROR loading dataset: {e}")
            return None

    def analyze_current_values(self, df):
        """Analyze current values in the Current edu. level column"""
        print(f"\n🔍 Analyzing '{self.column_name}' column...")
        
        if self.column_name not in df.columns:
            print(f"❌ ERROR: Column '{self.column_name}' not found in dataset")
            return None
        
        # Get value counts
        value_counts = df[self.column_name].value_counts(dropna=False)
        
        print(f"📊 Found {len(value_counts)} unique values (including NaN)")
        print(f"📊 Missing values: {df[self.column_name].isnull().sum():,} ({(df[self.column_name].isnull().sum()/len(df)*100):.2f}%)")
        
        # Identify Vietnamese values that need mapping
        vietnamese_patterns = ['học', 'THCS', 'THPT', 'Đại', 'Cao', 'Nhà trẻ', 'MG', 'Thạc', 'Tiến', 'Khác', 'nghề', 'cấp']
        vietnamese_values = []
        
        for value in value_counts.index:
            if pd.notna(value):
                str_value = str(value)
                if any(pattern in str_value for pattern in vietnamese_patterns):
                    vietnamese_values.append((value, value_counts[value]))
        
        print(f"\n🇻🇳 Vietnamese values found that need mapping:")
        for value, count in vietnamese_values:
            print(f"   '{value}': {count:,} occurrences")
        
        return value_counts, vietnamese_values

    def load_mapping_file(self):
        """Load the predefined mapping CSV file"""
        print(f"\n📋 LOADING MAPPING FILE")
        print("=" * 40)
        
        mapping_file_path = "data/edu_level_mapping.csv"
        
        if not os.path.exists(mapping_file_path):
            print(f"❌ ERROR: Mapping file not found at {mapping_file_path}")
            return None
        
        try:
            # Load the mapping file
            mapping_df = pd.read_csv(mapping_file_path, encoding='utf-8-sig')
            print(f"✅ Mapping file loaded: {len(mapping_df)} rows")
            
            # Verify expected columns
            expected_columns = ['year', 'original_value', 'corrected_value']
            if not all(col in mapping_df.columns for col in expected_columns):
                print(f"❌ ERROR: Expected columns {expected_columns} not found")
                print(f"📋 Found columns: {list(mapping_df.columns)}")
                return None
            
            print(f"✅ Mapping file format verified")
            print(f"📊 Mappings per year:")
            year_counts = mapping_df['year'].value_counts().sort_index()
            for year, count in year_counts.items():
                print(f"   {year}: {count} mappings")
            
            return mapping_df
            
        except Exception as e:
            print(f"❌ Error reading mapping file: {e}")
            return None

    def create_mapping_dict(self, mapping_df):
        """Create mapping dictionary from the CSV file"""
        print(f"\n🗺️  Creating mapping dictionary...")
        
        # Clean the data - remove rows with missing values
        mapping_df = mapping_df.dropna(subset=['original_value', 'corrected_value'])
        
        # Create comprehensive mapping dictionary from all years
        mapping_dict = {}
        duplicate_check = {}
        
        for _, row in mapping_df.iterrows():
            original_val = str(row['original_value']).strip()
            corrected_val = str(row['corrected_value']).strip()
            year = row['year']
            
            if original_val and corrected_val:
                # Check for conflicts (same original mapping to different values)
                if original_val in duplicate_check:
                    if duplicate_check[original_val] != corrected_val:
                        print(f"⚠️  Mapping conflict for '{original_val}': '{duplicate_check[original_val]}' vs '{corrected_val}' (year {year})")
                        # Keep the most recent mapping (latest year)
                        continue
                
                mapping_dict[original_val] = corrected_val
                duplicate_check[original_val] = corrected_val
        
        print(f"✅ Created mapping dictionary with {len(mapping_dict)} unique mappings")
        
        # Show preview
        print(f"\n📋 Mapping preview (first 10):")
        for i, (original, corrected) in enumerate(list(mapping_dict.items())[:10]):
            print(f"   '{original}' → '{corrected}'")
        
        if len(mapping_dict) > 10:
            print(f"   ... and {len(mapping_dict) - 10} more mappings")
        
        return mapping_dict

    def create_backup(self):
        """Create backup of the original dataset"""
        print(f"\n💾 Creating backup...")
        try:
            if not os.path.exists(self.backup_path):
                shutil.copy2(self.dataset_path, self.backup_path)
                print(f"✅ Backup created: {self.backup_path}")
            else:
                print(f"✅ Backup already exists: {self.backup_path}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False

    def apply_mappings(self, df, mapping_dict):
        """Apply the mappings to the dataset"""
        print(f"\n🔄 Applying mappings to '{self.column_name}' column...")
        
        # Count values before mapping
        before_counts = df[self.column_name].value_counts(dropna=False)
        
        # Apply mappings
        mapped_count = 0
        for original_value, new_value in mapping_dict.items():
            mask = df[self.column_name] == original_value
            count = mask.sum()
            if count > 0:
                df.loc[mask, self.column_name] = new_value
                mapped_count += count
                print(f"   ✓ '{original_value}' → '{new_value}' ({count:,} records)")
        
        # Count values after mapping
        after_counts = df[self.column_name].value_counts(dropna=False)
        
        print(f"\n📊 MAPPING SUMMARY:")
        print(f"   Total records mapped: {mapped_count:,}")
        print(f"   Unique values before: {len(before_counts)}")
        print(f"   Unique values after: {len(after_counts)}")
        
        return df, mapped_count, before_counts, after_counts

    def generate_report(self, mapped_count, before_counts, after_counts, mapping_dict):
        """Generate detailed mapping report"""
        print(f"\n📄 Generating mapping report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.report_dir, f"current_edu_level_mapping_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CURRENT EDUCATION LEVEL MAPPING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mapping Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Column: {self.column_name}\n\n")
            
            f.write("MAPPING SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total records mapped: {mapped_count:,}\n")
            f.write(f"Unique values before mapping: {len(before_counts)}\n")
            f.write(f"Unique values after mapping: {len(after_counts)}\n")
            f.write(f"Mappings applied: {len(mapping_dict)}\n\n")
            
            f.write("APPLIED MAPPINGS:\n")
            f.write("-" * 25 + "\n")
            for original, new in mapping_dict.items():
                count = before_counts.get(original, 0)
                f.write(f"'{original}' → '{new}' ({count:,} records)\n")
            
            f.write("\nVALUE DISTRIBUTION BEFORE MAPPING:\n")
            f.write("-" * 40 + "\n")
            for value, count in before_counts.head(20).items():
                if pd.isna(value):
                    f.write(f"[MISSING]: {count:,}\n")
                else:
                    f.write(f"'{value}': {count:,}\n")
            
            f.write("\nVALUE DISTRIBUTION AFTER MAPPING:\n")
            f.write("-" * 39 + "\n")
            for value, count in after_counts.head(20).items():
                if pd.isna(value):
                    f.write(f"[MISSING]: {count:,}\n")
                else:
                    f.write(f"'{value}': {count:,}\n")
        
        print(f"✅ Report saved: {report_file}")
        return report_file

    def save_updated_dataset(self, df):
        """Save the updated dataset"""
        print(f"\n💾 Saving updated dataset...")
        try:
            df.to_csv(self.dataset_path, index=False)
            print(f"✅ Updated dataset saved: {self.dataset_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False

    def run(self):
        """Main execution function"""
        self.print_header("CURRENT EDUCATION LEVEL MAPPING")
        
        # Step 1: Load dataset
        self.print_step(1, "LOAD DATASET")
        df = self.load_dataset()
        if df is None:
            return False
        
        # Step 2: Analyze current values
        self.print_step(2, "ANALYZE CURRENT VALUES")
        analysis_result = self.analyze_current_values(df)
        if analysis_result is None:
            return False
        value_counts, vietnamese_values = analysis_result
        
        if not vietnamese_values:
            print("✅ No Vietnamese values found that need mapping!")
            print("🏁 The column appears to already be properly standardized.")
            return True
        
        # Step 3: Load mapping file
        self.print_step(3, "LOAD MAPPING FILE")
        mapping_df = self.load_mapping_file()
        if mapping_df is None:
            return False
        
        # Step 4: Create mapping dictionary
        self.print_step(4, "CREATE MAPPING DICTIONARY")
        mapping_dict = self.create_mapping_dict(mapping_df)
        
        if not mapping_dict:
            print("❌ No valid mappings found in the file")
            return False
        
        # Step 5: Create backup
        self.print_step(5, "CREATE BACKUP")
        if not self.create_backup():
            print("❌ Failed to create backup. Aborting for data safety.")
            return False
        
        # Step 6: Apply mappings
        self.print_step(6, "APPLY MAPPINGS")
        df, mapped_count, before_counts, after_counts = self.apply_mappings(df, mapping_dict)
        
        # Step 7: Generate report
        self.print_step(7, "GENERATE REPORT")
        report_file = self.generate_report(mapped_count, before_counts, after_counts, mapping_dict)
        
        # Step 8: Save updated dataset
        self.print_step(8, "SAVE UPDATED DATASET")
        if not self.save_updated_dataset(df):
            return False
        
        # Success summary
        self.print_header("🎉 MAPPING COMPLETED SUCCESSFULLY! 🎉")
        print(f"✅ {mapped_count:,} records updated")
        print(f"✅ {len(mapping_dict)} mappings applied")
        print(f"✅ Report saved: {report_file}")
        print(f"✅ Backup created: {self.backup_path}")
        print(f"✅ Dataset updated: {self.dataset_path}")
        
        return True

def main():
    """Main function"""
    print("CURRENT EDUCATION LEVEL MAPPING SCRIPT")
    print("=" * 50)
    print("This script will map Vietnamese values in 'Current edu. level' to English")
    print("using your provided CSV mapping file.")
    
    mapper = CurrentEduLevelMapper()
    success = mapper.run()
    
    if success:
        print("\n🏁 Current education level mapping completed successfully!")
    else:
        print("\n❌ Current education level mapping failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
