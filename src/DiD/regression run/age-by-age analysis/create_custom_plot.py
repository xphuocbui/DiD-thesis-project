"""
Custom Age-by-Age Treatment Effects Plot
========================================

This script creates a custom visualization of the age-by-age treatment effects
with specific styling and shaded regions for different school levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_age_by_age_results():
    """Load the age-by-age analysis results."""
    results_file = "results/age-by-age analysis/age_by_age_results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_custom_plot():
    """Create the custom age-by-age treatment effects plot."""
    
    # Load results
    results = load_age_by_age_results()
    age_effects = results['age_effects']
    
    # Prepare data for plotting
    ages = list(range(6, 19))
    coefficients = []
    std_errors = []
    significance = []
    
    for age in ages:
        if str(age) in age_effects:
            coefficients.append(age_effects[str(age)]['coefficient'])
            std_errors.append(age_effects[str(age)]['standard_error'])
            significance.append(age_effects[str(age)]['significant_5pct'])
        else:
            # Age 18 was dropped due to collinearity, set to 0 (reference)
            coefficients.append(0.0)
            std_errors.append(0.0)
            significance.append(False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by significance and exposure
    colors = []
    for age, significant in zip(ages, significance):
        if age <= 11:  # Primary school age (exposed)
            if significant:
                colors.append('green')
            else:
                colors.append('lightgreen')
        else:  # Secondary school age (not exposed)
            if significant:
                colors.append('red')
            else:
                colors.append('gray')
    
    # Plot coefficients
    scatter = ax.scatter(ages, coefficients, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add confidence intervals
    for age, coef, se in zip(ages, coefficients, std_errors):
        if se > 0:  # Only plot CI if we have standard error
            ax.plot([age, age], [coef - 1.96*se, coef + 1.96*se], 'k-', alpha=0.5, linewidth=2)
            # Add caps to confidence intervals
            ax.plot([age-0.1, age+0.1], [coef - 1.96*se, coef - 1.96*se], 'k-', alpha=0.5, linewidth=2)
            ax.plot([age-0.1, age+0.1], [coef + 1.96*se, coef + 1.96*se], 'k-', alpha=0.5, linewidth=2)
    
    # Add shaded regions for different school levels
    ax.axvspan(6, 11.5, alpha=0.2, color='green', label='Primary school age (exposed to reform)')
    ax.axvspan(11.5, 18.5, alpha=0.2, color='orange', label='Secondary school age (not exposed)')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Formatting
    ax.set_xlabel('Age in 2013', fontsize=14, fontweight='bold')
    ax.set_ylabel('Treatment Effect on School Enrollment', fontsize=14, fontweight='bold')
    ax.set_title('Age-by-Age Treatment Effects: Vietnam Education Reform 2013', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    ax.set_xlim(5.5, 18.5)
    ax.set_xticks(ages)
    ax.set_xticklabels([str(age) for age in ages])
    
    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Primary school age (exposed to reform)'),
        Patch(facecolor='orange', alpha=0.2, label='Secondary school age (not exposed)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
               markeredgecolor='black', label='Significant positive effect'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='black', label='Significant effect (secondary age)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
               markeredgecolor='black', label='Not significant'),
        Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='No effect (reference)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Add annotations for key findings
    # Annotate the strongest effect (age 6)
    max_coef_idx = coefficients.index(max(coefficients))
    max_age = ages[max_coef_idx]
    max_coef = coefficients[max_coef_idx]
    
    ax.annotate(f'Strongest effect:\nAge {max_age} (+{max_coef:.1%})', 
                xy=(max_age, max_coef), 
                xytext=(max_age + 2, max_coef + 0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=11, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Annotate the transition point
    ax.annotate('Reform exposure\ntransition', 
                xy=(11.5, 0.2), 
                xytext=(13, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = "results/age-by-age analysis/custom_age_effects_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Custom plot saved to: {output_file}")
    
    # Don't show plot interactively to avoid blocking
    # plt.show()
    
    return fig, ax

def print_summary_stats():
    """Print summary statistics for the plot."""
    results = load_age_by_age_results()
    age_effects = results['age_effects']
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR CUSTOM PLOT")
    print("="*60)
    
    # Primary school ages (6-11) - exposed
    primary_ages = [6, 7, 8, 9, 10, 11]
    primary_effects = []
    primary_significant = 0
    
    for age in primary_ages:
        if str(age) in age_effects:
            effect = age_effects[str(age)]['coefficient']
            significant = age_effects[str(age)]['significant_5pct']
            primary_effects.append(effect)
            if significant:
                primary_significant += 1
            print(f"Age {age}: {effect:+.3f} ({'significant' if significant else 'not significant'})")
    
    print(f"\nPrimary school ages (6-11) summary:")
    print(f"  Mean effect: {np.mean(primary_effects):+.3f}")
    print(f"  Significant effects: {primary_significant}/{len(primary_ages)}")
    
    # Secondary school ages (12-18) - not exposed
    secondary_ages = [12, 13, 14, 15, 16, 17, 18]
    secondary_effects = []
    secondary_significant = 0
    
    print(f"\nSecondary school ages (12-18):")
    for age in secondary_ages:
        if str(age) in age_effects:
            effect = age_effects[str(age)]['coefficient']
            significant = age_effects[str(age)]['significant_5pct']
            secondary_effects.append(effect)
            if significant:
                secondary_significant += 1
            print(f"Age {age}: {effect:+.3f} ({'significant' if significant else 'not significant'})")
        else:
            print(f"Age {age}: 0.000 (reference - dropped)")
            secondary_effects.append(0.0)
    
    print(f"\nSecondary school ages (12-18) summary:")
    print(f"  Mean effect: {np.mean(secondary_effects):+.3f}")
    print(f"  Significant effects: {secondary_significant}/{len(secondary_ages)}")
    
    print("="*60)

if __name__ == "__main__":
    print("Creating custom age-by-age treatment effects plot...")
    
    try:
        # Create the plot
        fig, ax = create_custom_plot()
        
        # Print summary statistics
        print_summary_stats()
        
        print("\nCustom plot creation completed successfully!")
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        raise
