#!/usr/bin/env python3
"""
Variance Analysis: How concept distribution variance affects robust importance.
This script analyzes the probability that 90% of concept mass is important
as a function of Gaussian distribution variance for each concept.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from tqdm import tqdm

from ibymdt_cav import (
    ConceptBank, SKITGlobal, secure_importance, robust_importance_hoeffding
)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_linear_data():
    """Create the same linear data as before."""
    np.random.seed(42)
    
    # Dataset parameters
    n_samples = 800
    d_hidden = 5  # Hidden representation dimension
    m_concepts = 8  # Number of concepts
    
    # Create a single concept bank
    C = np.random.randn(m_concepts, d_hidden)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)  # Normalize concepts
    concept_bank = ConceptBank(C)
    
    # Generate hidden representations
    H = np.random.randn(n_samples, d_hidden)
    
    # Project to concept space
    Z = concept_bank.project(H)
    
    # Define true important concepts (concepts 1, 3, and 6)
    true_concepts = [1, 3, 6]
    
    # Create Y with linear dependency on true concepts + noise
    noise_level = 0.3
    Y = (2.5 * Z[:, 1] +       # Strong dependency on concept 1
         0.3 * Z[:, 3] +       # Medium dependency on concept 3  
         0.4 * Z[:, 6] +       # Weaker dependency on concept 6
         noise_level * np.random.randn(n_samples))
    
    return Z, Y, true_concepts, concept_bank

def analyze_concept_variance_effect(Z, Y, concept_idx, variance_range, n_samples=100, mass_threshold=0.7, confidence=0.95):
    """
    Analyze how concept distribution variance affects importance probability.
    
    Args:
        Z: Concept projections
        Y: Target variable
        concept_idx: Index of concept to analyze
        variance_range: Array of variance values to test
        n_samples: Number of samples for robust importance test
        mass_threshold: Threshold for mass importance (0.9 = 90%)
        confidence: Confidence level for bounds
    
    Returns:
        Dictionary with variance analysis results
    """
    results = {
        'variances': [],
        'empirical_probabilities': [],
        'lower_bounds': [],
        'upper_bounds': [],
        'mass_is_important': [],
        'epsilons': []
    }
    
    print(f"\nAnalyzing Concept {concept_idx} variance effects...")
    print(f"Testing {len(variance_range)} variance levels from {variance_range[0]:.3f} to {variance_range[-1]:.3f}")
    
    for variance in tqdm(variance_range, desc=f"Concept {concept_idx}"):
        # Convert variance to standard deviation
        noise_std = np.sqrt(variance)
        
        # Test robust importance with this noise level
        robust_result = robust_importance_hoeffding(
            Y, Z, [concept_idx],  # Single concept
            noise_std=noise_std,
            n_samples=n_samples,
            mass_threshold=mass_threshold,
            confidence=confidence
        )
        
        # Store results
        results['variances'].append(variance)
        results['empirical_probabilities'].append(robust_result['empirical_probability'])
        results['lower_bounds'].append(robust_result['lower_bound'])
        results['upper_bounds'].append(robust_result['upper_bound'])
        results['mass_is_important'].append(robust_result['mass_is_important'])
        results['epsilons'].append(robust_result['epsilon'])
    
    return results

def run_variance_analysis(Z, Y, true_concepts, output_dir="results"):
    """Run variance analysis for all concepts."""
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CONCEPT VARIANCE ANALYSIS")
    print("="*60)
    
    # Define variance range (logarithmic scale for better coverage)
    variance_range = np.logspace(-2, 1, 20)  # From 0.01 to 10
    print(f"Variance range: {variance_range[0]:.3f} to {variance_range[-1]:.3f}")
    print(f"Number of variance levels: {len(variance_range)}")
    
    # Store results for all concepts
    all_results = {}
    
    # Analyze each concept
    for concept_idx in range(Z.shape[1]):
        is_true_concept = concept_idx in true_concepts
        
        print(f"\n--- Analyzing Concept {concept_idx} (True concept: {is_true_concept}) ---")
        
        # Run variance analysis for this concept
        concept_results = analyze_concept_variance_effect(
            Z, Y, concept_idx, variance_range,
            n_samples=100, mass_threshold=0.7, confidence=0.95
        )
        
        concept_results['is_true_concept'] = is_true_concept
        concept_results['concept_idx'] = concept_idx
        all_results[concept_idx] = concept_results
        
        # Print summary for this concept
        min_prob = min(concept_results['empirical_probabilities'])
        max_prob = max(concept_results['empirical_probabilities'])
        print(f"Empirical probability range: {min_prob:.3f} to {max_prob:.3f}")
        
        # Count how many variance levels result in important mass
        important_count = sum(concept_results['mass_is_important'])
        print(f"Important mass achieved in {important_count}/{len(variance_range)} variance levels")
    
    return all_results, variance_range

def visualize_variance_analysis(all_results, variance_range, true_concepts, output_dir="results"):
    """Create comprehensive visualizations of variance analysis."""
    print(f"\nCreating variance analysis visualizations...")
    
    # Create a large figure with subplots
    n_concepts = len(all_results)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for concept_idx, ax in enumerate(axes):
        if concept_idx >= n_concepts:
            ax.set_visible(False)
            continue
            
        results = all_results[concept_idx]
        is_true = results['is_true_concept']
        
        # Plot empirical probability with confidence bands
        ax.plot(variance_range, results['empirical_probabilities'], 
               'o-', color='red' if is_true else 'blue', 
               linewidth=2, markersize=4, alpha=0.8,
               label='Empirical probability')
        
        # Add confidence bands
        ax.fill_between(variance_range, 
                       results['lower_bounds'], 
                       results['upper_bounds'],
                       alpha=0.3, color='red' if is_true else 'blue',
                       label='95% Confidence interval')
        
        # Add 90% threshold line
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='90% threshold')
        
        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Concept Distribution Variance')
        ax.set_ylabel('Probability')
        ax.set_title(f'Concept {concept_idx}{"★" if is_true else ""}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # Add legend only to first subplot
        if concept_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Concept Importance Probability vs Distribution Variance', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_analysis_all_concepts.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create focused plot for true concepts only
    fig, axes = plt.subplots(1, len(true_concepts), figsize=(5*len(true_concepts), 6))
    if len(true_concepts) == 1:
        axes = [axes]
    
    for i, concept_idx in enumerate(true_concepts):
        results = all_results[concept_idx]
        ax = axes[i]
        
        # Plot with error bars
        ax.errorbar(variance_range, results['empirical_probabilities'],
                   yerr=[np.array(results['empirical_probabilities']) - np.array(results['lower_bounds']),
                         np.array(results['upper_bounds']) - np.array(results['empirical_probabilities'])],
                   fmt='o-', color='red', linewidth=2, markersize=6, capsize=4,
                   label='Empirical probability ± 95% CI')
        
        # Add threshold
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='70% mass threshold')
        
        # Highlight regions where mass is important
        important_mask = np.array(results['mass_is_important'])
        if np.any(important_mask):
            important_variances = np.array(variance_range)[important_mask]
            ax.scatter(important_variances, 
                      np.array(results['empirical_probabilities'])[important_mask],
                      color='gold', s=100, marker='*', zorder=5,
                      label='Mass is important')
        
        ax.set_xscale('log')
        ax.set_xlabel('Concept Distribution Variance')
        ax.set_ylabel('Probability that 90% of Mass is Important')
        ax.set_title(f'True Concept {concept_idx}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
    
    plt.suptitle('True Concepts: Importance Probability vs Variance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_analysis_true_concepts.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Probability curves for all concepts
    for concept_idx, results in all_results.items():
        is_true = results['is_true_concept']
        color = 'red' if is_true else 'blue'
        alpha = 1.0 if is_true else 0.3
        linewidth = 2 if is_true else 1
        linestyle = '-' if is_true else '--'
        
        ax1.plot(variance_range, results['empirical_probabilities'],
                color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle,
                label=f'Concept {concept_idx}' if is_true else '')
    
    ax1.axhline(y=0.9, color='green', linestyle='--', label='90% threshold')
    ax1.set_xscale('log')
    ax1.set_xlabel('Concept Distribution Variance')
    ax1.set_ylabel('Empirical Probability')
    ax1.set_title('All Concepts: Probability vs Variance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Confidence interval width analysis
    for concept_idx, results in all_results.items():
        is_true = results['is_true_concept']
        if is_true:  # Only show true concepts for clarity
            ci_width = np.array(results['upper_bounds']) - np.array(results['lower_bounds'])
            ax2.plot(variance_range, ci_width, 'o-', label=f'Concept {concept_idx}', linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Concept Distribution Variance')
    ax2.set_ylabel('95% Confidence Interval Width')
    ax2.set_title('True Concepts: Confidence Interval Width')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_analysis_summary.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def create_variance_report(all_results, variance_range, true_concepts, output_dir="results"):
    """Create detailed report of variance analysis."""
    report_path = Path(output_dir) / "variance_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("CONCEPT VARIANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"TRUE IMPORTANT CONCEPTS: {true_concepts}\n")
        f.write(f"VARIANCE RANGE: {variance_range[0]:.3f} to {variance_range[-1]:.3f}\n")
        f.write(f"NUMBER OF VARIANCE LEVELS: {len(variance_range)}\n\n")
        
        f.write("ANALYSIS METHODOLOGY:\n")
        f.write("-" * 30 + "\n")
        f.write("• For each concept and each variance level:\n")
        f.write("  1. Add Gaussian noise with specified variance to concept values\n")
        f.write("  2. Test robust importance using Hoeffding inequality\n")
        f.write("  3. Estimate probability that 90% of concept mass is important\n")
        f.write("  4. Calculate 95% confidence bounds using Hoeffding inequality\n\n")
        
        f.write("DETAILED RESULTS BY CONCEPT:\n")
        f.write("-" * 40 + "\n")
        
        for concept_idx, results in all_results.items():
            is_true = results['is_true_concept']
            f.write(f"\nCONCEPT {concept_idx} {'★ (TRUE CONCEPT)' if is_true else '(False concept)'}:\n")
            f.write("-" * 30 + "\n")
            
            # Summary statistics
            probs = results['empirical_probabilities']
            lower_bounds = results['lower_bounds']
            important_count = sum(results['mass_is_important'])
            
            f.write(f"Probability range: {min(probs):.3f} to {max(probs):.3f}\n")
            f.write(f"Lower bound range: {min(lower_bounds):.3f} to {max(lower_bounds):.3f}\n")
            f.write(f"Mass important in {important_count}/{len(variance_range)} variance levels\n")
            
            # Find optimal variance range (if any)
            important_indices = [i for i, important in enumerate(results['mass_is_important']) if important]
            if important_indices:
                optimal_variances = [variance_range[i] for i in important_indices]
                f.write(f"Optimal variance range: {min(optimal_variances):.3f} to {max(optimal_variances):.3f}\n")
            else:
                f.write("No variance levels achieved 90% mass importance\n")
            
            # Detailed breakdown for key variance levels
            f.write("\nKey variance levels:\n")
            key_indices = [0, len(variance_range)//4, len(variance_range)//2, 3*len(variance_range)//4, -1]
            for i in key_indices:
                var = variance_range[i]
                prob = probs[i]
                lower = lower_bounds[i]
                upper = results['upper_bounds'][i]
                important = results['mass_is_important'][i]
                f.write(f"  Variance {var:.3f}: prob={prob:.3f}, CI=[{lower:.3f},{upper:.3f}], important={important}\n")
        
        f.write(f"\nSUMMARY INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        
        # Analyze trends for true concepts
        true_concept_trends = []
        for concept_idx in true_concepts:
            results = all_results[concept_idx]
            probs = results['empirical_probabilities']
            # Check if probability decreases with variance (expected behavior)
            correlation = np.corrcoef(np.log(variance_range), probs)[0, 1]
            true_concept_trends.append(correlation)
            
        avg_correlation = np.mean(true_concept_trends)
        f.write(f"• Average correlation between log(variance) and probability for true concepts: {avg_correlation:.3f}\n")
        f.write("• Negative correlation indicates probability decreases as variance increases (expected)\n")
        f.write("• Positive correlation may indicate robustness or non-linear effects\n")
        
        # Count concepts that achieve importance
        concepts_with_importance = sum(1 for results in all_results.values() 
                                     if any(results['mass_is_important']))
        f.write(f"• {concepts_with_importance}/{len(all_results)} concepts achieve 90% mass importance at some variance level\n")
        
        true_concepts_with_importance = sum(1 for concept_idx in true_concepts 
                                          if any(all_results[concept_idx]['mass_is_important']))
        f.write(f"• {true_concepts_with_importance}/{len(true_concepts)} true concepts achieve 90% mass importance\n")
    
    print(f"Variance analysis report saved to {report_path}")

def main():
    """Run the concept variance analysis."""
    print("CONCEPT DISTRIBUTION VARIANCE ANALYSIS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Setup the same linear data
        Z, Y, true_concepts, concept_bank = setup_linear_data()
        
        print(f"Data setup complete:")
        print(f"• Samples: {Z.shape[0]}")
        print(f"• Concepts: {Z.shape[1]}")
        print(f"• True important concepts: {true_concepts}")
        
        # Run variance analysis
        all_results, variance_range = run_variance_analysis(Z, Y, true_concepts, output_dir)
        
        # Create visualizations
        visualize_variance_analysis(all_results, variance_range, true_concepts, output_dir)
        
        # Create detailed report
        create_variance_report(all_results, variance_range, true_concepts, output_dir)
        
        print(f"\n{'='*60}")
        print("VARIANCE ANALYSIS COMPLETE")
        print("="*60)
        print("✅ Analysis completed successfully!")
        
        # Summary of findings
        print(f"\n📊 SUMMARY FOR TRUE CONCEPTS:")
        for concept_idx in true_concepts:
            results = all_results[concept_idx]
            probs = results['empirical_probabilities']
            important_count = sum(results['mass_is_important'])
            min_prob = min(probs)
            max_prob = max(probs)
            
            print(f"• Concept {concept_idx}:")
            print(f"  - Probability range: {min_prob:.3f} to {max_prob:.3f}")
            print(f"  - Important in {important_count}/{len(variance_range)} variance levels")
            
            # Find best variance level
            best_idx = np.argmax(probs)
            best_variance = variance_range[best_idx]
            best_prob = probs[best_idx]
            print(f"  - Best variance: {best_variance:.3f} (prob={best_prob:.3f})")
        
        print(f"\n📁 Results saved in '{output_dir}/' directory")
        print("📈 Key outputs:")
        print("  • variance_analysis_all_concepts.pdf - Individual concept analysis")
        print("  • variance_analysis_true_concepts.pdf - Focused on true concepts")  
        print("  • variance_analysis_summary.pdf - Comparative analysis")
        print("  • variance_analysis_report.txt - Detailed numerical results")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
