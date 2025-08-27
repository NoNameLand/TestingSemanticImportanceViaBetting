#!/usr/bin/env python3
"""
Simplified test script for semantic importance testing with linear relationships.
This focuses on testing robust importance of concepts in linear relationships
using concept distributions rather than single concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from ibymdt_cav import (
    ConceptBank, SKITGlobal, secure_importance, robust_importance_hoeffding
)

# Simple ConceptDist class for our test
class ConceptDist:
    def __init__(self, weights, noise_std=0.2):
        """
        A distribution around a weighted combination of concepts
        Args:
            weights: Weight vector for concepts 
            noise_std: Standard deviation of gaussian noise
        """
        self.weights = weights
        self.noise_std = noise_std
    
    def sample(self, base_vector):
        """
        Sample from this concept distribution
        Args:
            base_vector: Base concept vector to add noise to
        Returns:
            Noisy version of the weighted concept
        """
        # Apply weights and add gaussian noise
        weighted = np.dot(base_vector, self.weights)  
        noise = np.random.normal(0, self.noise_std)
        return weighted + noise

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_simple_linear_data():
    """Create simple synthetic data with linear relationships and noise."""
    np.random.seed(42)
    
    # Dataset parameters
    n_samples = 800
    d_hidden = 5  # Hidden representation dimension
    m_concepts = 8  # Number of concepts
    
    print(f"Generating linear data: {n_samples} samples, {d_hidden}D hidden, {m_concepts} concepts")
    
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
         1.8 * Z[:, 3] +       # Medium dependency on concept 3  
         1.2 * Z[:, 6] +       # Weaker dependency on concept 6
         noise_level * np.random.randn(n_samples))
    
    print(f"True important concepts: {true_concepts}")
    print(f"Y statistics: mean={Y.mean():.3f}, std={Y.std():.3f}")
    
    return Z, Y, true_concepts, concept_bank

def visualize_linear_relationships(Z, Y, true_concepts, output_dir="results"):
    """Visualize the linear relationships between concepts and Y."""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\nVisualizing concept-Y relationships...")
    
    # Calculate correlations
    correlations = [np.corrcoef(Y, Z[:, j])[0, 1] for j in range(Z.shape[1])]
    
    # Create correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of correlations
    colors = ['red' if j in true_concepts else 'blue' for j in range(len(correlations))]
    bars = ax1.bar(range(len(correlations)), correlations, color=colors, alpha=0.7)
    ax1.set_xlabel('Concept Index')
    ax1.set_ylabel('Correlation with Y')
    ax1.set_title('Concept-Y Correlations')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    true_patch = mpatches.Patch(color='red', alpha=0.7, label='True concepts')
    false_patch = mpatches.Patch(color='blue', alpha=0.7, label='Other concepts')
    ax1.legend(handles=[true_patch, false_patch])
    
    # Scatter plot for strongest correlation
    strongest_idx = np.argmax(np.abs(correlations))
    ax2.scatter(Z[:, strongest_idx], Y, alpha=0.6, s=20)
    ax2.set_xlabel(f'Concept {strongest_idx} values')
    ax2.set_ylabel('Y values')
    ax2.set_title(f'Y vs Concept {strongest_idx} (r={correlations[strongest_idx]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(Z[:, strongest_idx], Y)
    line = slope * Z[:, strongest_idx] + intercept
    ax2.plot(Z[:, strongest_idx], line, 'r', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/linear_relationships.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlations: {correlations}")
    print(f"Visualization saved to {output_dir}/linear_relationships.pdf")

def test_single_concept_importance(Z, Y, true_concepts, output_dir="results"):
    """Test importance of individual concepts using e-value accumulation."""
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print("SINGLE CONCEPT IMPORTANCE TESTING (E-VALUE ACCUMULATION)")
    print("="*50)
    
    results = {}
    
    for j in range(Z.shape[1]):
        print(f"\nTesting Concept {j}... ", end="")
        
        # Use different random seed for each concept to ensure different behavior
        np.random.seed(42 + j)
        
        # Create SKIT instance with e-value accumulation
        skit = SKITGlobal(alpha=0.01)
        
        # Randomly pair up samples for e-value accumulation
        indices = np.random.permutation(len(Y))
        n_pairs = min(100, len(Y) // 2)  # Limit pairs to avoid too much computation
        
        rejection_step = None
        e_values = []
        accumulated_e_values = []
        
        for i in range(n_pairs):
            idx1, idx2 = indices[2*i], indices[2*i + 1]
            y1, y2 = Y[idx1], Y[idx2]
            z1, z2 = Z[idx1, j], Z[idx2, j]
            
            # Add pair to e-value accumulation
            e_val, accumulated_e = skit.step_pair(y1, z1, y2, z2)
            e_values.append(e_val)
            accumulated_e_values.append(accumulated_e)
            
            if skit.rejected() and rejection_step is None:
                rejection_step = i + 1
                break
        
        results[j] = {
            'rejected': skit.rejected(),
            'accumulated_e': accumulated_e_values[-1] if accumulated_e_values else 1.0,
            'rejection_step': rejection_step,
            'e_values': e_values,
            'accumulated_e_trajectory': accumulated_e_values,
            'is_true_concept': j in true_concepts
        }
        
        status = "REJECTED" if skit.rejected() else "NOT REJECTED"
        true_marker = "★" if j in true_concepts else " "
        final_e = accumulated_e_values[-1] if accumulated_e_values else 1.0
        print(f"{true_marker} E={final_e:8.3f} {status}")
        
        if skit.rejected():
            print(f"  ✓ Rejected at step {rejection_step} with E-value {final_e:.3f}")
        else:
            print(f"  ○ Not rejected after {len(e_values)} pairs, final E-value {final_e:.3f}")
        
        # Debug: Show correlation for this concept
        corr = np.corrcoef(Y, Z[:, j])[0, 1]
        print(f"    Overall correlation: {corr:.3f}")
    
    # Visualize single concept results
    visualize_single_concept_results_evalue(results, output_dir)
    
    return results

def test_concept_distribution_importance(Z, Y, true_concepts, concept_bank, output_dir="results"):
    """Test robust importance of each individual concept using its distribution with Hoeffding inequality."""
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print("ROBUST INDIVIDUAL CONCEPT DISTRIBUTION TESTING")  
    print("="*50)
    
    dist_results = {}
    
    # Test each concept individually
    for concept_idx in range(Z.shape[1]):
        concept_name = f"Concept {concept_idx}"
        is_true_concept = concept_idx in true_concepts
        
        print(f"\nTesting {concept_name}...")
        print(f"  Is true concept: {is_true_concept}")
        
        # Test robust importance with Hoeffding inequality for this individual concept
        print("  Running robust importance test with Hoeffding bounds...")
        robust_result = robust_importance_hoeffding(
            Y, Z, [concept_idx],  # Single concept index
            noise_std=0.2, 
            n_samples=50, 
            mass_threshold=0.9, 
            confidence=0.95
        )
        
        robust_result['is_true_concept'] = is_true_concept
        robust_result['concept_name'] = concept_name
        robust_result['concept_idx'] = concept_idx
        
        dist_results[concept_name] = robust_result
        
        print(f"  Empirical probability: {robust_result['empirical_probability']:.3f}")
        print(f"  95% Confidence interval: [{robust_result['lower_bound']:.3f}, {robust_result['upper_bound']:.3f}]")
        print(f"  90% of mass is important: {robust_result['mass_is_important']}")
        print(f"  Hoeffding epsilon: {robust_result['epsilon']:.3f}")
        
        if robust_result['mass_is_important']:
            print(f"  ✅ Concept {concept_idx} has robustly important distribution mass")
        else:
            print(f"  ❌ Concept {concept_idx} does not have robustly important distribution mass")
    
    # Visualize distribution results
    visualize_individual_concept_distribution_results(dist_results, true_concepts, output_dir)
    
    return dist_results

def visualize_single_concept_results_evalue(results, output_dir):
    """Visualize single concept importance results using e-values."""
    print("\nVisualizing single concept e-value results...")
    
    # E-value trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accumulated e-value trajectories
    for concept_id, result in results.items():
        trajectory = result['accumulated_e_trajectory']
        if len(trajectory) > 0:
            steps = range(1, len(trajectory) + 1)
            color = 'red' if result['is_true_concept'] else 'blue'
            alpha = 1.0 if result['is_true_concept'] else 0.3
            linewidth = 2 if result['is_true_concept'] else 1
            
            ax1.plot(steps, trajectory, color=color, alpha=alpha, 
                    linewidth=linewidth, label=f'Concept {concept_id}' if result['is_true_concept'] else "")
    
    # Add rejection threshold (1/alpha = 1/0.01 = 100)
    ax1.axhline(y=100, color='black', linestyle='--', label='Rejection threshold (1/α)')
    ax1.set_xlabel('Steps (Pairs)')
    ax1.set_ylabel('Accumulated E-value')
    ax1.set_title('E-value Accumulation Trajectories')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final accumulated e-value bar plot
    concept_ids = list(results.keys())
    final_e_values = [results[cid]['accumulated_e'] for cid in concept_ids]
    colors = ['red' if results[cid]['is_true_concept'] else 'blue' for cid in concept_ids]
    
    bars = ax2.bar(concept_ids, final_e_values, color=colors, alpha=0.7)
    ax2.axhline(y=100, color='black', linestyle='--', label='Rejection threshold')
    ax2.set_xlabel('Concept ID')
    ax2.set_ylabel('Final Accumulated E-value')
    ax2.set_title('Final E-values by Concept')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    true_patch = mpatches.Patch(color='red', alpha=0.7, label='True concepts')
    false_patch = mpatches.Patch(color='blue', alpha=0.7, label='Other concepts')
    ax1.legend(handles=[true_patch, false_patch] + ax1.get_legend().get_patches())
    ax2.legend(handles=[true_patch, false_patch] + ax2.get_legend().get_patches())
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/single_concept_evalue_results.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_single_concept_results(results, output_dir):
    """Visualize single concept importance results."""
    print("\nVisualizing single concept results...")
    
    # Wealth trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot wealth trajectories
    for concept_id, result in results.items():
        trajectory = result['wealth_trajectory']
        if len(trajectory) > 0:
            steps = range(1, len(trajectory) + 1)
            color = 'red' if result['is_true_concept'] else 'blue'
            alpha = 1.0 if result['is_true_concept'] else 0.3
            linewidth = 2 if result['is_true_concept'] else 1
            
            ax1.plot(steps, trajectory, color=color, alpha=alpha, 
                    linewidth=linewidth, label=f'Concept {concept_id}' if result['is_true_concept'] else "")
    
    # Add rejection threshold
    ax1.axhline(y=1/0.01, color='black', linestyle='--', label='Rejection threshold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Wealth')
    ax1.set_title('Wealth Trajectories')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final wealth bar plot
    concept_ids = list(results.keys())
    final_wealths = [results[cid]['final_wealth'] for cid in concept_ids]
    colors = ['red' if results[cid]['is_true_concept'] else 'blue' for cid in concept_ids]
    
    bars = ax2.bar(concept_ids, final_wealths, color=colors, alpha=0.7)
    ax2.axhline(y=1/0.01, color='black', linestyle='--', label='Rejection threshold')
    ax2.set_xlabel('Concept ID')
    ax2.set_ylabel('Final Wealth')
    ax2.set_title('Final Wealth by Concept')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/single_concept_results.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_individual_concept_distribution_results(dist_results, true_concepts, output_dir):
    """Visualize individual concept distribution importance results with Hoeffding bounds."""
    print("Visualizing individual concept distribution results...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Extract data
    concept_names = list(dist_results.keys())
    concept_indices = [dist_results[name]['concept_idx'] for name in concept_names]
    empirical_probs = [dist_results[name]['empirical_probability'] for name in concept_names]
    lower_bounds = [dist_results[name]['lower_bound'] for name in concept_names]
    upper_bounds = [dist_results[name]['upper_bound'] for name in concept_names]
    mass_important = [dist_results[name]['mass_is_important'] for name in concept_names]
    is_true = [dist_results[name]['is_true_concept'] for name in concept_names]
    
    # Color by whether concept is true
    colors = ['red' if true else 'blue' for true in is_true]
    
    # Empirical probability plot
    x_pos = concept_indices
    bars1 = ax1.bar(x_pos, empirical_probs, color=colors, alpha=0.7)
    
    # Add error bars for confidence intervals
    yerr = [[ep - lb for ep, lb in zip(empirical_probs, lower_bounds)],
            [ub - ep for ub, ep in zip(upper_bounds, empirical_probs)]]
    ax1.errorbar(x_pos, empirical_probs, yerr=yerr, fmt='none', color='black', capsize=5)
    
    ax1.axhline(y=0.9, color='green', linestyle='--', label='90% mass threshold')
    ax1.set_xlabel('Concept Index')
    ax1.set_ylabel('Empirical Probability')
    ax1.set_title('Individual Concept Distribution Importance')
    ax1.set_xticks(x_pos)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Lower bound plot
    bars2 = ax2.bar(x_pos, lower_bounds, color=colors, alpha=0.7)
    ax2.axhline(y=0.9, color='green', linestyle='--', label='90% mass threshold')
    ax2.set_xlabel('Concept Index')
    ax2.set_ylabel('95% Confidence Lower Bound')
    ax2.set_title('Hoeffding Lower Bounds')
    ax2.set_xticks(x_pos)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mass importance classification
    mass_colors = ['green' if important else 'red' for important in mass_important]
    bars3 = ax3.bar(x_pos, [1 if important else 0 for important in mass_important], 
                   color=mass_colors, alpha=0.7)
    ax3.set_xlabel('Concept Index')
    ax3.set_ylabel('90% Mass is Important')
    ax3.set_title('Distribution Mass Importance')
    ax3.set_xticks(x_pos)
    ax3.set_ylim([-0.1, 1.1])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    ax3.grid(True, alpha=0.3)
    
    # Add overall legend
    import matplotlib.patches as mpatches
    true_patch = mpatches.Patch(color='red', alpha=0.7, label='True concepts')
    false_patch = mpatches.Patch(color='blue', alpha=0.7, label='False concepts')
    fig.legend(handles=[true_patch, false_patch], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_dir}/individual_concept_distribution_results.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_distribution_results_hoeffding(dist_results, output_dir):
    """Visualize concept distribution importance results with Hoeffding bounds."""
    print("Visualizing distribution results with Hoeffding bounds...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Extract data
    subset_names = list(dist_results.keys())
    empirical_probs = [dist_results[name]['empirical_probability'] for name in subset_names]
    lower_bounds = [dist_results[name]['lower_bound'] for name in subset_names]
    upper_bounds = [dist_results[name]['upper_bound'] for name in subset_names]
    mass_important = [dist_results[name]['mass_is_important'] for name in subset_names]
    contains_true = [dist_results[name]['contains_true_concepts'] for name in subset_names]
    
    # Color by whether subset contains true concepts
    colors = ['red' if true else 'blue' for true in contains_true]
    
    # Empirical probability plot
    x_pos = range(len(subset_names))
    bars1 = ax1.bar(x_pos, empirical_probs, color=colors, alpha=0.7)
    
    # Add error bars for confidence intervals
    yerr = [[ep - lb for ep, lb in zip(empirical_probs, lower_bounds)],
            [ub - ep for ub, ep in zip(upper_bounds, empirical_probs)]]
    ax1.errorbar(x_pos, empirical_probs, yerr=yerr, fmt='none', color='black', capsize=5)
    
    ax1.axhline(y=0.9, color='green', linestyle='--', label='90% mass threshold')
    ax1.set_xlabel('Concept Subset')
    ax1.set_ylabel('Empirical Probability')
    ax1.set_title('Empirical Probability of Importance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(subset_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Lower bound plot
    bars2 = ax2.bar(x_pos, lower_bounds, color=colors, alpha=0.7)
    ax2.axhline(y=0.9, color='green', linestyle='--', label='90% mass threshold')
    ax2.set_xlabel('Concept Subset')
    ax2.set_ylabel('95% Confidence Lower Bound')
    ax2.set_title('Hoeffding Lower Bounds')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(subset_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mass importance classification
    mass_colors = ['green' if important else 'red' for important in mass_important]
    bars3 = ax3.bar(x_pos, [1 if important else 0 for important in mass_important], 
                   color=mass_colors, alpha=0.7)
    ax3.set_xlabel('Concept Subset')
    ax3.set_ylabel('90% Mass is Important')
    ax3.set_title('Mass Importance Classification')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(subset_names, rotation=45, ha='right')
    ax3.set_ylim([-0.1, 1.1])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    ax3.grid(True, alpha=0.3)
    
    # Add overall legend
    import matplotlib.patches as mpatches
    true_patch = mpatches.Patch(color='red', alpha=0.7, label='Contains true concepts')
    false_patch = mpatches.Patch(color='blue', alpha=0.7, label='No true concepts')
    fig.legend(handles=[true_patch, false_patch], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_dir}/distribution_results_hoeffding.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_distribution_results(dist_results, output_dir):
    """Visualize concept distribution importance results."""
    print("Visualizing distribution results...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    subset_names = list(dist_results.keys())
    rejection_rates = [dist_results[name]['rejection_rate_hat'] for name in subset_names]
    lower_bounds = [dist_results[name]['rejection_rate_lower_bound'] for name in subset_names]
    contains_true = [dist_results[name]['contains_true_concepts'] for name in subset_names]
    
    # Color by whether subset contains true concepts
    colors = ['red' if true else 'blue' for true in contains_true]
    
    # Rejection rate plot
    x_pos = range(len(subset_names))
    bars1 = ax1.bar(x_pos, rejection_rates, color=colors, alpha=0.7)
    ax1.set_xlabel('Concept Subset')
    ax1.set_ylabel('Rejection Rate')
    ax1.set_title('Estimated Rejection Rates')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(subset_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Lower bound plot
    bars2 = ax2.bar(x_pos, lower_bounds, color=colors, alpha=0.7)
    ax2.set_xlabel('Concept Subset')
    ax2.set_ylabel('Lower Bound on Rejection Rate')
    ax2.set_title('95% Confidence Lower Bounds')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(subset_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    true_patch = mpatches.Patch(color='red', alpha=0.7, label='Contains true concepts')
    false_patch = mpatches.Patch(color='blue', alpha=0.7, label='No true concepts')
    ax1.legend(handles=[true_patch, false_patch])
    ax2.legend(handles=[true_patch, false_patch])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribution_results.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def create_simple_report(single_results, dist_results, true_concepts, output_dir="results"):
    """Create a simple summary report."""
    report_path = Path(output_dir) / "robust_importance_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("ROBUST LINEAR IMPORTANCE TESTING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"TRUE IMPORTANT CONCEPTS: {true_concepts}\n\n")
        
        f.write("SINGLE CONCEPT RESULTS (E-VALUE ACCUMULATION):\n")
        f.write("-" * 40 + "\n")
        
        # Single concept summary
        rejected_concepts = [cid for cid, result in single_results.items() if result['rejected']]
        true_positives = sum(1 for cid in true_concepts if single_results[cid]['rejected'])
        false_positives = sum(1 for cid in rejected_concepts if cid not in true_concepts)
        
        f.write(f"Rejected concepts: {rejected_concepts}\n")
        f.write(f"True positives: {true_positives}/{len(true_concepts)}\n")
        f.write(f"False positives: {false_positives}\n\n")
        
        for cid, result in single_results.items():
            status = "REJECTED" if result['rejected'] else "NOT REJECTED" 
            true_marker = "★" if cid in true_concepts else " "
            e_val = result['accumulated_e']
            step = result['rejection_step'] if result['rejection_step'] else "N/A"
            f.write(f"Concept {cid} {true_marker}: {status} (E={e_val:.3f}, step={step})\n")
        
        f.write(f"\nROBUST INDIVIDUAL CONCEPT DISTRIBUTION RESULTS (HOEFFDING):\n")
        f.write("-" * 60 + "\n")
        
        for name, result in dist_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Concept index: {result['concept_indices'][0]}\n")
            f.write(f"  Is true concept: {result['is_true_concept']}\n")
            f.write(f"  Empirical probability: {result['empirical_probability']:.3f}\n")
            f.write(f"  95% Confidence interval: [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]\n")
            f.write(f"  90% of mass is important: {result['mass_is_important']}\n")
            f.write(f"  Hoeffding epsilon: {result['epsilon']:.3f}\n")
            f.write(f"  Samples used: {result['n_samples']}\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write("• E-value accumulation tests individual concept importance\n")
        f.write("• Hoeffding inequality provides confidence bounds on concept distributions\n") 
        f.write("• 90% mass threshold determines if concept distribution is robustly important\n")
        f.write("• True concept combinations should exceed mass threshold with high confidence\n")
    
    print(f"Robust importance report saved to {report_path}")

def main():
    """Run the simplified semantic importance testing."""
    print("SIMPLIFIED LINEAR ROBUST IMPORTANCE TESTING")
    print("=" * 60)
    
    # Create output directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Setup simple linear data
        Z, Y, true_concepts, concept_bank = setup_simple_linear_data()
        
        # Visualize linear relationships
        visualize_linear_relationships(Z, Y, true_concepts)
        
        # Test single concept importance
        single_results = test_single_concept_importance(Z, Y, true_concepts)
        
        # Test concept distribution importance
        dist_results = test_concept_distribution_importance(Z, Y, true_concepts, concept_bank)
        
        # Create simple report
        create_simple_report(single_results, dist_results, true_concepts)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        print("✅ Simplified testing completed successfully!")
        print(f"📊 True concepts: {true_concepts}")
        
        # Single concept summary
        rejected_single = [cid for cid, result in single_results.items() if result['rejected']]
        print(f"🔍 Single concepts rejected: {rejected_single}")
        
        # Distribution summary
        print("📈 Individual concept distribution results:")
        for name, result in dist_results.items():
            prob = result['empirical_probability']
            lower_bound = result['lower_bound']
            mass_important = result['mass_is_important']
            is_true = result['is_true_concept']
            concept_idx = result['concept_indices'][0]
            marker = "★" if is_true else " "
            importance_marker = "🟢" if mass_important else "🔴"
            print(f"   {marker} Concept {concept_idx}: prob={prob:.3f}, bound={lower_bound:.3f} {importance_marker}")
        
        print("\n📁 Results saved in 'results/' directory")
        print("\n🔬 METHODOLOGY:")
        print("• E-value accumulation for linear relation testing")
        print("• Individual concept distribution sampling with noise")
        print("• Hoeffding inequality for robust importance bounds")
        print("• 90% mass threshold for concept distribution importance")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
