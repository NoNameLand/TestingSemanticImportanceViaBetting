#!/usr/bin/env python3
"""
Enhanced test script for semantic importance testing using the IBYDMT framework.
This creates comprehensive toy examples with multiple CAV models, visualizations,
and robust importance testing.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from pathlib import Path

from ibymdt_cav import (
    ConceptBank, ConceptDist, SKITGlobal, SKITConditional, XSKITLocal,
    JointHZConditional, greedy_e_fdr, secure_importance
)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_enhanced_toy_data():
    """Create larger synthetic dataset with multiple scenarios."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Enhanced synthetic dataset
    n_samples = 2000  # Larger dataset
    d_hidden = 50     # Higher dimensional hidden representations
    m_concepts = 10   # More concepts
    
    print(f"Generating enhanced data: {n_samples} samples, {d_hidden}D hidden, {m_concepts} concepts")
    
    # Create multiple concept banks (different CAV models)
    concept_banks = {}
    
    # 1. Random concept bank
    C_random = np.random.randn(m_concepts, d_hidden)
    concept_banks['random'] = ConceptBank(C_random)
    
    # 2. Structured concept bank (concepts have interpretable structure)
    C_structured = np.zeros((m_concepts, d_hidden))
    for i in range(m_concepts):
        # Each concept focuses on different parts of the hidden space
        start_idx = (i * d_hidden) // m_concepts
        end_idx = ((i + 1) * d_hidden) // m_concepts
        C_structured[i, start_idx:end_idx] = np.random.randn(end_idx - start_idx)
    concept_banks['structured'] = ConceptBank(C_structured)
    
    # 3. Correlated concept bank (concepts are somewhat correlated)
    base_concepts = np.random.randn(3, d_hidden)  # 3 base patterns
    C_correlated = np.zeros((m_concepts, d_hidden))
    for i in range(m_concepts):
        # Mix base patterns with some noise
        weights = np.random.dirichlet([1, 1, 1])  # Random mixing weights
        C_correlated[i] = weights @ base_concepts + 0.3 * np.random.randn(d_hidden)
    concept_banks['correlated'] = ConceptBank(C_correlated)
    
    # Generate hidden representations H with different patterns
    scenarios = {}
    
    # Scenario 1: Linear relationships
    H1 = np.random.randn(n_samples, d_hidden)
    Z1_dict = {name: bank.project(H1) for name, bank in concept_banks.items()}
    # Y depends strongly on concepts 0, 2, and 5
    Y1 = (3.0 * Z1_dict['random'][:, 0] + 
          2.0 * Z1_dict['random'][:, 2] + 
          1.5 * Z1_dict['random'][:, 5] + 
          0.2 * np.random.randn(n_samples))
    scenarios['linear'] = {'H': H1, 'Z_dict': Z1_dict, 'Y': Y1, 'true_concepts': [0, 2, 5]}
    
    # Scenario 2: Nonlinear relationships
    H2 = np.random.randn(n_samples, d_hidden)
    Z2_dict = {name: bank.project(H2) for name, bank in concept_banks.items()}
    # Y has nonlinear dependency on concepts 1, 3, 7
    Y2 = (2.0 * np.tanh(Z2_dict['random'][:, 1]) + 
          1.5 * Z2_dict['random'][:, 3]**2 + 
          1.0 * np.sin(Z2_dict['random'][:, 7] * np.pi) + 
          0.3 * np.random.randn(n_samples))
    scenarios['nonlinear'] = {'H': H2, 'Z_dict': Z2_dict, 'Y': Y2, 'true_concepts': [1, 3, 7]}
    
    # Scenario 3: Interaction effects
    H3 = np.random.randn(n_samples, d_hidden)
    Z3_dict = {name: bank.project(H3) for name, bank in concept_banks.items()}
    # Y depends on interactions between concepts
    Y3 = (2.0 * Z3_dict['random'][:, 0] * Z3_dict['random'][:, 4] + 
          1.5 * Z3_dict['random'][:, 6] + 
          0.25 * np.random.randn(n_samples))
    scenarios['interaction'] = {'H': H3, 'Z_dict': Z3_dict, 'Y': Y3, 'true_concepts': [0, 4, 6]}
    
    print(f"Created {len(scenarios)} scenarios with {len(concept_banks)} concept banks each")
    print("Scenarios:")
    for name, data in scenarios.items():
        print(f"  {name}: true concepts {data['true_concepts']}")
    
    return scenarios, concept_banks

def visualize_concept_distributions(scenarios, concept_banks, output_dir="results/visualizations"):
    """Visualize concept distributions and their relationships with Y."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VISUALIZING CONCEPT DISTRIBUTIONS")
    print("="*60)
    
    for scenario_name, scenario_data in scenarios.items():
        Y = scenario_data['Y']
        true_concepts = scenario_data['true_concepts']
        
        for bank_name, Z_dict in scenario_data['Z_dict'].items():
            Z = Z_dict
            
            # Create figure for this scenario-bank combination
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(f'Concept Distributions: {scenario_name} scenario, {bank_name} bank', fontsize=16)
            
            for j in range(min(10, Z.shape[1])):  # Show first 10 concepts
                row, col = j // 4, j % 4
                if row >= 3: break
                
                ax = axes[row, col]
                
                # Plot concept distribution
                ax.hist(Z[:, j], bins=50, alpha=0.7, density=True, label=f'Concept {j}')
                
                # Overlay normal fit
                mu, sigma = stats.norm.fit(Z[:, j])
                x = np.linspace(Z[:, j].min(), Z[:, j].max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', alpha=0.8, label=f'Normal fit')
                
                # Highlight if it's a true concept
                if j in true_concepts:
                    ax.set_facecolor('#ffe6e6')  # Light red background
                    ax.set_title(f'Concept {j} (TRUE)', fontweight='bold', color='red')
                else:
                    ax.set_title(f'Concept {j}')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for j in range(10, 12):
                if j < 12:
                    row, col = j // 4, j % 4
                    if row < 3:
                        fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/concept_dist_{scenario_name}_{bank_name}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create concept-Y correlation heatmaps
    for scenario_name, scenario_data in scenarios.items():
        Y = scenario_data['Y']
        true_concepts = scenario_data['true_concepts']
        
        fig, axes = plt.subplots(1, len(concept_banks), figsize=(15, 5))
        if len(concept_banks) == 1:
            axes = [axes]
        
        for idx, (bank_name, Z) in enumerate(scenario_data['Z_dict'].items()):
            # Compute correlations
            correlations = [np.corrcoef(Y, Z[:, j])[0, 1] for j in range(Z.shape[1])]
            
            # Create heatmap data
            corr_matrix = np.array(correlations).reshape(1, -1)
            
            # Plot heatmap
            im = axes[idx].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            axes[idx].set_title(f'{bank_name} bank')
            axes[idx].set_xlabel('Concept Index')
            axes[idx].set_yticks([0])
            axes[idx].set_yticklabels(['Y Correlation'])
            
            # Set x-ticks
            axes[idx].set_xticks(range(len(correlations)))
            axes[idx].set_xticklabels(range(len(correlations)))
            
            # Highlight true concepts
            for true_concept in true_concepts:
                if true_concept < len(correlations):
                    axes[idx].add_patch(plt.Rectangle((true_concept-0.4, -0.4), 0.8, 0.8, 
                                                    fill=False, edgecolor='yellow', lw=3))
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Concept-Y Correlations: {scenario_name} scenario', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlations_{scenario_name}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def test_enhanced_global_skit(scenarios, concept_banks, output_dir="results/skit_results"):
    """Enhanced global SKIT testing with multiple scenarios and banks."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ENHANCED GLOBAL SKIT TESTING")
    print("="*60)
    
    all_results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n--- Testing {scenario_name.upper()} scenario ---")
        true_concepts = scenario_data['true_concepts']
        Y = scenario_data['Y']
        
        scenario_results = {}
        
        for bank_name, Z in scenario_data['Z_dict'].items():
            print(f"\nUsing {bank_name} concept bank:")
            bank_results = {}
            
            # Test each concept
            for j in range(Z.shape[1]):
                # Create SKIT instance
                skit = SKITGlobal(alpha=0.01, use_ons=True)  # Lower alpha for more power
                
                # Randomly pair up samples
                indices = np.random.permutation(len(Y))
                n_pairs = len(Y) // 2
                
                rejection_step = None
                wealth_trajectory = []
                
                for i in range(n_pairs):
                    idx1, idx2 = indices[2*i], indices[2*i + 1]
                    y1, y2 = Y[idx1], Y[idx2]
                    z1, z2 = Z[idx1, j], Z[idx2, j]
                    
                    kappa, K = skit.step_pair(y1, z1, y2, z2)
                    wealth_trajectory.append(K)
                    
                    if skit.rejected() and rejection_step is None:
                        rejection_step = i + 1
                
                bank_results[j] = {
                    'rejected': skit.rejected(),
                    'final_wealth': skit.K,
                    'rejection_step': rejection_step,
                    'wealth_trajectory': wealth_trajectory,
                    'is_true_concept': j in true_concepts
                }
                
                status = "REJECTED" if skit.rejected() else "NOT REJECTED"
                true_marker = "★" if j in true_concepts else " "
                print(f"  Concept {j:2d} {true_marker}: K={skit.K:8.3f} {status}")
            
            scenario_results[bank_name] = bank_results
        
        all_results[scenario_name] = scenario_results
    
    # Visualize results
    visualize_skit_results(all_results, output_dir)
    
    return all_results

def visualize_skit_results(all_results, output_dir):
    """Create comprehensive visualizations of SKIT results."""
    print(f"\nCreating SKIT result visualizations...")
    
    # 1. Wealth trajectory plots
    for scenario_name, scenario_results in all_results.items():
        fig, axes = plt.subplots(2, len(scenario_results), figsize=(15, 10))
        if len(scenario_results) == 1:
            axes = axes.reshape(-1, 1)
        
        for col, (bank_name, bank_results) in enumerate(scenario_results.items()):
            # Top plot: wealth trajectories for true concepts
            ax_true = axes[0, col]
            ax_false = axes[1, col]
            
            for concept_id, result in bank_results.items():
                trajectory = result['wealth_trajectory']
                if len(trajectory) > 0:
                    steps = range(1, len(trajectory) + 1)
                    if result['is_true_concept']:
                        ax_true.plot(steps, trajectory, label=f'Concept {concept_id}', linewidth=2)
                    else:
                        ax_false.plot(steps, trajectory, alpha=0.3, linewidth=1)
            
            # Add rejection threshold
            ax_true.axhline(y=1/0.01, color='red', linestyle='--', label='Rejection threshold')
            ax_false.axhline(y=1/0.01, color='red', linestyle='--', label='Rejection threshold')
            
            ax_true.set_title(f'{bank_name}: True Concepts')
            ax_true.set_ylabel('Wealth')
            ax_true.legend()
            ax_true.grid(True, alpha=0.3)
            ax_true.set_yscale('log')
            
            ax_false.set_title(f'{bank_name}: False Concepts')
            ax_false.set_xlabel('Steps')
            ax_false.set_ylabel('Wealth')
            ax_false.grid(True, alpha=0.3)
            ax_false.set_yscale('log')
        
        plt.suptitle(f'Wealth Trajectories: {scenario_name} scenario', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/wealth_trajectories_{scenario_name}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Final wealth comparison
    fig, axes = plt.subplots(1, len(all_results), figsize=(15, 5))
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, (scenario_name, scenario_results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Collect data for plotting
        concept_ids = list(range(10))  # Assume 10 concepts
        bank_names = list(scenario_results.keys())
        
        # Create grouped bar plot
        x = np.arange(len(concept_ids))
        width = 0.25
        
        for bank_idx, (bank_name, bank_results) in enumerate(scenario_results.items()):
            wealths = [bank_results[cid]['final_wealth'] for cid in concept_ids]
            true_concepts = [bank_results[cid]['is_true_concept'] for cid in concept_ids]
            
            # Color bars differently for true vs false concepts
            colors = ['red' if is_true else 'blue' for is_true in true_concepts]
            
            bars = ax.bar(x + bank_idx * width, wealths, width, label=bank_name, 
                         color=colors, alpha=0.7)
        
        ax.axhline(y=1/0.01, color='black', linestyle='--', label='Rejection threshold')
        ax.set_xlabel('Concept ID')
        ax.set_ylabel('Final Wealth')
        ax.set_title(f'{scenario_name} scenario')
        ax.set_xticks(x + width)
        ax.set_xticklabels(concept_ids)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Final Wealth by Concept', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_wealths.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def test_robust_importance(scenarios, concept_banks, output_dir="results/robust_importance"):
    """Test robust/secure importance with multiple runs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ROBUST IMPORTANCE TESTING")
    print("="*60)
    
    robust_results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n--- Robust testing: {scenario_name.upper()} scenario ---")
        Y = scenario_data['Y']
        true_concepts = scenario_data['true_concepts']
        
        scenario_robust = {}
        
        for bank_name, Z in scenario_data['Z_dict'].items():
            print(f"\nUsing {bank_name} concept bank:")
            
            bank_robust = {}
            
            for j in range(min(5, Z.shape[1])):  # Test first 5 concepts for speed
                print(f"  Testing concept {j}... ", end="")
                
                def skit_run_function():
                    """Single SKIT run for concept j."""
                    skit = SKITGlobal(alpha=0.01, use_ons=True)
                    
                    # Use subset of data for faster computation
                    n_subset = min(500, len(Y))
                    indices = np.random.choice(len(Y), n_subset, replace=False)
                    Y_sub = Y[indices]
                    Z_sub = Z[indices, j]
                    
                    # Pair up samples
                    pairs = len(indices) // 2
                    for i in range(pairs):
                        idx1, idx2 = indices[2*i], indices[2*i + 1]
                        y1, y2 = Y_sub[2*i], Y_sub[2*i + 1]
                        z1, z2 = Z_sub[2*i], Z_sub[2*i + 1]
                        
                        skit.step_pair(y1, z1, y2, z2)
                        
                        if skit.rejected():
                            break
                    
                    return skit.rejected()
                
                # Run robust importance test
                robust_result = secure_importance(skit_run_function, R=20, delta=0.05)
                robust_result['is_true_concept'] = j in true_concepts
                bank_robust[j] = robust_result
                
                status = f"P(reject)≥{robust_result['rejection_rate_lower_bound']:.3f}"
                true_marker = "★" if j in true_concepts else " "
                print(f"{true_marker} {status}")
            
            scenario_robust[bank_name] = bank_robust
        
        robust_results[scenario_name] = scenario_robust
    
    # Visualize robust results
    visualize_robust_results(robust_results, output_dir)
    
    return robust_results

def visualize_robust_results(robust_results, output_dir):
    """Visualize robust importance results."""
    print(f"\nCreating robust importance visualizations...")
    
    # Create summary plot
    fig, axes = plt.subplots(2, len(robust_results), figsize=(15, 10))
    if len(robust_results) == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (scenario_name, scenario_results) in enumerate(robust_results.items()):
        ax_est = axes[0, col]
        ax_bound = axes[1, col]
        
        for bank_name, bank_results in scenario_results.items():
            concept_ids = list(bank_results.keys())
            rejection_rates = [bank_results[cid]['rejection_rate_hat'] for cid in concept_ids]
            lower_bounds = [bank_results[cid]['rejection_rate_lower_bound'] for cid in concept_ids]
            is_true = [bank_results[cid]['is_true_concept'] for cid in concept_ids]
            
            # Color by true/false concept
            colors = ['red' if true else 'blue' for true in is_true]
            
            ax_est.bar([f'C{cid}' for cid in concept_ids], rejection_rates, 
                      color=colors, alpha=0.7, label=bank_name)
            ax_bound.bar([f'C{cid}' for cid in concept_ids], lower_bounds,
                        color=colors, alpha=0.7, label=bank_name)
        
        ax_est.set_title(f'{scenario_name}: Estimated Rejection Rate')
        ax_est.set_ylabel('Rejection Rate')
        ax_est.legend()
        ax_est.grid(True, alpha=0.3)
        
        ax_bound.set_title(f'{scenario_name}: Lower Bound')
        ax_bound.set_xlabel('Concept')
        ax_bound.set_ylabel('Lower Bound')
        ax_bound.legend()
        ax_bound.grid(True, alpha=0.3)
    
    plt.suptitle('Robust Importance Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/robust_importance_summary.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(scenarios, concept_banks, global_results, robust_results, 
                               output_dir="results"):
    """Create a comprehensive report of all results."""
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE REPORT")
    print("="*60)
    
    report_path = Path(output_dir) / "comprehensive_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE SEMANTIC IMPORTANCE TESTING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write(f"Number of scenarios: {len(scenarios)}\n")
        f.write(f"Number of concept banks: {len(concept_banks)}\n")
        
        for scenario_name, scenario_data in scenarios.items():
            f.write(f"\n{scenario_name} scenario:\n")
            f.write(f"  True concepts: {scenario_data['true_concepts']}\n")
            f.write(f"  Samples: {len(scenario_data['Y'])}\n")
            f.write(f"  Y statistics: mean={scenario_data['Y'].mean():.3f}, "
                   f"std={scenario_data['Y'].std():.3f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("GLOBAL SKIT RESULTS:\n\n")
        
        for scenario_name, scenario_results in global_results.items():
            f.write(f"\n{scenario_name.upper()} SCENARIO:\n")
            true_concepts = scenarios[scenario_name]['true_concepts']
            
            for bank_name, bank_results in scenario_results.items():
                f.write(f"\n  {bank_name} bank:\n")
                
                # Calculate performance metrics
                true_positives = sum(1 for cid in true_concepts 
                                   if bank_results[cid]['rejected'])
                false_positives = sum(1 for cid in range(10) 
                                    if cid not in true_concepts and bank_results[cid]['rejected'])
                true_negatives = sum(1 for cid in range(10) 
                                   if cid not in true_concepts and not bank_results[cid]['rejected'])
                false_negatives = sum(1 for cid in true_concepts 
                                    if not bank_results[cid]['rejected'])
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"    Precision: {precision:.3f}\n")
                f.write(f"    Recall: {recall:.3f}\n")
                f.write(f"    F1-score: {f1_score:.3f}\n")
                f.write(f"    TP: {true_positives}, FP: {false_positives}, "
                       f"TN: {true_negatives}, FN: {false_negatives}\n")
                
                f.write(f"    Rejected concepts: {[cid for cid in range(10) if bank_results[cid]['rejected']]}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ROBUST IMPORTANCE RESULTS:\n\n")
        
        for scenario_name, scenario_results in robust_results.items():
            f.write(f"\n{scenario_name.upper()} SCENARIO:\n")
            
            for bank_name, bank_results in scenario_results.items():
                f.write(f"\n  {bank_name} bank:\n")
                
                for concept_id, result in bank_results.items():
                    true_marker = "★" if result['is_true_concept'] else " "
                    f.write(f"    Concept {concept_id} {true_marker}: "
                           f"rate={result['rejection_rate_hat']:.3f}, "
                           f"lower_bound={result['rejection_rate_lower_bound']:.3f}\n")
    
    print(f"Comprehensive report saved to {report_path}")

def main():
    """Run the enhanced semantic importance testing suite."""
    print("ENHANCED SEMANTIC IMPORTANCE TESTING SUITE")
    print("=" * 60)
    print("Creating comprehensive tests with multiple CAV models and robust importance")
    
    # Create output directories
    Path("results").mkdir(exist_ok=True)
    Path("results/visualizations").mkdir(exist_ok=True)
    Path("results/skit_results").mkdir(exist_ok=True)
    Path("results/robust_importance").mkdir(exist_ok=True)
    
    try:
        # Setup enhanced data
        scenarios, concept_banks = setup_enhanced_toy_data()
        
        # Visualize concept distributions
        visualize_concept_distributions(scenarios, concept_banks)
        
        # Run enhanced global SKIT testing
        global_results = test_enhanced_global_skit(scenarios, concept_banks)
        
        # Run robust importance testing
        robust_results = test_robust_importance(scenarios, concept_banks)
        
        # Create comprehensive report
        create_comprehensive_report(scenarios, concept_banks, global_results, robust_results)
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print("="*60)
        print("✅ Enhanced testing suite completed successfully!")
        print("📊 Results saved in the 'results/' directory:")
        print("   - results/visualizations/ : Concept distributions and correlations")
        print("   - results/skit_results/ : SKIT wealth trajectories and results")
        print("   - results/robust_importance/ : Robust importance analysis")
        print("   - results/comprehensive_report.txt : Detailed text report")
        
    except Exception as e:
        print(f"\n❌ Enhanced test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
