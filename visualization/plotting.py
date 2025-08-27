"""
Visualization module for experiment results.
Creates plots and visualizations for concept importance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class ExperimentVisualizer:
    """Creates comprehensive visualizations for experiment results."""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self, scenario_data: Dict[str, Any], 
                                experiment_results: Dict[str, Any],
                                experiment_name: str = "experiment"):
        """Create all standard visualizations for an experiment."""
        print(f"\nCreating visualizations for {experiment_name}...")
        
        # Create experiment-specific output directory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Concept distributions
        if 'Z_dict' in scenario_data:
            self.plot_concept_distributions(scenario_data, exp_dir)
        
        # 2. Concept-Y correlations
        if 'Z_dict' in scenario_data and 'Y' in scenario_data:
            self.plot_concept_correlations(scenario_data, exp_dir)
        
        # 3. SKIT wealth trajectories
        if 'scenarios' in experiment_results:
            self.plot_skit_trajectories(experiment_results, exp_dir)
        
        # 4. Performance comparison
        if 'summary' in experiment_results:
            self.plot_performance_comparison(experiment_results, exp_dir)
        
        # 5. Robust importance results
        if 'scenarios' in experiment_results:
            self.plot_robust_importance(experiment_results, exp_dir)
        
        print(f"Visualizations saved to: {exp_dir}")
    
    def plot_concept_distributions(self, scenario_data: Dict[str, Any], output_dir: Path):
        """Plot distributions of concept values."""
        Y = scenario_data.get('Y')
        true_concepts = scenario_data.get('true_concepts', [])
        
        for bank_name, Z in scenario_data['Z_dict'].items():
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(f'Concept Distributions: {bank_name} bank', fontsize=16)
            
            for j in range(min(12, Z.shape[1])):
                row, col = j // 4, j % 4
                ax = axes[row, col]
                
                # Plot concept distribution
                ax.hist(Z[:, j], bins=50, alpha=0.7, density=True, 
                       color='red' if j in true_concepts else 'blue')
                
                # Overlay normal fit
                mu, sigma = stats.norm.fit(Z[:, j])
                x = np.linspace(Z[:, j].min(), Z[:, j].max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', alpha=0.8, 
                       linewidth=2, label='Normal fit')
                
                # Highlight true concepts
                if j in true_concepts:
                    ax.set_facecolor('#ffe6e6')
                    ax.set_title(f'Concept {j} (TRUE)', fontweight='bold', color='red')
                else:
                    ax.set_title(f'Concept {j}')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for j in range(12, 12):
                if j < 12:
                    row, col = j // 4, j % 4
                    if row < 3 and col < 4:
                        fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            plt.savefig(output_dir / f"concept_distributions_{bank_name}.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_concept_correlations(self, scenario_data: Dict[str, Any], output_dir: Path):
        """Plot correlations between concepts and target Y."""
        Y = scenario_data['Y']
        true_concepts = scenario_data.get('true_concepts', [])
        
        fig, axes = plt.subplots(1, len(scenario_data['Z_dict']), figsize=(15, 5))
        if len(scenario_data['Z_dict']) == 1:
            axes = [axes]
        
        for idx, (bank_name, Z) in enumerate(scenario_data['Z_dict'].items()):
            # Compute correlations
            correlations = []
            p_values = []
            for j in range(Z.shape[1]):
                corr, p_val = stats.pearsonr(Y, Z[:, j])
                correlations.append(corr)
                p_values.append(p_val)
            
            # Create correlation plot
            concept_ids = list(range(len(correlations)))
            colors = ['red' if j in true_concepts else 'blue' for j in concept_ids]
            
            bars = axes[idx].bar(concept_ids, correlations, color=colors, alpha=0.7)
            
            # Add significance indicators
            for j, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., 
                                 height + 0.01 * np.sign(height),
                                 marker, ha='center', va='bottom' if height > 0 else 'top')
            
            axes[idx].set_title(f'{bank_name} bank')
            axes[idx].set_xlabel('Concept Index')
            axes[idx].set_ylabel('Correlation with Y')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.suptitle('Concept-Y Correlations (* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "concept_correlations.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_skit_trajectories(self, experiment_results: Dict[str, Any], output_dir: Path):
        """Plot SKIT wealth trajectories."""
        for scenario_name, scenario_data in experiment_results['scenarios'].items():
            if 'global_skit' not in scenario_data:
                continue
            
            n_banks = len(scenario_data['global_skit'])
            fig, axes = plt.subplots(2, n_banks, figsize=(5*n_banks, 10))
            # For n_banks==1, axes is a 1D array; for >1, it's 2D
            
            for col, (bank_name, bank_results) in enumerate(scenario_data['global_skit'].items()):
                ax_true = axes[0, col] if n_banks > 1 else axes[0]
                ax_false = axes[1, col] if n_banks > 1 else axes[1]
                
                # Plot trajectories for true and false concepts separately
                for concept_id, result in bank_results.items():
                    trajectory = result.get('wealth_trajectory', [])
                    if len(trajectory) > 0:
                        steps = range(1, len(trajectory) + 1)
                        if result['is_true_concept']:
                            ax_true.plot(steps, trajectory, label=f'Concept {concept_id}', 
                                       linewidth=2, alpha=0.8)
                        else:
                            ax_false.plot(steps, trajectory, alpha=0.3, linewidth=1)
                
                # Add rejection threshold
                threshold = 1 / 0.01  # Assuming alpha=0.01
                ax_true.axhline(y=threshold, color='red', linestyle='--', 
                              label='Rejection threshold')
                ax_false.axhline(y=threshold, color='red', linestyle='--', 
                               label='Rejection threshold')
                
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
            
            plt.suptitle(f'SKIT Wealth Trajectories: {scenario_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / f"skit_trajectories_{scenario_name}.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_performance_comparison(self, experiment_results: Dict[str, Any], output_dir: Path):
        """Plot performance comparison across scenarios and concept banks."""
        summary = experiment_results['summary']
        
        # Prepare data for plotting
        scenario_names = []
        bank_names = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for scenario_name, perf_data in summary['performance_by_scenario'].items():
            for bank_name, metrics in perf_data.items():
                scenario_names.append(scenario_name)
                bank_names.append(bank_name)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Scenario': scenario_names,
            'Bank': bank_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        })
        
        # Plot performance metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            if len(set(scenario_names)) > 1 and len(set(bank_names)) > 1:
                # Grouped bar plot
                pivot_df = df.pivot(index='Scenario', columns='Bank', values=metric)
                pivot_df.plot(kind='bar', ax=axes[i], rot=45)
            else:
                # Simple bar plot
                axes[i].bar(range(len(df)), df[metric])
                axes[i].set_xticks(range(len(df)))
                axes[i].set_xticklabels([f"{s}-{b}" for s, b in zip(scenario_names, bank_names)], 
                                       rotation=45)
            
            axes[i].set_title(metric)
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        plt.suptitle('Performance Comparison Across Scenarios and Banks', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Overall performance summary
        overall = summary['overall_performance']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        metrics = ['mean_precision', 'mean_recall', 'mean_f1']
        values = [overall[m] for m in metrics]
        errors = [overall[m.replace('mean', 'std')] for m in metrics]
        labels = ['Precision', 'Recall', 'F1-Score']
        
        bars = ax.bar(labels, values, yerr=errors, capsize=10, alpha=0.7, 
                     color=['red', 'blue', 'green'])
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Summary')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "overall_performance.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_robust_importance(self, experiment_results: Dict[str, Any], output_dir: Path):
        """Plot robust importance testing results."""
        for scenario_name, scenario_data in experiment_results['scenarios'].items():
            if 'robust_importance' not in scenario_data:
                continue
            
            for bank_name, robust_results in scenario_data['robust_importance'].items():
                if not robust_results:
                    continue
                concept_ids = list(robust_results.keys())
                rejection_rates = [robust_results[cid].get('rejection_rate_hat', 0.0) for cid in concept_ids]
                lower_bounds = [robust_results[cid].get('rejection_rate_lower_bound', 0.0) for cid in concept_ids]
                upper_bounds = [robust_results[cid].get('rejection_rate_upper_bound', robust_results[cid].get('rejection_rate_lower_bound', 0.0)) for cid in concept_ids]
                is_true = [robust_results[cid].get('is_true_concept', False) for cid in concept_ids]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Plot 1: Rejection rates with confidence intervals
                colors = ['red' if true else 'blue' for true in is_true]
                x_pos = np.arange(len(concept_ids))

                ax1.bar(x_pos, rejection_rates, color=colors, alpha=0.7)
                ax1.errorbar(x_pos, rejection_rates, 
                           yerr=[np.array(rejection_rates) - np.array(lower_bounds),
                                 np.array(upper_bounds) - np.array(rejection_rates)],
                           fmt='none', color='black', capsize=5)

                ax1.set_xlabel('Concept ID')
                ax1.set_ylabel('Rejection Rate')
                ax1.set_title(f'Robust Importance: {scenario_name} - {bank_name}')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([f'C{cid}' for cid in concept_ids])
                ax1.grid(True, alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', alpha=0.7, label='True Concepts'),
                                 Patch(facecolor='blue', alpha=0.7, label='False Concepts')]
                ax1.legend(handles=legend_elements)

                # Plot 2: Lower bounds only
                ax2.bar(x_pos, lower_bounds, color=colors, alpha=0.7)
                ax2.set_xlabel('Concept ID')
                ax2.set_ylabel('Lower Bound on Rejection Rate')
                ax2.set_title('Conservative Importance Estimates')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([f'C{cid}' for cid in concept_ids])
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_dir / f"robust_importance_{scenario_name}_{bank_name}.pdf", 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_relationship_analysis(self, scenario_data: Dict[str, Any], 
                                 config: Any, output_dir: Path):
        """Plot analysis of the relationships between concepts and Y."""
        if not hasattr(config, 'relationships'):
            return
        
        Y = scenario_data['Y']
        Z_dict = scenario_data['Z_dict']
        
        # For each relationship, create a detailed analysis
        for i, rel_config in enumerate(config.relationships):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Relationship Analysis: {rel_config.name}', fontsize=16)
            
            # Use primary concept bank
            primary_bank = list(Z_dict.keys())[0]
            Z = Z_dict[primary_bank]
            
            if rel_config.name == "linear":
                self._plot_linear_relationship(axes, Y, Z, rel_config)
            elif rel_config.name == "trigonometric":
                self._plot_trigonometric_relationship(axes, Y, Z, rel_config)
            elif rel_config.name == "polynomial":
                self._plot_polynomial_relationship(axes, Y, Z, rel_config)
            elif rel_config.name == "interaction":
                self._plot_interaction_relationship(axes, Y, Z, rel_config)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"relationship_analysis_{i}_{rel_config.name}.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_linear_relationship(self, axes, Y, Z, rel_config):
        """Plot linear relationship analysis."""
        indices = rel_config.concept_indices
        coeffs = rel_config.params.get("coefficients", [1.0] * len(indices))
        
        for i, (idx, coeff) in enumerate(zip(indices[:4], coeffs[:4])):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.scatter(Z[:, idx], Y, alpha=0.5, s=20)
            ax.set_xlabel(f'Concept {idx}')
            ax.set_ylabel('Y')
            ax.set_title(f'Y vs Concept {idx} (coeff={coeff:.2f})')
            
            # Add regression line
            z_sorted = np.sort(Z[:, idx])
            y_pred = coeff * z_sorted
            ax.plot(z_sorted, y_pred, 'r-', linewidth=2, alpha=0.8)
            ax.grid(True, alpha=0.3)
    
    def _plot_trigonometric_relationship(self, axes, Y, Z, rel_config):
        """Plot trigonometric relationship analysis."""
        indices = rel_config.concept_indices
        functions = rel_config.params.get("functions", ["sin"] * len(indices))
        frequencies = rel_config.params.get("frequencies", [1.0] * len(indices))
        
        for i, (idx, func, freq) in enumerate(zip(indices[:4], functions[:4], frequencies[:4])):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.scatter(Z[:, idx], Y, alpha=0.5, s=20)
            ax.set_xlabel(f'Concept {idx}')
            ax.set_ylabel('Y')
            ax.set_title(f'Y vs Concept {idx} ({func}, freq={freq:.2f})')
            
            # Add function curve
            z_sorted = np.sort(Z[:, idx])
            if func == "sin":
                y_pred = np.sin(freq * z_sorted)
            elif func == "cos":
                y_pred = np.cos(freq * z_sorted)
            else:
                y_pred = np.tanh(freq * z_sorted)
            
            ax.plot(z_sorted, y_pred, 'r-', linewidth=2, alpha=0.8)
            ax.grid(True, alpha=0.3)
    
    def _plot_polynomial_relationship(self, axes, Y, Z, rel_config):
        """Plot polynomial relationship analysis."""
        indices = rel_config.concept_indices
        degree = rel_config.params.get("degree", 2)
        
        for i, idx in enumerate(indices[:4]):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.scatter(Z[:, idx], Y, alpha=0.5, s=20)
            ax.set_xlabel(f'Concept {idx}')
            ax.set_ylabel('Y')
            ax.set_title(f'Y vs Concept {idx} (degree={degree})')
            
            # Fit and plot polynomial
            z_range = np.linspace(Z[:, idx].min(), Z[:, idx].max(), 100)
            poly_coeffs = np.polyfit(Z[:, idx], Y, degree)
            y_pred = np.polyval(poly_coeffs, z_range)
            ax.plot(z_range, y_pred, 'r-', linewidth=2, alpha=0.8)
            ax.grid(True, alpha=0.3)
    
    def _plot_interaction_relationship(self, axes, Y, Z, rel_config):
        """Plot interaction relationship analysis."""
        pairs = rel_config.concept_indices[:4]  # Limit to first 4 pairs
        
        for i, pair in enumerate(pairs):
            if i >= 4:
                break
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            idx1, idx2 = pair
            interaction = Z[:, idx1] * Z[:, idx2]  # Assume product interaction
            
            ax.scatter(interaction, Y, alpha=0.5, s=20)
            ax.set_xlabel(f'Concept {idx1} × Concept {idx2}')
            ax.set_ylabel('Y')
            ax.set_title(f'Y vs Interaction ({idx1}, {idx2})')
            ax.grid(True, alpha=0.3)
