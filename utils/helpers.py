"""
Utility functions and helper classes for the modular framework.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime


def save_experiment_config(config: Any, filepath: Union[str, Path]):
    """Save experiment configuration to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary for JSON serialization
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__.copy()
        
        # Handle nested objects
        for key, value in config_dict.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            elif isinstance(value, list):
                config_dict[key] = [
                    item.__dict__ if hasattr(item, '__dict__') else item 
                    for item in value
                ]
    else:
        config_dict = config
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_experiment_config(filepath: Union[str, Path]):
    """Load experiment configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def calculate_effect_size(Y: np.ndarray, Z: np.ndarray, concept_indices: List[int]) -> Dict[str, float]:
    """Calculate effect sizes for concepts."""
    effect_sizes = {}
    
    for idx in concept_indices:
        # Correlation-based effect size
        corr = np.corrcoef(Y, Z[:, idx])[0, 1]
        effect_sizes[f'concept_{idx}_correlation'] = float(corr)
        
        # R-squared (proportion of variance explained)
        r_squared = corr ** 2
        effect_sizes[f'concept_{idx}_r_squared'] = float(r_squared)
        
        # Cohen's conventions for correlation effect sizes
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            effect_category = "negligible"
        elif abs_corr < 0.3:
            effect_category = "small"
        elif abs_corr < 0.5:
            effect_category = "medium"
        else:
            effect_category = "large"
        
        effect_sizes[f'concept_{idx}_effect_category'] = effect_category
    
    return effect_sizes


def generate_concept_importance_ranking(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate ranking of concept importance based on multiple criteria."""
    rankings = []
    
    if 'scenarios' not in results:
        return rankings
    
    for scenario_name, scenario_data in results['scenarios'].items():
        if 'global_skit' not in scenario_data:
            continue
        
        for bank_name, bank_results in scenario_data['global_skit'].items():
            concept_scores = []
            
            for concept_id, result in bank_results.items():
                score = 0.0
                
                # SKIT rejection (high weight)
                if result['rejected']:
                    score += 1.0
                
                # Final wealth (normalized, log scale)
                wealth = result['final_wealth']
                if wealth > 1:
                    score += min(np.log10(wealth) / 3.0, 1.0)  # Cap at 1.0
                
                # Early rejection bonus
                if result['rejection_step'] is not None:
                    max_steps = len(result['wealth_trajectory'])
                    early_bonus = 1.0 - (result['rejection_step'] / max_steps)
                    score += 0.5 * early_bonus
                
                concept_scores.append({
                    'concept_id': concept_id,
                    'score': score,
                    'is_true_concept': result['is_true_concept'],
                    'rejected': result['rejected'],
                    'final_wealth': wealth
                })
            
            # Sort by score (descending)
            concept_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Add ranking information
            for rank, item in enumerate(concept_scores):
                item['rank'] = rank + 1
            
            rankings.append({
                'scenario': scenario_name,
                'bank': bank_name,
                'concept_ranking': concept_scores,
                'true_concepts': scenario_data.get('true_concepts', [])
            })
    
    return rankings


def calculate_discovery_metrics(rankings: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate concept discovery metrics from rankings."""
    metrics = {
        'mean_average_precision': 0.0,
        'mean_reciprocal_rank': 0.0,
        'precision_at_k': {},
        'recall_at_k': {}
    }
    
    if not rankings:
        return metrics
    
    all_ap = []  # Average Precision scores
    all_rr = []  # Reciprocal Rank scores
    k_values = [1, 3, 5, 10]
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    
    for ranking_data in rankings:
        concept_ranking = ranking_data['concept_ranking']
        true_concepts = set(ranking_data['true_concepts'])
        
        if not true_concepts:
            continue
        
        # Calculate Average Precision (AP)
        relevant_found = 0
        precision_sum = 0.0
        
        for i, concept_info in enumerate(concept_ranking):
            if concept_info['concept_id'] in true_concepts:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / len(true_concepts) if len(true_concepts) > 0 else 0.0
        all_ap.append(ap)
        
        # Calculate Reciprocal Rank (RR)
        rr = 0.0
        for i, concept_info in enumerate(concept_ranking):
            if concept_info['concept_id'] in true_concepts:
                rr = 1.0 / (i + 1)
                break
        all_rr.append(rr)
        
        # Calculate Precision@K and Recall@K
        for k in k_values:
            top_k_concepts = set([concept_ranking[i]['concept_id'] 
                                for i in range(min(k, len(concept_ranking)))])
            
            precision = len(top_k_concepts & true_concepts) / k
            recall = len(top_k_concepts & true_concepts) / len(true_concepts)
            
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)
    
    # Calculate means
    metrics['mean_average_precision'] = float(np.mean(all_ap)) if all_ap else 0.0
    metrics['mean_reciprocal_rank'] = float(np.mean(all_rr)) if all_rr else 0.0
    
    for k in k_values:
        metrics['precision_at_k'][k] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0
        metrics['recall_at_k'][k] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
    
    return metrics


def create_detailed_report(results: Dict[str, Any], output_path: Union[str, Path]):
    """Create a detailed analysis report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("DETAILED CONCEPT IMPORTANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'config' in results:
            config = results['config']
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Name: {getattr(config, 'name', 'Unknown')}\n")
            f.write(f"Description: {getattr(config, 'description', 'N/A')}\n")
            f.write(f"Samples: {getattr(config, 'n_samples', 'N/A')}\n")
            f.write(f"Hidden dimensions: {getattr(config, 'd_hidden', 'N/A')}\n")
            f.write(f"Concepts: {getattr(config, 'm_concepts', 'N/A')}\n")
            f.write(f"Noise std: {getattr(config, 'noise_std', 'N/A')}\n")
            
            if hasattr(config, 'relationships'):
                f.write("\nRELATIONSHIPS:\n")
                for i, rel in enumerate(config.relationships):
                    f.write(f"  {i+1}. {rel.name}: {rel.concept_indices} (weight: {rel.weight})\n")
        
        # Generate and include rankings
        rankings = generate_concept_importance_ranking(results)
        if rankings:
            f.write("\nCONCEPT IMPORTANCE RANKINGS:\n")
            f.write("-" * 30 + "\n")
            
            for ranking_data in rankings:
                scenario = ranking_data['scenario']
                bank = ranking_data['bank']
                true_concepts = ranking_data['true_concepts']
                
                f.write(f"\n{scenario} - {bank}:\n")
                f.write(f"True concepts: {true_concepts}\n")
                f.write("Ranking (Concept ID | Score | True? | Rejected?):\n")
                
                for concept_info in ranking_data['concept_ranking'][:10]:  # Top 10
                    cid = concept_info['concept_id']
                    score = concept_info['score']
                    is_true = "✓" if concept_info['is_true_concept'] else "✗"
                    rejected = "✓" if concept_info['rejected'] else "✗"
                    f.write(f"  {concept_info['rank']:2d}. C{cid:2d} | {score:5.3f} | {is_true} | {rejected}\n")
            
            # Discovery metrics
            discovery_metrics = calculate_discovery_metrics(rankings)
            f.write("\nDISCOVERY METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean Average Precision: {discovery_metrics['mean_average_precision']:.4f}\n")
            f.write(f"Mean Reciprocal Rank: {discovery_metrics['mean_reciprocal_rank']:.4f}\n")
            
            f.write("\nPrecision@K:\n")
            for k, prec in discovery_metrics['precision_at_k'].items():
                f.write(f"  P@{k}: {prec:.4f}\n")
            
            f.write("\nRecall@K:\n")
            for k, rec in discovery_metrics['recall_at_k'].items():
                f.write(f"  R@{k}: {rec:.4f}\n")
        
        # Performance summary
        if 'summary' in results:
            summary = results['summary']
            f.write("\nPERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            overall = summary.get('overall_performance', {})
            f.write(f"Overall Precision: {overall.get('mean_precision', 0):.4f} ± {overall.get('std_precision', 0):.4f}\n")
            f.write(f"Overall Recall: {overall.get('mean_recall', 0):.4f} ± {overall.get('std_recall', 0):.4f}\n")
            f.write(f"Overall F1-Score: {overall.get('mean_f1', 0):.4f} ± {overall.get('std_f1', 0):.4f}\n")


class ExperimentLogger:
    """Logger for tracking experiment progress and results."""
    
    def __init__(self, log_file: Union[str, Path] = "results/experiment.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 60 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Also print to console
        print(f"[{level}] {message}")
    
    def log_experiment_start(self, config_name: str):
        """Log the start of an experiment."""
        self.log(f"Starting experiment: {config_name}")
    
    def log_experiment_end(self, config_name: str, success: bool):
        """Log the end of an experiment."""
        status = "COMPLETED" if success else "FAILED"
        self.log(f"Experiment {config_name} {status}")
    
    def log_performance(self, config_name: str, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.log(f"Performance for {config_name}:")
        for metric, value in metrics.items():
            self.log(f"  {metric}: {value:.4f}")


def validate_experiment_config(config) -> List[str]:
    """Validate experiment configuration and return list of issues."""
    issues = []
    
    # Check required fields
    required_fields = ['name', 'n_samples', 'd_hidden', 'm_concepts']
    for field in required_fields:
        if not hasattr(config, field):
            issues.append(f"Missing required field: {field}")
    
    # Check relationships
    if hasattr(config, 'relationships') and config.relationships:
        for i, rel in enumerate(config.relationships):
            if not hasattr(rel, 'concept_indices'):
                issues.append(f"Relationship {i} missing concept_indices")
            elif max(rel.concept_indices) >= config.m_concepts:
                issues.append(f"Relationship {i} references concept index >= m_concepts")
    
    # Check concept distributions
    if hasattr(config, 'concept_distributions'):
        for idx, dist in config.concept_distributions.items():
            if idx >= config.m_concepts:
                issues.append(f"Concept distribution for index {idx} >= m_concepts")
    
    # Check reasonable parameter ranges
    if hasattr(config, 'n_samples') and config.n_samples < 100:
        issues.append("Very small sample size (< 100) may lead to unreliable results")
    
    if hasattr(config, 'noise_std') and config.noise_std < 0:
        issues.append("Noise std must be non-negative")
    
    return issues
