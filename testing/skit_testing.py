"""
Testing module for running SKIT experiments.
Handles global SKIT, conditional SKIT, and robust importance testing.
"""

import numpy as np
from typing import Dict, List, Any, Callable
from pathlib import Path

from ibymdt_cav_new import SKITGlobal, SKITConditional, secure_importance
from config.experiment_config import ExperimentConfig


class SKITTester:
    """Handles various SKIT testing procedures."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def test_global_skit(self, Y: np.ndarray, Z: np.ndarray, 
                        true_concepts: List[int]) -> Dict[str, Any]:
        """Run global SKIT testing on all concepts."""
        results = {}
        skit_config = self.config.skit_config
        
        for j in range(Z.shape[1]):
            # Create SKIT instance
            skit = SKITGlobal(alpha=skit_config.alpha, use_ons=skit_config.use_ons)
            
            # Determine number of pairs to use
            max_pairs = len(Y) // 2
            if skit_config.n_pairs is not None:
                n_pairs = min(skit_config.n_pairs, max_pairs)
            else:
                n_pairs = int(max_pairs * skit_config.subsample_ratio)
            
            # Randomly pair up samples
            indices = np.random.permutation(len(Y))[:2*n_pairs]
            
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
            
            results[j] = {
                'rejected': skit.rejected(),
                'final_wealth': skit.K,
                'rejection_step': rejection_step,
                'wealth_trajectory': wealth_trajectory,
                'is_true_concept': j in true_concepts,
                'concept_id': j
            }
        
        return results
    
    def test_robust_importance(self, Y: np.ndarray, Z: np.ndarray, 
                             true_concepts: List[int]) -> Dict[str, Any]:
        """Run robust importance testing."""
        results = {}
        robust_config = self.config.robust_config
        
        # Test subset of concepts for computational efficiency
        concepts_to_test = range(min(8, Z.shape[1]))
        
        for j in concepts_to_test:
            def skit_run_function():
                """Single SKIT run for concept j."""
                skit = SKITGlobal(alpha=self.config.skit_config.alpha, 
                                use_ons=self.config.skit_config.use_ons)
                
                # Use subset of data for faster computation
                n_subset = min(robust_config.subsample_size, len(Y))
                indices = np.random.choice(len(Y), n_subset, replace=False)
                Y_sub = Y[indices]
                Z_sub = Z[indices, j]
                
                # Pair up samples
                pairs = len(indices) // 2
                for i in range(pairs):
                    y1, y2 = Y_sub[2*i], Y_sub[2*i + 1]
                    z1, z2 = Z_sub[2*i], Z_sub[2*i + 1]
                    
                    skit.step_pair(y1, z1, y2, z2)
                    
                    if skit.rejected():
                        break
                
                return skit.rejected()
            
            # Run robust importance test
            robust_result = secure_importance(
                skit_run_function, 
                R=robust_config.n_runs, 
                delta=1.0 - robust_config.confidence
            )
            robust_result['is_true_concept'] = j in true_concepts
            robust_result['concept_id'] = j
            results[j] = robust_result
        
        return results


class ExperimentRunner:
    """Main class for running complete experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tester = SKITTester(config)
        self.results = {}
    
    def run_experiment(self, scenarios: Dict[str, Any], concept_banks: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete experiment with all scenarios and concept banks."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {self.config.name}")
        print(f"Description: {self.config.description}")
        print("="*60)
        
        experiment_results = {
            'config': self.config,
            'scenarios': {},
            'summary': {}
        }
        
        # If scenarios is actually a single scenario (not a dict of scenarios)
        if 'name' in scenarios:  # Single scenario
            scenarios = {scenarios['name']: scenarios}
        
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            
            scenario_results = {
                'global_skit': {},
                'robust_importance': {},
                'true_concepts': scenario_data['true_concepts'],
                'metadata': {
                    'n_samples': len(scenario_data['Y']),
                    'n_concepts': scenario_data['Z'].shape[1],
                    'y_stats': {
                        'mean': float(np.mean(scenario_data['Y'])),
                        'std': float(np.std(scenario_data['Y'])),
                        'min': float(np.min(scenario_data['Y'])),
                        'max': float(np.max(scenario_data['Y']))
                    }
                }
            }
            
            # Test the concept bank
            Z = scenario_data['Z']
            print(f"\n  Testing with concept bank:")
            
            # Global SKIT testing
            print("    Running global SKIT...")
            global_results = self.tester.test_global_skit(
                scenario_data['Y'], Z, scenario_data['true_concepts']
            )
            scenario_results['global_skit']['primary'] = global_results
            
            # Print summary
            rejected_concepts = [j for j, r in global_results.items() if r['rejected']]
            true_rejected = [j for j in rejected_concepts if j in scenario_data['true_concepts']]
            false_rejected = [j for j in rejected_concepts if j not in scenario_data['true_concepts']]
            
            print(f"      Rejected: {rejected_concepts}")
            print(f"      True positives: {true_rejected}")
            print(f"      False positives: {false_rejected}")
            
            # Robust importance testing
            print("    Running robust importance testing...")
            robust_results = self.tester.test_robust_importance(
                scenario_data['Y'], Z, scenario_data['true_concepts']
            )
            scenario_results['robust_importance']['primary'] = robust_results
            
            # Print robust summary
            for concept_id, result in robust_results.items():
                rate = result['rejection_rate_hat']
                lb = result['rejection_rate_lower_bound']
                true_marker = "★" if result['is_true_concept'] else " "
                print(f"      Concept {concept_id} {true_marker}: "
                      f"rate={rate:.3f}, lower_bound={lb:.3f}")
            
            experiment_results['scenarios'][scenario_name] = scenario_results
        
        # Calculate experiment summary
        experiment_results['summary'] = self._calculate_summary(experiment_results)
        
        # Save results if requested
        if self.config.save_results:
            self._save_results(experiment_results)
        
        self.results = experiment_results
        return experiment_results
    
    def _calculate_summary(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all scenarios and banks."""
        summary = {
            'total_scenarios': len(experiment_results['scenarios']),
            'performance_by_scenario': {},
            'overall_performance': {}
        }
        
        all_precision = []
        all_recall = []
        all_f1 = []
        
        for scenario_name, scenario_data in experiment_results['scenarios'].items():
            scenario_perf = {}
            
            for bank_name, bank_results in scenario_data['global_skit'].items():
                true_concepts = set(scenario_data['true_concepts'])
                
                # Calculate performance metrics
                rejected = set([j for j, r in bank_results.items() if r['rejected']])
                
                tp = len(rejected & true_concepts)
                fp = len(rejected - true_concepts)
                fn = len(true_concepts - rejected)
                tn = len(set(bank_results.keys()) - true_concepts - rejected)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                scenario_perf[bank_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                }
                
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
            
            summary['performance_by_scenario'][scenario_name] = scenario_perf
        
        # Overall performance
        summary['overall_performance'] = {
            'mean_precision': float(np.mean(all_precision)) if all_precision else 0.0,
            'mean_recall': float(np.mean(all_recall)) if all_recall else 0.0,
            'mean_f1': float(np.mean(all_f1)) if all_f1 else 0.0,
            'std_precision': float(np.std(all_precision)) if all_precision else 0.0,
            'std_recall': float(np.std(all_recall)) if all_recall else 0.0,
            'std_f1': float(np.std(all_f1)) if all_f1 else 0.0
        }
        
        return summary
    
    def _save_results(self, experiment_results: Dict[str, Any]):
        """Save experiment results to files."""
        import json
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to results
        experiment_results['timestamp'] = datetime.now().isoformat()
        experiment_results['experiment_name'] = self.config.name
        
        # Save main results as JSON
        results_file = output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=self._json_serializer)
        
        # Save human-readable summary
        summary_file = output_dir / "experiment_summary.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(f, experiment_results)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _write_summary_report(self, file, experiment_results: Dict[str, Any]):
        """Write human-readable summary report."""
        file.write(f"EXPERIMENT SUMMARY: {self.config.name}\n")
        file.write("=" * 60 + "\n\n")
        
        file.write(f"Description: {self.config.description}\n")
        file.write(f"Timestamp: {experiment_results.get('timestamp', 'N/A')}\n\n")
        
        # Configuration summary
        file.write("CONFIGURATION:\n")
        file.write(f"  Samples: {self.config.n_samples}\n")
        file.write(f"  Hidden dimensions: {self.config.d_hidden}\n")
        file.write(f"  Concepts: {self.config.m_concepts}\n")
        file.write(f"  Noise std: {self.config.noise_std}\n")
        file.write(f"  SKIT alpha: {self.config.skit_config.alpha}\n")
        
        file.write(f"\nRELATIONSHIPS:\n")
        for i, rel in enumerate(self.config.relationships):
            file.write(f"  {i+1}. {rel.name}: indices {rel.concept_indices}, weight {rel.weight}\n")
        
        # Overall performance
        overall = experiment_results['summary']['overall_performance']
        file.write(f"\nOVERALL PERFORMANCE:\n")
        file.write(f"  Mean Precision: {overall['mean_precision']:.3f} ± {overall['std_precision']:.3f}\n")
        file.write(f"  Mean Recall: {overall['mean_recall']:.3f} ± {overall['std_recall']:.3f}\n")
        file.write(f"  Mean F1-Score: {overall['mean_f1']:.3f} ± {overall['std_f1']:.3f}\n")
        
        # Detailed results by scenario
        for scenario_name, scenario_data in experiment_results['scenarios'].items():
            file.write(f"\nSCENARIO: {scenario_name}\n")
            file.write(f"  True concepts: {scenario_data['true_concepts']}\n")
            
            for bank_name, perf in experiment_results['summary']['performance_by_scenario'][scenario_name].items():
                file.write(f"  {bank_name} bank:\n")
                file.write(f"    Precision: {perf['precision']:.3f}\n")
                file.write(f"    Recall: {perf['recall']:.3f}\n")
                file.write(f"    F1-Score: {perf['f1_score']:.3f}\n")
                file.write(f"    TP: {perf['true_positives']}, FP: {perf['false_positives']}, ")
                file.write(f"TN: {perf['true_negatives']}, FN: {perf['false_negatives']}\n")
