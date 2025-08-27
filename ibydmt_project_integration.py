#!/usr/bin/env python3
"""
IBYDMT Project Integration Script

This script provides multiple ways to run and integrate IBYDMT tests:
1. Direct integration using the minimal runner
2. Parameter sweep testing
3. Multiple test scenarios for your project integration

This works with minimal dependencies and provides robust error handling.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import our minimal runner
sys.path.insert(0, str(Path(__file__).parent))
from minimal_runner import MinimalIBYDMTRunner


class IBYDMTProjectIntegrator:
    """
    High-level integrator for incorporating IBYDMT testing into your projects.
    
    This class provides easy-to-use methods for running IBYDMT tests with
    different parameter combinations and collecting results for analysis.
    """
    
    def __init__(self, workdir: Optional[str] = None):
        """Initialize the integrator."""
        self.workdir = Path(workdir) if workdir else Path(__file__).parent / "results"
        self.runner = MinimalIBYDMTRunner(workdir=str(self.workdir))
        self.test_history: List[Dict[str, Any]] = []
        
    def run_test_scenario(self, scenario_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a predefined test scenario.
        
        Available scenarios:
        - 'quick': Fast test with synthetic data
        - 'synthetic_full': Comprehensive synthetic data testing
        - 'real_data': Test with real datasets (CUB)
        - 'comparison': Compare different test types
        """
        scenarios = {
            'quick': {
                'configs': ['synthetic'],
                'test_types': ['global'],
                'concept_types': ['importance']
            },
            'synthetic_full': {
                'configs': ['synthetic', 'gaussian'],
                'test_types': ['global', 'global_cond'],
                'concept_types': ['importance', 'rank']
            },
            'real_data': {
                'configs': ['cub'],
                'test_types': ['global'],
                'concept_types': ['importance']
            },
            'comparison': {
                'configs': ['synthetic'],
                'test_types': ['global', 'global_cond', 'local_cond'],
                'concept_types': ['importance']
            }
        }
        
        if scenario_name not in scenarios:
            available = list(scenarios.keys())
            return {
                'status': 'error',
                'error': f'Unknown scenario. Available: {available}'
            }
        
        # Override scenario parameters with kwargs
        params = scenarios[scenario_name].copy()
        params.update(kwargs)
        
        print(f"Running scenario: {scenario_name}")
        print(f"Parameters: {params}")
        
        start_time = datetime.now()
        results = self.runner.run_batch_tests(**params)
        end_time = datetime.now()
        
        # Calculate scenario-level statistics
        total = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        
        scenario_result = {
            'scenario_name': scenario_name,
            'status': 'success' if successful > 0 else 'failed',
            'total_tests': total,
            'successful_tests': successful,
            'success_rate': successful / total if total > 0 else 0,
            'duration': (end_time - start_time).total_seconds(),
            'timestamp': start_time.isoformat(),
            'individual_results': results,
            'parameters': params
        }
        
        self.test_history.append(scenario_result)
        return scenario_result
    
    def parameter_sweep(self, 
                       configs: List[str],
                       test_types: Optional[List[str]] = None,
                       concept_types: Optional[List[str]] = None,
                       save_results: bool = True) -> Dict[str, Any]:
        """
        Run tests with all combinations of the given parameters.
        
        Args:
            configs: List of configuration names to test
            test_types: List of test types (defaults to ['global'])
            concept_types: List of concept types (defaults to ['importance'])
            save_results: Whether to automatically save results
            
        Returns:
            Dictionary with results and analysis
        """
        if test_types is None:
            test_types = ['global']
        if concept_types is None:
            concept_types = ['importance']
        
        print(f"Parameter sweep: {len(configs)} configs × {len(test_types)} test types × {len(concept_types)} concept types")
        print(f"Total combinations: {len(configs) * len(test_types) * len(concept_types)}")
        
        start_time = datetime.now()
        
        # Run the batch tests
        results = self.runner.run_batch_tests(
            configs=configs,
            test_types=test_types,
            concept_types=concept_types
        )
        
        end_time = datetime.now()
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        sweep_result = {
            'type': 'parameter_sweep',
            'timestamp': start_time.isoformat(),
            'duration': (end_time - start_time).total_seconds(),
            'parameters': {
                'configs': configs,
                'test_types': test_types,
                'concept_types': concept_types
            },
            'results': results,
            'analysis': analysis
        }
        
        if save_results:
            filename = f"parameter_sweep_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            self.save_to_file(sweep_result, filename)
        
        return sweep_result
    
    def compare_configurations(self, configs: List[str], 
                              test_type: str = 'global',
                              concept_type: str = 'importance') -> Dict[str, Any]:
        """
        Compare different configurations with the same test parameters.
        
        Args:
            configs: List of configurations to compare
            test_type: Test type to use for all configs
            concept_type: Concept type to use for all configs
            
        Returns:
            Comparison results with rankings
        """
        print(f"Comparing configurations: {configs}")
        print(f"Using test_type='{test_type}', concept_type='{concept_type}'")
        
        results = []
        
        for config in configs:
            print(f"\nTesting configuration: {config}")
            result = self.runner.run_original_test(config, test_type, concept_type)
            result['config_name'] = config
            results.append(result)
        
        # Analyze results
        successful_configs = [r for r in results if r.get('status') == 'success']
        failed_configs = [r for r in results if r.get('status') != 'success']
        
        # Rank by success and duration
        successful_configs.sort(key=lambda x: x.get('duration', float('inf')))
        
        comparison = {
            'type': 'configuration_comparison',
            'timestamp': datetime.now().isoformat(),
            'test_parameters': {
                'test_type': test_type,
                'concept_type': concept_type
            },
            'total_configs': len(configs),
            'successful_configs': len(successful_configs),
            'failed_configs': len(failed_configs),
            'results': results,
            'ranking': [r['config_name'] for r in successful_configs],
            'fastest_config': successful_configs[0]['config_name'] if successful_configs else None,
            'failed_config_names': [r['config_name'] for r in failed_configs]
        }
        
        return comparison
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and provide insights."""
        if not results:
            return {'error': 'No results to analyze'}
        
        total = len(results)
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        timeouts = [r for r in results if r.get('status') == 'timeout']
        
        analysis = {
            'total_tests': total,
            'successful': len(successful),
            'failed': len(failed),
            'timeouts': len(timeouts),
            'success_rate': len(successful) / total,
            'failure_rate': len(failed) / total,
            'timeout_rate': len(timeouts) / total
        }
        
        # Duration analysis for successful tests
        if successful:
            durations = [r.get('duration', 0) for r in successful]
            analysis['duration_stats'] = {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'total': sum(durations)
            }
        
        # Configuration success rates
        config_stats = {}
        for result in results:
            config = result.get('config_name', 'unknown')
            if config not in config_stats:
                config_stats[config] = {'total': 0, 'success': 0, 'durations': []}
            
            config_stats[config]['total'] += 1
            if result.get('status') == 'success':
                config_stats[config]['success'] += 1
                config_stats[config]['durations'].append(result.get('duration', 0))
        
        # Calculate success rates and average durations
        for config, stats in config_stats.items():
            stats['success_rate'] = stats['success'] / stats['total']
            if stats['durations']:
                stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
            else:
                stats['avg_duration'] = None
        
        analysis['config_stats'] = config_stats
        
        # Common error analysis
        error_messages = [r.get('error', '') for r in failed if r.get('error')]
        if error_messages:
            # Group similar errors
            error_counts = {}
            for error in error_messages:
                # Use first 100 characters as error key
                error_key = error[:100]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            analysis['common_errors'] = dict(sorted(
                error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3])  # Top 3 errors
        
        return analysis
    
    def save_to_file(self, data: Dict[str, Any], filename: str) -> str:
        """Save data to a JSON file in the results directory."""
        filepath = self.workdir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Data saved to: {filepath}")
        return str(filepath)
    
    def generate_report(self, include_history: bool = True) -> str:
        """Generate a text report of all testing activities."""
        report_lines = [
            "IBYDMT Testing Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Working Directory: {self.workdir}",
            ""
        ]
        
        if not self.test_history and not include_history:
            report_lines.append("No test history available.")
            return "\n".join(report_lines)
        
        if include_history and self.test_history:
            report_lines.extend([
                f"Test History ({len(self.test_history)} sessions):",
                "-" * 30
            ])
            
            for i, session in enumerate(self.test_history, 1):
                report_lines.extend([
                    f"\n{i}. {session.get('scenario_name', 'Unknown')}",
                    f"   Status: {session.get('status', 'unknown')}",
                    f"   Success Rate: {session.get('success_rate', 0):.1%}",
                    f"   Duration: {session.get('duration', 0):.1f}s",
                    f"   Tests: {session.get('successful_tests', 0)}/{session.get('total_tests', 0)}"
                ])
        
        # Add summary statistics
        if self.test_history:
            total_sessions = len(self.test_history)
            successful_sessions = sum(1 for s in self.test_history if s.get('status') == 'success')
            total_tests = sum(s.get('total_tests', 0) for s in self.test_history)
            total_successful = sum(s.get('successful_tests', 0) for s in self.test_history)
            
            success_rate = total_successful/total_tests if total_tests > 0 else 0
            report_lines.extend([
                "\nOverall Summary:",
                "-" * 20,
                f"Total Sessions: {total_sessions}",
                f"Successful Sessions: {successful_sessions} ({successful_sessions/total_sessions:.1%})",
                f"Total Tests Run: {total_tests}",
                f"Total Successful Tests: {total_successful} ({success_rate:.1%})"
            ])
        
        return "\n".join(report_lines)


def main():
    """Demonstration of the project integrator."""
    print("IBYDMT Project Integrator")
    print("=" * 40)
    
    # Initialize integrator
    integrator = IBYDMTProjectIntegrator()
    
    # Example 1: Quick test
    print("\n1. Running quick test scenario...")
    quick_result = integrator.run_test_scenario('quick')
    print(f"Quick test result: {quick_result['status']}")
    print(f"Success rate: {quick_result['success_rate']:.1%}")
    
    # Example 2: Compare configurations
    print("\n2. Comparing synthetic configurations...")
    comparison = integrator.compare_configurations(['synthetic', 'gaussian'])
    print(f"Best configuration: {comparison.get('fastest_config', 'None')}")
    
    # Example 3: Parameter sweep
    print("\n3. Running parameter sweep...")
    sweep_result = integrator.parameter_sweep(
        configs=['synthetic'],
        test_types=['global'],
        concept_types=['importance', 'rank']
    )
    print(f"Sweep completed: {sweep_result['analysis']['success_rate']:.1%} success rate")
    
    # Generate and save report
    report = integrator.generate_report()
    report_file = integrator.save_to_file({'report': report}, 'integration_report.json')
    
    print(f"\n4. Report generated and saved to: {report_file}")
    print("\nIntegration examples completed!")
    
    # Show usage examples
    print("\n" + "="*50)
    print("Usage in your own projects:")
    print("="*50)
    print("""
# Basic usage
from ibydmt_project_integration import IBYDMTProjectIntegrator

integrator = IBYDMTProjectIntegrator()

# Quick test
result = integrator.run_test_scenario('quick')

# Custom parameter sweep
sweep = integrator.parameter_sweep(
    configs=['synthetic', 'gaussian'],
    test_types=['global', 'global_cond'],
    concept_types=['importance']
)

# Analyze results
analysis = integrator.analyze_results(sweep['results'])
print(f"Success rate: {analysis['success_rate']:.1%}")
""")


if __name__ == "__main__":
    main()
