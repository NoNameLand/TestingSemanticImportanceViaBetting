#!/usr/bin/env python3
"""
Complete IBYDMT Integration Example

This script demonstrates how to use all the integration tools
to run IBYDMT tests and integrate them into your projects.
"""

import sys
from pathlib import Path

# Import our integration modules
sys.path.insert(0, str(Path(__file__).parent))
from ibydmt_project_integration import IBYDMTProjectIntegrator
from minimal_runner import MinimalIBYDMTRunner


def example_1_quick_validation():
    """Example 1: Quick validation that IBYDMT is working."""
    print("="*60)
    print("EXAMPLE 1: Quick Validation")
    print("="*60)
    
    runner = MinimalIBYDMTRunner()
    
    # Check if everything is set up correctly
    print("1. Checking dependencies...")
    deps_ok = runner.check_dependencies()
    
    print("\n2. Listing available configurations...")
    configs = runner.list_configurations()
    print(f"   Found {len(configs)} configurations: {configs}")
    
    if deps_ok and configs:
        print("\n3. Running quick test...")
        result = runner.run_original_test("synthetic", "global", "importance")
        
        if result["status"] == "success":
            print("   ✓ Quick test PASSED!")
            print(f"   Duration: {result.get('duration', 0):.1f} seconds")
            return True
        else:
            print(f"   ✗ Quick test FAILED: {result.get('error', 'Unknown error')}")
            return False
    else:
        print("   ✗ Setup issues detected")
        return False


def example_2_parameter_exploration():
    """Example 2: Explore different parameter combinations."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Parameter Exploration")
    print("="*60)
    
    integrator = IBYDMTProjectIntegrator()
    
    print("Testing different configurations with the same parameters...")
    
    # Test multiple configurations
    configs_to_test = ["synthetic", "gaussian"]
    
    sweep_result = integrator.parameter_sweep(
        configs=configs_to_test,
        test_types=["global"],
        concept_types=["importance"]
    )
    
    analysis = sweep_result["analysis"]
    
    print(f"\nResults:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    print(f"  Average duration: {analysis.get('duration_stats', {}).get('mean', 0):.1f}s")
    
    # Show per-configuration results
    print(f"\nPer-configuration results:")
    for config, stats in analysis.get("config_stats", {}).items():
        print(f"  {config}: {stats['success_rate']:.1%} success rate")
    
    return sweep_result


def example_3_batch_testing():
    """Example 3: Run batch tests for comprehensive evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Testing")
    print("="*60)
    
    integrator = IBYDMTProjectIntegrator()
    
    # Run different test scenarios
    scenarios = ["quick", "synthetic_full"]
    
    results = {}
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")
        try:
            result = integrator.run_test_scenario(scenario)
            results[scenario] = result
            
            print(f"  Status: {result['status']}")
            print(f"  Success rate: {result['success_rate']:.1%}")
            print(f"  Duration: {result['duration']:.1f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[scenario] = {"status": "error", "error": str(e)}
    
    return results


def example_4_configuration_comparison():
    """Example 4: Compare different configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Comparison")
    print("="*60)
    
    integrator = IBYDMTProjectIntegrator()
    
    configs_to_compare = ["synthetic", "gaussian"]
    
    print(f"Comparing configurations: {configs_to_compare}")
    
    comparison = integrator.compare_configurations(
        configs=configs_to_compare,
        test_type="global",
        concept_type="importance"
    )
    
    print(f"\nComparison Results:")
    print(f"  Total configs tested: {comparison['total_configs']}")
    print(f"  Successful configs: {comparison['successful_configs']}")
    print(f"  Failed configs: {comparison['failed_configs']}")
    
    if comparison['fastest_config']:
        print(f"  Fastest config: {comparison['fastest_config']}")
    
    if comparison['failed_config_names']:
        print(f"  Failed configs: {comparison['failed_config_names']}")
    
    return comparison


def example_5_multiple_test_types():
    """Example 5: Test different test types with the same configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Multiple Test Types")
    print("="*60)
    
    integrator = IBYDMTProjectIntegrator()
    
    # Test all test types with synthetic data
    test_types = ["global", "global_cond", "local_cond"]
    
    print(f"Testing all test types with synthetic config: {test_types}")
    
    sweep_result = integrator.parameter_sweep(
        configs=["synthetic"],
        test_types=test_types,
        concept_types=["importance"]
    )
    
    analysis = sweep_result["analysis"]
    
    print(f"\nResults by test type:")
    
    # Group results by test type
    results_by_test_type = {}
    for result in sweep_result["results"]:
        test_type = result.get("test_type", "unknown")
        if test_type not in results_by_test_type:
            results_by_test_type[test_type] = []
        results_by_test_type[test_type].append(result)
    
    for test_type, results in results_by_test_type.items():
        successful = sum(1 for r in results if r.get("status") == "success")
        total = len(results)
        avg_duration = sum(r.get("duration", 0) for r in results if r.get("status") == "success")
        avg_duration = avg_duration / successful if successful > 0 else 0
        
        print(f"  {test_type}: {successful}/{total} success, avg {avg_duration:.1f}s")
    
    return sweep_result


def example_6_your_project_integration():
    """Example 6: Template for integrating into your own project."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Your Project Integration Template")
    print("="*60)
    
    print("""
This is a template showing how to integrate IBYDMT into your own project:

```python
from ibydmt_project_integration import IBYDMTProjectIntegrator

def run_ibydmt_analysis(your_data_config, analysis_params):
    # Initialize integrator with your results directory
    integrator = IBYDMTProjectIntegrator(workdir="./my_project_results")
    
    # Step 1: Validate setup
    quick_test = integrator.run_test_scenario('quick')
    if quick_test['status'] != 'success':
        raise RuntimeError("IBYDMT setup validation failed")
    
    # Step 2: Find best configuration for your use case
    configs_to_try = ['synthetic', 'gaussian', 'cub']
    comparison = integrator.compare_configurations(configs_to_try)
    best_config = comparison['fastest_config']
    
    # Step 3: Run comprehensive tests with best configuration
    if best_config:
        sweep_result = integrator.parameter_sweep(
            configs=[best_config],
            test_types=['global', 'global_cond'],
            concept_types=['importance', 'rank']
        )
        
        # Step 4: Analyze results for your project
        analysis = sweep_result['analysis']
        
        # Extract insights for your project
        success_rate = analysis['success_rate']
        best_test_type = max(analysis['config_stats'].items(), 
                           key=lambda x: x[1]['success_rate'])
        
        return {
            'recommended_config': best_config,
            'success_rate': success_rate,
            'best_test_type': best_test_type,
            'full_results': sweep_result
        }
    
    return None

# Usage in your project:
# results = run_ibydmt_analysis(my_config, my_params)
# if results['success_rate'] > 0.8:
#     print(f"IBYDMT analysis successful with {results['recommended_config']}")
```
""")


def main():
    """Run all examples."""
    print("IBYDMT Complete Integration Examples")
    print("="*80)
    print("This script demonstrates all integration capabilities")
    print("="*80)
    
    # Run examples
    examples = [
        example_1_quick_validation,
        example_2_parameter_exploration, 
        example_3_batch_testing,
        example_4_configuration_comparison,
        example_5_multiple_test_types,
        example_6_your_project_integration
    ]
    
    results = {}
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}...")
            result = example_func()
            results[example_func.__name__] = {
                "status": "success",
                "result": result
            }
        except Exception as e:
            print(f"Example {i} failed: {e}")
            results[example_func.__name__] = {
                "status": "error", 
                "error": str(e)
            }
    
    # Summary
    print("\n" + "="*80)
    print("EXAMPLES SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)
    
    print(f"Completed: {successful}/{total} examples")
    
    for name, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} {name}")
        if result["status"] == "error":
            print(f"    Error: {result['error']}")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\nYou can now integrate IBYDMT into your projects using these patterns.")
    else:
        print(f"\n⚠️  {total - successful} examples had issues.")
        print("Check the error messages above and ensure IBYDMT is properly set up.")
    
    print("\nNext steps:")
    print("1. Use the minimal_runner.py for simple testing")
    print("2. Use ibydmt_project_integration.py for advanced integration") 
    print("3. Adapt example_6 template for your specific project needs")
    print("4. Check the results/ directory for saved test results")


if __name__ == "__main__":
    main()
