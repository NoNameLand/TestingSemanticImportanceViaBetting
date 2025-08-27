#!/usr/bin/env python3
"""
IBYDMT Integration Library

This module provides utilities to integrate IBYDMT testing capabilities
into your own projects. It allows you to run multiple tests with different
parameters and collect results for analysis.

Example usage:
    from ibydmt_integration import IBYDMTIntegrator
    
    integrator = IBYDMTIntegrator()
    
    # Run tests with parameter sweep
    results = integrator.parameter_sweep(
        configs=['synthetic', 'gaussian'],
        test_types=['global'],
        concept_types=['importance', 'rank']
    )
    
    # Analyze results
    summary = integrator.analyze_results(results)
    print(summary)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Add IBYDMT to Python path
IBYDMT_DIR = Path(__file__).parent / "IBYDMT"
sys.path.insert(0, str(IBYDMT_DIR))


@dataclass
class TestConfiguration:
    """Configuration for a single test run."""
    config_name: str
    test_type: str
    concept_type: str
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Result of a single test run."""
    config: TestConfiguration
    status: str  # 'success', 'error', 'timeout'
    start_time: datetime
    end_time: datetime
    duration: float
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class IBYDMTIntegrator:
    """Main integration class for running IBYDMT tests with parameter sweeps."""
    
    def __init__(self, workdir: Optional[str] = None, timeout: int = 300):
        """
        Initialize the integrator.
        
        Args:
            workdir: Working directory for results
            timeout: Timeout in seconds for each test
        """
        self.workdir = Path(workdir) if workdir else Path(__file__).parent / "results"
        self.timeout = timeout
        self.results_history: List[TestResult] = []
        self.setup_logging()
        
        # Create results directory
        self.workdir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=getattr(logging, level), format=log_format)
        self.logger = logging.getLogger(__name__)
        
    def run_single_test(self, config: TestConfiguration) -> TestResult:
        """
        Run a single test with the given configuration.
        
        Args:
            config: Test configuration
            
        Returns:
            Test result
        """
        start_time = datetime.now()
        self.logger.info(f"Starting test: {config.config_name} | {config.test_type} | {config.concept_type}")
        
        try:
            # Import here to avoid issues if IBYDMT is not available
            from ibydmt.tester import ConceptTester
            
            tester = ConceptTester(config.config_name)
            tester.test(config.test_type, config.concept_type, workdir=str(self.workdir))
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = TestResult(
                config=config,
                status="success",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                result_path=str(self.workdir / config.config_name.lower())
            )
            
            self.logger.info(f"Test completed successfully in {duration:.2f}s")
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = TestResult(
                config=config,
                status="error",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                error_message=str(e)
            )
            
            self.logger.error(f"Test failed after {duration:.2f}s: {e}")
        
        self.results_history.append(result)
        return result
    
    def parameter_sweep(self, 
                       configs: List[str],
                       test_types: List[str],
                       concept_types: List[str],
                       custom_params: Optional[Dict[str, Any]] = None) -> List[TestResult]:
        """
        Run tests with all combinations of the given parameters.
        
        Args:
            configs: List of configuration names
            test_types: List of test types
            concept_types: List of concept types
            custom_params: Optional custom parameters for all tests
            
        Returns:
            List of test results
        """
        test_configs = []
        
        # Generate all combinations
        for config_name in configs:
            for test_type in test_types:
                for concept_type in concept_types:
                    test_config = TestConfiguration(
                        config_name=config_name,
                        test_type=test_type,
                        concept_type=concept_type,
                        custom_params=custom_params
                    )
                    test_configs.append(test_config)
        
        total_tests = len(test_configs)
        self.logger.info(f"Starting parameter sweep with {total_tests} test combinations")
        
        results = []
        for i, config in enumerate(test_configs, 1):
            self.logger.info(f"Running test {i}/{total_tests}")
            result = self.run_single_test(config)
            results.append(result)
            
            # Log progress
            successful = sum(1 for r in results if r.status == "success")
            failed = len(results) - successful
            self.logger.info(f"Progress: {i}/{total_tests} | Success: {successful} | Failed: {failed}")
        
        return results
    
    def run_predefined_experiments(self) -> Dict[str, List[TestResult]]:
        """
        Run predefined experiment sets for common use cases.
        
        Returns:
            Dictionary mapping experiment names to results
        """
        experiments = {
            "quick_test": {
                "configs": ["synthetic"],
                "test_types": ["global"],
                "concept_types": ["importance"]
            },
            "synthetic_full": {
                "configs": ["synthetic", "gaussian"],
                "test_types": ["global", "global_cond"],
                "concept_types": ["importance", "rank"]
            },
            "real_data_basic": {
                "configs": ["cub"],
                "test_types": ["global"],
                "concept_types": ["importance"]
            }
        }
        
        all_results = {}
        
        for exp_name, params in experiments.items():
            self.logger.info(f"Running experiment: {exp_name}")
            try:
                results = self.parameter_sweep(
                    configs=params["configs"],
                    test_types=params["test_types"],
                    concept_types=params["concept_types"]
                )
                all_results[exp_name] = results
            except Exception as e:
                self.logger.error(f"Experiment {exp_name} failed: {e}")
                all_results[exp_name] = []
        
        return all_results
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Analyze test results and provide summary statistics.
        
        Args:
            results: List of test results
            
        Returns:
            Analysis summary
        """
        if not results:
            return {"error": "No results to analyze"}
        
        total_tests = len(results)
        successful = sum(1 for r in results if r.status == "success")
        failed = total_tests - successful
        
        # Duration statistics
        durations = [r.duration for r in results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Success rate by configuration
        config_stats = {}
        for result in results:
            config_name = result.config.config_name
            if config_name not in config_stats:
                config_stats[config_name] = {"total": 0, "success": 0}
            config_stats[config_name]["total"] += 1
            if result.status == "success":
                config_stats[config_name]["success"] += 1
        
        # Calculate success rates
        for config_name in config_stats:
            stats = config_stats[config_name]
            stats["success_rate"] = stats["success"] / stats["total"]
        
        # Error analysis
        error_types = {}
        for result in results:
            if result.status == "error" and result.error_message:
                error_key = result.error_message[:100]  # First 100 chars
                error_types[error_key] = error_types.get(error_key, 0) + 1
        
        summary = {
            "total_tests": total_tests,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_tests,
            "duration_stats": {
                "average": avg_duration,
                "maximum": max_duration,
                "minimum": min_duration
            },
            "config_stats": config_stats,
            "common_errors": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
        return summary
    
    def save_results(self, results: List[TestResult], filename: Optional[str] = None) -> str:
        """
        Save test results to a JSON file.
        
        Args:
            results: Test results to save
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        filepath = self.workdir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                "config": {
                    "config_name": result.config.config_name,
                    "test_type": result.config.test_type,
                    "concept_type": result.config.concept_type,
                    "custom_params": result.config.custom_params
                },
                "status": result.status,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration": result.duration,
                "error_message": result.error_message,
                "result_path": result.result_path,
                "metrics": result.metrics
            }
            serializable_results.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def load_results(self, filepath: str) -> List[TestResult]:
        """
        Load test results from a JSON file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            List of test results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            config = TestConfiguration(
                config_name=item["config"]["config_name"],
                test_type=item["config"]["test_type"],
                concept_type=item["config"]["concept_type"],
                custom_params=item["config"]["custom_params"]
            )
            
            result = TestResult(
                config=config,
                status=item["status"],
                start_time=datetime.fromisoformat(item["start_time"]),
                end_time=datetime.fromisoformat(item["end_time"]),
                duration=item["duration"],
                error_message=item["error_message"],
                result_path=item["result_path"],
                metrics=item["metrics"]
            )
            results.append(result)
        
        return results
    
    def get_available_configurations(self) -> Dict[str, List[str]]:
        """
        Get available configurations for testing.
        
        Returns:
            Dictionary with available configs, test types, and concept types
        """
        try:
            configs_dir = IBYDMT_DIR / "configs"
            config_files = [f.stem for f in configs_dir.glob("*.py") 
                           if not f.name.startswith("__")]
            
            return {
                "configs": config_files,
                "test_types": ["global", "global_cond", "local_cond"],
                "concept_types": ["importance", "rank", "both"]
            }
        except Exception as e:
            self.logger.error(f"Error getting configurations: {e}")
            return {
                "configs": [],
                "test_types": [],
                "concept_types": []
            }


def main():
    """Example usage of the integration library."""
    print("IBYDMT Integration Library")
    print("=" * 30)
    
    # Initialize integrator
    integrator = IBYDMTIntegrator()
    
    # Show available configurations
    available = integrator.get_available_configurations()
    print("Available configurations:")
    for key, values in available.items():
        print(f"  {key}: {values}")
    
    # Run a quick test
    print("\nRunning quick test...")
    quick_results = integrator.parameter_sweep(
        configs=["synthetic"],
        test_types=["global"],
        concept_types=["importance"]
    )
    
    # Analyze results
    analysis = integrator.analyze_results(quick_results)
    print("\nTest Analysis:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Success rate: {analysis['success_rate']:.2%}")
    print(f"  Average duration: {analysis['duration_stats']['average']:.2f}s")
    
    # Save results
    saved_file = integrator.save_results(quick_results)
    print(f"\nResults saved to: {saved_file}")


if __name__ == "__main__":
    main()
