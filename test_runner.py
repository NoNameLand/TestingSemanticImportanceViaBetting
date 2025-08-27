#!/usr/bin/env python3
"""
IBYDMT Test Runner

This script provides an easy interface to run tests from the IBYDMT project
and integrate it with your own projects.

Usage examples:
    # Run a single test
    python test_runner.py --config_name synthetic --test_type global --concept_type importance

    # Run multiple tests with different parameters
    python test_runner.py --batch_test

    # Run with custom parameters
    python test_runner.py --config_name cub --test_type global_cond --concept_type rank --workdir ./results
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add IBYDMT to Python path
IBYDMT_DIR = Path(__file__).parent / "IBYDMT"
sys.path.insert(0, str(IBYDMT_DIR))

# Now we can import IBYDMT modules
try:
    from ibydmt.tester import ConceptTester
    from ibydmt.utils.config import TestType, ConceptType
except ImportError as e:
    print(f"Error importing IBYDMT modules: {e}")
    print("Make sure all dependencies are installed and IBYDMT directory exists.")
    sys.exit(1)


class IBYDMTTestRunner:
    """A wrapper class to run IBYDMT tests with various configurations."""
    
    def __init__(self, workdir: Optional[str] = None):
        self.workdir = workdir or str(Path(__file__).parent / "results")
        self.setup_logging()
        
    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.root.setLevel(level)
        
        # Configure IBYDMT loggers
        loggers = [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict
            if "ibydmt" in name
        ]
        for logger in loggers:
            logger.setLevel(level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logging.root.addHandler(stream_handler)
    
    def run_single_test(self, config_name: str, test_type: str, concept_type: str) -> Dict[str, Any]:
        """
        Run a single test with specified parameters.
        
        Args:
            config_name: Configuration to use (e.g., 'synthetic', 'cub', 'awa2')
            test_type: Type of test ('global', 'global_cond', 'local_cond')
            concept_type: Type of concept ('importance', 'rank', 'both')
        
        Returns:
            Dictionary with test results and metadata
        """
        print(f"Running test: {config_name} | {test_type} | {concept_type}")
        
        try:
            tester = ConceptTester(config_name)
            tester.test(test_type, concept_type, workdir=self.workdir)
            
            return {
                "status": "success",
                "config_name": config_name,
                "test_type": test_type,
                "concept_type": concept_type,
                "workdir": self.workdir
            }
        except Exception as e:
            print(f"Error running test: {e}")
            return {
                "status": "error",
                "error": str(e),
                "config_name": config_name,
                "test_type": test_type,
                "concept_type": concept_type
            }
    
    def run_batch_tests(self, configs: Optional[List[str]] = None, test_types: Optional[List[str]] = None, 
                       concept_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run multiple tests with different parameter combinations.
        
        Args:
            configs: List of configuration names
            test_types: List of test types
            concept_types: List of concept types
        
        Returns:
            List of test results
        """
        # Default parameter sets
        if configs is None:
            configs = ["synthetic", "gaussian"]  # Start with simpler configs
        if test_types is None:
            test_types = ["global"]  # Start with global tests
        if concept_types is None:
            concept_types = ["importance"]
        
        results = []
        total_tests = len(configs) * len(test_types) * len(concept_types)
        current_test = 0
        
        print(f"Running {total_tests} tests...")
        
        for config in configs:
            for test_type in test_types:
                for concept_type in concept_types:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}")
                    
                    result = self.run_single_test(config, test_type, concept_type)
                    results.append(result)
                    
                    # Print progress
                    if result["status"] == "success":
                        print("✓ Test completed successfully")
                    else:
                        print(f"✗ Test failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration names."""
        configs_dir = IBYDMT_DIR / "configs"
        config_files = [f.stem for f in configs_dir.glob("*.py") 
                       if not f.name.startswith("__")]
        return config_files
    
    def get_test_types(self) -> List[str]:
        """Get list of available test types."""
        return ["global", "global_cond", "local_cond"]
    
    def get_concept_types(self) -> List[str]:
        """Get list of available concept types."""
        return ["importance", "rank", "both"]


def main():
    parser = argparse.ArgumentParser(description="IBYDMT Test Runner")
    parser.add_argument("--config_name", type=str, help="Configuration name")
    parser.add_argument("--test_type", type=str, help="Test type")
    parser.add_argument("--concept_type", type=str, help="Concept type")
    parser.add_argument("--workdir", type=str, help="Working directory for results")
    parser.add_argument("--batch_test", action="store_true", 
                       help="Run batch tests with default parameters")
    parser.add_argument("--list_configs", action="store_true",
                       help="List available configurations")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IBYDMTTestRunner(workdir=args.workdir)
    runner.setup_logging(args.log_level)
    
    if args.list_configs:
        print("Available configurations:")
        for config in runner.get_available_configs():
            print(f"  - {config}")
        print("\nAvailable test types:")
        for test_type in runner.get_test_types():
            print(f"  - {test_type}")
        print("\nAvailable concept types:")
        for concept_type in runner.get_concept_types():
            print(f"  - {concept_type}")
        return
    
    if args.batch_test:
        print("Running batch tests...")
        results = runner.run_batch_tests()
        
        # Print summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        print(f"\n=== Test Summary ===")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed tests:")
            for result in results:
                if result["status"] == "error":
                    print(f"  - {result['config_name']} | {result['test_type']} | {result['concept_type']}: {result['error']}")
    
    elif args.config_name and args.test_type and args.concept_type:
        result = runner.run_single_test(args.config_name, args.test_type, args.concept_type)
        if result["status"] == "success":
            print("Test completed successfully!")
        else:
            print(f"Test failed: {result['error']}")
            sys.exit(1)
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python test_runner.py --config_name synthetic --test_type global --concept_type importance")
        print("  python test_runner.py --batch_test")
        print("  python test_runner.py --list_configs")


if __name__ == "__main__":
    main()
