#!/usr/bin/env python3
"""
Minimal IBYDMT Test Runner

This script provides a minimal interface to test IBYDMT functionality
without requiring heavy dependencies. It focuses on the core testing
capabilities and provides a simple way to run tests.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add IBYDMT to Python path
IBYDMT_DIR = Path(__file__).parent / "IBYDMT"
sys.path.insert(0, str(IBYDMT_DIR))

class MinimalIBYDMTRunner:
    """Minimal runner for IBYDMT tests that works with basic Python setup."""
    
    def __init__(self, workdir=None):
        self.workdir = Path(workdir) if workdir else Path(__file__).parent / "results"
        self.workdir.mkdir(parents=True, exist_ok=True)
        
    def check_dependencies(self):
        """Check if basic dependencies are available."""
        required_modules = ['os', 'sys', 'json', 'argparse']
        optional_modules = ['numpy', 'torch', 'sklearn']
        
        print("Checking dependencies...")
        
        # Check required modules
        for module in required_modules:
            try:
                __import__(module)
                print(f"✓ {module}: Available")
            except ImportError:
                print(f"✗ {module}: Missing (required)")
                return False
        
        # Check optional modules
        missing_optional = []
        for module in optional_modules:
            try:
                __import__(module)
                print(f"✓ {module}: Available")
            except ImportError:
                print(f"! {module}: Missing (optional)")
                missing_optional.append(module)
        
        if missing_optional:
            print(f"\nWarning: Missing optional modules: {missing_optional}")
            print("Some IBYDMT functionality may not work.")
            
        return True
    
    def list_configurations(self):
        """List available IBYDMT configurations."""
        configs_dir = IBYDMT_DIR / "configs"
        
        if not configs_dir.exists():
            print(f"Error: Configs directory not found at {configs_dir}")
            return []
        
        config_files = []
        for file in configs_dir.glob("*.py"):
            if not file.name.startswith("__"):
                config_files.append(file.stem)
        
        return sorted(config_files)
    
    def run_original_test(self, config_name, test_type, concept_type):
        """Run test using the original IBYDMT test.py script."""
        test_script = IBYDMT_DIR / "test.py"
        
        if not test_script.exists():
            return {
                "status": "error",
                "error": f"Original test script not found at {test_script}"
            }
        
        # Build command - use system python3 instead of venv
        cmd = [
            "/usr/bin/python3",  # Use absolute path to system python3
            "test.py",  # Relative path since we set cwd to IBYDMT_DIR
            "--config_name", config_name,
            "--test_type", test_type,
            "--concept_type", concept_type,
            "--workdir", str(self.workdir)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            start_time = datetime.now()
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(IBYDMT_DIR)  # Set working directory to IBYDMT
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "config_name": config_name,
                    "test_type": test_type,
                    "concept_type": concept_type
                }
            else:
                return {
                    "status": "error",
                    "duration": duration,
                    "error": f"Command failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "config_name": config_name,
                    "test_type": test_type,
                    "concept_type": concept_type
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Test timed out after 5 minutes",
                "config_name": config_name,
                "test_type": test_type,
                "concept_type": concept_type
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "config_name": config_name,
                "test_type": test_type,
                "concept_type": concept_type
            }
    
    def run_batch_tests(self, configs=None, test_types=None, concept_types=None):
        """Run multiple tests with different parameter combinations."""
        # Default parameters - start with simplest ones
        if configs is None:
            configs = ["synthetic"]  # Start with synthetic only
        if test_types is None:
            test_types = ["global"]  # Start with global only
        if concept_types is None:
            concept_types = ["importance"]  # Start with importance only
        
        results = []
        total_tests = len(configs) * len(test_types) * len(concept_types)
        current_test = 0
        
        print(f"\nRunning {total_tests} test combinations...")
        print("=" * 50)
        
        for config in configs:
            for test_type in test_types:
                for concept_type in concept_types:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}: {config} | {test_type} | {concept_type}")
                    print("-" * 40)
                    
                    result = self.run_original_test(config, test_type, concept_type)
                    results.append(result)
                    
                    # Print immediate result
                    if result["status"] == "success":
                        print(f"✓ SUCCESS in {result.get('duration', 0):.1f}s")
                    elif result["status"] == "timeout":
                        print("⏱ TIMEOUT")
                    else:
                        print(f"✗ FAILED: {result.get('error', 'Unknown error')}")
                    
                    # Show some output if available
                    if result.get("stderr") and len(result["stderr"]) > 0:
                        print(f"Warnings/Errors: {result['stderr'][:200]}...")
        
        return results
    
    def save_results(self, results, filename=None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        filepath = self.workdir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self, results):
        """Print a summary of test results."""
        if not results:
            print("No results to summarize.")
            return
        
        total = len(results)
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        timeouts = sum(1 for r in results if r["status"] == "timeout")
        
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests:     {total}")
        print(f"Successful:      {successful} ({successful/total:.1%})")
        print(f"Failed:          {failed} ({failed/total:.1%})")
        print(f"Timeouts:        {timeouts} ({timeouts/total:.1%})")
        
        if successful > 0:
            durations = [r.get("duration", 0) for r in results if r["status"] == "success"]
            avg_duration = sum(durations) / len(durations)
            print(f"Avg duration:    {avg_duration:.1f}s")
        
        # Show failed tests
        if failed > 0:
            print(f"\nFailed tests:")
            for result in results:
                if result["status"] == "error":
                    config = result.get("config_name", "unknown")
                    test_type = result.get("test_type", "unknown")
                    concept_type = result.get("concept_type", "unknown")
                    error = result.get("error", "Unknown error")[:100]
                    print(f"  - {config}|{test_type}|{concept_type}: {error}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal IBYDMT Test Runner")
    parser.add_argument("--config_name", type=str, help="Configuration name")
    parser.add_argument("--test_type", type=str, help="Test type (global, global_cond, local_cond)")
    parser.add_argument("--concept_type", type=str, help="Concept type (importance, rank, both)")
    parser.add_argument("--workdir", type=str, help="Working directory for results")
    parser.add_argument("--list_configs", action="store_true", help="List available configurations")
    parser.add_argument("--check_deps", action="store_true", help="Check dependencies")
    parser.add_argument("--batch_test", action="store_true", help="Run batch tests")
    parser.add_argument("--quick_test", action="store_true", help="Run one quick test")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MinimalIBYDMTRunner(workdir=args.workdir)
    
    print("Minimal IBYDMT Test Runner")
    print("=" * 30)
    
    if args.check_deps:
        if runner.check_dependencies():
            print("\n✓ Basic dependencies check passed!")
        else:
            print("\n✗ Dependency check failed!")
            return 1
        
    if args.list_configs:
        configs = runner.list_configurations()
        if configs:
            print("\nAvailable configurations:")
            for config in configs:
                print(f"  - {config}")
        else:
            print("\nNo configurations found or IBYDMT directory missing.")
        
        print("\nAvailable test types: global, global_cond, local_cond")
        print("Available concept types: importance, rank, both")
        return 0
    
    if args.quick_test:
        print("\nRunning quick test with synthetic data...")
        result = runner.run_original_test("synthetic", "global", "importance")
        
        if result["status"] == "success":
            print("✓ Quick test passed!")
            print(f"Duration: {result.get('duration', 0):.1f}s")
        else:
            print(f"✗ Quick test failed: {result.get('error', 'Unknown error')}")
            if result.get("stderr"):
                print(f"Error details: {result['stderr']}")
        
        return 0 if result["status"] == "success" else 1
    
    if args.batch_test:
        print("\nRunning batch tests...")
        results = runner.run_batch_tests()
        
        runner.print_summary(results)
        runner.save_results(results)
        
        return 0
    
    if args.config_name and args.test_type and args.concept_type:
        print(f"\nRunning single test: {args.config_name} | {args.test_type} | {args.concept_type}")
        
        result = runner.run_original_test(args.config_name, args.test_type, args.concept_type)
        
        if result["status"] == "success":
            print("✓ Test completed successfully!")
            print(f"Duration: {result.get('duration', 0):.1f}s")
            if result.get("stdout"):
                print(f"Output: {result['stdout'][:500]}...")
        else:
            print(f"✗ Test failed: {result.get('error', 'Unknown error')}")
            if result.get("stderr"):
                print(f"Error details: {result['stderr']}")
        
        return 0 if result["status"] == "success" else 1
    
    # Show help if no specific action
    parser.print_help()
    print("\nQuick start examples:")
    print("  python minimal_runner.py --check_deps")
    print("  python minimal_runner.py --list_configs")
    print("  python minimal_runner.py --quick_test")
    print("  python minimal_runner.py --batch_test")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
