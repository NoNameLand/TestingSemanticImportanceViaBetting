#!/usr/bin/env python3
"""
Quick test script to verify IBYDMT setup is working correctly.
"""

import os
import sys
from pathlib import Path

# Add IBYDMT to Python path
IBYDMT_DIR = Path(__file__).parent / "IBYDMT"
sys.path.insert(0, str(IBYDMT_DIR))

def test_imports():
    """Test if we can import IBYDMT modules."""
    print("Testing imports...")
    
    try:
        import configs
        print("✓ configs module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import configs: {e}")
        return False
    
    try:
        from ibydmt.tester import ConceptTester
        print("✓ ConceptTester imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ConceptTester: {e}")
        return False
    
    try:
        from ibydmt.utils.config import get_config
        print("✓ Config utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config utilities: {e}")
        return False
    
    return True

def test_config_loading():
    """Test if we can load configurations."""
    print("\nTesting configuration loading...")
    
    try:
        from ibydmt.utils.config import get_config
        
        # Try to load synthetic config (should be simplest)
        config = get_config("synthetic")
        print(f"✓ Loaded synthetic config: {config.name}")
        
        # Try to load gaussian config
        config = get_config("gaussian")
        print(f"✓ Loaded gaussian config: {config.name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load configs: {e}")
        return False

def test_tester_creation():
    """Test if we can create a ConceptTester."""
    print("\nTesting ConceptTester creation...")
    
    try:
        from ibydmt.tester import ConceptTester
        
        # Create tester with synthetic config
        tester = ConceptTester("synthetic")
        print("✓ ConceptTester created successfully with synthetic config")
        
        # Check if it has the expected config
        print(f"✓ Config name: {tester.config.name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create ConceptTester: {e}")
        return False

def list_available_configs():
    """List all available configurations."""
    print("\nListing available configurations...")
    
    try:
        configs_dir = IBYDMT_DIR / "configs"
        config_files = [f.stem for f in configs_dir.glob("*.py") 
                       if not f.name.startswith("__")]
        
        print("Available configurations:")
        for config in config_files:
            print(f"  - {config}")
        
        return config_files
    except Exception as e:
        print(f"✗ Failed to list configs: {e}")
        return []

def main():
    """Run all tests."""
    print("IBYDMT Setup Verification")
    print("=" * 30)
    
    # Check if IBYDMT directory exists
    if not IBYDMT_DIR.exists():
        print(f"✗ IBYDMT directory not found at {IBYDMT_DIR}")
        print("Make sure you have cloned the IBYDMT repository to the correct location.")
        sys.exit(1)
    
    print(f"✓ IBYDMT directory found at {IBYDMT_DIR}")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
    
    if test_config_loading():
        tests_passed += 1
    
    if test_tester_creation():
        tests_passed += 1
    
    configs = list_available_configs()
    if configs:
        tests_passed += 1
    
    # Summary
    print(f"\n{'=' * 30}")
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! IBYDMT setup is working correctly.")
        print("\nYou can now run:")
        print("  python test_runner.py --list_configs")
        print("  python test_runner.py --batch_test")
        print("  python ibydmt_integration.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed:")
        print("   pip install torch numpy scipy scikit-learn matplotlib pandas tqdm joblib")
        print("2. Check that the IBYDMT directory structure is correct")
        print("3. Make sure you're using the correct Python environment")

if __name__ == "__main__":
    main()
