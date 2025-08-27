#!/usr/bin/env python3
"""
TRULY simplified test script for semantic importance testing with linear relationships.
Focus on robust importance with concept distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Simple implementations
class ConceptBank:
    def __init__(self, concepts):
        self.C = concepts
        self.m, self.d = concepts.shape
        
    def project(self, H):
        return H @ self.C.T

class ConceptDist:
    def __init__(self, weights, noise_std=0.2):
        self.weights = weights
        self.noise_std = noise_std

def correlation_test(Y, Z_j, threshold=0.3):
    """Simple correlation-based test."""
    corr = abs(np.corrcoef(Y.flatten(), Z_j.flatten())[0,1])
    return corr > threshold, corr

def secure_importance_test(Y, Z, concept_indices, noise_std=0.2, n_runs=25):
    """Test robust importance using concept distributions."""
    successes = 0
    
    for _ in range(n_runs):
        # Subsample data
        n_sub = int(0.7 * len(Y))
        indices = np.random.choice(len(Y), size=n_sub, replace=False)
        Y_sub = Y[indices]
        Z_sub = Z[indices]
        
        # Create noisy concept combination
        Z_important = np.zeros(n_sub)
        for i in range(n_sub):
            base_values = Z_sub[i, concept_indices]
            noisy_values = base_values + np.random.normal(0, noise_std, len(concept_indices))
            Z_important[i] = np.mean(noisy_values)
        
        # Test correlation
        rejected, _ = correlation_test(Y_sub, Z_important)
        if rejected:
            successes += 1
    
    return successes / n_runs

def main():
    print("SIMPLIFIED LINEAR ROBUST IMPORTANCE TESTING")
    print("=" * 60)
    
    # Setup
    np.random.seed(42)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate data
    n_samples = 800
    d_hidden = 5
    m_concepts = 8
    
    print(f"Data: {n_samples} samples, {d_hidden}D hidden, {m_concepts} concepts")
    
    # Create concept bank
    C = np.random.randn(m_concepts, d_hidden)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    concept_bank = ConceptBank(C)
    
    # Generate hidden representations and project to concept space
    H = np.random.randn(n_samples, d_hidden)
    Z = concept_bank.project(H)
    
    # Create Y with linear dependency on concepts 1, 3, 6
    true_concepts = [1, 3, 6]
    noise_level = 0.3
    Y = (2.5 * Z[:, 1] + 1.8 * Z[:, 3] + 1.2 * Z[:, 6] + 
         noise_level * np.random.randn(n_samples))
    
    print(f"True important concepts: {true_concepts}")
    print(f"Y statistics: mean={Y.mean():.3f}, std={Y.std():.3f}")
    
    # Test individual concepts
    print("\nSINGLE CONCEPT TESTS:")
    print("-" * 40)
    for j in range(m_concepts):
        rejected, corr = correlation_test(Y, Z[:, j])
        marker = "★" if j in true_concepts else " "
        status = "REJECTED" if rejected else "NOT REJECTED"
        print(f"Concept {j}: {marker} corr={corr:.3f} {status}")
    
    # Test robust importance of concept subsets
    print("\nROBUST IMPORTANCE TESTS:")
    print("-" * 40)
    
    # Test true subset
    rejection_rate = secure_importance_test(Y, Z, true_concepts)
    print(f"True concepts {true_concepts}: rejection_rate={rejection_rate:.3f}")
    
    # Test false subsets
    for false_subset in [[0, 2], [4, 7], [0, 5, 7]]:
        rejection_rate = secure_importance_test(Y, Z, false_subset)
        print(f"False concepts {false_subset}: rejection_rate={rejection_rate:.3f}")
    
    # Test mixed subsets
    for mixed_subset in [[1, 2], [3, 4], [1, 3, 7]]:
        has_true = any(c in true_concepts for c in mixed_subset)
        rejection_rate = secure_importance_test(Y, Z, mixed_subset)
        marker = "★" if has_true else " "
        print(f"Mixed concepts {mixed_subset}: {marker} rejection_rate={rejection_rate:.3f}")
    
    print("\nSUMMARY:")
    print("- High rejection rate (>0.5) indicates semantic importance")
    print("- Concept distributions add robustness to importance testing")
    print("- True concepts should have higher rejection rates than false ones")

if __name__ == "__main__":
    main()
