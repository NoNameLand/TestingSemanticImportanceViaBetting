#!/usr/bin/env python3
"""
Debug script to understand why all concepts are getting the same e-values.
"""

import numpy as np
from ibymdt_cav import SKITGlobal

def debug_e_value_calculation():
    """Debug the e-value calculation step by step."""
    print("DEBUGGING E-VALUE CALCULATION")
    print("=" * 50)
    
    # Use the same data as the main test
    np.random.seed(42)
    
    # Simple test data - create Y that's strongly correlated with one concept
    n_samples = 100
    
    # Concept 0: strong correlation
    Z0 = np.random.randn(n_samples)
    Y = 2.0 * Z0 + 0.1 * np.random.randn(n_samples)  # Strong correlation
    
    # Concept 1: no correlation  
    Z1 = np.random.randn(n_samples)
    
    print(f"True correlation Y-Z0: {np.corrcoef(Y, Z0)[0,1]:.3f}")
    print(f"True correlation Y-Z1: {np.corrcoef(Y, Z1)[0,1]:.3f}")
    
    # Test both concepts
    for concept_idx, Z_concept in enumerate([Z0, Z1]):
        print(f"\nTesting Concept {concept_idx}:")
        
        skit = SKITGlobal(alpha=0.01)
        np.random.seed(42 + concept_idx)
        indices = np.random.permutation(len(Y))
        
        for pair_idx in range(5):  # Just first 5 pairs
            idx1, idx2 = indices[2*pair_idx], indices[2*pair_idx + 1]
            y1, y2 = Y[idx1], Y[idx2]
            z1, z2 = Z_concept[idx1], Z_concept[idx2]
            
            # Calculate pair correlation
            pair_corr = np.corrcoef([y1, y2], [z1, z2])[0,1]
            
            e_val, accumulated_e = skit.step_pair(y1, z1, y2, z2)
            
            print(f"  Pair {pair_idx+1}: corr={pair_corr:.3f}, e_val={e_val:.3f}, accum_e={accumulated_e:.3f}")
            
            if skit.rejected():
                print(f"  REJECTED at pair {pair_idx+1}")
                break

if __name__ == "__main__":
    debug_e_value_calculation()
