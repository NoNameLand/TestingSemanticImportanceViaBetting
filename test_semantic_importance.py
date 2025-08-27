#!/usr/bin/env python3
"""
Test script for semantic importance testing using the IBYDMT framework.
This creates toy examples to demonstrate global SKIT, conditional c-SKIT, and local x-SKIT.
"""

import numpy as np
import torch
import torch.nn as nn
from ibymdt_cav import (
    ConceptBank, ConceptDist, SKITGlobal, SKITConditional, XSKITLocal,
    JointHZConditional, greedy_e_fdr, secure_importance
)

def setup_toy_data():
    """Create synthetic data for testing semantic importance."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Synthetic dataset
    n_samples = 500
    d_hidden = 20  # hidden representation dimension
    m_concepts = 5  # number of concepts
    
    # Create concept vectors (random but fixed)
    C = np.random.randn(m_concepts, d_hidden)
    concept_bank = ConceptBank(C)
    
    # Generate hidden representations H
    H = np.random.randn(n_samples, d_hidden)
    
    # Project to concept space
    Z = concept_bank.project(H)
    
    # Create ground truth: Y depends on concepts 0 and 2, but not others
    # Y = f(Z_0, Z_2) + noise, where other concepts are irrelevant
    Y = (2.0 * Z[:, 0] + 1.5 * Z[:, 2] + 0.3 * np.random.randn(n_samples))
    
    print(f"Generated data: {n_samples} samples, {d_hidden}D hidden, {m_concepts} concepts")
    print(f"Ground truth: Y depends on concepts 0 and 2")
    
    return H, Z, Y, concept_bank

def test_global_skit(Z, Y):
    """Test global SKIT for marginal independence testing."""
    print("\n" + "="*60)
    print("TESTING GLOBAL SKIT (marginal independence)")
    print("="*60)
    
    results = {}
    
    # Test each concept for marginal independence with Y
    for j in range(Z.shape[1]):
        print(f"\nTesting concept {j} (expect rejection for j=0,2)")
        
        # Create SKIT instance
        skit = SKITGlobal(alpha=0.05, use_ons=True)
        
        # Randomly pair up samples for SKIT
        indices = np.random.permutation(len(Y))
        n_pairs = len(Y) // 2
        
        rejection_steps = []
        
        for i in range(n_pairs):
            idx1, idx2 = indices[2*i], indices[2*i + 1]
            y1, y2 = Y[idx1], Y[idx2]
            z1, z2 = Z[idx1, j], Z[idx2, j]
            
            kappa, K = skit.step_pair(y1, z1, y2, z2)
            
            if skit.rejected():
                rejection_steps.append(i + 1)
                break
        
        rejected = skit.rejected()
        results[j] = {
            'rejected': rejected,
            'final_wealth': skit.K,
            'rejection_step': rejection_steps[0] if rejection_steps else None
        }
        
        status = "REJECTED" if rejected else "NOT REJECTED"
        print(f"  Final wealth K = {skit.K:.3f}, Status: {status}")
        if rejection_steps:
            print(f"  Rejected at step {rejection_steps[0]}")
    
    print(f"\nGlobal SKIT Results Summary:")
    for j, res in results.items():
        print(f"  Concept {j}: {'✓' if res['rejected'] else '✗'} (K={res['final_wealth']:.3f})")
    
    return results

def test_conditional_skit(Z, Y):
    """Test c-SKIT for conditional independence testing."""
    print("\n" + "="*60)
    print("TESTING CONDITIONAL c-SKIT")
    print("="*60)
    
    # Fit concept distribution for conditional sampling
    concept_dist = ConceptDist(kind="gaussian")
    concept_dist.fit(Z)
    
    results = {}
    
    # Test each concept for conditional independence Y ⊥ Z_j | Z_{-j}
    for j in range(Z.shape[1]):
        print(f"\nTesting conditional independence for concept {j}")
        
        # Create conditional SKIT instance
        cskit = SKITConditional(alpha=0.05, use_ons=True)
        cskit.set_concept_index(j)
        
        # Sample subset of data
        n_test = min(100, len(Y))
        indices = np.random.choice(len(Y), n_test, replace=False)
        
        rejection_steps = []
        
        for i, idx in enumerate(indices):
            y = Y[idx]
            z_full = Z[idx]
            
            # Use the conditional sampler
            def sampler(z_minus, j_idx):
                return concept_dist.sample_Zj_given_Zminus(z_minus, j_idx)
            
            kappa, K = cskit.step(y, z_full, sampler)
            
            if cskit.rejected():
                rejection_steps.append(i + 1)
                break
        
        rejected = cskit.rejected()
        results[j] = {
            'rejected': rejected,
            'final_wealth': cskit.K,
            'rejection_step': rejection_steps[0] if rejection_steps else None
        }
        
        status = "REJECTED" if rejected else "NOT REJECTED"
        print(f"  Final wealth K = {cskit.K:.3f}, Status: {status}")
        if rejection_steps:
            print(f"  Rejected at step {rejection_steps[0]}")
    
    print(f"\nc-SKIT Results Summary:")
    for j, res in results.items():
        print(f"  Concept {j}: {'✓' if res['rejected'] else '✗'} (K={res['final_wealth']:.3f})")
    
    return results

def test_local_xskit(H, Z, Y):
    """Test x-SKIT for local explanations."""
    print("\n" + "="*60)
    print("TESTING LOCAL x-SKIT")
    print("="*60)
    
    # Fit joint (H, Z) distribution for conditional sampling
    joint_dist = JointHZConditional()
    joint_dist.fit(H, Z)
    
    # Create a simple prediction function g(h) that mimics the true relationship
    def g_function(h):
        # Project h to concepts and use ground truth relationship
        z_sample = ConceptBank(np.random.randn(Z.shape[1], H.shape[1])).project(h.reshape(1, -1))[0]
        return 2.0 * z_sample[0] + 1.5 * z_sample[2] + 0.1 * np.random.randn()
    
    # Pick a test instance
    test_idx = 42
    z_obs = Z[test_idx]
    print(f"Testing local importance for instance {test_idx}")
    print(f"Observed concepts: {z_obs}")
    
    results = {}
    
    # Test importance of each concept for this specific instance
    for j in range(Z.shape[1]):
        print(f"\nTesting local importance of concept {j}")
        
        xskit = XSKITLocal(alpha=0.05, use_ons=True)
        
        # Base set S (all concepts except j)
        S = [k for k in range(Z.shape[1]) if k != j]
        
        n_samples = 50  # number of samples to draw
        rejection_steps = []
        
        for i in range(n_samples):
            kappa, K = xskit.step(
                g_fn=g_function,
                sample_H_given_ZC=joint_dist.sample_H_given_ZC,
                z_obs=z_obs,
                S=S,
                j_idx=j
            )
            
            if xskit.rejected():
                rejection_steps.append(i + 1)
                break
        
        rejected = xskit.rejected()
        results[j] = {
            'rejected': rejected,
            'final_wealth': xskit.K,
            'rejection_step': rejection_steps[0] if rejection_steps else None
        }
        
        status = "REJECTED" if rejected else "NOT REJECTED"
        print(f"  Final wealth K = {xskit.K:.3f}, Status: {status}")
        if rejection_steps:
            print(f"  Rejected at step {rejection_steps[0]}")
    
    print(f"\nx-SKIT Results Summary:")
    for j, res in results.items():
        print(f"  Concept {j}: {'✓' if res['rejected'] else '✗'} (K={res['final_wealth']:.3f})")
    
    return results

def test_fdr_control(global_results):
    """Test FDR control using the wealth from global SKIT."""
    print("\n" + "="*60)
    print("TESTING FDR CONTROL")
    print("="*60)
    
    # Extract final wealths
    wealth_dict = {j: res['final_wealth'] for j, res in global_results.items()}
    
    print("Final wealths from global SKIT:")
    for j, K in wealth_dict.items():
        print(f"  Concept {j}: K = {K:.3f}")
    
    # Apply greedy e-FDR procedure
    rejected_list = greedy_e_fdr(wealth_dict, alpha=0.05)
    
    print(f"\nFDR-controlled rejections:")
    if rejected_list:
        for concept_id, threshold_rank in rejected_list:
            print(f"  Concept {concept_id} (rejected at rank {threshold_rank})")
    else:
        print("  No concepts rejected under FDR control")
    
    return rejected_list

def test_secure_importance():
    """Test the secure importance wrapper with repeated runs."""
    print("\n" + "="*60)
    print("TESTING SECURE IMPORTANCE")
    print("="*60)
    
    # Create a simple test function that should reject with some probability
    def mock_skit_run():
        # Simulate a SKIT run that rejects with probability ~0.7
        return np.random.random() < 0.7
    
    # Run secure importance test
    result = secure_importance(mock_skit_run, R=30, delta=0.05)
    
    print("Secure importance results:")
    print(f"  Rejection rate (estimated): {result['rejection_rate_hat']:.3f}")
    print(f"  Rejection rate (lower bound): {result['rejection_rate_lower_bound']:.3f}")
    print(f"  Prob. not important (upper bound): {result['prob_not_important_upper_bound']:.3f}")
    
    return result

def main():
    """Run all tests."""
    print("SEMANTIC IMPORTANCE TESTING")
    print("="*60)
    print("Testing the IBYDMT framework with synthetic data")
    
    # Set up toy data
    H, Z, Y, concept_bank = setup_toy_data()
    
    # Run tests
    try:
        global_results = test_global_skit(Z, Y)
        conditional_results = test_conditional_skit(Z, Y)
        local_results = test_local_xskit(H, Z, Y)
        fdr_results = test_fdr_control(global_results)
        secure_results = test_secure_importance()
        
        # Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print("Expected: Concepts 0 and 2 should be important (ground truth)")
        print("\nActual results:")
        print("Global SKIT rejections:", [j for j, res in global_results.items() if res['rejected']])
        print("c-SKIT rejections:", [j for j, res in conditional_results.items() if res['rejected']])
        print("x-SKIT rejections:", [j for j, res in local_results.items() if res['rejected']])
        print("FDR-controlled rejections:", [j for j, _ in fdr_results])
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
