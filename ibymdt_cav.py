#!/usr/bin/env python3
"""
Minimal version of IBYDMT CAV functionality for the simplified test.
"""

import numpy as np
from sklearn.neighbors import KernelDensity

class ConceptBank:
    """Simple concept bank for projecting representations to concept space."""
    
    def __init__(self, concepts):
        """
        Args:
            concepts: (m, d) array where each row is a concept vector
        """
        self.C = concepts
        self.m, self.d = concepts.shape
        
    def project(self, H):
        """
        Project representations H to concept space.
        Args:
            H: (n, d) representations
        Returns:
            Z: (n, m) concept projections
        """
        return H @ self.C.T
    
    def __len__(self):
        return self.m

class EValueAccumulator:
    """
    E-value accumulation test for testing linear relations.
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.e_values = []
        self.accumulated_e = 1.0
        self.rejected = False
        
    def add_e_value(self, Y_pair, Z_pair):
        """
        Add an e-value based on a pair of observations.
        Uses a more realistic correlation-based e-value for linear relations.
        """
        if len(Y_pair) != 2 or len(Z_pair) != 2:
            raise ValueError("Need exactly 2 observations per pair")
            
        # Calculate empirical correlation for this pair
        if np.var(Y_pair) > 1e-12 and np.var(Z_pair) > 1e-12:
            corr = np.corrcoef(Y_pair, Z_pair)[0, 1]
            
            # More nuanced e-value based on correlation strength
            abs_corr = abs(corr)
            if abs_corr > 0.8:  # Very strong correlation
                e_val = 1.0 + 3.0 * abs_corr
            elif abs_corr > 0.5:  # Strong correlation  
                e_val = 1.0 + 1.5 * abs_corr
            elif abs_corr > 0.3:  # Moderate correlation
                e_val = 1.0 + 0.5 * abs_corr
            else:  # Weak correlation
                e_val = 1.0 + 0.1 * abs_corr
        else:
            e_val = 1.0  # No correlation observable
            
        self.e_values.append(e_val)
        self.accumulated_e *= e_val
        
        # Check for rejection
        if self.accumulated_e >= 1.0 / self.alpha:
            self.rejected = True
            
        return e_val
    
    def is_rejected(self):
        return self.rejected
    
    def get_accumulated_e(self):
        return self.accumulated_e

class SKITGlobal:
    """
    Global SKIT test using e-value accumulation based on running statistics.
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.e_accumulator = EValueAccumulator(alpha)
        self.Y_samples = []
        self.Z_samples = []
        
    def step_pair(self, y1, z1, y2, z2):
        """Add a single pair and return e-value and accumulated e-value."""
        # Collect samples for running correlation
        self.Y_samples.extend([y1, y2])
        self.Z_samples.extend([z1, z2])
        
        # Calculate running correlation
        if len(self.Y_samples) >= 4:  # Need at least 4 points for meaningful correlation
            running_corr = np.corrcoef(self.Y_samples, self.Z_samples)[0, 1]
            
            # Convert running correlation to e-value
            abs_corr = abs(running_corr)
            if abs_corr > 0.8:  # Very strong correlation
                e_val = 1.2  # Conservative e-value
            elif abs_corr > 0.6:  # Strong correlation  
                e_val = 1.1
            elif abs_corr > 0.4:  # Moderate correlation
                e_val = 1.05
            elif abs_corr > 0.2:  # Weak correlation
                e_val = 1.01
            else:  # Very weak correlation
                e_val = 1.0
        else:
            e_val = 1.0  # Not enough data yet
        
        # Update accumulation manually
        self.e_accumulator.e_values.append(e_val)
        self.e_accumulator.accumulated_e *= e_val
        
        # Check for rejection
        if self.e_accumulator.accumulated_e >= 1.0 / self.alpha:
            self.e_accumulator.rejected = True
            
        return e_val, self.e_accumulator.accumulated_e
    
    def test(self, Y, Z_j):
        """
        Test independence between Y and Z_j using e-value accumulation
        """
        if len(Y) != len(Z_j):
            raise ValueError("Y and Z_j must have same length")
            
        # Process pairs sequentially
        n_pairs = len(Y) // 2
        for i in range(n_pairs):
            y1, y2 = Y[2*i], Y[2*i+1]
            z1, z2 = Z_j[2*i], Z_j[2*i+1]
            self.step_pair(y1, z1, y2, z2)
            
            if self.rejected():
                break
        
        return self.rejected(), self.e_accumulator.get_accumulated_e()
    
    def rejected(self):
        return self.e_accumulator.is_rejected()

def robust_importance_hoeffding(Y, Z, concept_indices, noise_std=0.2, n_samples=50, mass_threshold=0.9, confidence=0.95):
    """
    Calculate robust importance using Hoeffding inequality for individual concepts.
    Tests the probability that 90% of the mass of the concept distribution is important.
    
    Args:
        Y: Target variable
        Z: Concept matrix
        concept_indices: Indices of concepts to test (should be single concept for individual testing)
        noise_std: Standard deviation for concept distribution
        n_samples: Number of samples from concept distribution
        mass_threshold: Threshold for mass (default 0.9 for 90%)
        confidence: Confidence level (default 0.95)
    
    Returns:
        dict with robust importance results
    """
    import math
    
    # Sample from concept distribution
    importance_scores = []
    
    if len(concept_indices) == 1:
        concept_idx = concept_indices[0]
        # For individual concepts, we sample noisy versions of the concept values
        print(f"    Sampling from distribution of concept {concept_idx}...")
    else:
        print(f"    Sampling from distribution of concept combination {concept_indices}...")
    
    for sample_idx in range(n_samples):
        # Use different random seed for each sample to get variation
        np.random.seed(1000 + sample_idx * 17 + sum(concept_indices))
        
        # Subsample data for this test
        n_sub = min(150, len(Y))
        indices = np.random.choice(len(Y), size=n_sub, replace=False)
        Y_sub = Y[indices]
        Z_sub = Z[indices]
        
        # Create concept distribution sample
        if len(concept_indices) == 1:
            # For individual concept: add noise to the concept values
            concept_idx = concept_indices[0]
            Z_concept_base = Z_sub[:, concept_idx]
            Z_concept_noisy = Z_concept_base + np.random.normal(0, noise_std, size=len(Z_concept_base))
        else:
            # For multiple concepts: weighted combination with noise
            Z_concept_noisy = np.zeros(n_sub)
            for i in range(n_sub):
                base_values = Z_sub[i, concept_indices]
                noisy_values = base_values + np.random.normal(0, noise_std, len(concept_indices))
                Z_concept_noisy[i] = np.mean(noisy_values)
        
        # Test importance with e-value accumulation
        skit = SKITGlobal(alpha=0.1)  # More lenient for more variation
        
        # Use fewer pairs for computational efficiency
        n_pairs = min(15, len(Y_sub) // 2)
        indices_pairs = np.random.permutation(len(Y_sub))
        
        rejected = False
        for pair_idx in range(n_pairs):
            if 2*pair_idx + 1 >= len(Y_sub):
                break
            idx1, idx2 = indices_pairs[2*pair_idx], indices_pairs[2*pair_idx + 1]
            y1, y2 = Y_sub[idx1], Y_sub[idx2]
            z1, z2 = Z_concept_noisy[idx1], Z_concept_noisy[idx2]
            
            e_val, accumulated_e = skit.step_pair(y1, z1, y2, z2)
            if skit.rejected():
                rejected = True
                break
        
        # Convert to importance score (0 or 1)
        importance_scores.append(1.0 if rejected else 0.0)
    
    # Calculate empirical probability
    empirical_prob = float(np.mean(importance_scores))
    
    # Apply Hoeffding inequality for confidence bound
    epsilon = math.sqrt(-math.log((1 - confidence) / 2) / (2 * n_samples))
    lower_bound = float(max(0.0, empirical_prob - epsilon))
    upper_bound = float(min(1.0, empirical_prob + epsilon))
    
    # Check if 90% of mass is likely important
    mass_is_important = lower_bound >= mass_threshold
    
    result = {
        'empirical_probability': empirical_prob,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'epsilon': epsilon,
        'mass_threshold': mass_threshold,
        'mass_is_important': mass_is_important,
        'confidence': confidence,
        'n_samples': n_samples,
        'concept_indices': concept_indices,
        'importance_scores': importance_scores  # Store for debugging
    }
    
    return result

def secure_importance(run_fn, R=25, delta=0.05):
    """
    Secure importance testing using multiple runs with Hoeffding bounds.
    """
    successes = 0
    for _ in range(R):
        if run_fn():
            successes += 1
    
    # Calculate rejection rate and Hoeffding bound
    rejection_rate_hat = successes / R
    
    # Hoeffding bound: P(|p_hat - p| >= epsilon) <= 2 * exp(-2 * R * epsilon^2)
    # For confidence 1-delta: epsilon = sqrt(-log(delta/2) / (2*R))
    epsilon = np.sqrt(-np.log(delta / 2) / (2 * R))
    rejection_rate_lower_bound = max(0.0, rejection_rate_hat - epsilon)
    rejection_rate_upper_bound = min(1.0, rejection_rate_hat + epsilon)
    
    return {
        'rejection_rate_hat': rejection_rate_hat,
        'rejection_rate_lower_bound': rejection_rate_lower_bound, 
        'rejection_rate_upper_bound': rejection_rate_upper_bound,
        'successes': successes,
        'total_runs': R,
        'confidence': 1 - delta,
        'epsilon': epsilon
    }
