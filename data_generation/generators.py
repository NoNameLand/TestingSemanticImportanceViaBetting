"""
Data generation module for concept importance experiments.
Handles distribution sampling and relationship creation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from scipy.stats import beta, gamma

from config.experiment_config import (
    DistributionConfig, RelationshipConfig, ConceptBankConfig, ExperimentConfig
)
from ibymdt_cav_new import ConceptBank


class DistributionSampler:
    """Handles sampling from various distributions."""
    
    @staticmethod
    def sample(dist_config: DistributionConfig, size: Tuple[int, ...]) -> np.ndarray:
        """Sample from a distribution configuration."""
        if dist_config.name == "normal":
            mean = dist_config.params.get("mean", 0.0)
            std = dist_config.params.get("std", 1.0)
            return np.random.normal(mean, std, size)
        
        elif dist_config.name == "uniform":
            low = dist_config.params.get("low", -1.0)
            high = dist_config.params.get("high", 1.0)
            return np.random.uniform(low, high, size)
        
        elif dist_config.name == "beta":
            alpha = dist_config.params.get("alpha", 2.0)
            beta_param = dist_config.params.get("beta", 2.0)
            scale = dist_config.params.get("scale", 2.0)
            shift = dist_config.params.get("shift", -1.0)
            samples = beta.rvs(alpha, beta_param, size=size)
            return samples * scale + shift
        
        elif dist_config.name == "gamma":
            shape = dist_config.params.get("shape", 2.0)
            scale = dist_config.params.get("scale", 1.0)
            return gamma.rvs(shape, scale=scale, size=size)
        
        elif dist_config.name == "mixture_gaussian":
            means = dist_config.params["means"]
            stds = dist_config.params["stds"]
            weights = np.array(dist_config.params["weights"])
            weights = weights / weights.sum()  # Normalize
            
            samples = np.zeros(size)
            for i in range(len(samples)):
                component = np.random.choice(len(means), p=weights)
                samples[i] = np.random.normal(means[component], stds[component])
            return samples
        
        else:
            raise ValueError(f"Unknown distribution: {dist_config.name}")


class RelationshipBuilder:
    """Builds Y from Z using various relationship types."""
    
    def __init__(self):
        self.fitted_models = {}
    
    def create_relationship(self, Z: np.ndarray, rel_config: RelationshipConfig) -> np.ndarray:
        """Create Y values based on relationship configuration."""
        if rel_config.name == "linear":
            return self._linear_relationship(Z, rel_config)
        elif rel_config.name == "polynomial":
            return self._polynomial_relationship(Z, rel_config)
        elif rel_config.name == "trigonometric":
            return self._trigonometric_relationship(Z, rel_config)
        elif rel_config.name == "exponential":
            return self._exponential_relationship(Z, rel_config)
        elif rel_config.name == "interaction":
            return self._interaction_relationship(Z, rel_config)
        elif rel_config.name == "neural_network":
            return self._neural_network_relationship(Z, rel_config)
        else:
            raise ValueError(f"Unknown relationship type: {rel_config.name}")
    
    def _linear_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Linear combination of concepts."""
        coeffs = config.params["coefficients"]
        indices = config.concept_indices
        
        if len(coeffs) != len(indices):
            raise ValueError("Number of coefficients must match number of concept indices")
        
        result = np.zeros(Z.shape[0])
        for i, idx in enumerate(indices):
            result += coeffs[i] * Z[:, idx]
        
        return config.weight * result
    
    def _polynomial_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Polynomial relationship."""
        degree = config.params["degree"]
        indices = config.concept_indices
        coeffs = config.params.get("coefficients")
        
        # If no coefficients provided, generate random ones
        if coeffs is None:
            n_terms = len(indices) * degree
            coeffs = np.random.randn(n_terms)
        
        result = np.zeros(Z.shape[0])
        coeff_idx = 0
        
        for idx in indices:
            for d in range(1, degree + 1):
                if coeff_idx < len(coeffs):
                    result += coeffs[coeff_idx] * (Z[:, idx] ** d)
                    coeff_idx += 1
        
        return config.weight * result
    
    def _trigonometric_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Trigonometric relationships."""
        functions = config.params["functions"]
        frequencies = config.params["frequencies"]
        indices = config.concept_indices
        
        result = np.zeros(Z.shape[0])
        
        for i, idx in enumerate(indices):
            freq = frequencies[i] if i < len(frequencies) else 1.0
            func_name = functions[i] if i < len(functions) else "sin"
            
            if func_name == "sin":
                result += np.sin(freq * Z[:, idx])
            elif func_name == "cos":
                result += np.cos(freq * Z[:, idx])
            elif func_name == "tan":
                result += np.tanh(freq * Z[:, idx])  # Use tanh to avoid infinities
            else:
                result += np.sin(freq * Z[:, idx])  # Default to sin
        
        return config.weight * result
    
    def _exponential_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Exponential relationships."""
        base = config.params.get("base", np.e)
        coeffs = config.params.get("coefficients")
        indices = config.concept_indices
        
        if coeffs is None:
            coeffs = np.ones(len(indices))
        
        result = np.zeros(Z.shape[0])
        for i, idx in enumerate(indices):
            coeff = coeffs[i] if i < len(coeffs) else 1.0
            # Clip to avoid overflow
            exponent = np.clip(coeff * Z[:, idx], -10, 10)
            result += np.power(base, exponent)
        
        return config.weight * result
    
    def _interaction_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Interaction effects between concept pairs."""
        pairs = config.concept_indices  # Should be list of tuples
        interaction_type = config.params.get("interaction_type", "product")
        
        result = np.zeros(Z.shape[0])
        
        for pair in pairs:
            if len(pair) != 2:
                continue
            
            idx1, idx2 = pair
            if interaction_type == "product":
                result += Z[:, idx1] * Z[:, idx2]
            elif interaction_type == "max":
                result += np.maximum(Z[:, idx1], Z[:, idx2])
            elif interaction_type == "min":
                result += np.minimum(Z[:, idx1], Z[:, idx2])
            else:
                result += Z[:, idx1] * Z[:, idx2]  # Default to product
        
        return config.weight * result
    
    def _neural_network_relationship(self, Z: np.ndarray, config: RelationshipConfig) -> np.ndarray:
        """Neural network-based relationship."""
        indices = config.concept_indices
        hidden_sizes = config.params.get("hidden_sizes", [10, 5])
        activation = config.params.get("activation", "relu")
        
        # Use the same key for consistent models across calls
        key = f"nn_{hash(tuple(indices))}_{tuple(hidden_sizes)}_{activation}"
        
        if key not in self.fitted_models:
            # Create and train a simple neural network
            input_size = len(indices)
            
            class SimpleNN(nn.Module):
                def __init__(self, input_size, hidden_sizes, activation):
                    super().__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_sizes:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        if activation == "relu":
                            layers.append(nn.ReLU())
                        elif activation == "tanh":
                            layers.append(nn.Tanh())
                        elif activation == "sigmoid":
                            layers.append(nn.Sigmoid())
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 1))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x).squeeze()
            
            # Create model
            model = SimpleNN(input_size, hidden_sizes, activation)
            
            # Generate some random weights (in practice, you might train this)
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.randn_like(param.data) * 0.5
            
            self.fitted_models[key] = model
        
        # Use the model
        model = self.fitted_models[key]
        Z_subset = Z[:, indices]
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(Z_subset)
            result = model(X_tensor).numpy()
        
        return config.weight * result


class ConceptBankGenerator:
    """Generates concept banks with different structures."""
    
    @staticmethod
    def generate(config: ConceptBankConfig, m_concepts: int, d_hidden: int) -> ConceptBank:
        """Generate a concept bank based on configuration."""
        if config.generation_type == "random":
            C = np.random.randn(m_concepts, d_hidden)
        
        elif config.generation_type == "structured":
            overlap_ratio = config.params.get("overlap_ratio", 0.1)
            C = ConceptBankGenerator._structured_concepts(m_concepts, d_hidden, overlap_ratio)
        
        elif config.generation_type == "correlated":
            n_base = config.params.get("n_base_patterns", 3)
            noise = config.params.get("noise_level", 0.3)
            C = ConceptBankGenerator._correlated_concepts(m_concepts, d_hidden, n_base, noise)
        
        elif config.generation_type == "orthogonal":
            perturbation = config.params.get("perturbation", 0.1)
            C = ConceptBankGenerator._orthogonal_concepts(m_concepts, d_hidden, perturbation)
        
        else:
            raise ValueError(f"Unknown concept bank type: {config.generation_type}")
        
        return ConceptBank(C)
    
    @staticmethod
    def _structured_concepts(m_concepts: int, d_hidden: int, overlap_ratio: float) -> np.ndarray:
        """Generate structured concepts with controlled overlap."""
        C = np.zeros((m_concepts, d_hidden))
        
        # Calculate how many dimensions each concept gets
        base_dims_per_concept = d_hidden // m_concepts
        overlap_dims = int(base_dims_per_concept * overlap_ratio)
        
        for i in range(m_concepts):
            start_idx = i * base_dims_per_concept
            end_idx = min((i + 1) * base_dims_per_concept + overlap_dims, d_hidden)
            
            # Fill with random values in assigned dimensions
            C[i, start_idx:end_idx] = np.random.randn(end_idx - start_idx)
        
        return C
    
    @staticmethod
    def _correlated_concepts(m_concepts: int, d_hidden: int, n_base: int, noise_level: float) -> np.ndarray:
        """Generate correlated concepts from base patterns."""
        base_patterns = np.random.randn(n_base, d_hidden)
        C = np.zeros((m_concepts, d_hidden))
        
        for i in range(m_concepts):
            # Random mixing weights
            weights = np.random.dirichlet(np.ones(n_base))
            # Mix base patterns
            C[i] = weights @ base_patterns
            # Add noise
            C[i] += noise_level * np.random.randn(d_hidden)
        
        return C
    
    @staticmethod
    def _orthogonal_concepts(m_concepts: int, d_hidden: int, perturbation: float) -> np.ndarray:
        """Generate approximately orthogonal concepts."""
        # Start with random matrix
        C = np.random.randn(m_concepts, d_hidden)
        
        # Gram-Schmidt orthogonalization
        for i in range(min(m_concepts, d_hidden)):
            # Orthogonalize against previous vectors
            for j in range(i):
                projection = np.dot(C[i], C[j]) * C[j]
                C[i] -= projection
            
            # Normalize
            norm = np.linalg.norm(C[i])
            if norm > 1e-10:
                C[i] /= norm
        
        # Add small perturbations to break perfect orthogonality
        C += perturbation * np.random.randn(m_concepts, d_hidden)
        
        return C


class ExperimentDataGenerator:
    """Main data generator for experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.relationship_builder = RelationshipBuilder()
    
    def generate_scenario(self, scenario_name: str = "experiment") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a complete experimental scenario."""
        np.random.seed(42)  # For reproducibility
        
        # Validate and fix indices before generating data
        self.config.validate_and_fix_indices()
        
        # Generate hidden representations
        H = np.random.randn(self.config.n_samples, self.config.d_hidden)
        
        # Generate concept bank (use single bank for simplicity)
        bank_config = self.config.concept_bank_config or ConceptBankConfig.random()
        bank = ConceptBankGenerator.generate(
            bank_config, self.config.m_concepts, self.config.d_hidden
        )
        
        # Project H onto concepts, then sample from configured distributions
        Z_base = bank.project(H)
        Z = self._sample_concept_distributions(Z_base)
        
        Y = self._generate_target(Z)
        
        # Determine true concepts
        true_concepts = set()
        if self.config.relationships is not None:
            for rel_config in self.config.relationships:
                if hasattr(rel_config.concept_indices[0], '__iter__'):
                    # Handle interaction case where indices might be pairs
                    for item in rel_config.concept_indices:
                        if isinstance(item, tuple):
                            true_concepts.update(item)
                        else:
                            true_concepts.add(item)
                else:
                    true_concepts.update(rel_config.concept_indices)
        
        scenario = {
            'name': scenario_name,
            'H': H,
            'Z': Z,
            'Y': Y,
            'true_concepts': sorted(list(true_concepts)),
            'config': self.config
        }
        
        concept_banks = {'primary': bank}
        
        return scenario, concept_banks
    
    def _sample_concept_distributions(self, Z_base: np.ndarray) -> np.ndarray:
        """Sample concepts according to configured distributions."""
        Z = np.zeros_like(Z_base)
        
        distributions = self.config.concept_distributions or {}
        
        for j in range(Z_base.shape[1]):
            if j in distributions:
                dist_config = distributions[j]
            else:
                # Default to standard normal
                dist_config = DistributionConfig.normal(0, 1)
            
            Z[:, j] = DistributionSampler.sample(dist_config, (Z_base.shape[0],))
        
        return Z
    
    def _generate_target(self, Z: np.ndarray) -> np.ndarray:
        """Generate target Y from concepts Z using configured relationships."""
        Y = np.zeros(Z.shape[0])
        
        relationships = self.config.relationships or []
        for rel_config in relationships:
            Y_component = self.relationship_builder.create_relationship(Z, rel_config)
            Y += Y_component
        
        # Add noise
        Y += self.config.noise_std * np.random.randn(Z.shape[0])
        
        return Y
