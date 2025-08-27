"""
Configuration system for semantic importance experiments.
Defines distributions, relationships, and experiment parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DistributionConfig:
    """Configuration for concept distributions."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def normal(cls, mean: float = 0.0, std: float = 1.0):
        return cls("normal", {"mean": mean, "std": std})
    
    @classmethod
    def uniform(cls, low: float = -1.0, high: float = 1.0):
        return cls("uniform", {"low": low, "high": high})
    
    @classmethod
    def beta(cls, alpha: float = 2.0, beta: float = 2.0, scale: float = 2.0, shift: float = -1.0):
        """Beta distribution scaled to [shift, shift+scale]"""
        return cls("beta", {"alpha": alpha, "beta": beta, "scale": scale, "shift": shift})
    
    @classmethod
    def gamma(cls, shape: float = 2.0, scale: float = 1.0):
        return cls("gamma", {"shape": shape, "scale": scale})
    
    @classmethod
    def mixture_gaussian(cls, means: List[float], stds: List[float], weights: List[float]):
        """Mixture of Gaussians"""
        return cls("mixture_gaussian", {"means": means, "stds": stds, "weights": weights})


@dataclass
class RelationshipConfig:
    """Configuration for Y-Z relationships."""
    name: str
    concept_indices: List[int]
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    @classmethod
    def linear(cls, concept_indices: List[int], coefficients: List[float]):
        return cls("linear", concept_indices, {"coefficients": coefficients})
    
    @classmethod
    def polynomial(cls, concept_indices: List[int], degree: int = 2, coefficients: Optional[List[float]] = None):
        return cls("polynomial", concept_indices, {"degree": degree, "coefficients": coefficients})
    
    @classmethod
    def trigonometric(cls, concept_indices: List[int], functions: List[str] = None, frequencies: List[float] = None):
        """functions: ['sin', 'cos', 'tan'], frequencies: multipliers for each function"""
        if functions is None:
            functions = ["sin"] * len(concept_indices)
        if frequencies is None:
            frequencies = [1.0] * len(concept_indices)
        return cls("trigonometric", concept_indices, {"functions": functions, "frequencies": frequencies})
    
    @classmethod
    def exponential(cls, concept_indices: List[int], base: float = np.e, coefficients: List[float] = None):
        return cls("exponential", concept_indices, {"base": base, "coefficients": coefficients})
    
    @classmethod
    def interaction(cls, concept_pairs: List[tuple], interaction_type: str = "product"):
        """interaction_type: 'product', 'max', 'min'"""
        return cls("interaction", concept_pairs, {"interaction_type": interaction_type})
    
    @classmethod
    def neural_network(cls, concept_indices: List[int], hidden_sizes: List[int] = None, activation: str = "relu"):
        """Simple neural network relationship"""
        if hidden_sizes is None:
            hidden_sizes = [10, 5]
        return cls("neural_network", concept_indices, {"hidden_sizes": hidden_sizes, "activation": activation})


@dataclass
class ConceptBankConfig:
    """Configuration for concept bank generation."""
    name: str
    generation_type: str  # "random", "structured", "correlated", "orthogonal"
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def random(cls, name: str = "random"):
        return cls(name, "random", {})
    
    @classmethod
    def structured(cls, name: str = "structured", overlap_ratio: float = 0.1):
        """Each concept focuses on different parts with some overlap"""
        return cls(name, "structured", {"overlap_ratio": overlap_ratio})
    
    @classmethod
    def correlated(cls, name: str = "correlated", n_base_patterns: int = 3, noise_level: float = 0.3):
        return cls(name, "correlated", {"n_base_patterns": n_base_patterns, "noise_level": noise_level})
    
    @classmethod
    def orthogonal(cls, name: str = "orthogonal", perturbation: float = 0.1):
        """Orthogonal concepts with small perturbations"""
        return cls(name, "orthogonal", {"perturbation": perturbation})


@dataclass
class SKITConfig:
    """Configuration for SKIT testing parameters."""
    alpha: float = 0.01
    use_ons: bool = True
    n_pairs: Optional[int] = None  # If None, use all available pairs
    subsample_ratio: float = 1.0  # Fraction of data to use


@dataclass
class RobustTestConfig:
    """Configuration for robust importance testing."""
    n_runs: int = 25
    confidence: float = 0.95
    subsample_size: int = 500


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    name: str = "experiment"
    description: str = "An experiment"
    n_samples: int = 1000
    d_hidden: int = 50
    m_concepts: int = 10
    
    # Data configuration
    concept_distributions: Optional[Dict[int, DistributionConfig]] = None
    concept_bank_config: Optional[ConceptBankConfig] = None
    relationships: Optional[List[RelationshipConfig]] = None
    noise_std: float = 0.1
    
    # Experiment settings
    test_size: float = 0.2
    random_state: int = 42
    
    # Analysis settings  
    create_visualizations: bool = True
    save_results: bool = True
    
    # Testing configurations
    skit_config: Optional[SKITConfig] = None
    robust_config: Optional[RobustTestConfig] = None
    output_dir: str = "results"
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.concept_distributions is None:
            # Default: all concepts have standard normal distribution
            self.concept_distributions = {
                i: DistributionConfig.normal(0, 1) for i in range(self.m_concepts)
            }
        
        if self.concept_bank_config is None:
            self.concept_bank_config = ConceptBankConfig.random()
            
        if self.relationships is None:
            # Default: linear relationship with first few concepts
            n_concepts_to_use = min(3, self.m_concepts)
            indices = list(range(n_concepts_to_use))
            coeffs = [1.0] * n_concepts_to_use
            self.relationships = [RelationshipConfig.linear(indices, coeffs)]
        
        if self.skit_config is None:
            self.skit_config = SKITConfig()
            
        if self.robust_config is None:
            self.robust_config = RobustTestConfig()
    
    def validate_and_fix_indices(self):
        """Validate and fix concept indices to be within bounds."""
        if self.relationships is None:
            return
            
        for rel in self.relationships:
            # Fix indices that are out of bounds
            valid_indices = [idx for idx in rel.concept_indices if idx < self.m_concepts]
            
            # If we lost indices, replace with valid ones
            while len(valid_indices) < len(rel.concept_indices):
                for i in range(self.m_concepts):
                    if i not in valid_indices:
                        valid_indices.append(i)
                        break
                if len(valid_indices) >= len(rel.concept_indices):
                    break
            
            rel.concept_indices = valid_indices[:len(rel.concept_indices)]


# Predefined experiment configurations
class ExperimentPresets:
    """Collection of predefined experiment configurations."""
    
    @staticmethod
    def basic_linear():
        """Basic linear relationships with normal distributions."""
        config = ExperimentConfig(
            name="basic_linear",
            description="Linear relationships with normal concept distributions"
        )
        config.relationships = [
            RelationshipConfig.linear([0, 2, 4], [3.0, 2.0, 1.5])
        ]
        return config
    
    @staticmethod
    def nonlinear_mixed():
        """Mixed nonlinear relationships."""
        config = ExperimentConfig(
            name="nonlinear_mixed",
            description="Mixed nonlinear relationships including trig and polynomial"
        )
        config.relationships = [
            RelationshipConfig.trigonometric([1, 3], ["sin", "cos"], [1.0, 2.0]),
            RelationshipConfig.polynomial([4, 6], degree=2),
            RelationshipConfig.exponential([8])
        ]
        return config
    
    @staticmethod
    def interaction_effects():
        """Interaction effects between concepts."""
        config = ExperimentConfig(
            name="interaction_effects",
            description="Concept interactions and combinations"
        )
        config.relationships = [
            RelationshipConfig.interaction([(0, 4), (2, 7)], "product"),
            RelationshipConfig.linear([1, 6], [2.0, 1.5])
        ]
        return config
    
    @staticmethod
    def distribution_variety():
        """Various concept distributions."""
        config = ExperimentConfig(
            name="distribution_variety",
            description="Different concept distributions with linear relationships"
        )
        config.concept_distributions = {
            0: DistributionConfig.normal(0, 1),
            1: DistributionConfig.uniform(-2, 2),
            2: DistributionConfig.beta(2, 5, 4, -2),
            3: DistributionConfig.gamma(2, 1),
            4: DistributionConfig.mixture_gaussian([-1, 1], [0.5, 0.5], [0.6, 0.4])
        }
        config.relationships = [
            RelationshipConfig.linear([0, 1, 2, 3, 4], [1.0, 1.5, 2.0, 1.2, 1.8])
        ]
        return config
    
    @staticmethod
    def neural_network_relationship():
        """Neural network-based relationships."""
        config = ExperimentConfig(
            name="neural_network",
            description="Neural network relationships between concepts and output"
        )
        config.relationships = [
            RelationshipConfig.neural_network([0, 2, 4, 6], [20, 10], "relu"),
            RelationshipConfig.linear([1, 8], [1.0, 0.5])  # Add some linear component
        ]
        return config
