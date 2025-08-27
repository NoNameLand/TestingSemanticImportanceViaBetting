#!/usr/bin/env python3
"""
Demo: Wealth gain experiment using tanh as in the article, with robust testing and wealth trajectory plots.
"""
import sys
sys.path.append('.')
import numpy as np
from config.experiment_config import ExperimentConfig, RelationshipConfig, DistributionConfig
from run_modular_experiments import run_single_experiment
from visualization.plotting import ExperimentVisualizer

# 1. Define experiment config with tanh relationship (wealth gain style)
config = ExperimentConfig(
    name="wealth_gain_tanh",
    description="Wealth gain experiment using tanh relationship as in the article",
    n_samples=500,
    d_hidden=20,
    m_concepts=5,
    create_visualizations=True,
    save_results=False
)

# Use normal distributions for all concepts
config.concept_distributions = {i: DistributionConfig.normal(0, 1) for i in range(config.m_concepts)}

# Use a tanh relationship for Y (simulate wealth gain style)
# Y = tanh(Z_0 + 0.5*Z_1 - 0.5*Z_2)
config.relationships = [
    RelationshipConfig.linear([0, 1, 2], [1.0, 0.5, -0.5]),
]

# After linear combination, apply tanh nonlinearity in the data generator
# We'll override the target generation for this experiment
from data_generation.generators import ExperimentDataGenerator

def custom_generate_target(Z):
    return np.tanh(Z[:, 0] + 0.5 * Z[:, 1] - 0.5 * Z[:, 2])

# Patch the generator for this experiment
orig_generate_target = ExperimentDataGenerator._generate_target
def patched_generate_target(self, Z):
    return custom_generate_target(Z)
ExperimentDataGenerator._generate_target = patched_generate_target

# 2. Run the experiment
results = run_single_experiment(config, visualize=True)

# Restore the original method for safety
ExperimentDataGenerator._generate_target = orig_generate_target

# 3. Plot robust testing and wealth trajectories
if results:
    visualizer = ExperimentVisualizer(config, results)
    visualizer.plot_robust_testing()
    visualizer.plot_wealth_trajectories()
    print("\n✅ Wealth gain tanh experiment completed and plots generated.")
else:
    print("❌ Experiment failed.")
