# Modular Concept Importance Testing Framework

A flexible, modular framework for testing concept importance in machine learning models using Statistical Kernel Independence Tests (SKIT).

## Overview

This framework allows you to:
- Configure different concept distributions (normal, uniform, beta, gamma, mixture models)
- Define various relationships between concepts and outputs (linear, polynomial, trigonometric, neural network, interactions)
- Generate different types of concept banks (random, structured, correlated, orthogonal)
- Run comprehensive SKIT testing with robust importance analysis
- Create detailed visualizations and reports

## Project Structure

```
├── config/
│   └── experiment_config.py      # Configuration classes and presets
├── data_generation/
│   └── generators.py            # Data generation and concept sampling
├── testing/
│   └── skit_testing.py         # SKIT testing implementation
├── visualization/
│   └── plotting.py             # Visualization and plotting functions
├── utils/
│   └── helpers.py              # Utility functions and analysis tools
├── run_modular_experiments.py  # Main experiment runner
├── ibymdt_cav.py               # Core SKIT implementation
└── README.md                   # This file
```

## Key Features

### 1. Flexible Configuration System

**Concept Distributions:**
- Normal distributions with custom mean/std
- Uniform distributions  
- Beta distributions (scaled and shifted)
- Gamma distributions
- Mixture of Gaussians

**Relationship Types:**
- Linear combinations
- Polynomial relationships
- Trigonometric functions (sin, cos, tanh)
- Exponential functions
- Interaction effects between concepts
- Neural network relationships

**Concept Bank Types:**
- Random: Standard random concept vectors
- Structured: Concepts focus on different regions of feature space
- Correlated: Concepts share common base patterns
- Orthogonal: Approximately orthogonal concepts with perturbations

### 2. Comprehensive Testing

- **Global SKIT**: Tests independence between concepts and outputs
- **Robust Importance**: Multiple runs with confidence bounds using Hoeffding inequalities
- **Performance Metrics**: Precision, recall, F1-scores
- **Discovery Metrics**: Mean Average Precision, Reciprocal Rank, Precision@K

### 3. Rich Visualizations

- Concept distribution plots
- Concept-Y correlation analysis
- SKIT wealth trajectories
- Performance comparisons
- Robust importance confidence intervals
- Relationship analysis plots

## Quick Start

### Running Preset Experiments

```python
from config.experiment_config import ExperimentPresets
from run_modular_experiments import run_single_experiment

# Run a basic linear experiment
config = ExperimentPresets.basic_linear()
results = run_single_experiment(config)

# Run nonlinear mixed relationships
config = ExperimentPresets.nonlinear_mixed()
results = run_single_experiment(config)
```

### Creating Custom Experiments

```python
from config.experiment_config import *

# Create custom experiment configuration
config = ExperimentConfig(
    name="my_experiment",
    description="Custom experiment with specific distributions and relationships",
    n_samples=2000,
    d_hidden=50,
    m_concepts=10
)

# Define concept distributions
config.concept_distributions = {
    0: DistributionConfig.beta(2, 5, 4, -2),  # Beta distribution
    1: DistributionConfig.gamma(3, 1),        # Gamma distribution  
    2: DistributionConfig.mixture_gaussian([-1, 1], [0.5, 0.8], [0.3, 0.7])
}

# Define relationships
config.relationships = [
    RelationshipConfig.polynomial([0, 1], degree=3),
    RelationshipConfig.trigonometric([2, 4], ["sin", "cos"], [2.0, 1.5]),
    RelationshipConfig.interaction([(0, 1), (2, 4)], "product")
]

# Run experiment
results = run_single_experiment(config)
```

### Running the Full Framework

```bash
python run_modular_experiments.py
```

This will run a comprehensive suite of experiments demonstrating all features.

## Configuration Examples

### Distribution Configurations

```python
# Normal distribution
DistributionConfig.normal(mean=0.5, std=1.2)

# Uniform distribution  
DistributionConfig.uniform(low=-2.0, high=3.0)

# Beta distribution (scaled to [-2, 2])
DistributionConfig.beta(alpha=2.0, beta=5.0, scale=4.0, shift=-2.0)

# Mixture of Gaussians
DistributionConfig.mixture_gaussian(
    means=[-1.0, 1.0], 
    stds=[0.5, 0.8], 
    weights=[0.3, 0.7]
)
```

### Relationship Configurations

```python
# Linear relationship: Y = 3*Z0 + 2*Z2 + 1.5*Z5
RelationshipConfig.linear([0, 2, 5], [3.0, 2.0, 1.5])

# Polynomial: Y = Z1^2 + Z3^3 (degree 2, coefficients auto-generated)
RelationshipConfig.polynomial([1, 3], degree=3)

# Trigonometric: Y = sin(2*Z1) + cos(1.5*Z4)  
RelationshipConfig.trigonometric([1, 4], ["sin", "cos"], [2.0, 1.5])

# Interactions: Y = Z0*Z4 + Z2*Z7
RelationshipConfig.interaction([(0, 4), (2, 7)], "product")

# Neural network with concepts [0,2,4,6] as input
RelationshipConfig.neural_network([0, 2, 4, 6], [20, 10], "relu")
```

### Concept Bank Configurations

```python
# Random concept bank
ConceptBankConfig.random("random_bank")

# Structured with 10% overlap between concept regions  
ConceptBankConfig.structured("structured_bank", overlap_ratio=0.1)

# Correlated concepts from 3 base patterns with 30% noise
ConceptBankConfig.correlated("correlated_bank", n_base_patterns=3, noise_level=0.3)

# Orthogonal concepts with 20% perturbation
ConceptBankConfig.orthogonal("orthogonal_bank", perturbation=0.2)
```

## Results and Analysis

Results are automatically saved to structured directories:

```
results/
├── experiment_name/
│   ├── experiment_results.json       # Complete results data
│   ├── experiment_summary.txt        # Human-readable summary
│   └── detailed_analysis.txt         # Detailed analysis report
├── visualizations/
│   ├── experiment_name/
│   │   ├── concept_distributions_*.pdf
│   │   ├── concept_correlations.pdf
│   │   ├── skit_trajectories_*.pdf
│   │   ├── performance_comparison.pdf
│   │   └── robust_importance_*.pdf
│   └── ...
└── experiment.log                   # Experiment log
```

### Key Metrics

- **Precision/Recall/F1**: Standard classification metrics for concept detection
- **Mean Average Precision (MAP)**: Ranking quality for concept importance
- **Mean Reciprocal Rank (MRR)**: Average rank of first true concept found  
- **Precision@K/Recall@K**: Performance in top-K concept identification
- **Confidence Intervals**: Robust bounds on importance probabilities

## Extending the Framework

### Adding New Distribution Types

Add to `data_generation/generators.py`:

```python
# In DistributionSampler.sample()
elif dist_config.name == "my_distribution":
    param1 = dist_config.params.get("param1", default_value)
    return my_sampling_function(param1, size)
```

### Adding New Relationship Types

Add to `data_generation/generators.py`:

```python  
# In RelationshipBuilder
elif rel_config.name == "my_relationship":
    return self._my_relationship_function(Z, rel_config)

def _my_relationship_function(self, Z, config):
    # Implement your relationship
    pass
```

### Adding Custom Visualizations

Add to `visualization/plotting.py`:

```python
def plot_my_analysis(self, data, output_dir):
    # Create your custom plots
    pass
```

## Dependencies

- numpy
- scipy  
- matplotlib
- seaborn
- pandas
- torch (for neural network relationships)
- scikit-learn
- pathlib

## Original IBYDMT Integration

The original IBYDMT project files are also included for reference:
- `IBYDMT/` - Original repository
- `minimal_runner.py` - Simple test runner
- Various integration scripts

## License

[Add your license information here]

## Usage Examples

### Quick Start (Recommended)

**Option 1: Minimal Runner (Works with basic Python)**
```bash
cd /root/ariel/projects/IBYDMT_proj
python3 minimal_runner.py --check_deps
python3 minimal_runner.py --quick_test
python3 minimal_runner.py --batch_test
```

**Option 2: Complete Examples**
```bash
python3 complete_example.py
```

**Option 3: Advanced Integration (Requires more dependencies)**
```bash
python3 test_runner.py --list_configs
python3 test_runner.py --batch_test
```

### Single Test Execution

Run a single test with specific parameters:

```bash
/root/ariel/projects/IBYDMT_proj/.venv/bin/python test_runner.py \
    --config_name synthetic \
    --test_type global \
    --concept_type importance
```

### Batch Testing

Run multiple tests with default parameters:

```bash
/root/ariel/projects/IBYDMT_proj/.venv/bin/python test_runner.py --batch_test
```

### Advanced Integration

For more sophisticated testing and integration with your projects:

```python
from ibydmt_integration import IBYDMTIntegrator

# Initialize integrator
integrator = IBYDMTIntegrator(workdir="./my_results")

# Run parameter sweep
results = integrator.parameter_sweep(
    configs=['synthetic', 'gaussian'],
    test_types=['global'],
    concept_types=['importance', 'rank']
)

# Analyze results
analysis = integrator.analyze_results(results)
print(f"Success rate: {analysis['success_rate']:.2%}")

# Save results for later analysis
integrator.save_results(results, "my_experiment.json")
```

## Available Configurations

The IBYDMT project includes several pre-configured datasets and test scenarios:

- **synthetic**: Simple synthetic data for quick testing
- **gaussian**: Gaussian-distributed synthetic data
- **cub**: Caltech-UCSD Birds (CUB) dataset
- **awa2**: Animals with Attributes 2 dataset
- **imagenette**: Subset of ImageNet

## Test Types

1. **global**: Global concept importance testing
2. **global_cond**: Global conditional concept importance testing
3. **local_cond**: Local conditional concept importance testing

## Concept Types

1. **importance**: Test concept importance
2. **rank**: Test concept ranking
3. **both**: Test both importance and ranking

## Integration with Your Projects

### Example 1: Parameter Exploration

```python
from ibydmt_integration import IBYDMTIntegrator

integrator = IBYDMTIntegrator()

# Test different configurations
configs_to_test = ['synthetic', 'gaussian']
results = integrator.parameter_sweep(
    configs=configs_to_test,
    test_types=['global', 'global_cond'],
    concept_types=['importance']
)

# Analyze which configuration works best
analysis = integrator.analyze_results(results)
best_config = max(analysis['config_stats'].items(), 
                  key=lambda x: x[1]['success_rate'])
print(f"Best configuration: {best_config[0]}")
```

### Example 2: Automated Testing Pipeline

```python
import schedule
import time
from ibydmt_integration import IBYDMTIntegrator

def run_daily_tests():
    integrator = IBYDMTIntegrator()
    
    # Run predefined experiments
    experiments = integrator.run_predefined_experiments()
    
    # Save results with timestamp
    for exp_name, results in experiments.items():
        if results:
            filename = f"{exp_name}_{datetime.now().strftime('%Y%m%d')}.json"
            integrator.save_results(results, filename)

# Schedule daily tests
schedule.every().day.at("02:00").do(run_daily_tests)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Example 3: Custom Test Configuration

```python
from ibydmt_integration import TestConfiguration, IBYDMTIntegrator

# Create custom test configuration
custom_config = TestConfiguration(
    config_name="synthetic",
    test_type="global",
    concept_type="importance",
    custom_params={"significance_level": 0.01}
)

integrator = IBYDMTIntegrator()
result = integrator.run_single_test(custom_config)

if result.status == "success":
    print(f"Test completed in {result.duration:.2f} seconds")
    print(f"Results saved to: {result.result_path}")
else:
    print(f"Test failed: {result.error_message}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Make sure you're using the virtual environment
   - Install missing dependencies as shown above

2. **CUDA/GPU Issues**:
   - The code will automatically fall back to CPU if CUDA is not available
   - For faster processing, ensure PyTorch is installed with CUDA support

3. **Memory Issues**:
   - Some datasets (like CUB) require significant memory
   - Start with synthetic datasets for testing

4. **SSL Certificate Issues**:
   - Use `--trusted-host` flags when installing packages
   - Some environments may require additional SSL configuration

### Getting Help

1. Check the original IBYDMT repository: https://github.com/Sulam-Group/IBYDMT
2. Review the paper for methodological details
3. Run `verify_setup.py` to diagnose setup issues

## File Descriptions

- **test_runner.py**: Simple command-line interface for running tests
- **ibydmt_integration.py**: Comprehensive integration library with advanced features
- **verify_setup.py**: Diagnostic script to verify the installation
- **README.md**: This documentation file

## Next Steps

1. Run `verify_setup.py` to ensure everything is working
2. Try running a simple test with `test_runner.py`
3. Explore the integration library for more advanced usage
4. Integrate the testing capabilities into your own projects

We hope you find this repositroy useful. 
