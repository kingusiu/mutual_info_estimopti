# Multi-Parameter Optimization with the Mutual Information Neural Estimator

This repository showcases optimization using the mutual information neural estimator.

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Other dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

## Usage

The core scripts are located in the `scripts/` directory.


## Demo Script: `train_test_mine_two_param_noisy_channel.py`

This script demonstrates how to train and evaluate the Mutual Information Neural Estimation (MINE) model on a synthetic two-parameter noisy channel dataset.

### Running the script

```bash
python scripts/train_test_mine_two_param_noisy_channel.py
```

---

## What does this script do?

1. **Generates synthetic data** for a noisy channel with two parameters (noise and damping).
2. **Trains a MINE model** to estimate the mutual information between the true and observed signals for various parameter settings.
3. **Evaluates the model** on test data and compares the estimated mutual information to a ground truth computed using scikit-learn.
4. **Visualizes** the results, including scatter plots of the data, histograms of the parameters, and plots of mutual information as a function of the channel parameters.

---

The script expects a configuration YAML file (default: `config/noisy_channel.yaml`) specifying model and data parameters.

---

## Demo Script: `train_surrogate_with_optimizer.py`

This script demonstrates how to use a surrogate model to optimize mutual information with respect to channel parameters, following these main steps:

1. **Loads mutual information results** previously estimated by the MINE model for different parameter settings from disk.
2. **Trains a surrogate model** (MLP) to learn the mapping from channel parameters (theta) to mutual information values.
3. **Validates the surrogate** using a held-out validation set and visualizes the fit between parameters and mutual information.
4. **Optimizes the parameters** using the trained surrogate model and a custom optimizer to find the parameter values that maximize mutual information, subject to constraints.
5. **Visualizes the optimization results** by plotting the optimized parameters and their corresponding mutual information values.

The script expects configuration and result files to be present (see the `config/` and `results/` directories).  
It is useful for exploring how surrogate modeling and optimization can be combined with mutual information estimation in experimental design or parameter tuning

---

## Configuration

Edit the YAML config file to set:
- Number of samples per parameter setting
- Range and step size for the two parameters (noise, damping)
- Model hyperparameters (activation, learning rate, batch size, etc.)

---

---

## What is MINE?

**MINE (Mutual Information Neural Estimation)** is a neural network-based approach for estimating the mutual information between two random variables.  
It leverages the Donsker-Varadhan representation of the Kullback-Leibler divergence and trains a neural network to maximize a lower bound on the mutual information.  
This makes it suitable for high-dimensional and complex data where traditional estimation methods struggle.

- **Reference:** [MINE: Mutual Information Neural Estimation (Belghazi et al., 2018)](https://arxiv.org/abs/1801.04062)


---

## License

See [LICENSE](../LICENSE) for details.
