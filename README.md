# mutual_info_estimopti

This repository provides optimization showcases using mutual information estimation techniques.

## Overview

The main focus is on training and testing Mutual Information Neural Estimation (MINE) models for various optimization tasks.

## Getting Started

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

### Training MINE

To train a MINE model, run:

```bash
python scripts/train_test_mine.py --mode train --config configs/mine_config.yaml
```

### Testing MINE

To test a trained MINE model, run:

```bash
python scripts/train_test_mine.py --mode test --config configs/mine_config.yaml --checkpoint path/to/checkpoint.pth
```

### Arguments

- `--mode`: `train` or `test`
- `--config`: Path to configuration YAML file
- `--checkpoint`: (Optional) Path to model checkpoint for testing

## Configuration

Model and training parameters are set in YAML files under the `configs/` directory.

## Results

Results and logs are saved in the `results/` directory after training or testing.

## License

See [LICENSE](LICENSE) for details.

# Script: `train_test_mine_two_param_noisy_channel.py`

This script demonstrates how to train and evaluate a Mutual Information Neural Estimation (MINE) model on a synthetic two-parameter noisy channel dataset. It is part of the `mutual_info_estimopti` project, which explores mutual information estimation techniques for optimization and analysis tasks.

---

## What is MINE?

**MINE (Mutual Information Neural Estimation)** is a neural network-based approach for estimating the mutual information between two random variables.  
It leverages the Donsker-Varadhan representation of the Kullback-Leibler divergence and trains a neural network to maximize a lower bound on the mutual information.  
This makes it suitable for high-dimensional and complex data where traditional estimation methods struggle.

- **Reference:** [MINE: Mutual Information Neural Estimation (Belghazi et al., 2018)](https://arxiv.org/abs/1801.04062)

---

## What does this script do?

1. **Generates synthetic data** for a noisy channel with two parameters (noise and damping).
2. **Trains a MINE model** to estimate the mutual information between the true and observed signals for various parameter settings.
3. **Evaluates the model** on test data and compares the estimated mutual information to a ground truth computed using scikit-learn.
4. **Visualizes** the results, including scatter plots of the data, histograms of the parameters, and 3D plots of mutual information as a function of the channel parameters.

---

## Usage

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- pyyaml

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the script

```bash
python scripts/train_test_mine_two_param_noisy_channel.py
```

The script expects a configuration YAML file (default: `config/noisy_channel.yaml`) specifying model and data parameters.

---

## Configuration

Edit the YAML config file to set:
- Number of samples per parameter setting
- Range and step size for the two parameters (noise, damping)
- Model hyperparameters (activation, learning rate, batch size, etc.)

---

## Output

- **Logs:** Training and testing mutual information estimates for each parameter setting.
- **Plots:** 
  - Scatter plots of input/output data
  - Histograms of parameter distributions
  - Plots of estimated mutual information

---

## References

- [MINE: Mutual Information Neural Estimation (Belghazi et al., 2018)](https://arxiv.org/abs/1801.04062)

---

## License

See [LICENSE](../LICENSE) for details.
