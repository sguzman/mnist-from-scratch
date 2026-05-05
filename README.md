# MNIST from scratch in Rust

A high-performance, from-scratch implementation of classic machine learning
models for digit classification on the MNIST dataset, built with Rust and
`ndarray`.

## Features

- **Custom Dataset Pipeline**: Manual parser for the IDX binary format and
  automated dataset downloader.
- **Model Ladder**:
  - **Perceptron**: Multiclass mistake-driven linear classifier.
  - **Softmax Regression**: Probabilistic linear model with cross-entropy loss.
  - **Multi-Layer Perceptron (MLP)**: 3-layer neural network with ReLU
    activation and manual backpropagation.
- **Professional CLI**: Comprehensive toolkit for training, evaluating, and
  visual inspection.
- **Single Image Inference**: Predict digits from any 28x28 PNG file.
- **Persistence**: Models are saved as JSON files for easy sharing and
  deployment.

## Installation

Ensure you have Rust and Cargo installed.

```bash
git clone https://github.com/your-username/mnist-from-scratch
cd mnist-from-scratch
cargo build --release
```

## Usage

### 1. Fetch data

```bash
cargo run --bin mnist-lab -- fetch
```

### 2. Train a model

```bash
cargo run --bin mnist-lab -- train --model mlp --epochs 10 --lr 0.01
```

### 3. Evaluate

```bash
cargo run --bin mnist-lab -- eval --path model.json --model-type mlp
```

### 4. Predict from Image

```bash
cargo run --bin mnist-lab -- predict --image path/to/digit.png --model-type mlp
```

### 5. Inspect Dataset

```bash
cargo run --bin mnist-lab -- inspect --index 42 --dataset test
```

## Performance Comparison

| Model | Parameters | Test Accuracy | Training Time | Notes |
|-------|------------|---------------|---------------|-------|
| **Binary Perceptron (0 vs 1)** | ~1.5k | ~99% | < 1s | Simple baseline. |
| **Multiclass Perceptron** | 7.8k | ~88% | Fast | Mistake-driven. |
| **Softmax Regression** | 7.8k | ~92% | Fast | Probabilistic. |
| **MLP (784-128-10)** | ~100k | **~96%+** | Moderate | Backprop. |

## Model Ladder

### 1. Perceptron (Binary & Multiclass)

The simplest model tier. It uses a mistake-driven update rule. If the prediction
is wrong, it pushes the weights toward the correct features and away from the
incorrect ones.

### 2. Softmax Regression

A multiclass generalization of logistic regression. It uses the Softmax function
to turn raw scores into probabilities and minimizes cross-entropy loss using
stochastic gradient descent.

### 3. Multi-Layer Perceptron (MLP)

A "serious" neural network. It includes a hidden layer with ReLU activation.
Training is implemented using the backpropagation algorithm (manual gradient
calculation) to update weights across multiple layers.

## Subcommands

- `fetch`: Downloads the MNIST dataset to `.cache/`.
- `train`: Trains a model. Supports epochs, learning rate, and digit filtering.
- `evaluate` (or `eval`): Tests a model and displays a confusion matrix.
- `test`: Runs a standard battery of validation tests.
- `run` (or `predict`): Performs inference on a single image file.
- `inspect`: Visualizes dataset digits in the terminal.

## License

MIT
