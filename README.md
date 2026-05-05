# MNIST from scratch in Rust

A high-performance, from-scratch implementation of classic machine learning models for digit classification on the MNIST dataset, built with Rust and `ndarray`.

## Features

- **Custom Dataset Pipeline**: Manual parser for the IDX binary format and automated dataset downloader.
- **Model Ladder**:
  - **Perceptron**: Multiclass mistake-driven linear classifier.
  - **Softmax Regression**: Probabilistic linear model with cross-entropy loss.
  - **Multi-Layer Perceptron (MLP)**: 3-layer neural network with ReLU activation and manual backpropagation.
- **Professional CLI**: Comprehensive toolkit for training, evaluating, and visual inspection.
- **Single Image Inference**: Predict digits from any 28x28 PNG file.
- **Persistence**: Models are saved as JSON files for easy sharing and deployment.

## Installation

Ensure you have Rust and Cargo installed.

```bash
git clone https://github.com/your-username/mnist-from-scratch
cd mnist-from-scratch
cargo build --release
```

## Usage

### 1. Train a model
Train the Multi-Layer Perceptron on the full dataset:
```bash
cargo run --bin mnist-lab -- train --model mlp --epochs 10 --lr 0.01
```

Or train a binary perceptron on specific digits (e.g., 0 and 1):
```bash
cargo run --bin mnist-lab -- train --model perceptron --digits 0,1 --epochs 5
```

### 2. Evaluate
Generate a confusion matrix and calculate accuracy on the test set:
```bash
cargo run --bin mnist-lab -- eval --path model.json --model-type mlp
```

### 3. Predict from Image
Predict the digit in a PNG file:
```bash
cargo run --bin mnist-lab -- predict --image path/to/digit.png --model-type mlp
```

### 4. Inspect Dataset
View an ASCII representation of a digit from the dataset:
```bash
cargo run --bin mnist-lab -- inspect --index 42 --dataset test
```

## Architecture

The project is structured as a library (`src/lib.rs`) with a CLI binary (`src/bin/mnist-lab.rs`).

- `src/data/`: IDX parsing and MNIST dataset management.
- `src/models/`: Implementation of Perceptron, Softmax, and MLP architectures.
- `src/train/`: Metrics calculation (Accuracy, Confusion Matrix) and training loops.
- `src/io/`: JSON serialization logic for model weights.

## Performance

| Model | Epochs | Test Accuracy |
|-------|--------|---------------|
| Perceptron | 1 | ~88% |
| Softmax Regression | 5 | ~91% |
| MLP (784-128-10) | 10 | ~96% |

## License

MIT
