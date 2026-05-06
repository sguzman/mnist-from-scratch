# 🧠 Training Guide: How Iron Digits Learns

This guide provides a deep dive into the machine learning engineering behind
**Iron Digits**. We'll explore the mathematical foundations, the training loop
implementation, and how to optimize your models for maximum accuracy.

---

## 1. The Core Philosophy

In most modern ML projects, training is a black box handled by frameworks like
PyTorch. In **Iron Digits**, we've unrolled that box. Training is the process
of iteratively adjusting thousands of small numbers (weights and biases) so
that a specific input (a 28x28 grid of pixels) results in the correct output
(a digit 0-9).

### The Feedback Loop

1. **Forward Pass**: The image pixels are fed through the model's mathematical
   layers.
2. **Loss Calculation**: We measure how far the model's prediction was from the
   truth.
3. **Backward Pass (Backprop)**: We calculate the "blame"—which weights
   contributed most to the error?
4. **Optimizer Step**: We tweak the weights slightly in the direction that
   reduces the error.

---

## 2. Model Architectures

### A. Perceptron (The Fast Learner)

The simplest model. It treats each class as a template.

- **Training Method**: Mistake-driven.
- **Logic**: If the model predicts `7` but the answer was `3`:
  - It adds the image pixels to the `3` template.
  - It subtracts the image pixels from the `7` template.
- **Best for**: Rapid baselines (~88% accuracy).

### B. Softmax Regression (The Probabilistic Model)

A linear model that outputs probabilities (0.0 to 1.0) for each digit.

- **Loss Function**: Cross-Entropy Loss.
- **Training Method**: Gradient Descent.
- **Logic**: It calculates a "score" for each digit, turns them into
  probabilities via the Softmax function, and then uses the difference between
  its probabilities and the "truth" to update its weights.
- **Best for**: Understanding linear feature templates (~92% accuracy).

### C. Multi-Layer Perceptron (The Deep Thinker)

A 3-layer neural network with a hidden layer of "neurons."

- **Activation**: ReLU (Rectified Linear Unit) to introduce non-linearity.
- **Training Method**: Backpropagation via the Chain Rule.
- **Logic**: The error at the output is propagated backward through the hidden
  layer, calculating gradients for both sets of weights.
- **Best for**: High-performance digit recognition (~97%+ accuracy).

---

## 3. The Training Walkthrough

Follow these steps to train your own high-accuracy model.

### Step 1: Data Preparation

Before training, you must have the MNIST dataset.

```bash
cargo run --release --bin mnist-lab -- fetch
```

### Step 2: Choosing Hyperparameters

Hyperparameters are the "knobs" you turn to control learning.

- **Learning Rate (`--lr`)**: How big of a "step" the model takes when updating
  weights.
  - *Too high*: The model overshoots and becomes unstable.
  - *Too low*: The model takes forever to learn.
  - *Sweet spot*: `0.01` for MLP, `0.1` for Softmax.
- **Epochs (`--epochs`)**: How many times the model sees the entire training
  set (60,000 images).
  - *Sweet spot*: `10-15` for most models.

### Step 3: Execution

Run the training command. The laboratory will show a progress bar and log the
accuracy as it goes.

```bash
cargo run --release --bin mnist-lab -- train --model-type mlp --epochs 10 --lr 0.01
```

---

## 4. Understanding the Math (Backpropagation)

For the **MLP**, we implement the gradient of the loss function manually.

### The ReLU Derivative

During the backward pass, we only pass gradients through "active" neurons.

```rust
// Only pass gradient if the neuron was "on" (value > 0)
for (i, &val) in a1.iter().enumerate() {
    if val <= 0.0 {
        da1[i] = 0.0;
    }
}
```

### Weight Updates

We use **Stochastic Gradient Descent (SGD)**. We update the weights immediately
after seeing *every single image*.

```rust
// New Weight = Old Weight - (Learning Rate * Gradient)
row[j] -= (d_scores[i] * learning_rate) * xj;
```

---

## 5. Tips for High Accuracy

1. **Shuffle the Data**: The laboratory automatically shuffles the training set
   so the model doesn't "memorize" the order of the images.
2. **Normalize**: Pixels are scaled from `0-255` down to `0.0-1.0` so the math
   doesn't explode.
3. **Start Small**: Use the `perceptron` first to ensure your environment is
   working, then move to `mlp` for the "production" model.

---

*You are now a master of the Iron Digits training pipeline. Happy learning!*
