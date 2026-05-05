use crate::models::ArgMax;
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Mlp {
    pub w1: Array2<f32>, // (hidden_dim, input_dim)
    pub b1: Array1<f32>, // (hidden_dim)
    pub w2: Array2<f32>, // (output_dim, hidden_dim)
    pub b2: Array1<f32>, // (output_dim)
    pub learning_rate: f32,
}

impl Mlp {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, learning_rate: f32) -> Self {
        let mut rng = thread_rng();
        // He initialization for ReLU
        let std1 = (2.0 / input_dim as f32).sqrt();
        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.gen_range(-std1..std1));
        let b1 = Array1::zeros(hidden_dim);

        // Xavier initialization for output layer
        let std2 = (1.0 / hidden_dim as f32).sqrt();
        let w2 = Array2::from_shape_fn((output_dim, hidden_dim), |_| rng.gen_range(-std2..std2));
        let b2 = Array1::zeros(output_dim);

        Self {
            w1,
            b1,
            w2,
            b2,
            learning_rate,
        }
    }

    pub fn forward(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        let z1 = self.w1.dot(x) + &self.b1;
        let a1 = z1.mapv(|v| v.max(0.0)); // ReLU
        let z2 = self.w2.dot(&a1) + &self.b2;
        
        let max_score = z2.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = z2.mapv(|s| (s - max_score).exp());
        let sum_exp = exp_scores.sum();
        let probs = exp_scores / sum_exp;

        (a1, z2, probs)
    }

    pub fn predict(&self, x: &Array1<f32>) -> usize {
        let (_, _, probs) = self.forward(x);
        probs.argmax().unwrap_or(0)
    }

    pub fn train_step(&mut self, x: &Array1<f32>, y: usize) {
        let (a1, _z2, probs) = self.forward(x);

        // Output layer error
        let mut dz2 = probs;
        dz2[y] -= 1.0;

        // Hidden layer error
        let mut da1 = self.w2.t().dot(&dz2);
        // ReLU derivative
        for (i, &val) in a1.iter().enumerate() {
            if val <= 0.0 {
                da1[i] = 0.0;
            }
        }

        // Update weights/biases
        // w2 -= lr * dz2.outer(a1)
        for i in 0..self.w2.nrows() {
            let mut row = self.w2.row_mut(i);
            let grad_i = dz2[i] * self.learning_rate;
            for (j, &aj) in a1.iter().enumerate() {
                row[j] -= grad_i * aj;
            }
            self.b2[i] -= grad_i;
        }

        // w1 -= lr * da1.outer(x)
        for i in 0..self.w1.nrows() {
            let mut row = self.w1.row_mut(i);
            let grad_i = da1[i] * self.learning_rate;
            for (j, &xj) in x.iter().enumerate() {
                row[j] -= grad_i * xj;
            }
            self.b1[i] -= grad_i;
        }
    }
}
