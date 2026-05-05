use crate::models::ArgMax;
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SoftmaxRegression {
    pub weights: Array2<f32>, // (num_classes, input_dim)
    pub bias: Array1<f32>,    // (num_classes)
    pub learning_rate: f32,
}

impl SoftmaxRegression {
    pub fn new(input_dim: usize, num_classes: usize, learning_rate: f32) -> Self {
        let mut rng = thread_rng();
        let weights = Array2::from_shape_fn((num_classes, input_dim), |_| rng.gen_range(-0.01..0.01));
        let bias = Array1::zeros(num_classes);
        
        Self {
            weights,
            bias,
            learning_rate,
        }
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let scores = self.weights.dot(x) + &self.bias;
        let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = scores.mapv(|s| (s - max_score).exp());
        let sum_exp = exp_scores.sum();
        exp_scores / sum_exp
    }

    pub fn predict(&self, x: &Array1<f32>) -> usize {
        self.forward(x).argmax().unwrap_or(0)
    }

    pub fn train_step(&mut self, x: &Array1<f32>, y: usize) {
        let probs = self.forward(x);
        
        // Gradient of cross-entropy w.r.t. scores: p_i - y_i
        let mut d_scores = probs;
        d_scores[y] -= 1.0;

        // Gradient w.r.t. weights: d_scores.outer(x)
        for i in 0..self.weights.nrows() {
            let mut row = self.weights.row_mut(i);
            let grad_i = d_scores[i] * self.learning_rate;
            for (j, &xj) in x.iter().enumerate() {
                row[j] -= grad_i * xj;
            }
            self.bias[i] -= grad_i;
        }
    }
}
