use crate::models::ArgMax;
use ndarray::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Perceptron {
    pub weights: Array2<f32>, // (num_classes, input_dim)
    pub bias: Array1<f32>,    // (num_classes)
    pub learning_rate: f32,
}

impl Perceptron {
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

    pub fn forward(&self, x: &Array1<f32>) -> usize {
        let scores = self.weights.dot(x) + &self.bias;
        scores.argmax().unwrap_or(0)
    }

    pub fn train_step(&mut self, x: &Array1<f32>, y: usize) {
        let prediction = self.forward(x);
        if prediction != y {
            // weights[y] += lr * x
            // weights[pred] -= lr * x
            let mut row_y = self.weights.row_mut(y);
            for (i, &xi) in x.iter().enumerate() {
                row_y[i] += xi * self.learning_rate;
            }
            self.bias[y] += self.learning_rate;

            let mut row_p = self.weights.row_mut(prediction);
            for (i, &xi) in x.iter().enumerate() {
                row_p[i] -= xi * self.learning_rate;
            }
            self.bias[prediction] -= self.learning_rate;
        }
    }
}
