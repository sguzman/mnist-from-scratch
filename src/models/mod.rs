pub mod perceptron;
pub mod softmax;
pub mod mlp;

pub use perceptron::Perceptron;
pub use softmax::SoftmaxRegression;
pub use mlp::Mlp;

use ndarray::prelude::*;

pub trait ArgMax {
    fn argmax(&self) -> Option<usize>;
}

impl ArgMax for Array1<f32> {
    fn argmax(&self) -> Option<usize> {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
    }
}
