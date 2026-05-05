use ndarray::prelude::*;

pub struct Metrics {
    pub accuracy: f32,
    pub confusion_matrix: Array2<usize>,
}

impl Metrics {
    pub fn calculate<F>(predict: F, images: &Array2<f32>, labels: &Array1<u8>, num_classes: usize) -> Self 
    where F: Fn(&Array1<f32>) -> usize {
        let mut correct = 0;
        let mut confusion = Array2::zeros((num_classes, num_classes));

        for i in 0..images.nrows() {
            let x = images.row(i).to_owned();
            let y = labels[i] as usize;
            let pred = predict(&x);

            if pred == y {
                correct += 1;
            }
            if pred < num_classes && y < num_classes {
                confusion[[y, pred]] += 1;
            }
        }

        Self {
            accuracy: correct as f32 / images.nrows() as f32,
            confusion_matrix: confusion,
        }
    }

    pub fn print_confusion_matrix(&self) {
        println!("Confusion Matrix:");
        print!("    ");
        for i in 0..self.confusion_matrix.ncols() {
            print!("{:4} ", i);
        }
        println!("\n    {}", "-".repeat(self.confusion_matrix.ncols() * 5));

        for i in 0..self.confusion_matrix.nrows() {
            print!("{:2} | ", i);
            for j in 0..self.confusion_matrix.ncols() {
                print!("{:4} ", self.confusion_matrix[[i, j]]);
            }
            println!();
        }
    }
}
