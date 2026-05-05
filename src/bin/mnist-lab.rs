use anyhow::Result;
use clap::{Parser, Subcommand};
use mnist_from_scratch::data::mnist::MnistDataset;
use mnist_from_scratch::models::{Perceptron, SoftmaxRegression, Mlp};
use mnist_from_scratch::train::Metrics;
use mnist_from_scratch::io;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Axis};

#[derive(Parser)]
#[command(name = "mnist-from-scratch")]
#[command(about = "A from-scratch Rust implementation of classic digit classifiers on MNIST", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train {
        /// Model type: perceptron, softmax, mlp
        #[arg(short, long, default_value = "mlp")]
        model: String,
        /// Number of epochs
        #[arg(short, long, default_value_t = 5)]
        epochs: usize,
        /// Learning rate
        #[arg(short, long, default_value_t = 0.01)]
        lr: f32,
        /// Hidden layer size (for MLP)
        #[arg(long, default_value_t = 128)]
        hidden: usize,
        /// Output path for the model
        #[arg(short, long, default_value = "model.json")]
        output: PathBuf,
        /// Specific digits to train on (e.g. 0,1 for binary)
        #[arg(short, long)]
        digits: Option<String>,
    },
    /// Evaluate a model
    /// Evaluate a model
    Eval {
        /// Path to the model file
        #[arg(short, long, default_value = "model.json")]
        path: PathBuf,
        /// Model type: perceptron, softmax, mlp
        #[arg(short = 't', long, default_value = "mlp")]
        model_type: String,
    },
    /// Predict a single digit from an image file
    Predict {
        /// Path to the model file
        #[arg(short, long, default_value = "model.json")]
        path: PathBuf,
        /// Model type: perceptron, softmax, mlp
        #[arg(short = 't', long, default_value = "mlp")]
        model_type: String,
        /// Path to the image file (PNG)
        #[arg(short, long)]
        image: PathBuf,
    },
    /// Inspect a digit from the dataset
    Inspect {
        /// Index of the digit to inspect
        #[arg(short, long)]
        index: usize,
        /// Dataset type: train or test
        #[arg(short, long, default_value = "test")]
        dataset: String,
    },
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Train { model, epochs, lr, hidden, output, digits } => {
            let mut dataset = MnistDataset::load()?;
            
            if let Some(digits_str) = digits {
                let allowed_digits: Vec<u8> = digits_str.split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
                info!("Filtering dataset for digits: {:?}", allowed_digits);
                
                // Filter train
                let mut train_indices = Vec::new();
                for i in 0..dataset.train_labels.len() {
                    if allowed_digits.contains(&dataset.train_labels[i]) {
                        train_indices.push(i);
                    }
                }
                let filtered_train_images = dataset.train_images.select(Axis(0), &train_indices);
                let filtered_train_labels = dataset.train_labels.select(Axis(0), &train_indices);

                // Filter test
                let mut test_indices = Vec::new();
                for i in 0..dataset.test_labels.len() {
                    if allowed_digits.contains(&dataset.test_labels[i]) {
                        test_indices.push(i);
                    }
                }
                let filtered_test_images = dataset.test_images.select(Axis(0), &test_indices);
                let filtered_test_labels = dataset.test_labels.select(Axis(0), &test_indices);

                dataset.train_images = filtered_train_images;
                dataset.train_labels = filtered_train_labels;
                dataset.test_images = filtered_test_images;
                dataset.test_labels = filtered_test_labels;
            }

            info!("Training {} for {} epochs...", model, epochs);

            match model.as_str() {
                "perceptron" => {
                    let mut m = Perceptron::new(784, 10, lr);
                    train_perceptron(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                }
                "softmax" => {
                    let mut m = SoftmaxRegression::new(784, 10, lr);
                    train_softmax(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                }
                "mlp" => {
                    let mut m = Mlp::new(784, hidden, 10, lr);
                    train_mlp(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                }
                _ => anyhow::bail!("Unknown model type: {}", model),
            }
        }
        Commands::Eval { path, model_type } => {
            let dataset = MnistDataset::load()?;
            info!("Evaluating {} from {:?}...", model_type, path);

            match model_type.as_str() {
                "perceptron" => {
                    let m: Perceptron = io::load_model(path)?;
                    let metrics = Metrics::calculate(|x| m.forward(x), &dataset.test_images, &dataset.test_labels, 10);
                    info!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
                    metrics.print_confusion_matrix();
                }
                "softmax" => {
                    let m: SoftmaxRegression = io::load_model(path)?;
                    let metrics = Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10);
                    info!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
                    metrics.print_confusion_matrix();
                }
                "mlp" => {
                    let m: Mlp = io::load_model(path)?;
                    let metrics = Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10);
                    info!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
                    metrics.print_confusion_matrix();
                }
                _ => anyhow::bail!("Unknown model type: {}", model_type),
            }
        }
        Commands::Predict { path, model_type, image } => {
            info!("Predicting using {} from {:?} on {:?}", model_type, path, image);
            let img = image::open(image)?.to_luma8();
            let img = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);
            let data: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
            let x = Array1::from_vec(data);

            let pred = match model_type.as_str() {
                "perceptron" => {
                    let m: Perceptron = io::load_model(path)?;
                    m.forward(&x)
                }
                "softmax" => {
                    let m: SoftmaxRegression = io::load_model(path)?;
                    m.predict(&x)
                }
                "mlp" => {
                    let m: Mlp = io::load_model(path)?;
                    m.predict(&x)
                }
                _ => anyhow::bail!("Unknown model type: {}", model_type),
            };
            info!("Prediction: {}", pred);
        }
        Commands::Inspect { index, dataset } => {
            let ds = MnistDataset::load()?;
            let (images, labels) = if dataset == "train" {
                (&ds.train_images, &ds.train_labels)
            } else {
                (&ds.test_images, &ds.test_labels)
            };

            if index >= images.nrows() {
                anyhow::bail!("Index out of bounds: {}", index);
            }

            let img = images.row(index);
            println!("Label: {}", labels[index]);
            for r in 0..28 {
                for c in 0..28 {
                    let val = img[r * 28 + c];
                    if val > 0.5 {
                        print!("##");
                    } else if val > 0.1 {
                        print!("..");
                    } else {
                        print!("  ");
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}

fn train_perceptron(model: &mut Perceptron, dataset: &MnistDataset, epochs: usize) -> Result<()> {
    for epoch in 1..=epochs {
        let pb = ProgressBar::new(dataset.train_images.nrows() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for i in 0..dataset.train_images.nrows() {
            let x = dataset.train_images.row(i).to_owned();
            let y = dataset.train_labels[i] as usize;
            model.train_step(&x, y);
            pb.inc(1);
        }
        pb.finish_with_message(format!("Epoch {} complete", epoch));

        let metrics = Metrics::calculate(|x| model.forward(x), &dataset.test_images, &dataset.test_labels, 10);
        info!("Epoch {}: Accuracy: {:.2}%", epoch, metrics.accuracy * 100.0);
    }
    Ok(())
}

fn train_softmax(model: &mut SoftmaxRegression, dataset: &MnistDataset, epochs: usize) -> Result<()> {
    for epoch in 1..=epochs {
        let pb = ProgressBar::new(dataset.train_images.nrows() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for i in 0..dataset.train_images.nrows() {
            let x = dataset.train_images.row(i).to_owned();
            let y = dataset.train_labels[i] as usize;
            model.train_step(&x, y);
            pb.inc(1);
        }
        pb.finish_with_message(format!("Epoch {} complete", epoch));

        let metrics = Metrics::calculate(|x| model.predict(x), &dataset.test_images, &dataset.test_labels, 10);
        info!("Epoch {}: Accuracy: {:.2}%", epoch, metrics.accuracy * 100.0);
    }
    Ok(())
}

fn train_mlp(model: &mut Mlp, dataset: &MnistDataset, epochs: usize) -> Result<()> {
    for epoch in 1..=epochs {
        let pb = ProgressBar::new(dataset.train_images.nrows() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());

        for i in 0..dataset.train_images.nrows() {
            let x = dataset.train_images.row(i).to_owned();
            let y = dataset.train_labels[i] as usize;
            model.train_step(&x, y);
            pb.inc(1);
        }
        pb.finish_with_message(format!("Epoch {} complete", epoch));

        let metrics = Metrics::calculate(|x| model.predict(x), &dataset.test_images, &dataset.test_labels, 10);
        info!("Epoch {}: Accuracy: {:.2}%", epoch, metrics.accuracy * 100.0);
    }
    Ok(())
}
