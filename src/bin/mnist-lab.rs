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
    /// Fetch and prepare the MNIST dataset
    Fetch,
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
        /// Export training metrics to this JSON file
        #[arg(long)]
        metrics_export: Option<PathBuf>,
    },
    /// Evaluate a model on the test set
    #[command(alias = "eval")]
    Evaluate {
        /// Path to the model file
        #[arg(short, long, default_value = "model.json")]
        path: PathBuf,
        /// Model type: perceptron, softmax, mlp
        #[arg(short = 't', long, default_value = "mlp")]
        model_type: String,
        /// Show indices of misclassified digits
        #[arg(long)]
        show_misclassified: bool,
    },
    /// Run a standard battery of tests on a model
    Test {
        /// Path to the model file
        #[arg(short, long, default_value = "model.json")]
        path: PathBuf,
        /// Model type: perceptron, softmax, mlp
        #[arg(short = 't', long, default_value = "mlp")]
        model_type: String,
    },
    /// Predict a single digit from an image file
    #[command(alias = "run")]
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
        Commands::Fetch => {
            MnistDataset::fetch()?;
            info!("MNIST dataset prepared in .cache/");
        }
        Commands::Train { model, epochs, lr, hidden, output, digits, metrics_export } => {
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

            let history = match model.as_str() {
                "perceptron" => {
                    let mut m = Perceptron::new(784, 10, lr);
                    let hist = train_perceptron(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                    hist
                }
                "softmax" => {
                    let mut m = SoftmaxRegression::new(784, 10, lr);
                    let hist = train_softmax(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                    hist
                }
                "mlp" => {
                    let mut m = Mlp::new(784, hidden, 10, lr);
                    let hist = train_mlp(&mut m, &dataset, epochs)?;
                    io::save_model(&m, output)?;
                    hist
                }
                _ => anyhow::bail!("Unknown model type: {}", model),
            };

            if let Some(export_path) = metrics_export {
                let json = serde_json::to_string_pretty(&history)?;
                std::fs::write(&export_path, json)?;
                info!("Training history exported to {:?}", export_path);
            }
        }
        Commands::Evaluate { path, model_type, show_misclassified } => {
            let dataset = MnistDataset::load()?;
            info!("Evaluating {} from {:?}...", model_type, path);

            let metrics = match model_type.as_str() {
                "perceptron" => {
                    let m: Perceptron = io::load_model(path)?;
                    Metrics::calculate(|x| m.forward(x), &dataset.test_images, &dataset.test_labels, 10)
                }
                "softmax" => {
                    let m: SoftmaxRegression = io::load_model(path)?;
                    Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10)
                }
                "mlp" => {
                    let m: Mlp = io::load_model(path)?;
                    Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10)
                }
                _ => anyhow::bail!("Unknown model type: {}", model_type),
            };

            info!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
            metrics.print_confusion_matrix();

            if show_misclassified {
                info!("Misclassified examples (first 20):");
                for (i, (&pred, &true_val)) in metrics.predictions.iter().zip(dataset.test_labels.iter()).enumerate().take(20) {
                    if pred != true_val {
                        info!("Index {}: Predicted {}, True {}", i, pred, true_val);
                    }
                }
            }
        }
        Commands::Test { path, model_type } => {
            let dataset = MnistDataset::load()?;
            info!("Running standard battery of tests for {} using model {:?}", model_type, path);
            
            // For now, Test is just a simplified Evaluate
            let acc = match model_type.as_str() {
                "perceptron" => {
                    let m: Perceptron = io::load_model(path)?;
                    Metrics::calculate(|x| m.forward(x), &dataset.test_images, &dataset.test_labels, 10).accuracy
                }
                "softmax" => {
                    let m: SoftmaxRegression = io::load_model(path)?;
                    Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10).accuracy
                }
                "mlp" => {
                    let m: Mlp = io::load_model(path)?;
                    Metrics::calculate(|x| m.predict(x), &dataset.test_images, &dataset.test_labels, 10).accuracy
                }
                _ => anyhow::bail!("Unknown model type: {}", model_type),
            };
            
            info!("Standard Test Result: {}%", (acc * 100.0) as u32);
            if acc > 0.9 {
                info!("PASSED: High accuracy detected.");
            } else {
                info!("WARNING: Accuracy is lower than expected for high-tier performance.");
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
        Commands::Generate { digit, model_type, path } => {
            info!("Generating 'ideal' digit {} using {} model from {:?}", digit, model_type, path);
            let k = digit as usize;
            
            let img = match model_type.as_str() {
                "perceptron" => {
                    let m: Perceptron = io::load_model(path)?;
                    let weights = m.weights.row(k).to_owned();
                    let min = weights.fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = weights.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    (weights - min) / (max - min)
                }
                "softmax" => {
                    let m: SoftmaxRegression = io::load_model(path)?;
                    let weights = m.weights.row(k).to_owned();
                    let min = weights.fold(f32::INFINITY, |a, &b| a.min(b));
                    let max = weights.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    (weights - min) / (max - min)
                }
                "mlp" => {
                    let m: Mlp = io::load_model(path)?;
                    let mut x = Array1::from_elem(784, 0.5);
                    let lr = 0.5;
                    for _ in 0..200 {
                        let (a1, _z2, _) = m.forward(&x);
                        let w2_k = m.w2.row(k);
                        let mut dz1 = w2_k.to_owned();
                        for (i, &val) in a1.iter().enumerate() {
                            if val <= 0.0 {
                                dz1[i] = 0.0;
                            }
                        }
                        let grad = m.w1.t().dot(&dz1);
                        x += &(grad * lr);
                        x.mapv_inplace(|v| v.clamp(0.0, 1.0));
                    }
                    x
                }
                _ => anyhow::bail!("Unknown model type: {}", model_type),
            };
            
            println!("Generated Digit: {}", digit);
            render_image(&img);
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

            let img = images.row(index).to_owned();
            println!("Label: {}", labels[index]);
            render_image(&img);
        }
    }

    Ok(())
}

fn train_perceptron(model: &mut Perceptron, dataset: &MnistDataset, epochs: usize) -> Result<Vec<f32>> {
    let mut history = Vec::new();
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
        history.push(metrics.accuracy);
    }
    Ok(history)
}

fn train_softmax(model: &mut SoftmaxRegression, dataset: &MnistDataset, epochs: usize) -> Result<Vec<f32>> {
    let mut history = Vec::new();
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
        history.push(metrics.accuracy);
    }
    Ok(history)
}

fn train_mlp(model: &mut Mlp, dataset: &MnistDataset, epochs: usize) -> Result<Vec<f32>> {
    let mut history = Vec::new();
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
        history.push(metrics.accuracy);
    }
    Ok(history)
}
