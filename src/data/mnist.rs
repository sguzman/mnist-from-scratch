use crate::data::idx::parse_idx;
use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use tracing::info;

const BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist/";
const FILES: &[&str] = &[
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

pub struct MnistDataset {
    pub train_images: Array2<f32>,
    pub train_labels: Array1<u8>,
    pub test_images: Array2<f32>,
    pub test_labels: Array1<u8>,
}

impl MnistDataset {
    pub fn fetch() -> Result<()> {
        let cache_dir = PathBuf::from(".cache");
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }

        for file in FILES {
            let path = cache_dir.join(file);
            if !path.exists() {
                info!("Downloading {}...", file);
                let url = format!("{}{}", BASE_URL, file);
                let response = reqwest::blocking::get(url)?.bytes()?;
                let mut f = File::create(&path)?;
                f.write_all(&response)?;
            } else {
                info!("{} already exists in cache.", file);
            }
        }
        Ok(())
    }

    pub fn load() -> Result<Self> {
        Self::fetch()?;
        let cache_dir = PathBuf::from(".cache");

        info!("Parsing MNIST data...");
        let train_images = load_images(cache_dir.join("train-images-idx3-ubyte.gz"))?;
        let train_labels = load_labels(cache_dir.join("train-labels-idx1-ubyte.gz"))?;
        let test_images = load_images(cache_dir.join("t10k-images-idx3-ubyte.gz"))?;
        let test_labels = load_labels(cache_dir.join("t10k-labels-idx1-ubyte.gz"))?;

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}

fn load_images<P: AsRef<Path>>(path: P) -> Result<Array2<f32>> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    let idx = parse_idx(BufReader::new(decoder))?;
    
    let count = idx.dimensions[0] as usize;
    let rows = idx.dimensions[1] as usize;
    let cols = idx.dimensions[2] as usize;
    
    let data: Vec<f32> = idx.data.into_iter().map(|b| b as f32 / 255.0).collect();
    let arr = Array2::from_shape_vec((count, rows * cols), data)
        .context("Failed to reshape image data")?;
    
    Ok(arr)
}

fn load_labels<P: AsRef<Path>>(path: P) -> Result<Array1<u8>> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    let idx = parse_idx(BufReader::new(decoder))?;
    
    let count = idx.dimensions[0] as usize;
    let arr = Array1::from_shape_vec(count, idx.data)
        .context("Failed to reshape label data")?;
    
    Ok(arr)
}
