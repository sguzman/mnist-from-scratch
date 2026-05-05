use mnist_from_scratch::data::mnist::MnistDataset;
use image::{Luma, ImageBuffer};

fn main() -> anyhow::Result<()> {
    let ds = MnistDataset::load()?;
    let img_data = ds.test_images.row(0);
    let mut img = ImageBuffer::new(28, 28);
    for r in 0..28 {
        for c in 0..28 {
            let val = (img_data[r * 28 + c] * 255.0) as u8;
            img.put_pixel(c as u32, r as u32, Luma([val]));
        }
    }
    img.save("test.png")?;
    println!("Saved test.png (Label: {})", ds.test_labels[0]);
    Ok(())
}
