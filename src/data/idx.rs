use anyhow::{Context, Result};
use byteorder::{BigEndian, ReadBytesExt};
use std::io::Read;

#[derive(Debug)]
pub struct IdxFile {
    pub magic: u32,
    pub dimensions: Vec<u32>,
    pub data: Vec<u8>,
}

pub fn parse_idx<R: Read>(mut reader: R) -> Result<IdxFile> {
    let magic = reader.read_u32::<BigEndian>()?;
    let _zero = (magic >> 16) & 0xFFFF;
    let data_type = (magic >> 8) & 0xFF;
    let num_dims = magic & 0xFF;

    if data_type != 0x08 {
        anyhow::bail!("Unsupported data type in IDX file: 0x{:02x}", data_type);
    }

    let mut dimensions = Vec::with_capacity(num_dims as usize);
    for _ in 0..num_dims {
        dimensions.push(reader.read_u32::<BigEndian>()?);
    }

    let total_elements: usize = dimensions.iter().map(|&d| d as usize).product();
    let mut data = vec![0u8; total_elements];
    reader.read_exact(&mut data).context("Failed to read IDX data")?;

    Ok(IdxFile {
        magic,
        dimensions,
        data,
    })
}
