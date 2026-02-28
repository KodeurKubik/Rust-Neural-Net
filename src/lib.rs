pub type NUM = f64;

pub mod activation;
pub mod data;
pub mod layer;
pub mod matrix;
pub mod network;

pub use parquet;

#[cfg(test)]
mod tests;

//

use crate::network::Network;
use anyhow::Result;
use std::path::Path;

pub fn save_model<const IN: usize, const OUT: usize, L: serde::Serialize>(
    network: &Network<IN, OUT, L>,
    path: &Path,
) -> Result<()> {
    let bytes = bincode::serialize(network)?;
    std::fs::write(path, bytes)?;

    Ok(())
}

pub fn load_model<const IN: usize, const OUT: usize, L: serde::de::DeserializeOwned>(
    network: &mut Network<IN, OUT, L>,
    path: &Path,
) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let loaded: Network<IN, OUT, L> = bincode::deserialize(&bytes)?;
    *network = loaded;

    Ok(())
}
