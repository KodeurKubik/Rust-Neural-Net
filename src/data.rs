use anyhow::Result;
use parquet::file::reader::SerializedFileReader;
use std::fs::File;
use std::path::Path;

pub fn read_parquet(path: &Path) -> Result<SerializedFileReader<File>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;

    Ok(reader)
}
