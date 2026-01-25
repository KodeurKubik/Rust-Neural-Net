// use parquet::file::reader::{FileReader, SerializedFileReader};
// use std::fs::File;
// use std::path::Path;

// let file = File::open(&Path::new("./data/mnist/train.parquet")).unwrap();
// let reader = SerializedFileReader::new(file).unwrap();
// let mut iter = reader.get_row_iter(None).unwrap();
// while let Some(record) = iter.next() {
//     println!("{}", record.unwrap());
// }
