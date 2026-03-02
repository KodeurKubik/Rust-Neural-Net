use image::ImageReader;
use neuralnet::matrix::Matrix;
use neuralnet::optimizers::AdamBuilder;
use neuralnet::parquet::file::reader::FileReader;
use neuralnet::parquet::record::{Field, Row};
use neuralnet::{
    NUM,
    activation::{Linear, ReLU},
    data,
    network::Network,
};
use neuralnet::{load_model, save_model};
use std::io::Cursor;
use std::path::Path;

const LEARNING_RATE: NUM = 0.0001;
const ITERATIONS: usize = 101;
const BATCH_SIZE: usize = 64;

fn main() {
    let mut net = Network::new(AdamBuilder::new(LEARNING_RATE))
        .input_layer::<784, 128, ReLU>()
        .add_layer::<10, Linear>();

    let model_path = Path::new("./model.bin");
    if let Err(_) = load_model(&mut net, model_path) {
        eprintln!("Model not found on disk... starting from random weights instead");
    }

    let rows: Vec<(Vec<u8>, usize)> = {
        let reader = data::read_parquet(&Path::new("./data/train.parquet")).unwrap();
        reader
            .get_row_iter(None)
            .unwrap()
            .filter_map(|r| parse_row(r.unwrap()))
            .collect()
    };

    let (inputs, labels): (Vec<Matrix<784, 1>>, Vec<usize>) = rows
        .iter()
        .map(|(bytes, label)| (to_input(bytes), *label))
        .unzip();

    let test_rows: Vec<(Vec<u8>, usize)> = {
        let reader = data::read_parquet(&Path::new("./data/test.parquet")).unwrap();
        reader
            .get_row_iter(None)
            .unwrap()
            .filter_map(|r| parse_row(r.unwrap()))
            .collect()
    };

    let (test_inputs, test_labels): (Vec<Matrix<784, 1>>, Vec<usize>) = test_rows
        .iter()
        .map(|(bytes, label)| (to_input(bytes), *label))
        .unzip();

    for i in 0..ITERATIONS {
        for (inputs_batch, labels_batch) in inputs.chunks(BATCH_SIZE).zip(labels.chunks(BATCH_SIZE))
        {
            net.fit_batch(inputs_batch.to_vec(), labels_batch.to_vec());
        }

        if i % 10 == 0 {
            let losses = net.loss_batch(test_inputs.clone(), test_labels.clone());
            let avg_loss: NUM = losses.iter().sum::<NUM>() / losses.len() as NUM;

            println!("Iteration {i}, avg_loss={avg_loss}");
        }
    }

    save_model(&net, model_path).unwrap();
    println!("Model saved to {model_path:?}");
}

fn to_input(bytes: &Vec<u8>) -> Matrix<784, 1> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8();

    let pixels = img
        .pixels()
        .map(|p| p.0[0] as NUM / 255 as NUM)
        .collect::<Vec<NUM>>();
    let mut ready_pixels = [0 as NUM; 784];
    ready_pixels.copy_from_slice(&pixels);

    let data: [[NUM; 1]; 784] = std::array::from_fn(|i| [ready_pixels[i]]);
    Matrix::from(data)
}

fn parse_row(row: Row) -> Option<(Vec<u8>, usize)> {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut label: Option<usize> = None;

    for (name, field) in row.get_column_iter() {
        match (name.as_str(), field) {
            ("label", Field::Long(l)) => {
                label = Some(*l as usize);
            }
            ("image", Field::Group(image_row)) => {
                for (_, f) in image_row.get_column_iter() {
                    if let Field::Bytes(b) = f {
                        image_bytes = Some(b.data().to_vec());
                    }
                }
            }
            _ => {}
        }
    }

    Some((image_bytes?, label?))
}
