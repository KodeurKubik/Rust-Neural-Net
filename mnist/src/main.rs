use image::ImageReader;
use neuralnet::matrix::Matrix;
use neuralnet::parquet::file::reader::FileReader;
use neuralnet::parquet::record::{Field, Row};
use neuralnet::{
    NUM,
    activation::{Linear, ReLU},
    data,
    layer::Layer,
    network::Network,
};
use neuralnet::{load_model, save_model};
use std::io::Cursor;
use std::path::Path;

const LEARNING_RATE: NUM = 0.0001;
const ITERATIONS: usize = 11;

fn main() {
    let mut net = Network::new(LEARNING_RATE)
        .add_layer(Layer::<784, 128, ReLU>::new())
        .add_layer(Layer::<128, 10, Linear>::new());

    let model_path = Path::new("./model.bin");
    if let Err(_) = load_model(&mut net, model_path) {
        eprintln!("Model not found on disk... starting from random weights instead");
    }

    let rows: Vec<(Vec<u8>, usize)> = {
        let reader = data::read_parquet(&Path::new("./train.parquet")).unwrap();
        reader
            .get_row_iter(None)
            .unwrap()
            .filter_map(|r| parse_row(r.unwrap()))
            .collect()
    };

    let test_rows: Vec<(Vec<u8>, usize)> = {
        let reader = data::read_parquet(&Path::new("./test.parquet")).unwrap();
        reader
            .get_row_iter(None)
            .unwrap()
            .filter_map(|r| parse_row(r.unwrap()))
            .collect()
    };

    for i in 0..ITERATIONS {
        for (bytes, label) in &rows {
            net.fit(to_input(bytes), *label);
        }

        if i % 10 == 0 {
            let (inputs, correct_indices): (Vec<Matrix<784, 1>>, Vec<usize>) = test_rows
                .iter()
                .map(|(bytes, label)| (to_input(bytes), *label))
                .unzip();

            let losses = net.loss_batch(inputs, correct_indices);
            let avg_loss: NUM = losses.iter().sum::<NUM>() / losses.len() as NUM;

            println!("Iteration {i}, avg_loss={avg_loss}");
        } else {
            println!("Iteration {i}");
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

    !Matrix::from([ready_pixels])
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
