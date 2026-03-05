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
use ratatui::crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType};
use ratatui::{Frame, symbols};
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

const LEARNING_RATE: NUM = 0.001;
const ITERATIONS: usize = 51;
const BATCH_SIZE: usize = 64;

fn draw(frame: &mut Frame, iter_loss_points: &Vec<NUM>) {
    let area = frame.area();

    if iter_loss_points.is_empty() {
        frame.render_widget(
            Span::styled(
                "  waiting for first iteration...",
                Style::default().fg(Color::DarkGray),
            ),
            area,
        );
        return;
    }

    let data_points: Vec<(f64, f64)> = iter_loss_points
        .iter()
        .enumerate()
        .map(|(i, &loss)| (i as f64, loss as f64))
        .collect();

    let max_iter = (iter_loss_points.len() - 1).max(1) as f64;
    let max_loss = data_points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    // let y_padding = ((max_loss - min_loss) * 0.15).max(1e-4);
    let y_max = max_loss;
    let y_mid = (y_max) / 2.0;

    let current_loss = iter_loss_points.last().unwrap();
    let iteration = iter_loss_points.len() - 1;

    let datasets = vec![
        Dataset::default()
            .name(format!("iter {iteration}  loss {current_loss:.4}"))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&data_points),
    ];

    let x_labels = vec![
        Line::from("0"),
        Line::from(format!("{}", (max_iter / 2.0) as usize)),
        Line::from(format!("{}", max_iter as usize)),
    ];

    let y_labels = vec![
        Line::from(Span::styled(
            format!("0"),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            format!("{y_mid:.4}"),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            format!("{y_max:.4}"),
            Style::default().fg(Color::Cyan),
        )),
    ];

    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .labels(x_labels)
                .bounds([0.0, max_iter]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .labels(y_labels)
                .bounds([0f64, y_max]),
        );

    frame.render_widget(chart, area);
}

fn should_quit() -> bool {
    if event::poll(Duration::from_millis(0)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            return key.code == KeyCode::Char('q')
                || (key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL));
        }
    }
    false
}

fn main() {
    let mut net = Network::new(AdamBuilder::new(LEARNING_RATE))
        .input_layer::<784, 256, ReLU>()
        .add_layer::<128, ReLU>()
        .add_layer::<10, Linear>();

    let model_path = Path::new("./model.bin");
    if let Err(_) = load_model(&mut net, model_path) {
        eprintln!("Model not found on disk... starting from random weights instead");
    }

    let mut terminal = ratatui::init();
    let mut iter_loss_points: Vec<NUM> = Vec::new();
    let _ = terminal.draw(|frame| draw(frame, &iter_loss_points));

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

    let mut last_avg_loss = 0 as NUM;
    let mut emergency_exit = false;
    'training: for i in 0..ITERATIONS {
        let losses = net.loss_batch(test_inputs.clone(), test_labels.clone());
        let temp_sum = losses.iter().sum::<NUM>() / losses.len() as NUM;

        if i * 2 > ITERATIONS && temp_sum < last_avg_loss {
            println!("New best! Saving model with avg_loss={temp_sum} (previous {last_avg_loss})");
            let _ = save_model(&net, model_path);
        }

        last_avg_loss = temp_sum;
        iter_loss_points.push(last_avg_loss);

        let _ = terminal.draw(|frame| draw(frame, &iter_loss_points));

        for (inputs_batch, labels_batch) in inputs.chunks(BATCH_SIZE).zip(labels.chunks(BATCH_SIZE))
        {
            if should_quit() {
                emergency_exit = true;
                break 'training;
            }
            net.fit_batch(inputs_batch.to_vec(), labels_batch.to_vec());
        }
    }

    ratatui::restore();
    if !emergency_exit {
        save_model(&net, model_path).unwrap();
        println!("Model saved to {model_path:?} - avg_loss={last_avg_loss}");
    } else {
        println!("Not saving model as process what exited");
    }
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
