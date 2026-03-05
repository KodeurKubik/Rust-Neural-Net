use axum::{
    Json, Router,
    extract::State,
    routing::{get_service, post},
};
use neuralnet::{
    NUM,
    activation::{Linear, ReLU},
    layer::Layer,
    load_model,
    matrix::Matrix,
    network::Network,
    optimizers::AdamBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::services::ServeDir;

const LEARNING_RATE: NUM = 0.0001;

type AppState = Arc<
    Network<
        784,
        10,
        (
            (
                ((), Layer<784, 256, ReLU, AdamBuilder>),
                Layer<256, 128, ReLU, AdamBuilder>,
            ),
            Layer<128, 10, Linear, AdamBuilder>,
        ),
        AdamBuilder,
    >,
>;

fn load_net() -> AppState {
    let mut net = Network::new(AdamBuilder::new(LEARNING_RATE))
        .input_layer::<784, 256, ReLU>()
        .add_layer::<128, ReLU>()
        .add_layer::<10, Linear>();

    let model_path = std::path::Path::new("./models/mnist.bin");
    load_model(&mut net, model_path).expect("Could not load model");

    Arc::new(net)
}

#[derive(Deserialize)]
struct PredictRequest {
    pixels: Vec<NUM>,
}

#[derive(Serialize)]
struct PredictResponse {
    digit: u8,
    confidence: NUM,
    probabilities: Vec<NUM>,
}

async fn mnist_predict(
    State(net): State<AppState>,
    Json(body): Json<PredictRequest>,
) -> Json<PredictResponse> {
    let mut input = [0 as NUM; 784];
    input.clone_from_slice(&body.pixels);

    let data: [[NUM; 1]; 784] = std::array::from_fn(|i| [input[i]]);
    let prediction = net.predict(Matrix::from(data));
    let predicted = prediction.flatten();

    let digit = predicted
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u8)
        .unwrap();

    let confidence = predicted[digit as usize];

    Json(PredictResponse {
        digit,
        confidence,
        probabilities: predicted,
    })
}

#[tokio::main]
async fn main() {
    #[cfg(all(feature = "relative", not(debug_assertions)))]
    {
        let exe_path = std::env::current_exe().expect("Failed to get current executable path");

        let exe_dir = exe_path
            .parent()
            .expect("Failed to get executable directory");

        std::env::set_current_dir(exe_dir).expect("Failed to change working directory");
    }

    let net = load_net();

    let app = Router::new()
        .route("/predict/mnist", post(mnist_predict))
        .with_state(net)
        .fallback_service(get_service(ServeDir::new("static")));

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
