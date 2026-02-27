use neuralnet::{
    NUM,
    activation::{ReLU, Sigmoid, Softmax},
    layer::Layer,
    matrix::Matrix,
    network::Network,
};

fn main() {
    let mut net = Network::new(0.01)
        .add_layer(Layer::<784, 128, ReLU>::new())
        .add_layer(Layer::<128, 10, Sigmoid>::new());

    let input = Matrix::random();
    let correct_index = 2usize;

    // run it
    let output = net.run(input).softmax();
    let loss = -output[correct_index][0].ln();
    println!("Got {}, loss={}", output, loss);

    let mut correct_mat = Matrix::zero();
    correct_mat[correct_index][0] = 1 as NUM;
    let delta_output = output - correct_mat;
}
