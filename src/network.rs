use crate::{NUM, activation::Activation, layer::Layer, matrix::Matrix};

pub struct NetworkBuilder {
    learning_rate: NUM,
}

impl NetworkBuilder {
    pub fn add_layer<const IN: usize, const OUT: usize, A: Activation>(
        self,
        layer: Layer<IN, OUT, A>,
    ) -> Network<OUT, ((), Layer<IN, OUT, A>)> {
        Network {
            layers: ((), layer),
            learning_rate: self.learning_rate,
        }
    }
}

//

pub struct Network<const OUT: usize, L> {
    layers: L,
    learning_rate: NUM,
}

impl Network<0, ()> {
    pub fn new(learning_rate: NUM) -> NetworkBuilder {
        NetworkBuilder { learning_rate }
    }
}

impl<const OUT: usize, L> Network<OUT, L> {
    pub fn add_layer<const NEW_OUT: usize, A: Activation>(
        self,
        layer: Layer<OUT, NEW_OUT, A>,
    ) -> Network<NEW_OUT, (L, Layer<OUT, NEW_OUT, A>)> {
        Network {
            layers: (self.layers, layer),
            learning_rate: self.learning_rate,
        }
    }

    pub fn run<const IN: usize>(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>
    where
        L: Forward<IN, OUT>,
    {
        self.layers.forward(input)
    }
}

//

pub trait Forward<const IN: usize, const OUT: usize> {
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
}

impl<const IN: usize, const OUT: usize, A: Activation> Forward<IN, OUT> for Layer<IN, OUT, A> {
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        self.pass(input)
    }
}

impl<const N: usize> Forward<N, N> for () {
    fn forward(&mut self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }
}

impl<const IN: usize, const MID: usize, const OUT: usize, A: Activation, T> Forward<IN, OUT>
    for (T, Layer<MID, OUT, A>)
where
    T: Forward<IN, MID>,
{
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        let mid = self.0.forward(input);
        self.1.pass(mid)
    }
}
