use crate::{
    NUM,
    activation::{Activation, Softmax},
    layer::{Layer, LayerT},
    matrix::Matrix,
};

pub struct NetworkBuilder {
    learning_rate: NUM,
}

impl NetworkBuilder {
    pub fn add_layer<const IN: usize, const OUT: usize, A: Activation>(
        self,
        layer: Layer<IN, OUT, A>,
    ) -> Network<IN, OUT, ((), Layer<IN, OUT, A>)> {
        Network {
            layers: ((), layer),
            learning_rate: self.learning_rate,
        }
    }
}

//

pub struct Network<const IN: usize, const OUT: usize, L> {
    layers: L,
    learning_rate: NUM,
}

impl Network<0, 0, ()> {
    pub fn new(learning_rate: NUM) -> NetworkBuilder {
        NetworkBuilder { learning_rate }
    }
}

impl<const IN: usize, const OUT: usize, L> Network<IN, OUT, L>
where
    L: LayerT<IN, OUT>,
{
    pub fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        self.layers.predict(input).softmax()
    }

    pub fn predict_batch(&self, inputs: Vec<Matrix<IN, 1>>) -> Vec<Matrix<OUT, 1>>
    where
        L: Send + Sync,
    {
        use rayon::prelude::*;
        inputs
            .into_par_iter()
            .map(|input| self.predict(input))
            .collect()
    }

    pub fn loss(&self, input: Matrix<IN, 1>, correct_index: usize) -> NUM {
        let output = self.predict(input).softmax();
        let loss = -output[correct_index][0].ln();
        loss
    }

    pub fn loss_batch(&self, inputs: Vec<Matrix<IN, 1>>, correct_indices: Vec<usize>) -> Vec<NUM>
    where
        L: Send + Sync,
    {
        self.predict_batch(inputs)
            .iter()
            .zip(correct_indices.iter())
            .map(|(output, &correct)| -(output[correct][0]).ln())
            .collect()
    }

    pub fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        self.layers.forward(input)
    }

    pub fn fit_batch(&mut self, inputs: Vec<Matrix<IN, 1>>, correct_indices: Vec<usize>) {
        let batch_size = correct_indices.len();
        
        for (input, correct_index) in inputs.into_iter().zip(correct_indices.into_iter()) {
            let output = self.forward(input).softmax();
            let mut correct_mat = Matrix::zero();
            correct_mat[correct_index][0] = 1 as NUM;
            let delta = output - correct_mat;

            self.layers.accumulate(delta);
        }

        self.layers.apply(self.learning_rate, batch_size);
    }
}

impl<const IN: usize, const OUT: usize, L> Network<IN, OUT, L> {
    pub fn add_layer<const NEW_OUT: usize, A: Activation>(
        self,
        layer: Layer<OUT, NEW_OUT, A>,
    ) -> Network<IN, NEW_OUT, (L, Layer<OUT, NEW_OUT, A>)> {
        Network {
            layers: (self.layers, layer),
            learning_rate: self.learning_rate,
        }
    }
}

//

impl<const IN: usize, const OUT: usize, L: serde::Serialize> serde::Serialize
    for Network<IN, OUT, L>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Network", 2)?;
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("learning_rate", &self.learning_rate)?;
        state.end()
    }
}

impl<'de, const IN: usize, const OUT: usize, L: serde::de::DeserializeOwned> serde::Deserialize<'de>
    for Network<IN, OUT, L>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct NetworkData<L> {
            layers: L,
            learning_rate: NUM,
        }

        let data = NetworkData::<L>::deserialize(deserializer)?;

        Ok(Self {
            layers: data.layers,
            learning_rate: data.learning_rate,
        })
    }
}
