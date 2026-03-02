use crate::{
    NUM,
    activation::{Activation, Softmax},
    layer::{Dropout, Layer, LayerT},
    matrix::Matrix,
    optimizers::OptimizerFactory,
};

pub struct NetworkBuilder<O> {
    optimizer: O,
}

impl<O: Clone> NetworkBuilder<O> {
    pub fn input_layer<const IN: usize, const OUT: usize, A: Activation>(
        self,
    ) -> Network<IN, OUT, ((), Layer<IN, OUT, A, O>), O>
    where
        O: OptimizerFactory + Clone,
    {
        Network {
            layers: ((), Layer::new(&self.optimizer)),
            optimizer: self.optimizer,
        }
    }
}

//

pub struct Network<const IN: usize, const OUT: usize, L, O> {
    layers: L,
    optimizer: O,
}

impl<O> Network<0, 0, (), O> {
    pub fn new(optimizer: O) -> NetworkBuilder<O> {
        NetworkBuilder { optimizer }
    }
}

impl<const IN: usize, const OUT: usize, L, O: Clone> Network<IN, OUT, L, O> {
    pub fn add_layer<const NEW_OUT: usize, A: Activation>(
        self,
    ) -> Network<IN, NEW_OUT, (L, Layer<OUT, NEW_OUT, A, O>), O>
    where
        O: OptimizerFactory + Clone,
    {
        Network {
            layers: (self.layers, Layer::new(&self.optimizer)),
            optimizer: self.optimizer,
        }
    }

    pub fn add_dropout(self, rate: NUM) -> Network<IN, OUT, (L, Dropout<OUT>), O> {
        Network {
            layers: (self.layers, Dropout::new(rate)),
            optimizer: self.optimizer,
        }
    }
}

impl<const IN: usize, const OUT: usize, L, O> Network<IN, OUT, L, O>
where
    L: LayerT<IN, OUT>,
    O: Sync,
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
        let output = self.predict(input);
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

        self.layers.apply(batch_size);
    }
}

//

impl<const IN: usize, const OUT: usize, L: serde::Serialize, O: serde::Serialize> serde::Serialize
    for Network<IN, OUT, L, O>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Network", 2)?;
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("optimizer", &self.optimizer)?;
        state.end()
    }
}

impl<
    'de,
    const IN: usize,
    const OUT: usize,
    L: serde::de::DeserializeOwned,
    O: serde::de::DeserializeOwned,
> serde::Deserialize<'de> for Network<IN, OUT, L, O>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct NetworkData<L, O> {
            layers: L,
            optimizer: O,
        }

        let data = NetworkData::<L, O>::deserialize(deserializer)?;

        Ok(Self {
            layers: data.layers,
            optimizer: data.optimizer,
        })
    }
}
