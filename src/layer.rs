use crate::{
    NUM,
    activation::Activation,
    matrix::Matrix,
    optimizers::{Optimizer, OptimizerFactory},
};
use rand::RngExt;

pub struct Layer<const INPUT: usize, const OUTPUT: usize, A: Activation, O: OptimizerFactory> {
    weight: Matrix<OUTPUT, INPUT>,
    bias: Matrix<OUTPUT, 1>,

    cache_input: Option<Matrix<INPUT, 1>>,
    cache_pre: Option<Matrix<OUTPUT, 1>>,
    grad_weight: Option<Matrix<OUTPUT, INPUT>>,
    grad_bias: Option<Matrix<OUTPUT, 1>>,

    activation: std::marker::PhantomData<A>,
    weight_optim: O::Output<OUTPUT, INPUT>,
    bias_optim: O::Output<OUTPUT, 1>,
}

impl<const INPUT: usize, const OUTPUT: usize, A: Activation, O: OptimizerFactory>
    Layer<INPUT, OUTPUT, A, O>
{
    pub fn new(optimizer_factory: &O) -> Self {
        Self {
            weight: {
                let dist = rand_distr::Normal::new(0.0, (2.0 / INPUT as NUM).sqrt()).unwrap();
                let mut rng = rand::rng();
                Matrix::zero().map_mut(|_| rng.sample(dist))
            },
            bias: Matrix::zero(),

            cache_input: None,
            cache_pre: None,
            grad_weight: None,
            grad_bias: None,

            activation: std::marker::PhantomData,
            weight_optim: optimizer_factory.generate(),
            bias_optim: optimizer_factory.generate(),
        }
    }

    pub fn forward(&mut self, input: Matrix<INPUT, 1>) -> Matrix<OUTPUT, 1> {
        let mut z = &self.weight * &input;
        z += &self.bias;

        self.cache_input = Some(input);
        self.cache_pre = Some(z.clone());

        z.map(A::apply)
    }

    pub fn predict(&self, input: Matrix<INPUT, 1>) -> Matrix<OUTPUT, 1> {
        let mut z = &self.weight * &input;
        z += &self.bias;
        let a = z.map(A::apply);

        a
    }

    pub fn accumulate(&mut self, delta: Matrix<OUTPUT, 1>) -> Matrix<INPUT, 1> {
        let cache_input = self.cache_input.take().unwrap();
        let cache_pre = self.cache_pre.take().unwrap();

        #[allow(non_snake_case)]
        let dL_dW = delta.clone() * !cache_input;
        let delta_prev = (!&self.weight) * (cache_pre.map(A::derivative).elementmul(delta.clone()));

        if let Some(grad_weight) = &mut self.grad_weight {
            *grad_weight += dL_dW;
        } else {
            self.grad_weight = Some(dL_dW);
        }

        if let Some(grad_bias) = &mut self.grad_bias {
            *grad_bias += delta;
        } else {
            self.grad_bias = Some(delta);
        }

        delta_prev
    }

    pub fn apply(&mut self, batch_size: usize) {
        if let Some(grad_weight) = self.grad_weight.take() {
            self.weight_optim
                .step(&mut self.weight, grad_weight, batch_size);
        }

        if let Some(grad_bias) = self.grad_bias.take() {
            self.bias_optim.step(&mut self.bias, grad_bias, batch_size);
        }
    }
}

pub trait LayerT<const IN: usize, const OUT: usize> {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn accumulate(&mut self, delta: Matrix<OUT, 1>) -> Matrix<IN, 1>;
    fn apply(&mut self, batch_size: usize);
}

impl<const IN: usize, const MID: usize, const OUT: usize, A: Activation, T, O: OptimizerFactory>
    LayerT<IN, OUT> for (T, Layer<MID, OUT, A, O>)
where
    T: LayerT<IN, MID>,
{
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        let mid = self.0.predict(input);
        Layer::predict(&self.1, mid)
    }
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        let mid = self.0.forward(input);
        Layer::forward(&mut self.1, mid)
    }
    fn accumulate(&mut self, delta: Matrix<OUT, 1>) -> Matrix<IN, 1> {
        let mid = Layer::accumulate(&mut self.1, delta);
        self.0.accumulate(mid)
    }
    fn apply(&mut self, batch_size: usize) {
        Layer::apply(&mut self.1, batch_size);
        self.0.apply(batch_size);
    }
}

impl<const IN: usize, const OUT: usize, A: Activation, O: OptimizerFactory> LayerT<IN, OUT>
    for Layer<IN, OUT, A, O>
{
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::predict(&self, input)
    }
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::forward(self, input)
    }
    fn accumulate(&mut self, delta: Matrix<OUT, 1>) -> Matrix<IN, 1> {
        Layer::accumulate(self, delta)
    }
    fn apply(&mut self, batch_size: usize) {
        Layer::apply(self, batch_size);
    }
}

impl<const N: usize> LayerT<N, N> for () {
    fn predict(&self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }
    fn forward(&mut self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }
    fn accumulate(&mut self, _delta: Matrix<N, 1>) -> Matrix<N, 1> {
        Matrix::zero()
    }
    fn apply(&mut self, _batch_size: usize) {}
}

//

impl<const INPUT: usize, const OUTPUT: usize, A: Activation, O: OptimizerFactory> serde::Serialize
    for Layer<INPUT, OUTPUT, A, O>
where
    O::Output<OUTPUT, INPUT>: serde::Serialize,
    O::Output<OUTPUT, 1>: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Layer", 4)?;
        state.serialize_field("weight", &self.weight)?;
        state.serialize_field("bias", &self.bias)?;
        state.serialize_field("weight_optim", &self.weight_optim)?;
        state.serialize_field("bias_optim", &self.bias_optim)?;
        state.end()
    }
}

impl<'de, const INPUT: usize, const OUTPUT: usize, A: Activation, O: OptimizerFactory>
    serde::Deserialize<'de> for Layer<INPUT, OUTPUT, A, O>
where
    O::Output<OUTPUT, INPUT>: serde::de::DeserializeOwned,
    O::Output<OUTPUT, 1>: serde::de::DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct LayerData<const INPUT: usize, const OUTPUT: usize, W, B> {
            weight: Matrix<OUTPUT, INPUT>,
            bias: Matrix<OUTPUT, 1>,
            weight_optim: W,
            bias_optim: B,
        }

        let data = LayerData::<INPUT, OUTPUT, O::Output<OUTPUT, INPUT>, O::Output<OUTPUT, 1>>::deserialize(deserializer)?;

        Ok(Self {
            weight: data.weight,
            bias: data.bias,
            cache_input: None,
            cache_pre: None,
            grad_weight: None,
            grad_bias: None,
            activation: std::marker::PhantomData,
            bias_optim: data.bias_optim,
            weight_optim: data.weight_optim,
        })
    }
}

//

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Dropout<const N: usize> {
    rate: NUM,
    mask: Option<Matrix<N, 1>>,
}

impl<const N: usize> Dropout<N> {
    pub fn new(rate: NUM) -> Self {
        Self { rate, mask: None }
    }
}

impl<const N: usize> LayerT<N, N> for Dropout<N> {
    fn forward(&mut self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        let dist = rand_distr::Bernoulli::new(self.rate as f64).unwrap();
        let mut rng = rand::rng();

        let mask = input
            .clone()
            .map_mut(|_| if rng.sample(dist) { 0.0 } else { 1.0 });

        let scale = 1.0 / (1.0 - self.rate);
        let output = input.elementmul(mask.clone()) * scale;
        self.mask = Some(mask);
        output
    }

    fn predict(&self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }

    fn accumulate(&mut self, delta: Matrix<N, 1>) -> Matrix<N, 1> {
        let mask = self.mask.take().unwrap();
        delta.elementmul(mask)
    }

    fn apply(&mut self, _batch_size: usize) {}
}

impl<const IN: usize, const N: usize, T: LayerT<IN, N>> LayerT<IN, N> for (T, Dropout<N>) {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<N, 1> {
        let mid = self.0.predict(input);
        self.1.predict(mid)
    }
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<N, 1> {
        let mid = self.0.forward(input);
        self.1.forward(mid)
    }
    fn accumulate(&mut self, delta: Matrix<N, 1>) -> Matrix<IN, 1> {
        let mid = self.1.accumulate(delta);
        self.0.accumulate(mid)
    }
    fn apply(&mut self, batch_size: usize) {
        self.0.apply(batch_size);
    }
}
