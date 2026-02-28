use crate::{NUM, activation::Activation, matrix::Matrix};

pub struct Layer<const INPUT: usize, const OUTPUT: usize, A: Activation> {
    weight: Matrix<OUTPUT, INPUT>,
    bias: Matrix<OUTPUT, 1>,
    cache_input: Option<Matrix<INPUT, 1>>,
    cache_pre: Option<Matrix<OUTPUT, 1>>,
    grad_weight: Option<Matrix<OUTPUT, INPUT>>,
    grad_bias: Option<Matrix<OUTPUT, 1>>,
    activation: std::marker::PhantomData<A>,
}

impl<const INPUT: usize, const OUTPUT: usize, A: Activation> Layer<INPUT, OUTPUT, A> {
    pub fn new() -> Self {
        Self {
            weight: Matrix::random(),
            bias: Matrix::random(),
            cache_input: None,
            cache_pre: None,
            grad_weight: None,
            grad_bias: None,
            activation: std::marker::PhantomData,
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

    pub fn apply(&mut self, learning_rate: NUM, batch_size: usize) {
        if let Some(mut grad_weight) = self.grad_weight.take() {
            grad_weight *= learning_rate / batch_size as NUM;
            self.weight -= grad_weight;
        }

        if let Some(mut grad_bias) = self.grad_bias.take() {
            grad_bias *= learning_rate / batch_size as NUM;
            self.bias -= grad_bias;
        }
    }
}

pub trait LayerT<const IN: usize, const OUT: usize> {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn accumulate(&mut self, delta: Matrix<OUT, 1>) -> Matrix<IN, 1>;
    fn apply(&mut self, learning_rate: NUM, batch_size: usize);
}

impl<const IN: usize, const MID: usize, const OUT: usize, A: Activation, T> LayerT<IN, OUT>
    for (T, Layer<MID, OUT, A>)
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
    fn apply(&mut self, learning_rate: NUM, batch_size: usize) {
        Layer::apply(&mut self.1, learning_rate, batch_size);
        self.0.apply(learning_rate, batch_size);
    }
}

impl<const IN: usize, const OUT: usize, A: Activation> LayerT<IN, OUT> for Layer<IN, OUT, A> {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::predict(&self, input)
    }
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::forward(self, input)
    }
    fn accumulate(&mut self, delta: Matrix<OUT, 1>) -> Matrix<IN, 1> {
        Layer::accumulate(self, delta)
    }
    fn apply(&mut self, learning_rate: NUM, batch_size: usize) {
        Layer::apply(self, learning_rate, batch_size);
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
    fn apply(&mut self, _learning_rate: NUM, _batch_size: usize) {}
}

//

impl<const INPUT: usize, const OUTPUT: usize, A: Activation> serde::Serialize
    for Layer<INPUT, OUTPUT, A>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Layer", 2)?;
        state.serialize_field("weight", &self.weight)?;
        state.serialize_field("bias", &self.bias)?;
        state.end()
    }
}

impl<'de, const INPUT: usize, const OUTPUT: usize, A: Activation> serde::Deserialize<'de>
    for Layer<INPUT, OUTPUT, A>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct LayerData<const INPUT: usize, const OUTPUT: usize> {
            weight: Matrix<OUTPUT, INPUT>,
            bias: Matrix<OUTPUT, 1>,
        }

        let data = LayerData::<INPUT, OUTPUT>::deserialize(deserializer)?;

        Ok(Self {
            weight: data.weight,
            bias: data.bias,
            cache_input: None,
            cache_pre: None,
            grad_weight: None,
            grad_bias: None,
            activation: std::marker::PhantomData,
        })
    }
}
