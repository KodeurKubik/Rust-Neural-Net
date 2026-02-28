use crate::{NUM, activation::Activation, matrix::Matrix};

pub struct Layer<const INPUT: usize, const OUTPUT: usize, A: Activation> {
    weight: Matrix<OUTPUT, INPUT>,
    bias: Matrix<OUTPUT, 1>,
    cache_input: Option<Matrix<INPUT, 1>>,
    cache_pre: Option<Matrix<OUTPUT, 1>>,
    activation: std::marker::PhantomData<A>,
}

impl<const INPUT: usize, const OUTPUT: usize, A: Activation> Layer<INPUT, OUTPUT, A> {
    pub fn new() -> Self {
        Self {
            weight: Matrix::random(),
            bias: Matrix::random(),
            cache_input: None,
            cache_pre: None,
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

    pub fn backprop(&mut self, learning_rate: NUM, delta: Matrix<OUTPUT, 1>) -> Matrix<INPUT, 1> {
        let cache_input = self.cache_input.take().unwrap();
        let cache_pre = self.cache_pre.take().unwrap();

        #[allow(non_snake_case)]
        let mut dL_dW = delta.clone() * !cache_input;
        let delta_prev = (!&self.weight) * (cache_pre.map(A::derivative).elementmul(delta.clone()));

        dL_dW *= learning_rate;
        self.weight -= &dL_dW;
        self.bias -= &(delta * learning_rate);

        delta_prev
    }
}

pub trait LayerT<const IN: usize, const OUT: usize> {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1>;
    fn backprop(&mut self, learning_rate: NUM, delta: Matrix<OUT, 1>) -> Matrix<IN, 1>;
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
    fn backprop(&mut self, learning_rate: NUM, delta: Matrix<OUT, 1>) -> Matrix<IN, 1> {
        let mid = Layer::backprop(&mut self.1, learning_rate, delta);
        self.0.backprop(learning_rate, mid)
    }
}

impl<const IN: usize, const OUT: usize, A: Activation> LayerT<IN, OUT> for Layer<IN, OUT, A> {
    fn predict(&self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::predict(&self, input)
    }
    fn forward(&mut self, input: Matrix<IN, 1>) -> Matrix<OUT, 1> {
        Layer::forward(self, input)
    }
    fn backprop(&mut self, learning_rate: NUM, delta: Matrix<OUT, 1>) -> Matrix<IN, 1> {
        Layer::backprop(self, learning_rate, delta)
    }
}

impl<const N: usize> LayerT<N, N> for () {
    fn predict(&self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }
    fn forward(&mut self, input: Matrix<N, 1>) -> Matrix<N, 1> {
        input
    }
    fn backprop(&mut self, _learning_rate: NUM, _delta: Matrix<N, 1>) -> Matrix<N, 1> {
        Matrix::zero()
    }
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
            activation: std::marker::PhantomData,
        })
    }
}
