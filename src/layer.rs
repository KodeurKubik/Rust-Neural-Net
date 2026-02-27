use crate::{activation::Activation, matrix::Matrix};

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

    pub fn pass(&mut self, input: Matrix<INPUT, 1>) -> Matrix<OUTPUT, 1> {
        let z = self.weight.clone() * input.clone() + self.bias.clone();
        let a = z.map(A::apply);

        self.cache_input = Some(input);
        self.cache_pre = Some(z);

        a
    }
}
