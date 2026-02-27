use crate::{NUM, matrix::Matrix};

pub trait Softmax {
    fn softmax(&self) -> Self;
}

impl Softmax for Vec<NUM> {
    fn softmax(&self) -> Self {
        let sum: NUM = self.iter().map(|x| x.exp()).sum();
        self.iter().map(|x| x.exp() / sum).collect()
    }
}

impl<const N: usize> Softmax for Matrix<N, 1> {
    fn softmax(&self) -> Self {
        let sum: NUM = self.data.iter().map(|x| x[0].exp()).sum();
        let mut result = Matrix::zero();

        for i in 0..N {
            result.data[i][0] = self.data[i][0].exp() / sum;
        }

        result
    }
}

//

pub trait Activation {
    fn apply(x: NUM) -> NUM;
    fn derivative(x: NUM) -> NUM;
}

pub struct ReLU;
pub struct Sigmoid;
pub struct Tanh;
pub struct Linear;

impl Activation for ReLU {
    fn apply(x: NUM) -> NUM {
        x.max(0 as NUM)
    }
    fn derivative(x: NUM) -> NUM {
        if x > 0 as NUM { 1 as NUM } else { 0 as NUM }
    }
}

impl Activation for Sigmoid {
    fn apply(x: NUM) -> NUM {
        (1 as NUM) / ((1 as NUM) + (-x).exp())
    }
    fn derivative(x: NUM) -> NUM {
        x * (1 as NUM - x)
    }
}

impl Activation for Tanh {
    fn apply(x: NUM) -> NUM {
        x.tanh()
    }
    fn derivative(x: NUM) -> NUM {
        1 as NUM - (x * x)
    }
}

impl Activation for Linear {
    fn apply(x: NUM) -> NUM {
        x
    }
    fn derivative(_x: NUM) -> NUM {
        1 as NUM
    }
}
