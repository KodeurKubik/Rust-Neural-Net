use crate::{NUM, matrix::Matrix};

pub trait OptimizerFactory {
    type Output<const ROWS: usize, const COLS: usize>: Optimizer<ROWS, COLS>;
    fn generate<const ROWS: usize, const COLS: usize>(&self) -> Self::Output<ROWS, COLS>;
}

//

pub trait Optimizer<const ROWS: usize, const COLS: usize> {
    fn step(
        &mut self,
        weights: &mut Matrix<ROWS, COLS>,
        gradient: Matrix<ROWS, COLS>,
        batch_size: usize,
    );
}

/// Stochastic Gradient Descent (what a name)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct SGD {
    pub learning_rate: NUM,
}

impl SGD {
    pub fn new(learning_rate: NUM) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerFactory for SGD {
    type Output<const ROWS: usize, const COLS: usize> = SGD;
    fn generate<const ROWS: usize, const COLS: usize>(&self) -> Self::Output<ROWS, COLS> {
        self.clone()
    }
}

impl<const ROWS: usize, const COLS: usize> Optimizer<ROWS, COLS> for SGD {
    fn step(
        &mut self,
        weights: &mut Matrix<ROWS, COLS>,
        mut gradient: Matrix<ROWS, COLS>,
        batch_size: usize,
    ) {
        gradient *= self.learning_rate / batch_size as NUM;

        *weights -= gradient;
    }
}

/// Adaptive Moment Estimation (Adam)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct AdamBuilder {
    pub learning_rate: NUM,
    pub beta1: NUM,
    pub beta2: NUM,
    pub epsilon: NUM,
}

impl AdamBuilder {
    pub fn new(learning_rate: NUM) -> Self {
        Self::with_config(learning_rate, 0.9 as NUM, 0.999 as NUM, 1e-8 as NUM)
    }

    pub fn with_config(learning_rate: NUM, beta1: NUM, beta2: NUM, epsilon: NUM) -> Self {
        AdamBuilder {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }
}

impl OptimizerFactory for AdamBuilder {
    type Output<const ROWS: usize, const COLS: usize> = Adam<ROWS, COLS>;
    fn generate<const ROWS: usize, const COLS: usize>(&self) -> Self::Output<ROWS, COLS> {
        Adam {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            m: Matrix::zero(),
            v: Matrix::zero(),
            t: 0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Adam<const ROWS: usize, const COLS: usize> {
    pub learning_rate: NUM,
    pub beta1: NUM,
    pub beta2: NUM,
    pub epsilon: NUM,
    pub m: Matrix<ROWS, COLS>,
    pub v: Matrix<ROWS, COLS>,
    pub t: usize,
}

impl<const ROWS: usize, const COLS: usize> Optimizer<ROWS, COLS> for Adam<ROWS, COLS> {
    fn step(
        &mut self,
        weights: &mut Matrix<ROWS, COLS>,
        mut gradient: Matrix<ROWS, COLS>,
        batch_size: usize,
    ) {
        gradient *= 1 as NUM / batch_size as NUM;
        self.t += 1;

        let g_clone = gradient.clone();
        let g_squared = gradient.map(|x| x * x);

        self.m *= self.beta1;
        self.m += (1 as NUM - self.beta1) * g_clone;

        self.v *= self.beta2;
        self.v += (1 as NUM - self.beta2) * g_squared;

        let mut m_hat = self.m.clone();
        m_hat *= 1 as NUM / (1 as NUM - self.beta1.powi(self.t as i32));

        let mut v_hat = self.v.clone();
        v_hat *= 1 as NUM / (1 as NUM - self.beta2.powi(self.t as i32));

        let update = m_hat.elementmul(v_hat.map(|x| 1 as NUM / (x.sqrt() + self.epsilon)));
        *weights -= update * self.learning_rate;
    }
}
