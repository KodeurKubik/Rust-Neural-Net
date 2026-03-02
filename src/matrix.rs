use crate::NUM;
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Not, Sub, SubAssign},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    pub data: Box<[[NUM; COLS]; ROWS]>,
}

impl<const ROWS: usize, const COLS: usize> Display for Matrix<ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} Matrix: {:?}", ROWS, COLS, self.data)?;
        Ok(())
    }
}

impl<const ROWS: usize, const COLS: usize> From<[[NUM; COLS]; ROWS]> for Matrix<ROWS, COLS> {
    fn from(data: [[NUM; COLS]; ROWS]) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn zero() -> Self {
        Self {
            data: Box::new([[0 as NUM; COLS]; ROWS]),
        }
    }

    /// flattens a matrix to a Vec
    pub fn flatten(&self) -> Vec<NUM> {
        self.data.iter().flatten().copied().collect()
    }

    /// transposes a 'i x j' matrix into 'j x i'
    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut result = Matrix::<COLS, ROWS>::zero();

        for i in 0..COLS {
            for j in 0..ROWS {
                result.data[i][j] = self.data[j][i];
            }
        }

        result
    }

    /// Applies a function to each element of the matrix
    pub fn map(mut self, f: impl Fn(NUM) -> NUM) -> Self {
        for row in self.data.iter_mut() {
            for elem in row {
                *elem = f(*elem);
            }
        }

        self
    }
    /// Applies a function to each element of the matrix
    pub fn map_mut(mut self, mut f: impl FnMut(NUM) -> NUM) -> Self {
        for row in self.data.iter_mut() {
            for elem in row {
                *elem = f(*elem);
            }
        }

        self
    }

    pub fn elementmul(mut self, other: Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] *= other.data[i][j];
            }
        }

        self
    }
}

//

impl<const ROWS: usize, const COLS: usize> serde::Serialize for Matrix<ROWS, COLS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.data
            .iter()
            .flatten()
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
}

impl<'de, const ROWS: usize, const COLS: usize> serde::Deserialize<'de> for Matrix<ROWS, COLS> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let flat = Vec::<NUM>::deserialize(deserializer)?;

        if flat.len() != ROWS * COLS {
            return Err(serde::de::Error::custom(format!(
                "expected {}x{} = {} elements, got {}",
                ROWS,
                COLS,
                ROWS * COLS,
                flat.len()
            )));
        }

        let mut data = Box::new([[0 as NUM; COLS]; ROWS]);

        for i in 0..ROWS {
            for j in 0..COLS {
                data[i][j] = flat[i * COLS + j];
            }
        }

        Ok(Self { data })
    }
}

//

impl<const ROWS: usize, const COLS: usize> AddAssign for Matrix<ROWS, COLS> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] += rhs.data[i][j];
            }
        }
    }
}
impl<const ROWS: usize, const COLS: usize> AddAssign<&Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    fn add_assign(&mut self, rhs: &Matrix<ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] += rhs.data[i][j];
            }
        }
    }
}
impl<const ROWS: usize, const COLS: usize> Add for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const ROWS: usize, const COLS: usize> SubAssign for Matrix<ROWS, COLS> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] -= rhs.data[i][j];
            }
        }
    }
}
impl<const ROWS: usize, const COLS: usize> SubAssign<&Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    fn sub_assign(&mut self, rhs: &Matrix<ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] -= rhs.data[i][j];
            }
        }
    }
}
impl<const ROWS: usize, const COLS: usize> Sub for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<const ROWS: usize, const COLS: usize> MulAssign<NUM> for Matrix<ROWS, COLS> {
    fn mul_assign(&mut self, rhs: NUM) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] *= rhs;
            }
        }
    }
}
impl<const ROWS: usize, const COLS: usize> Mul<NUM> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn mul(mut self, rhs: NUM) -> Self::Output {
        self *= rhs;
        self
    }
}
impl<const ROWS: usize, const COLS: usize> Mul<Matrix<ROWS, COLS>> for NUM {
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        rhs * self
    }
}

impl<const A: usize, const B: usize, const C: usize> Mul<Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;
    fn mul(self, rhs: Matrix<B, C>) -> Self::Output {
        let mut result = Matrix::zero();
        for i in 0..A {
            for k in 0..B {
                let a_ik = self.data[i][k];
                for j in 0..C {
                    result.data[i][j] += a_ik * rhs.data[k][j];
                }
            }
        }
        result
    }
}
impl<const A: usize, const B: usize, const C: usize> Mul<&Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;
    fn mul(self, rhs: &Matrix<B, C>) -> Self::Output {
        let mut result = Matrix::zero();
        for i in 0..A {
            for k in 0..B {
                let a_ik = self.data[i][k];
                for j in 0..C {
                    result.data[i][j] += a_ik * rhs.data[k][j];
                }
            }
        }
        result
    }
}
impl<const A: usize, const B: usize, const C: usize> Mul<&Matrix<B, C>> for &Matrix<A, B> {
    type Output = Matrix<A, C>;
    fn mul(self, rhs: &Matrix<B, C>) -> Self::Output {
        let mut result = Matrix::zero();
        for i in 0..A {
            for k in 0..B {
                let a_ik = self.data[i][k];
                for j in 0..C {
                    result.data[i][j] += a_ik * rhs.data[k][j];
                }
            }
        }
        result
    }
}

/// Using ! (not) as a shortcut to transpose
impl<const ROWS: usize, const COLS: usize> Not for Matrix<ROWS, COLS> {
    type Output = Matrix<COLS, ROWS>;
    fn not(self) -> Self::Output {
        self.transpose()
    }
}
impl<const ROWS: usize, const COLS: usize> Not for &Matrix<ROWS, COLS> {
    type Output = Matrix<COLS, ROWS>;
    fn not(self) -> Self::Output {
        self.transpose()
    }
}

impl<const ROWS: usize, const COLS: usize> Index<usize> for Matrix<ROWS, COLS> {
    type Output = [NUM; COLS];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<usize> for Matrix<ROWS, COLS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
