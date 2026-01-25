use anyhow::{Result, anyhow};

/// Product of two matrices
/// matrix a has dimensions 'i x j' and b has dimensions 'j x k'
/// output matrix will have dimensions 'i x k'
pub fn matmul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    if a[0].len() != b.len() {
        return Err(anyhow!("Invalid matrices size for multiplication"));
    }

    let mut result: Vec<Vec<f32>> = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        let mut row: Vec<f32> = Vec::with_capacity(b[0].len());

        for j in 0..b[0].len() {
            let mut val = 0f32;

            for elem in 0..b.len() {
                val += a[i][elem] * b[elem][j];
            }

            row.push(val);
        }

        result.push(row);
    }

    Ok(result)
}

/// scalar the whole matrice by a f32
pub fn scalar(a: &f32, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = b.clone();

    for row in &mut result {
        for elem in row {
            *elem *= a;
        }
    }

    result
}

/// just adds two matrices a and b together
pub fn add(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = a.clone();

    let mut i = 0usize;
    for row in &mut result {
        let mut j = 0usize;

        for elem in row {
            *elem += b[i][j];

            j += 1;
        }

        i += 1;
    }

    result
}

/// just subs the matrix b from matrix a
pub fn subtract(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = a.clone();

    let mut i = 0usize;
    for row in &mut result {
        let mut j = 0usize;

        for elem in row {
            *elem -= b[i][j];

            j += 1;
        }

        i += 1;
    }

    result
}

/// transposes a 'i x j' matrix into 'j x i'
pub fn transpose(a: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(a[0].len());

    for i in 0..a[0].len() {
        let mut row = Vec::with_capacity(a.len());

        for j in 0..a.len() {
            row.push(a[j][i]);
        }

        result.push(row);
    }

    result
}

/// Applies a function to each element of the matrix
pub fn map(a: &Vec<Vec<f32>>, f: impl Fn(f32) -> f32) -> Vec<Vec<f32>> {
    let mut result = a.clone();

    for row in &mut result {
        for elem in row {
            *elem = f(*elem);
        }
    }

    result
}
