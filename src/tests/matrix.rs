use crate::matrix::Matrix;

// MatMul
#[test]
fn mul_2x2_2x1() {
    let a = Matrix::from([[1f64, -2f64], [-1f64, 0f64]]);
    let b = Matrix::from([[3f64], [5f64]]);

    let expect = Matrix::from([[-7f64], [-3f64]]);
    let res = a * b;

    assert_eq!(res, expect);
}

#[test]
fn mul_2x3_3x1() {
    let a = Matrix::from([[1f64, -2f64, 9f64], [-1f64, 0f64, -4f64]]);
    let b = Matrix::from([[3f64], [5f64], [2f64]]);

    let expect = Matrix::from([[11f64], [-11f64]]);
    let res = a * b;

    assert_eq!(res, expect);
}

// Add
#[test]
fn add_2x2() {
    let a = Matrix::from([[1f64, -2f64], [-1f64, 0f64]]);
    let b = Matrix::from([[0f64, 1f64], [1f64, 2f64]]);

    let expect = Matrix::from([[1f64, -1f64], [0f64, 2f64]]);
    let res = a + b;

    assert_eq!(res, expect);
}

// subtract
#[test]
fn subtract_2x2() {
    let a = Matrix::from([[1f64, -2f64], [-1f64, 0f64]]);
    let b = Matrix::from([[0f64, 1f64], [1f64, 2f64]]);

    let expect = Matrix::from([[1f64, -3f64], [-2f64, -2f64]]);
    let res = a - b;

    assert_eq!(res, expect);
}

// Transpose
#[test]
fn transpose_2x2() {
    let a = Matrix::from([[1f64, -2f64], [-1f64, 0f64]]);

    let expect = Matrix::from([[1f64, -1f64], [-2f64, 0f64]]);
    let res = a.transpose();

    assert_eq!(res, expect);
}

#[test]
fn transpose_2x3() {
    let a = Matrix::from([[1f64, -2f64, 9f64], [-1f64, 0f64, -4f64]]);

    let expect = Matrix::from([[1f64, -1f64], [-2f64, 0f64], [9f64, -4f64]]);
    let res = a.transpose();

    assert_eq!(res, expect);
}

#[test]
fn map_2x3() {
    let a = Matrix::from([[1f64, -2f64, 9f64], [-1f64, 0f64, -4f64]]);

    let expect = Matrix::from([[1f64, -1f64, 1f64], [-1f64, 0f64, -1f64]]);
    let res = a.map(|a| {
        if a <= -1f64 {
            return -1f64;
        } else if a == 0f64 {
            return 0f64;
        } else {
            return 1f64;
        }
    });

    assert_eq!(res, expect);
}
