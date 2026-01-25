use crate::matrix;

// MatMul
#[test]
fn mul_2x2_2x1() {
    let a = vec![vec![1f32, -2f32], vec![-1f32, 0f32]];
    let b = vec![vec![3f32], vec![5f32]];

    let expect = vec![vec![-7f32], vec![-3f32]];
    let res = matrix::matmul(&a, &b).unwrap();

    assert_eq!(res, expect);
}

#[test]
fn mul_2x3_3x1() {
    let a = vec![vec![1f32, -2f32, 9f32], vec![-1f32, 0f32, -4f32]];
    let b = vec![vec![3f32], vec![5f32], vec![2f32]];

    let expect = vec![vec![11f32], vec![-11f32]];
    let res = matrix::matmul(&a, &b).unwrap();

    assert_eq!(res, expect);
}

// Scalar
#[test]
fn scalar_2x2() {
    let a = 3f32;
    let b = vec![vec![1f32, -2f32], vec![-1f32, 0f32]];

    let expect = vec![vec![3f32, -6f32], vec![-3f32, 0f32]];
    let res = matrix::scalar(&a, &b);

    assert_eq!(res, expect);
}

// Add
#[test]
fn add_2x2() {
    let a = vec![vec![1f32, -2f32], vec![-1f32, 0f32]];
    let b = vec![vec![0f32, 1f32], vec![1f32, 2f32]];

    let expect = vec![vec![1f32, -1f32], vec![0f32, 2f32]];
    let res = matrix::add(&a, &b);

    assert_eq!(res, expect);
}

// subtract
#[test]
fn subtract_2x2() {
    let a = vec![vec![1f32, -2f32], vec![-1f32, 0f32]];
    let b = vec![vec![0f32, 1f32], vec![1f32, 2f32]];

    let expect = vec![vec![1f32, -3f32], vec![-2f32, -2f32]];
    let res = matrix::subtract(&a, &b);

    assert_eq!(res, expect);
}

// Transpose
#[test]
fn transpose_2x2() {
    let a = vec![vec![1f32, -2f32], vec![-1f32, 0f32]];

    let expect = vec![vec![1f32, -1f32], vec![-2f32, 0f32]];
    let res = matrix::transpose(&a);

    assert_eq!(res, expect);
}

#[test]
fn transpose_2x3() {
    let a = vec![vec![1f32, -2f32, 9f32], vec![-1f32, 0f32, -4f32]];

    let expect = vec![vec![1f32, -1f32], vec![-2f32, 0f32], vec![9f32, -4f32]];
    let res = matrix::transpose(&a);

    assert_eq!(res, expect);
}

#[test]
fn map_2x3() {
    let a = vec![vec![1f32, -2f32, 9f32], vec![-1f32, 0f32, -4f32]];

    let expect = vec![vec![1f32, -1f32, 1f32], vec![-1f32, 0f32, -1f32]];
    let res = matrix::map(&a, |a| {
        if a <= -1f32 {
            return -1f32;
        } else if a == 0f32 {
            return 0f32;
        } else {
            return 1f32;
        }
    });

    assert_eq!(res, expect);
}
