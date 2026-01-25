use criterion::{Criterion, criterion_group, criterion_main};
use neuralnet::matrix;

fn matmul() {
    let a = vec![vec![1f32, -2f32, 9f32], vec![-1f32, 0f32, -4f32]];
    let b = vec![vec![3f32], vec![5f32], vec![2f32]];
    matrix::matmul(&a, &b).unwrap();
}

fn add() {
    let a = vec![vec![1f32; 1000]; 1000];
    let b = vec![vec![2f32; 1000]; 1000];
    matrix::add(&a, &b);
}

fn subtract() {
    let a = vec![vec![1f32; 1000]; 1000];
    let b = vec![vec![2f32; 1000]; 1000];
    matrix::subtract(&a, &b);
}

fn scalar() {
    let a = -16f32;
    let b = vec![vec![2f32; 1000]; 1000];
    matrix::scalar(&a, &b);
}

fn transpose() {
    let a = vec![vec![1f32, -2f32, 9f32], vec![-1f32, 0f32, -4f32]];
    matrix::transpose(&a);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("matrix mul", |b| b.iter(|| matmul()));
    c.bench_function("matrix add", |b| b.iter(|| add()));
    c.bench_function("matrix subtract", |b| b.iter(|| subtract()));
    c.bench_function("matrix scalar", |b| b.iter(|| scalar()));
    c.bench_function("matrix transpose", |b| b.iter(|| transpose()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
