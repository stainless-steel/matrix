#![allow(non_snake_case)]
#![feature(test)]

extern crate matrix;
extern crate test;

#[bench]
fn multiply_matrix_matrix(bench: &mut test::Bencher) {
    let m = 100;

    let A = vec![1.0; m * m];
    let B = vec![1.0; m * m];
    let mut C = vec![1.0; m * m];

    bench.iter(|| {
        matrix::multiply(1.0, &A, &B, 1.0, &mut C, m, m, m)
    });
}

#[bench]
fn multiply_matrix_vector(bench: &mut test::Bencher) {
    let m = 100;

    let A = vec![1.0; m * m];
    let B = vec![1.0; m];
    let mut C = vec![1.0; m];

    bench.iter(|| {
        matrix::multiply(1.0, &A, &B, 1.0, &mut C, m, m, 1)
    });
}
