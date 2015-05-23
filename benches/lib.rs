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
        matrix::multiply(&A, &B, &mut C, m, m, m)
    });
}

#[bench]
fn multiply_matrix_vector(bench: &mut test::Bencher) {
    let m = 100;

    let A = vec![1.0; m * m];
    let B = vec![1.0; m * 1];
    let mut C = vec![1.0; m * 1];

    bench.iter(|| {
        matrix::multiply(&A, &B, &mut C, m, m, 1)
    });
}
