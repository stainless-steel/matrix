#![feature(test)]

extern crate matrix;
extern crate test;

use std::iter::repeat;

#[bench]
fn multiply_matrix_matrix(bench: &mut test::Bencher) {
    let m = 100;

    let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
    let b = repeat(1.0).take(m * m).collect::<Vec<_>>();
    let mut c = repeat(1.0).take(m * m).collect::<Vec<_>>();

    bench.iter(|| {
        matrix::multiply(&a, &b, &mut c, m, m, m)
    });
}

#[bench]
fn multiply_matrix_vector(bench: &mut test::Bencher) {
    let m = 100;

    let a = repeat(1.0).take(m * m).collect::<Vec<_>>();
    let b = repeat(1.0).take(m * 1).collect::<Vec<_>>();
    let mut c = repeat(1.0).take(m * 1).collect::<Vec<_>>();

    bench.iter(|| {
        matrix::multiply(&a, &b, &mut c, m, m, 1)
    });
}
