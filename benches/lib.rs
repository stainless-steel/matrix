extern crate matrix;
extern crate test;

use self::test::Bencher;

#[bench]
fn multiply_matrix_matrix(bench: &mut Bencher) {
    let m = 100;

    let a = Vec::from_elem(m * m, 1.0);
    let b = Vec::from_elem(m * m, 1.0);
    let mut c = Vec::from_elem(m * m, 1.0);

    bench.iter(|| {
        matrix::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, m, m)
    })
}

#[bench]
fn multiply_matrix_vector(bench: &mut Bencher) {
    let m = 100;

    let a = Vec::from_elem(m * m, 1.0);
    let b = Vec::from_elem(m * 1, 1.0);
    let mut c = Vec::from_elem(m * 1, 1.0);

    bench.iter(|| {
        matrix::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, m, 1)
    })
}
