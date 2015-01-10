//! Algorithms for manipulating real matrices.

#![allow(unstable)]

#[cfg(test)]
#[macro_use]
extern crate assert;

#[cfg(test)]
extern crate test;

extern crate blas;
extern crate lapack;

pub mod decomp;

/// Multiply two matrices.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is stored in an `m`-by-`n` matrix `c`.
#[inline]
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, p: usize, n: usize) {
    if n == 1 {
        blas::dgemv(b'N', m, p, 1.0, a, m, b, 1, 0.0, c, 1);
    } else {
        blas::dgemm(b'N', b'N', m, n, p, 1.0, a, m, b, p, 0.0, c, m);
    }
}

/// Multiply two matrices and add another matrix.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is summed up with an `m`-by-`n` matrix `c` and stored in an
/// `m`-by-`n` matrix `d`.
#[inline]
pub fn multiply_add(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64], m: usize, p: usize, n: usize) {
    if c.as_ptr() != d.as_ptr() {
        unsafe {
            use std::ptr::copy_nonoverlapping_memory as copy;
            copy(d.as_mut_ptr(), c.as_ptr(), m * n);
        }
    }

    if n == 1 {
        blas::dgemv(b'N', m, p, 1.0, a, m, b, 1, 1.0, d, 1);
    } else {
        blas::dgemm(b'N', b'N', m, n, p, 1.0, a, m, b, p, 1.0, d, m);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn multiply() {
        let (m, p, n) = (2, 4, 1);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0, 0.0];

        ::multiply(&a[], &b[], &mut c[], m, p, n);

        let expected_c = vec![50.0, 60.0];
        assert_equal!(c, expected_c);
    }

    #[test]
    fn multiply_add() {
        let (m, p, n) = (2, 3, 4);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut d = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        ::multiply_add(&a[], &b[], &c[], &mut d[], m, p, n);

        let expected_c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_equal!(c, expected_c);

        let expected_d = vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0];
        assert_equal!(d, expected_d);
    }
}

#[cfg(test)]
mod benches {
    use std::iter::repeat;
    use test;

    #[bench]
    fn multiply_matrix_matrix(bench: &mut test::Bencher) {
        let m = 100;

        let a = repeat(1.0).take(m * m).collect::<Vec<f64>>();
        let b = repeat(1.0).take(m * m).collect::<Vec<f64>>();
        let mut c = repeat(1.0).take(m * m).collect::<Vec<f64>>();

        bench.iter(|| {
            ::multiply(&a[], &b[], &mut c[], m, m, m)
        });
    }

    #[bench]
    fn multiply_matrix_vector(bench: &mut test::Bencher) {
        let m = 100;

        let a = repeat(1.0).take(m * m).collect::<Vec<f64>>();
        let b = repeat(1.0).take(m * 1).collect::<Vec<f64>>();
        let mut c = repeat(1.0).take(m * 1).collect::<Vec<f64>>();

        bench.iter(|| {
            ::multiply(&a[], &b[], &mut c[], m, m, 1)
        });
    }
}
