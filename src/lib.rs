//! Algorithms for manipulating real matrices.

extern crate blas;
extern crate lapack;

pub mod decomp;

/// Multiply two matrices.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is stored in an `m`-by-`n` matrix `c`.
#[inline]
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, p: usize, n: usize) {
    use blas::metal::{dgemv, dgemm, Trans};

    if n == 1 {
        dgemv(Trans::N, m, p, 1.0, a, m, b, 1, 0.0, c, 1);
    } else {
        dgemm(Trans::N, Trans::N, m, n, p, 1.0, a, m, b, p, 0.0, c, m);
    }
}

/// Multiply two matrices and add another matrix.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is summed up with an `m`-by-`n` matrix `c` and stored in an
/// `m`-by-`n` matrix `d`.
#[inline]
pub fn multiply_add(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64], m: usize, p: usize, n: usize) {
    use blas::metal::{dgemv, dgemm, Trans};

    if c.as_ptr() != d.as_ptr() {
        unsafe {
            use std::ptr::copy_nonoverlapping as copy;
            copy(c.as_ptr(), d.as_mut_ptr(), m * n);
        }
    }

    if n == 1 {
        dgemv(Trans::N, m, p, 1.0, a, m, b, 1, 1.0, d, 1);
    } else {
        dgemm(Trans::N, Trans::N, m, n, p, 1.0, a, m, b, p, 1.0, d, m);
    }
}
