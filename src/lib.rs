//! Algorithms for manipulating real matrices.

#![allow(non_snake_case)]

extern crate blas;
extern crate lapack;

mod decomposition;

pub use decomposition::symmetric_eigen;

/// An error.
#[derive(Clone, Copy)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// A result.
pub type Result<T> = std::result::Result<T, Error>;

/// Multiply two matrices.
///
/// The formula is as follows:
///
/// ```math
/// C = A * B.
/// ```
///
/// The slices `A`, `B`, and `C` should have `m × p`, `p × n`, and `m × n` elements, respectively.
#[inline]
pub fn multiply(A: &[f64], B: &[f64], C: &mut [f64], m: usize, p: usize, n: usize) {
    use blas::{dgemv, dgemm, Trans};

    if n == 1 {
        dgemv(Trans::N, m, p, 1.0, A, m, B, 1, 0.0, C, 1);
    } else {
        dgemm(Trans::N, Trans::N, m, n, p, 1.0, A, m, B, p, 0.0, C, m);
    }
}

/// Multiply two matrices and add another matrix.
///
/// The formula is as follows:
///
/// ```math
/// D = A * B + C.
/// ```
///
/// The slices `A`, `B`, `C`, and `D` should have `m × p`, `p × n`, `m × n`, and `m × n` elements,
/// respectively.
#[inline]
pub fn multiply_add(A: &[f64], B: &[f64], C: &[f64], D: &mut [f64], m: usize, p: usize, n: usize) {
    use blas::{dgemv, dgemm, Trans};

    if C.as_ptr() != D.as_ptr() {
        unsafe {
            std::ptr::copy_nonoverlapping(C.as_ptr(), D.as_mut_ptr(), m * n);
        }
    }

    if n == 1 {
        dgemv(Trans::N, m, p, 1.0, A, m, B, 1, 1.0, D, 1);
    } else {
        dgemm(Trans::N, Trans::N, m, n, p, 1.0, A, m, B, p, 1.0, D, m);
    }
}
