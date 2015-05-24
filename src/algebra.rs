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

    debug_assert_eq!(A.len(), m * p);
    debug_assert_eq!(B.len(), p * n);
    debug_assert_eq!(C.len(), m * n);

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
    use std::ptr;

    debug_assert_eq!(A.len(), m * p);
    debug_assert_eq!(B.len(), p * n);
    debug_assert_eq!(C.len(), m * n);
    debug_assert_eq!(D.len(), m * n);

    if C.as_ptr() != D.as_ptr() {
        unsafe {
            ptr::copy_nonoverlapping(C.as_ptr(), D.as_mut_ptr(), m * n);
        }
    }

    if n == 1 {
        dgemv(Trans::N, m, p, 1.0, A, m, B, 1, 1.0, D, 1);
    } else {
        dgemm(Trans::N, Trans::N, m, n, p, 1.0, A, m, B, p, 1.0, D, m);
    }
}

#[cfg(test)]
mod tests {
    use assert;

    #[test]
    fn multiply() {
        let (m, p, n) = (2, 4, 1);

        let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let B = vec![1.0, 2.0, 3.0, 4.0];
        let mut C = vec![0.0, 0.0];

        super::multiply(&A, &B, &mut C, m, p, n);

        assert::equal(&C, &vec![50.0, 60.0]);
    }

    #[test]
    fn multiply_add() {
        let (m, p, n) = (2, 3, 4);

        let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let B = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let C = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut D = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        super::multiply_add(&A, &B, &C, &mut D, m, p, n);

        assert::equal(&C, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert::equal(&D, &vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0]);
    }
}
