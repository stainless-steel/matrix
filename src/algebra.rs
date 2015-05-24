use blas;

/// Multiply a matrix by a matrix.
///
/// The formula is as follows:
///
/// ```math
/// C = α * A * B + β * C.
/// ```
///
/// The slices `A`, `B`, and `C` should have `m × p`, `p × n`, and `m × n` elements, respectively.
#[inline]
pub fn multiply(alpha: f64, A: &[f64], B: &[f64], beta: f64, C: &mut [f64], m: usize, p: usize,
                n: usize) {

    debug_assert_eq!(A.len(), m * p);
    debug_assert_eq!(B.len(), p * n);
    debug_assert_eq!(C.len(), m * n);

    if n == 1 {
        blas::dgemv(blas::Trans::N, m, p, alpha, A, m, B, 1, beta, C, 1);
    } else {
        blas::dgemm(blas::Trans::N, blas::Trans::N, m, n, p, alpha, A, m, B, p, beta, C, m);
    }
}

/// Multiply a vector by a scalar.
///
/// The slice `X` should have `m` elements.
#[inline]
pub fn scale(alpha: f64, X: &mut [f64], m: usize) {
    debug_assert_eq!(X.len(), m);
    blas::dscal(m, alpha, X, 1);
}

#[cfg(test)]
mod tests {
    use assert;

    #[test]
    fn multiply() {
        let (m, p, n) = (2, 3, 4);
        let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let B = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut C = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        super::multiply(1.0, &A, &B, 1.0, &mut C, m, p, n);

        assert::equal(&C, &vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0]);
    }

    #[test]
    fn scale() {
        let m = 4;
        let mut X = vec![1.0, 2.0, 3.0, 4.0];

        super::scale(2.0, &mut X, m);

        assert::equal(&X, &vec![2.0, 4.0, 6.0, 8.0]);
    }
}
