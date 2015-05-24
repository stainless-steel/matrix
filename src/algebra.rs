use blas;

/// Compute the scalar product of two vectors.
///
/// `X` and `Y` should contain `n` elements.
#[inline]
pub fn dot(X: &[f64], Y: &[f64], n: usize) -> f64 {
    debug_assert_eq!(X.len(), n);
    debug_assert_eq!(Y.len(), n);
    blas::ddot(n, X, 1, Y, 1)
}

/// Scale a vector by a scalar.
///
/// The formula is as follows:
///
/// ```math
/// X = α * X.
/// ```
///
/// `X` should have `n` elements.
#[inline]
pub fn scale(alpha: f64, X: &mut [f64], n: usize) {
    debug_assert_eq!(X.len(), n);
    blas::dscal(n, alpha, X, 1);
}

/// Compute the sum of two vectors.
///
/// The formula is as follows:
///
/// ```math
/// Y = α * X + Y.
/// ```
///
/// `X` and `Y` should have `n` elements.
#[inline]
pub fn sum(alpha: f64, X: &[f64], Y: &mut [f64], n: usize) {
    debug_assert_eq!(X.len(), n);
    debug_assert_eq!(Y.len(), n);
    blas::daxpy(n, alpha, X, 1, Y, 1)
}

/// Compute the matrix product of two matrices.
///
/// The formula is as follows:
///
/// ```math
/// C = α * A * B + β * C.
/// ```
///
/// `A`, `B`, and `C` should have `m × p`, `p × n`, and `m × n` elements, respectively.
#[inline]
pub fn times(alpha: f64, A: &[f64], B: &[f64], beta: f64, C: &mut [f64], m: usize, p: usize,
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

#[cfg(test)]
mod tests {
    use assert;

    #[test]
    fn dot() {
        assert::equal(super::dot(&[10.0, -4.0], &[5.0, 2.0], 2), 42.0);
    }

    #[test]
    fn times() {
        let (m, p, n) = (2, 3, 4);
        let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let B = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut C = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        super::times(1.0, &A, &B, 1.0, &mut C, m, p, n);

        assert::equal(&C, &vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0]);
    }

    #[test]
    fn scale() {
        let n = 4;
        let mut X = vec![1.0, 2.0, 3.0, 4.0];

        super::scale(2.0, &mut X, n);

        assert::equal(&X, &vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn sum() {
        let n = 4;
        let X = vec![1.0, 2.0, 3.0, 4.0];
        let mut Y = vec![-1.0, 2.0, -3.0, 4.0];

        super::sum(1.0, &X, &mut Y, n);

        assert::equal(&Y, &vec![0.0, 4.0, 0.0, 8.0]);
    }
}
