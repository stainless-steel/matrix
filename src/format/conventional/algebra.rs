use blas;

use algebra::MultiplyInto;
use format::Conventional;

impl MultiplyInto<Conventional<f64>, Conventional<f64>> for Conventional<f64> {
    fn multiply_into(&self, right: &Self, result: &mut Self) {
        let (m, p, n) = (self.rows, self.columns, right.columns);
        multiply(1.0, &self.values, &right.values, 1.0, &mut result.values, m, p, n)
    }
}

impl MultiplyInto<[f64], [f64]> for Conventional<f64> {
    fn multiply_into(&self, right: &[f64], result: &mut [f64]) {
        let (m, p) = (self.rows, self.columns);
        let n = right.len() / p;
        multiply(1.0, &self.values, right, 1.0, result, m, p, n)
    }
}

#[inline(always)]
fn multiply(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, p: usize,
            n: usize) {

    debug_assert_eq!(a.len(), m * p);
    debug_assert_eq!(b.len(), p * n);
    debug_assert_eq!(c.len(), m * n);
    if n == 1 {
        blas::dgemv(blas::Trans::N, m, p, alpha, a, m, b, 1, beta, c, 1);
    } else {
        blas::dgemm(blas::Trans::N, blas::Trans::N, m, n, p, alpha, a, m, b, p, beta, c, m);
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn multiply_into_conventional() {
        let matrix = Conventional::from_vec((2, 3), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]);
        let right = Conventional::from_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let mut result = Conventional::from_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);

        matrix.multiply_into(&right, &mut result);

        assert_eq!(result, Conventional::from_vec((2, 4), vec![
            23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0,
        ]));
    }
}
