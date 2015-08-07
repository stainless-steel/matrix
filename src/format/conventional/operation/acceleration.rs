use blas;

use format::Conventional;
use operation::{Multiply, MultiplyInto, ScaleSelf};

impl Multiply<[f64], Conventional<f64>> for Conventional<f64> {
    #[inline]
    fn multiply(&self, right: &[f64]) -> Self {
        let (m, p) = (self.rows, self.columns);
        let n = right.len() / p;
        let mut result = unsafe { Conventional::with_uninitialized((m, n)) };
        multiply(1.0, &self.values, right, 0.0, &mut result.values, m, p, n);
        result
    }
}

impl MultiplyInto<Conventional<f64>, [f64]> for Conventional<f64> {
    #[inline(always)]
    fn multiply_into(&self, right: &Self, result: &mut [f64]) {
        MultiplyInto::multiply_into(self, &*right as &[f64], result)
    }
}

impl MultiplyInto<Vec<f64>, [f64]> for Conventional<f64> {
    #[inline(always)]
    fn multiply_into(&self, right: &Vec<f64>, result: &mut [f64]) {
        MultiplyInto::multiply_into(self, &*right as &[f64], result)
    }
}

impl MultiplyInto<[f64], [f64]> for Conventional<f64> {
    #[inline]
    fn multiply_into(&self, right: &[f64], result: &mut [f64]) {
        let (m, p) = (self.rows, self.columns);
        let n = right.len() / p;
        multiply(1.0, &self.values, right, 1.0, result, m, p, n)
    }
}

impl ScaleSelf<f64> for [f64] {
    #[inline]
    fn scale_self(&mut self, alpha: f64) {
        blas::dscal(self.len(), alpha, self, 1);
    }
}

fn multiply(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, p: usize,
            n: usize) {

    debug_assert_eq!(a.len(), m * p);
    debug_assert_eq!(b.len(), p * n);
    debug_assert_eq!(c.len(), m * n);
    if n == 1 {
        blas::dgemv(b'N', m, p, alpha, a, m, b, 1, beta, c, 1);
    } else {
        blas::dgemm(b'N', b'N', m, n, p, alpha, a, m, b, p, beta, c, m);
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn multiply() {
        let matrix = Conventional::from_vec((2, 3), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]);
        let right = Conventional::from_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);

        assert_eq!(matrix.multiply(&right), Conventional::from_vec((2, 4), vec![
            22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0,
        ]));
    }

    #[test]
    fn multiply_into() {
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

    #[test]
    fn scale_self() {
        let mut matrix = Conventional::from_vec(2, vec![21.0, 21.0, 21.0, 21.0]);
        matrix.scale_self(2.0);
        assert_eq!(matrix, Conventional::from_vec(2, vec![42.0, 42.0, 42.0, 42.0]));
    }
}
