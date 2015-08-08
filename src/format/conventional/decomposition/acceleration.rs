use lapack;

use Result;
use decomposition::{SingularValue, SymmetricEigen};
use format::{Conventional, Diagonal};

macro_rules! success(
    ($info:expr) => (
        if $info < 0 {
            raise!("encountered invalid arguments");
        } else if $info > 0 {
            raise!("failed to converge");
        }
    );
);

impl SingularValue<f64> for Conventional<f64> {
    fn decompose(&self) -> Result<(Conventional<f64>, Diagonal<f64>, Conventional<f64>)> {
        let (m, n) = (self.rows, self.columns);
        let mut left = unsafe { Conventional::with_uninitialized(m) };
        let mut values = unsafe { Diagonal::with_uninitialized((m, n)) };
        let mut right = unsafe { Conventional::with_uninitialized(n) };
        try!(singular_value(&self, &mut left, &mut values, &mut right, m, n));
        Ok((left, values, right))
    }
}

impl SymmetricEigen<f64> for Conventional<f64> {
    fn decompose(&self) -> Result<(Conventional<f64>, Diagonal<f64>)> {
        debug_assert_eq!(self.rows, self.columns);
        let mut vectors = self.clone();
        let mut values = unsafe { Diagonal::with_uninitialized(self.rows) };
        try!(symmetric_eigen(&mut vectors, &mut values, self.rows));
        Ok((vectors, values))
    }
}

fn singular_value(matrix: &[f64], left: &mut [f64], values: &mut [f64], right: &mut [f64],
                  m: usize, n: usize) -> Result<()> {

    debug_assert_eq!(matrix.len(), m * n);
    debug_assert_eq!(left.len(), m * m);
    debug_assert_eq!(values.len(), min!(m, n));
    debug_assert_eq!(right.len(), n * n);

    let mut matrix = matrix.to_vec();

    let mut info = 0;
    let mut iwork = unsafe { buffer!(8 * min!(m, n)) };

    let mut work = [0.0];
    lapack::dgesdd(b'A', m, n, &mut matrix, m, values, left, m, right, n, &mut work, -1,
                   &mut iwork, &mut info);
    success!(info);

    let lwork = work[0] as usize;
    let mut work = unsafe { buffer!(lwork) };
    lapack::dgesdd(b'A', m, n, &mut matrix, m, values, left, m, right, n, &mut work,
                   lwork as isize, &mut iwork, &mut info);
    success!(info);

    Ok(())
}

fn symmetric_eigen(matrix: &mut [f64], values: &mut [f64], m: usize) -> Result<()> {
    debug_assert_eq!(matrix.len(), m * m);
    debug_assert_eq!(values.len(), m);

    let mut info = 0;

    let mut work = [0.0];
    let mut iwork = [0];
    lapack::dsyevd(b'V', b'U', m, matrix, m, values, &mut work, -1, &mut iwork, -1, &mut info);
    success!(info);

    let lwork = work[0] as usize;
    let liwork = iwork[0] as usize;
    let mut work = unsafe { buffer!(lwork) };
    let mut iwork = unsafe { buffer!(liwork) };
    lapack::dsyevd(b'V', b'U', m, matrix, m, values, &mut work, lwork as isize, &mut iwork,
                   liwork as isize, &mut info);
    success!(info);

    Ok(())
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    #[test]
    fn singular_value() {
        let matrix = Conventional::from_vec((4, 2), matrix![
            1.0, 2.0;
            3.0, 4.0;
            5.0, 6.0;
            7.0, 8.0;
        ]);

        let (left, values, right) = SingularValue::decompose(&matrix).unwrap();

        assert::close(&*left, &*vec![
            -1.524832333102012e-01, -3.499183718079640e-01, -5.473535103057272e-01,
            -7.447886488034903e-01, -8.226474722256604e-01, -4.213752876845798e-01,
            -2.010310314350211e-02,  3.811690813975744e-01, -3.945010222838286e-01,
             2.427965457043579e-01,  6.979099754427756e-01, -5.462054988633035e-01,
            -3.799591338775954e-01,  8.006558795100630e-01, -4.614343573873367e-01,
             4.073761175486993e-02,
        ], 1e-14);

        assert::close(&*values, &*vec![1.426909549926149e+01, 6.268282324175424e-01], 1e-14);

        assert::close(&*right, &*vec![
            -6.414230279950722e-01, 7.671873950721771e-01, -7.671873950721771e-01,
            -6.414230279950722e-01,
        ], 1e-14);
    }

    #[test]
    fn symmetric_eigen() {
        let matrix = Conventional::from_vec(4, matrix![
            1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0;
            1.0 / 2.0, 1.0, 2.0 / 3.0, 1.0 / 2.0;
            1.0 / 3.0, 2.0 / 3.0, 1.0, 3.0 / 4.0;
            1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0;
        ]);

        let (vectors, values) = SymmetricEigen::decompose(&matrix).unwrap();

        assert::close(&*vectors, &*vec![
             6.931852607427760e-02, -3.617963298359111e-01,  7.693670370857654e-01,
            -5.218933989868291e-01, -4.422228501075730e-01,  7.420398064553687e-01,
             4.863601702209238e-02, -5.014483167053618e-01, -8.104763801066263e-01,
            -1.877143925990472e-01,  3.009681045547824e-01,  4.661647178209991e-01,
             3.778384973436192e-01,  5.322063962074435e-01,  5.613618263961305e-01,
             5.087900565323598e-01,
        ], 1e-14);
        assert::close(&*values, &*vec![
            2.077754859180120e-01, 4.078328841178751e-01, 8.482291554779129e-01,
            2.536162474486201e+00,
        ], 1e-14);
    }
}
