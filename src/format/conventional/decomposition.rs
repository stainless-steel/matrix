use lapack;

use Result;
use decomposition::SymmetricEigen;
use format::Conventional;

impl<'l> SymmetricEigen for (&'l mut Conventional<f64>, &'l mut Conventional<f64>) {
    fn decompose(pair: &mut Self) -> Result<()> {
        let m = pair.0.rows;
        try!(decompose(pair.0, pair.1, m));
        Ok(())
    }
}

impl<'l> SymmetricEigen for (&'l mut Conventional<f64>, &'l mut [f64]) {
    fn decompose(pair: &mut Self) -> Result<()> {
        let m = pair.0.rows;
        try!(decompose(pair.0, pair.1, m));
        Ok(())
    }
}

impl<'l> SymmetricEigen for (&'l mut [f64], &'l mut Conventional<f64>) {
    fn decompose(pair: &mut Self) -> Result<()> {
        let m = pair.1.len();
        try!(decompose(pair.0, pair.1, m));
        Ok(())
    }
}

impl<'l> SymmetricEigen for (&'l mut [f64], &'l mut [f64]) {
    fn decompose(pair: &mut Self) -> Result<()> {
        let m = pair.1.len();
        try!(decompose(pair.0, pair.1, m));
        Ok(())
    }
}

fn decompose(matrix: &mut [f64], vector: &mut [f64], m: usize) -> Result<()> {
    use lapack::{Jobz, Uplo};

    debug_assert_eq!(matrix.len(), m * m);
    debug_assert_eq!(vector.len(), m);

    macro_rules! success(
        ($flag:expr) => (
            if $flag != 0 {
                raise!("encountered invalid arguments");
            } else if $flag > 0 {
                raise!("failed to converge");
            }
        );
    );

    let mut flag = 0;

    let mut work = [0.0];
    lapack::dsyev(Jobz::V, Uplo::U, m, matrix, m, vector, &mut work, -1isize as usize, &mut flag);
    success!(flag);

    let size = work[0] as usize;
    let mut work = Vec::with_capacity(size);
    unsafe { work.set_len(size) };
    lapack::dsyev(Jobz::V, Uplo::U, m, matrix, m, vector, &mut work, size, &mut flag);
    success!(flag);

    Ok(())
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    #[test]
    fn symmetric_eigen() {
        let mut matrix = Conventional::from_vec(5, vec![
            0.814723686393179, 0.097540404999410, 0.157613081677548, 0.141886338627215,
            0.655740699156587, 0.097540404999410, 0.278498218867048, 0.970592781760616,
            0.421761282626275, 0.035711678574190, 0.157613081677548, 0.970592781760616,
            0.957166948242946, 0.915735525189067, 0.849129305868777, 0.141886338627215,
            0.421761282626275, 0.915735525189067, 0.792207329559554, 0.933993247757551,
            0.655740699156587, 0.035711678574190, 0.849129305868777, 0.933993247757551,
            0.678735154857773,
        ]);
        let mut vector = Conventional::from_vec((1, 5), vec![0.0; 5]);

        assert::success(SymmetricEigen::decompose(&mut (&mut matrix, &mut vector)));

        assert::close(&matrix.values, &vec![
             0.200767588469279, -0.613521879994358,  0.529492579537623,  0.161735212201923,
            -0.526082320114459, -0.241005628008408, -0.272281143378657,  0.443280672960843,
            -0.675165120368165,  0.464148221418878,  0.509762909240926,  0.555609456752178,
             0.244072927029371, -0.492754485897426, -0.359251069377747, -0.766321363493223,
             0.386556170387878,  0.341170928524320,  0.084643789583352, -0.373849864790357,
             0.233456648876442,  0.302202482503382,  0.589211894835079,  0.517708631263932,
             0.488854547655902,
        ], 1e-14);
        assert::close(&vector.values, &vec![
            -0.671640666831794, -0.230366398529950, 0.397221322493687, 0.999582068576074,
             3.026535012212483,
        ], 1e-14);
    }
}
