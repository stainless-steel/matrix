/// An error.
#[derive(Copy)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// Perform the eigendecomposition of a symmetric matrix.
///
/// A symmetric `m`-by-`m` matrix `a` is decomposed; the resulting eigenvectors
/// and eigenvalus are stored in an `m`-by-`m` matrix `vecs` and an `m`-element
/// vector `vals`, respectively.
pub fn sym_eig(a: &[f64], vecs: &mut [f64], vals: &mut [f64], m: uint) -> Result<(), Error> {
    use std::iter::repeat;

    if a.as_ptr() != vecs.as_ptr() {
        // Only the upper triangular matrix is actually needed; however, copying
        // only that part might not be optimal for performance. Check!
        unsafe {
            use std::ptr::copy_nonoverlapping_memory as copy;
            copy(vecs.as_mut_ptr(), a.as_ptr(), m * m);
        }
    }

    // The size of the temporary array should be >= max(1, 3 * m - 1).
    // http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html
    let mut temp = repeat(0.0).take(4 * m).collect::<Vec<_>>();
    let mut flag = 0;

    ::lapack::dsyev(b'V', b'U', m, vecs, m, vals, temp.as_mut_slice(), 4 * m, &mut flag);

    if flag < 0 {
        Err(Error::InvalidArguments)
    } else if flag > 0 {
        Err(Error::FailedToConverge)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn sym_eig() {
        use std::iter::repeat;

        let m = 5;

        let a = vec![
            0.814723686393179, 0.097540404999410, 0.157613081677548,
            0.141886338627215, 0.655740699156587, 0.097540404999410,
            0.278498218867048, 0.970592781760616, 0.421761282626275,
            0.035711678574190, 0.157613081677548, 0.970592781760616,
            0.957166948242946, 0.915735525189067, 0.849129305868777,
            0.141886338627215, 0.421761282626275, 0.915735525189067,
            0.792207329559554, 0.933993247757551, 0.655740699156587,
            0.035711678574190, 0.849129305868777, 0.933993247757551,
            0.678735154857773,
        ];

        let mut vecs = repeat(0.0).take(m * m).collect::<Vec<_>>();
        let mut vals = repeat(0.0).take(m).collect::<Vec<_>>();

        assert_ok!(::decomp::sym_eig(a.as_slice(), vecs.as_mut_slice(), vals.as_mut_slice(), m));

        let expected_vecs = vec![
             0.200767588469279, -0.613521879994358,  0.529492579537623,
             0.161735212201923, -0.526082320114459, -0.241005628008408,
            -0.272281143378657,  0.443280672960843, -0.675165120368165,
             0.464148221418878,  0.509762909240926,  0.555609456752178,
             0.244072927029371, -0.492754485897426, -0.359251069377747,
            -0.766321363493223,  0.386556170387878,  0.341170928524320,
             0.084643789583352, -0.373849864790357,  0.233456648876442,
             0.302202482503382,  0.589211894835079,  0.517708631263932,
             0.488854547655902,
        ];
        assert_close!(vecs, expected_vecs);

        let expected_vals = vec![
            -0.671640666831794, -0.230366398529950, 0.397221322493687,
             0.999582068576074,  3.026535012212483,
        ];
        assert_close!(vals, expected_vals);
    }
}
