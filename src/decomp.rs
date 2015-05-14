/// An error.
#[derive(Clone, Copy)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// Perform the eigendecomposition of a symmetric matrix.
///
/// A symmetric `m`-by-`m` matrix `a` is decomposed; the resulting eigenvectors and eigenvalus are
/// stored in an `m`-by-`m` matrix `vecs` and an `m`-element vector `vals`, respectively.
pub fn sym_eig(a: &[f64], vecs: &mut [f64], vals: &mut [f64], m: usize) -> Result<(), Error> {
    use std::iter::repeat;
    use lapack::metal::{dsyev, Jobz, Uplo};

    if a.as_ptr() != vecs.as_ptr() {
        // Only the upper triangular matrix is actually needed; however, copying only that part
        // might not be optimal for performance. Check!
        unsafe {
            use std::ptr::copy_nonoverlapping as copy;
            copy(a.as_ptr(), vecs.as_mut_ptr(), m * m);
        }
    }

    // The size of the temporary array should be >= max(1, 3 * m - 1).
    // http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html
    let mut temp = repeat(0.0).take(4 * m).collect::<Vec<_>>();
    let mut flag = 0;

    dsyev(Jobz::V, Uplo::U, m, vecs, m, vals, &mut temp, 4 * m, &mut flag);

    if flag < 0 {
        Err(Error::InvalidArguments)
    } else if flag > 0 {
        Err(Error::FailedToConverge)
    } else {
        Ok(())
    }
}
