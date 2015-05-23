use std::ptr;

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
/// The slices `matrix`, `vectors`, and `values` should have `size × size`, `size × size`, and
/// `size` elements, respectively.
pub fn symmetric_eigen(matrix: &[f64], vectors: &mut [f64], values: &mut [f64],
                       size: usize) -> Result<(), Error> {

    use lapack::{dsyev, Jobz, Uplo};

    macro_rules! success(
        ($flag:expr) => (
            if $flag < 0 {
                return Err(Error::InvalidArguments)
            } else if $flag > 0 {
                return Err(Error::FailedToConverge)
            }
        );
    );

    if matrix.len() != size * size || vectors.len() != size * size || values.len() != size {
        return Err(Error::InvalidArguments)
    }

    if matrix.as_ptr() != vectors.as_ptr() {
        unsafe {
            // Only the upper triangular matrix is actually needed.
            ptr::copy_nonoverlapping(matrix.as_ptr(), vectors.as_mut_ptr(), size * size);
        }
    }

    let mut flag = 0;

    let mut work = [0.0];
    dsyev(Jobz::V, Uplo::U, size, vectors, size, values, &mut work, -1isize as usize, &mut flag);
    success!(flag);

    let lwork = work[0] as usize;
    let mut work = vec![0.0; lwork];
    dsyev(Jobz::V, Uplo::U, size, vectors, size, values, &mut work, lwork, &mut flag);
    success!(flag);

    Ok(())
}
