use std::ptr;
use {Error, Result};

/// Perform the eigendecomposition of a symmetric matrix.
///
/// The formula is as follows:
///
/// ```math
/// A = U * diag(L) * U^T.
/// ```
///
/// The slices `A`, `U`, and `L` should have `m × m`, `m × m`, and `m` elements, respectively.
pub fn symmetric_eigen(A: &[f64], U: &mut [f64], L: &mut [f64], m: usize) -> Result<()> {
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

    debug_assert_eq!(A.len(), m * m);
    debug_assert_eq!(U.len(), m * m);
    debug_assert_eq!(L.len(), m);

    if A.as_ptr() != U.as_ptr() {
        unsafe {
            // Only the upper triangular matrix is actually needed.
            ptr::copy_nonoverlapping(A.as_ptr(), U.as_mut_ptr(), m * m);
        }
    }

    let mut flag = 0;

    let mut work = [0.0];
    dsyev(Jobz::V, Uplo::U, m, U, m, L, &mut work, -1isize as usize, &mut flag);
    success!(flag);

    let size = work[0] as usize;
    let mut work = Vec::with_capacity(size);
    unsafe { work.set_len(size) };
    dsyev(Jobz::V, Uplo::U, m, U, m, L, &mut work, size, &mut flag);
    success!(flag);

    Ok(())
}
