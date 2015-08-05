//! Decompositions.

use Result;

/// The eigendecomposition of a symmetric matrix.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}
