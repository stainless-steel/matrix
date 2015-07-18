//! Decompositions.

use Result;

/// The eigendecomposition of symmetric matrices.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}
