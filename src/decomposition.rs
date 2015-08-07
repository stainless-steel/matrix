//! Decompositions.

use Result;

/// The eigendecomposition for symmetric matrices.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}
