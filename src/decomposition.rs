//! Decompositions.

use format::{Conventional, Diagonal};
use {Element, Result};

/// The singular-value decomposition.
pub trait SingularValue<T: Element> {
    /// Perform the decomposition.
    fn decompose(&self) -> Result<(Conventional<T>, Diagonal<T>, Conventional<T>)>;
}

/// The eigendecomposition for symmetric matrices.
pub trait SymmetricEigen<T: Element> {
    /// Perform the decomposition.
    fn decompose(&self) -> Result<(Conventional<T>, Diagonal<T>)>;
}
