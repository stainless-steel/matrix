//! Decompositions.

use format::{Conventional, Diagonal};
use {Element, Result};

/// The eigendecomposition for symmetric matrices.
pub trait SymmetricEigen<T: Element> {
    /// Perform the decomposition.
    fn decompose(&self) -> Result<(Conventional<T>, Diagonal<T>)>;
}
