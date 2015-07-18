//! Basic operations.

use Result;

/// A multiplication that adds the result to a third object.
pub trait MultiplyInto<Right: ?Sized, Output: ?Sized> {
    /// Perform the multiplication.
    fn multiply_into(&self, &Right, &mut Output);
}

/// A multiplication that overwrites the left operand with the result.
pub trait MultiplySelf<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_self(&mut self, &Right);
}

/// The eigendecomposition of a symmetric matrix.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}
