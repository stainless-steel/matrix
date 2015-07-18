//! Basic operations.

use Result;

/// A multiplication adding the result to a third object.
pub trait MultiplyInto<Right: ?Sized, Output: ?Sized> {
    /// Perform the multiplication.
    fn multiply_into(&self, &Right, &mut Output);
}

/// A multiplication overwriting the left operand with the result.
pub trait MultiplySelf<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_self(&mut self, &Right);
}

/// A multiplication overwriting the right operand with the result.
pub trait MultiplyThat<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_that(&self, &mut Right);
}

/// The eigendecomposition of a symmetric matrix.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}
