//! Basic operations.

use Result;

/// A multiplication.
pub trait Multiply<Right: ?Sized, Output> {
    /// Perform the multiplication.
    fn multiply(&self, &Right) -> Output;
}

/// A multiplication that adds the result to a third object.
pub trait MultiplyInto<Right: ?Sized, Output: ?Sized> {
    /// Perform the multiplication.
    fn multiply_into(&self, &Right, &mut Output);
}

/// A multiplication that overwrites the receiver with the result.
pub trait MultiplySelf<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_self(&mut self, &Right);
}

/// A scaling that overwrites the receiver with the result.
pub trait ScaleSelf<T> {
    /// Perform the scaling.
    fn scale_self(&mut self, T);
}

/// The eigendecomposition of a symmetric matrix.
pub trait SymmetricEigen {
    /// Perform the decomposition.
    fn decompose(&mut Self) -> Result<()>;
}

/// The transpose.
pub trait Transpose {
    /// Perform the transpose.
    fn transpose(&self) -> Self;
}
