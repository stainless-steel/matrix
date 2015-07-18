//! Basic operations.

/// A multiplication storing the result in a third object.
pub trait MultiplyInto<Right: ?Sized, Output: ?Sized> {
    /// Perform the multiplication.
    fn multiply_into(&self, &Right, &mut Output);
}

/// A multiplication storing the result in the left operand.
pub trait MultiplySelf<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_self(&mut self, &Right);
}

/// A multiplication storing the result in the right operand.
pub trait MultiplyThat<Right: ?Sized> {
    /// Perform the multiplication.
    fn multiply_that(&self, &mut Right);
}
