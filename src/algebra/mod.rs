//! Basic operations.

/// A multiplication storing the result in a third object.
pub trait MultiplyInto<Right, Output> {
    /// Perform the multiplication.
    fn multiply_into(&self, &Right, &mut Output);
}

/// A multiplication storing the result in the left operand.
pub trait MultiplySelf<Right> {
    /// Perform the multiplication.
    fn multiply_self(&mut self, &Right);
}

/// A multiplication storing the result in the right operand.
pub trait MultiplyThat<Right> {
    /// Perform the multiplication.
    fn multiply_that(&self, &mut Right);
}
