//! The diagonal storage.
//!
//! The storage is suitable for diagonal matrices.

use std::ops::{Deref, DerefMut};

use {Element, Matrix, Size};

/// A diagonal matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagonal<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub values: Vec<T>,
}

macro_rules! new(
    ($rows:expr, $columns:expr, $values:expr) => (
        Diagonal { rows: $rows, columns: $columns, values: $values }
    );
);

mod convert;

#[cfg(debug_assertions)]
impl<T: Element> ::storage::Validate for Diagonal<T> {
    fn validate(&self) {
        assert_eq!(self.values.len(), min!(self.rows, self.columns))
    }
}

size!(Diagonal);

impl<T: Element> Diagonal<T> {
    /// Create a matrix from a slice.
    pub fn from_slice<S: Size>(values: &[T], size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), min!(rows, columns));
        new!(rows, columns, values.to_vec())
    }

    /// Create a matrix from a vector.
    pub fn from_vec<S: Size>(values: Vec<T>, size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), min!(rows, columns));
        new!(rows, columns, values)
    }
}

impl<T: Element> Matrix for Diagonal<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        self.clone()
    }

    fn zero<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        new!(rows, columns, vec![T::zero(); min!(rows, columns)])
    }
}

impl<T: Element> Deref for Diagonal<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.values.deref()
    }
}

impl<T: Element> DerefMut for Diagonal<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn nonzeros() {
        let matrix = Diagonal::from_vec(vec![1.0, 2.0, 0.0, 3.0], 4);
        assert_eq!(matrix.nonzeros(), 3);
    }
}
