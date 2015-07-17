//! The conventional storage.
//!
//! The storage is suitable for dense matrices.

use std::convert::Into;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use {Element, Matrix, Position, Size};

/// A conventional matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Conventional<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values stored in the column-major order.
    pub values: Vec<T>,
}

size!(Conventional);

impl<T: Element> Conventional<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        Conventional { rows: rows, columns: columns, values: vec![T::zero(); rows * columns] }
    }

    /// Create a matrix from a slice.
    pub fn from_slice<S: Size>(values: &[T], size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), rows * columns);
        Conventional { rows: rows, columns: columns, values: values.to_vec() }
    }

    /// Create a matrix from a vector.
    pub fn from_vec<S: Size>(values: Vec<T>, size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), rows * columns);
        Conventional { rows: rows, columns: columns, values: values }
    }
}

impl<T: Element> Matrix for Conventional<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    fn transpose(&self) -> Self {
        let (rows, columns) = (self.rows, self.columns);
        let mut matrix = Conventional::from_slice(&self.values, (columns, rows));
        for i in 0..rows {
            for j in i..columns {
                matrix.values.swap(j * rows + i, i * rows + j);
            }
        }
        matrix
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Conventional::new(size)
    }
}

impl<T: Element, P: Position> Index<P> for Conventional<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: P) -> &Self::Output {
        let (i, j) = index.coordinates();
        &self.values[j * self.rows + i]
    }
}

impl<T: Element, P: Position> IndexMut<P> for Conventional<T> {
    #[inline]
    fn index_mut(&mut self, index: P) -> &mut Self::Output {
        let (i, j) = index.coordinates();
        &mut self.values[j * self.rows + i]
    }
}

impl<T: Element> Into<Vec<T>> for Conventional<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}

impl<T: Element> Deref for Conventional<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.values.deref()
    }
}

impl<T: Element> DerefMut for Conventional<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.deref_mut()
    }
}

#[cfg(tests)]
mod tests {
    use {Conventional, Matrix};

    #[test]
    fn nonzeros() {
        let matrix = Conventional::from_vec(vec![1.0, 2.0, 3.0, 0.0], 2);
        assert_eq!(matrix.nonzeros(), 3);
    }

    #[test]
    fn transpose() {
        let mut matrix = Conventional::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
        matrix = matrix.transpose();
        assert_eq!(matrix, Conventional::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], (2, 3)));
    }
}
