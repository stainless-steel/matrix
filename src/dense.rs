use std::convert::Into;
use std::ops::{Deref, DerefMut};

use {Element, Make, Shape};

/// A dense matrix.
///
/// The storage is suitable for generic matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values stored in the column-major order.
    pub values: Vec<T>,
}

matrix!(Dense);

impl<'l, T: Element> Make<&'l [T]> for Dense<T> {
    fn make(values: &'l [T], shape: Shape) -> Self {
        let (rows, columns) = match shape {
            Shape::Square(size) => (size, size),
            Shape::Rectangular(rows, columns) => (rows, columns),
        };
        debug_assert_eq!(values.len(), rows * columns);
        Dense { rows: rows, columns: columns, values: values.to_vec() }
    }
}

impl<T: Element> Make<Vec<T>> for Dense<T> {
    fn make(values: Vec<T>, shape: Shape) -> Self {
        let (rows, columns) = match shape {
            Shape::Square(size) => (size, size),
            Shape::Rectangular(rows, columns) => (rows, columns),
        };
        debug_assert_eq!(values.len(), rows * columns);
        Dense { rows: rows, columns: columns, values: values }
    }
}

impl<T: Element> Into<Vec<T>> for Dense<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}

impl<T: Element> Deref for Dense<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.values.deref()
    }
}

impl<T: Element> DerefMut for Dense<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.values.deref_mut()
    }
}
