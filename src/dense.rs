use std::convert::Into;
use std::ops::{Deref, DerefMut};

use Element;

/// A dense matrix.
#[derive(Clone, Debug)]
pub struct DenseMatrix<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

impl<T: Element> Into<Vec<T>> for DenseMatrix<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T: Element> Deref for DenseMatrix<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T: Element> DerefMut for DenseMatrix<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
