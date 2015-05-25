use std::convert::Into;
use std::ops::{Deref, DerefMut};

/// A dense matrix.
#[derive(Debug)]
pub struct Matrix<T> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

impl<T> Into<Vec<T>> for Matrix<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> Deref for Matrix<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T> DerefMut for Matrix<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
