use std::convert::Into;
use std::ops::{Deref, DerefMut};

/// A dense matrix.
#[derive(Debug)]
pub struct Dense<T> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

impl<T> Into<Vec<T>> for Dense<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> Deref for Dense<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T> DerefMut for Dense<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
