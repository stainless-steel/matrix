use std::convert::Into;
use std::ops::{Deref, DerefMut};

use Element;

/// A dense matrix.
///
/// The storage is suitable for generic matrices.
#[derive(Clone, Debug)]
pub struct Dense<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

matrix!(Dense);

impl<T: Element> Into<Vec<T>> for Dense<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T: Element> Deref for Dense<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T: Element> DerefMut for Dense<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
