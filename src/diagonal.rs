use num::Num;
use std::ops::{Deref, DerefMut};

use {Band, Dense};

/// A diagonal matrix.
#[derive(Debug)]
pub struct Diagonal<T> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub data: Vec<T>,
}

impl<T> From<Diagonal<T>> for Band<T> where T: Copy + Num {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Band<T> {
        Band {
            rows: diagonal.rows,
            columns: diagonal.columns,
            superdiagonals: 0,
            subdiagonals: 0,
            data: diagonal.data,
        }
    }
}

impl<T> From<Diagonal<T>> for Dense<T> where T: Copy + Num {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Dense<T> {
        <Diagonal<T> as Into<Band<T>>>::into(diagonal).into()
    }
}

impl<T> Deref for Diagonal<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T> DerefMut for Diagonal<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
