use std::ops::{Deref, DerefMut};

use {Band, Dense, Element};

/// A diagonal matrix.
#[derive(Clone, Debug)]
pub struct Diagonal<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub data: Vec<T>,
}

matrix!(Diagonal);

impl<T: Element> From<Diagonal<T>> for Band<T> {
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

impl<T: Element> From<Diagonal<T>> for Dense<T> {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Dense<T> {
        <Diagonal<T> as Into<Band<T>>>::into(diagonal).into()
    }
}

impl<T: Element> Deref for Diagonal<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T: Element> DerefMut for Diagonal<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}
