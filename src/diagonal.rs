//! Diagonal matrices.

use num::Num;
use std::ops::{Deref, DerefMut};

use {band, dense};

/// A diagonal matrix.
#[derive(Debug)]
pub struct Matrix<T> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub data: Vec<T>,
}

impl<T> From<Matrix<T>> for band::Matrix<T> where T: Copy + Num {
    #[inline]
    fn from(matrix: Matrix<T>) -> band::Matrix<T> {
        band::Matrix {
            rows: matrix.rows,
            columns: matrix.columns,
            superdiagonals: 0,
            subdiagonals: 0,
            data: matrix.data,
        }
    }
}

impl<T> From<Matrix<T>> for dense::Matrix<T> where T: Copy + Num {
    #[inline]
    fn from(matrix: Matrix<T>) -> dense::Matrix<T> {
        <Matrix<T> as Into<band::Matrix<T>>>::into(matrix).into()
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
