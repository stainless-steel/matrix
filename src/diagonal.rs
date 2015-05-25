//! Diagonal matrices.

use std::ops::{Deref, DerefMut};

use {band, dense};

/// A diagonal matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub values: Vec<f64>,
}

impl From<Matrix> for band::Matrix {
    #[inline]
    fn from(matrix: Matrix) -> band::Matrix {
        band::Matrix {
            rows: matrix.rows,
            columns: matrix.columns,
            superdiagonals: 0,
            subdiagonals: 0,
            values: matrix.values,
        }
    }
}

impl From<Matrix> for dense::Matrix {
    #[inline]
    fn from(matrix: Matrix) -> dense::Matrix {
        <Matrix as Into<band::Matrix>>::into(matrix).into()
    }
}

impl Deref for Matrix {
    type Target = [f64];

    #[inline]
    fn deref(&self) -> &[f64] {
        self.values.deref()
    }
}

impl DerefMut for Matrix {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f64] {
        self.values.deref_mut()
    }
}
