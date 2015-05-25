//! Dense matrices.

use std::convert::Into;
use std::ops::{Deref, DerefMut};

/// A dense matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The data stored in the column-major order.
    pub data: Vec<f64>,
}

impl Into<Vec<f64>> for Matrix {
    #[inline]
    fn into(self) -> Vec<f64> {
        self.data
    }
}

impl Deref for Matrix {
    type Target = [f64];

    #[inline]
    fn deref(&self) -> &[f64] {
        self.data.deref()
    }
}

impl DerefMut for Matrix {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f64] {
        self.data.deref_mut()
    }
}
