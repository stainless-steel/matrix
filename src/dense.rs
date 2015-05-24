use std::convert::Into;
use std::ops::{Deref, DerefMut};

use generic;

/// A dense matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The actual data.
    pub data: Data,
}

pub type Data = Vec<f64>;

impl Matrix {
}

impl generic::Matrix for Matrix {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn columns(&self) -> usize {
        self.columns
    }
}

impl Into<Data> for Matrix {
    #[inline]
    fn into(self) -> Data {
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
