//! Generic matrices.

use std::convert::Into;

use dense;

/// A generic matrix.
pub trait Matrix: Into<dense::Matrix> {
    /// Return the number of rows.
    fn rows(&self) -> usize;
    /// Return the number of columns.
    fn columns(&self) -> usize;
}
