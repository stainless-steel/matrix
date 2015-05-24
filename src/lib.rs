//! Algorithms for manipulating real matrices.

#![allow(non_snake_case)]

#[cfg(test)]
extern crate assert;

extern crate blas;
extern crate lapack;

mod generic;
mod dense;
mod sparse;

mod algebra;
mod decomposition;

pub use generic::Matrix as GenericMatrix;

pub use dense::Matrix as DenseMatrix;
pub use dense::Data as DenseData;

pub use sparse::Matrix as SparseMatrix;
pub use sparse::Data as SparseData;
pub use sparse::CompressedDimension;

pub use algebra::{multiply, multiply_add};
pub use decomposition::symmetric_eigen;

/// An error.
#[derive(Clone, Copy)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// A result.
pub type Result<T> = std::result::Result<T, Error>;
