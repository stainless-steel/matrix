//! Algorithms for manipulating real matrices.

#![allow(non_snake_case)]

extern crate blas;
extern crate lapack;

mod algebra;
mod decomposition;
mod storage;

pub use algebra::{multiply, multiply_add};
pub use decomposition::symmetric_eigen;
pub use storage::{Sparse, CompressedData};

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
