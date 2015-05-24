//! Algorithms for manipulating real matrices.

#![allow(non_snake_case)]

#[cfg(test)]
extern crate assert;

extern crate blas;
extern crate lapack;

pub mod dense;
pub mod generic;
pub mod sparse;

mod algebra;
mod decomposition;

pub use algebra::{dot, scale, sum, times};
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
