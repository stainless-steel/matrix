//! Matrix laboratory.
//!
//! ## Formats
//!
//! The following storage formats are supported:
//!
//! * [Banded][banded], suitable for matrices with a small number of
//!   superdiagonals and/or subdiagonals;
//!
//! * [Compressed][compressed], suitable for generic sparse matrices;
//!
//! * [Conventional][conventional], suitable for dense matrices;
//!
//! * [Diagonal][diagonal], suitable for diagonal matrices; and
//!
//! * [Packed][packed], suitable for symmetric, Hermitian, and triangular
//!   matrices.
//!
//! ## Example
//!
//! ```
//! #[macro_use]
//! extern crate matrix;
//!
//! use matrix::prelude::*;
//!
//! # fn main() {
//! let mut sparse = Compressed::zero((2, 4));
//! sparse.set((0, 0), 42.0);
//! sparse.set((1, 3), 69.0);
//!
//! let dense = Conventional::from(&sparse);
//! assert!(
//!     &*dense == &*matrix![
//!         42.0, 0.0, 0.0,  0.0;
//!          0.0, 0.0, 0.0, 69.0;
//!     ]
//! );
//! # }
//! ```
//!
//! [banded]: format/banded/index.html
//! [compressed]: format/compressed/index.html
//! [conventional]: format/conventional/index.html
//! [diagonal]: format/diagonal/index.html
//! [packed]: format/packed/index.html

#[cfg(test)]
extern crate assert;

#[cfg(feature = "acceleration")]
extern crate blas;

#[cfg(feature = "acceleration")]
extern crate lapack;

#[cfg(feature = "acceleration-src")]
extern crate openblas_src;

extern crate num_complex;
extern crate num_traits;

pub use num_traits::Num as Number;

/// A complex number with 32-bit parts.
#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex<f32>;

/// A complex number with 64-bit parts.
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex<f64>;

use std::convert::Into;
use std::{error, fmt};

use format::Conventional;

/// A matrix.
pub trait Matrix: Into<Conventional<<Self as Matrix>::Element>> + Size {
    /// The element type.
    type Element: Element;

    /// Count nonzero elements.
    fn nonzeros(&self) -> usize;

    /// Create a zero matrix.
    fn zero<S: Size>(S) -> Self;
}

/// A macro for composing matrices in the natural order.
///
/// The data of a generic matrix is conventionally stored in the column-major
/// order; see `format::conventional`. Consequently, the vector
///
/// ```norun
/// vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
/// ]
/// ```
///
/// corresponds to the following matrix with four rows and two columns:
///
/// ```math
/// ┌            ┐
/// │  1.0  5.0  │
/// │  2.0  6.0  │
/// │  3.0  7.0  │
/// │  4.0  8.0  │
/// └            ┘
/// ```
///
/// The macro allows one to write such a matrix in the natural order:
///
/// ```norun
/// matrix![
///     1.0, 5.0;
///     2.0, 6.0;
///     3.0, 7.0;
///     4.0, 8.0;
/// ]
/// ```
#[macro_export]
macro_rules! matrix {
    ($([$tail:expr,];)* -> [$($head:expr,)*]) => (
        vec![$($head,)* $($tail,)*]
    );
    ($([$middle:expr, $($tail:expr,)*];)* -> [$($head:expr,)*]) => (
        matrix!($([$($tail,)*];)* -> [$($head,)* $($middle,)*])
    );
    ($($($item:expr),*;)*) => (
        matrix!($([$($item,)*];)* -> [])
    );
    ($($($item:expr,)*;)*) => (
        matrix!($([$($item,)*];)* -> [])
    );
}

/// An error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(String);

/// A result.
pub type Result<T> = std::result::Result<T, Error>;

macro_rules! raise(
    ($message:expr) => (
        return Err(::Error($message.to_string()));
    );
);

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl error::Error for Error {
    #[inline]
    fn description(&self) -> &str {
        &self.0
    }
}

mod element;
mod position;
mod size;

pub use element::Element;
pub use position::Position;
pub use size::Size;

pub mod decomposition;
pub mod format;
pub mod operation;
pub mod prelude;
