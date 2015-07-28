//! Matrix laboratory.

#[cfg(test)]
extern crate assert;

#[cfg(feature = "acceleration")]
extern crate blas;

#[cfg(feature = "complex")]
extern crate complex;

#[cfg(feature = "acceleration")]
extern crate lapack;

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
mod number;
mod position;
mod size;

pub use element::Element;
pub use number::Number;
pub use position::Position;
pub use size::Size;

pub mod format;
pub mod operation;
pub mod prelude;
