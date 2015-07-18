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
