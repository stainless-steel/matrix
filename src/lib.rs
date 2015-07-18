//! Matrix laboratory.

#[cfg(feature = "complex")]
extern crate complex;

use std::convert::Into;

/// A matrix.
pub trait Matrix: Into<Conventional<<Self as Matrix>::Element>> + Size {
    /// The element type.
    type Element: Element;

    /// Count the number of nonzero elements.
    fn nonzeros(&self) -> usize;

    /// Compute the matrix transpose.
    fn transpose(&self) -> Self;

    /// Create a zero matrix.
    fn zero<S: Size>(S) -> Self;
}

mod element;
mod position;
mod size;

pub use element::Element;
pub use position::Position;
pub use size::Size;

pub mod storage;

pub use storage::Banded;
pub use storage::Compressed;
pub use storage::Conventional;
pub use storage::Diagonal;
pub use storage::Packed;
