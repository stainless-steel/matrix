//! Matrix storage schemes.

#[cfg(feature = "complex")]
extern crate complex;

use std::convert::Into;

/// A matrix.
pub trait Matrix: Into<Conventional<<Self as Matrix>::Element>> + Size {
    /// The element type.
    type Element: Element;

    /// Count the number of nonzero elements.
    fn nonzeros(&self) -> usize;

    /// Transpose the matrix.
    fn transpose(&self) -> Self;

    /// Create a zero matrix.
    fn zero<S: Size>(S) -> Self;
}

#[cfg(debug_assertions)]
trait Validate {
    fn validate(&self);
}

#[cfg(debug_assertions)]
macro_rules! validate(
    ($matrix:expr) => ({
        use ::Validate;
        let matrix = $matrix;
        matrix.validate();
        matrix
    });
);

#[cfg(not(debug_assertions))]
macro_rules! validate(
    ($matrix:expr) => ($matrix);
);

macro_rules! size(
    ($kind:ident, $rows:ident, $columns:ident) => (
        impl<T: ::Element> ::Size for $kind<T> {
            #[inline(always)]
            fn rows(&self) -> usize {
                self.$rows
            }

            #[inline(always)]
            fn columns(&self) -> usize {
                self.$columns
            }
        }
    );
    ($kind:ident) => (
        size!($kind, rows, columns);
    );
);

macro_rules! min(
    ($left:expr, $right:expr) => ({
        let (left, right) = ($left, $right);
        if left > right { right } else { left }
    });
);

mod element;
mod position;
mod size;

pub mod banded;
pub mod compressed;
pub mod conventional;
pub mod diagonal;
pub mod packed;

pub use element::Element;
pub use position::Position;
pub use size::Size;

pub use banded::Banded;
pub use compressed::Compressed;
pub use conventional::Conventional;
pub use diagonal::Diagonal;
pub use packed::Packed;
