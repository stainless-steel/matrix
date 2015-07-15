//! Matrix storage schemes.

#[cfg(feature = "complex")]
extern crate complex;

use std::convert::Into;

/// A matrix.
pub trait Matrix: Size {
    /// The element type.
    type Element: Element;

    /// Construct a zero matrix.
    fn zero<S: Size>(S) -> Self;
}

/// A sparse matrix.
pub trait Sparse: Matrix + Into<Dense<<Self as Matrix>::Element>> {
    /// Return the number of nonzero elements.
    fn nonzeros(&self) -> usize;
}

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
        if left < right { left } else { right }
    });
);

mod element;
mod position;
mod size;

pub mod band;
pub mod compressed;
pub mod dense;
pub mod diagonal;
pub mod triangular;

pub use band::Band;
pub use compressed::Compressed;
pub use dense::Dense;
pub use diagonal::Diagonal;
pub use element::Element;
pub use position::Position;
pub use size::Size;
pub use triangular::Triangular;
