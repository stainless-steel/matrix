//! Matrix storage schemes.

#[cfg(feature = "complex")]
extern crate complex;

use std::convert::Into;

/// A matrix.
pub trait Matrix: Size {
    /// The element type.
    type Element: Element;
}

/// A sparse matrix.
pub trait Sparse: Matrix + Into<Dense<<Self as Matrix>::Element>> {
    /// Return the number of nonzero elements.
    fn nonzeros(&self) -> usize;
}

/// A major dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Major {
    /// The column major.
    Column,
    /// The row major.
    Row,
}

/// A part of a matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// The lower triangular part.
    Lower,
    /// The upper triangular part.
    Upper,
}

macro_rules! matrix(
    ($kind:ident, $rows:ident, $columns:ident) => (
        impl<T: ::Element> ::Matrix for $kind<T> {
            type Element = T;
        }

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
        matrix!($kind, rows, columns);
    );
);

macro_rules! min(
    ($left:expr, $right:expr) => ({
        let (left, right) = ($left, $right);
        if left < right { left } else { right }
    });
);

mod band;
mod compressed;
mod dense;
mod diagonal;
mod element;
mod position;
mod size;
mod triangular;

pub use band::Band;
pub use compressed::{Compressed, CompressedIterator};
pub use dense::Dense;
pub use diagonal::Diagonal;
pub use element::Element;
pub use position::Position;
pub use size::Size;
pub use triangular::Triangular;
