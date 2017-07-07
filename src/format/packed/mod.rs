//! The packed format.
//!
//! The format is suitable for symmetric, Hermitian, and triangular matrices.
//! The format is compatible with the [one][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node123.html
//! [2]: http://www.netlib.org/lapack

use {Element, Matrix, Size};
use position::Position;

/// A packed matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed<T: Element> {
    /// The number of rows or columns.
    pub size: usize,
    /// The format variant.
    pub variant: Variant,
    /// The values of the lower triangle when `variant = Lower` or upper
    /// triangle when `variant = Upper` stored by columns.
    pub values: Vec<T>,
}

macro_rules! new(
    ($size:expr, $variant:expr, $values:expr) => (
        Packed { size: $size, variant: $variant, values: $values }
    );
);

mod convert;
mod operation;

/// A variant of a packed matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    /// The lower-triangular variant.
    Lower,
    /// The upper-triangular variant.
    Upper,
}

#[cfg(debug_assertions)]
impl<T: Element> ::format::Validate for Packed<T> {
    fn validate(&self) {
        assert_eq!(self.values.len(), triangular_elems(self.size));
    }
}

size!(Packed, size, size);

/// Computes the offset into the array for the upper triangular representation
/// Copied from http://www.netlib.org/lapack/lug/node123.html
#[inline]
pub fn upper_triangular_offset(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    debug_assert!(i <= n);
    debug_assert!(j <= n);
    // Source material uses 1-based indexing
    let i = i + 1;
    let j = j + 1;
    i + j * (j - 1) / 2 - 1
}

/// Computes the offset into the array for the lower triangular representation
/// Copied from http://www.netlib.org/lapack/lug/node123.html
#[inline]
pub fn lower_triangular_offset(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(j <= i);
    debug_assert!(i <= n);
    debug_assert!(j <= n);
    // Source material uses 1-based indexing
    let i = i + 1;
    let j = j + 1;
    i + (2 * n - j) * (j - 1) / 2 - 1
}

/// Computes how many unique elements are in a symmetric square matrix of size nxn
#[inline]
pub fn triangular_elems(n: usize) -> usize {
    n * (n + 1) / 2
}

impl<T: Element> Packed<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, variant: Variant) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        new!(rows, variant, vec![T::zero(); triangular_elems(rows)])
    }

    /// Read an element from the matrix
    pub fn get<P: Position>(&self, position: P) -> Option<&T> {
        let (mut i, mut j) = position.coordinates();
        let index = match &self.variant {
            &Variant::Upper => {
                if i > j {
                    upper_triangular_offset(self.size, j, i)
                } else {
                    upper_triangular_offset(self.size, i, j)
                }
            },
            &Variant::Lower => {
                if j > i {
                    lower_triangular_offset(self.size, j, i)
                } else {
                    lower_triangular_offset(self.size, i, j)
                }
            },
        };
        self.values.get(index)
    }
}

impl<T: Element> Matrix for Packed<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Packed::new(size, Variant::Lower)
    }
}

impl Variant {
    /// Return the other variant.
    #[inline]
    pub fn flip(&self) -> Self {
        match *self {
            Variant::Lower => Variant::Upper,
            Variant::Upper => Variant::Lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use format::packed::Variant;
    use prelude::*;

    #[test]
    fn nonzeros() {
        let matrix = new!(4, Variant::Lower, vec![
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        assert_eq!(matrix.nonzeros(), 7);
    }
}
