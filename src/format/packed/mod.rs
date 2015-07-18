//! The packed format.
//!
//! The format is suitable for symmetric, Hermitian, and triangular matrices.
//! The format is compatible with the [one][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node123.html
//! [2]: http://www.netlib.org/lapack

use {Element, Matrix, Size};

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

macro_rules! arithmetic(
    ($count:expr, $first:expr, $last:expr) => (
        $count * ($first + $last) / 2
    );
);

macro_rules! storage(
    ($size:expr) => (arithmetic!($size, 1, $size))
);

mod convert;

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
        assert_eq!(self.values.len(), storage!(self.size));
    }
}

size!(Packed, size, size);

impl<T: Element> Packed<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, variant: Variant) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        new!(rows, variant, vec![T::zero(); storage!(rows)])
    }
}

impl<T: Element> Matrix for Packed<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    fn transpose(&self) -> Self {
        let &Packed { size, variant, .. } = self;
        let lower = variant == Variant::Lower;
        let mut matrix = Packed::new(size, variant.flip());
        let mut k = 0;
        for j in 0..size {
            for i in j..size {
                if lower {
                    matrix.values[arithmetic!(i, 1, i) + j] = self.values[k];
                } else {
                    matrix.values[k] = self.values[arithmetic!(i, 1, i) + j];
                }
                k += 1;
            }
        }
        matrix
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

    #[test]
    fn transpose_lower() {
        let matrix = new!(4, Variant::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Variant::Upper, vec![
            1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0,
        ]));
    }

    #[test]
    fn transpose_upper() {
        let matrix = new!(4, Variant::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Variant::Lower, vec![
            1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 8.0, 6.0, 9.0, 10.0,
        ]));
    }
}
