//! The packed storage.
//!
//! The storage is suitable for symmetric, Hermitian, and triangular matrices.
//! Data are stored in the [format][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node123.html
//! [2]: http://www.netlib.org/lapack

use {Element, Matrix, Size};

/// A packed matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed<T: Element> {
    /// The number of rows or columns.
    pub size: usize,
    /// The storage format.
    pub format: Format,
    /// The values of the lower triangle when `format = Lower` or upper triangle
    /// when `format = Upper` stored by columns.
    pub values: Vec<T>,
}

macro_rules! new(
    ($size:expr, $format:expr, $values:expr) => (
        Packed { size: $size, format: $format, values: $values }
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

/// A format of a packed matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// The lower-triangular format.
    Lower,
    /// The upper-triangular format.
    Upper,
}

#[cfg(debug_assertions)]
impl<T: Element> ::storage::Validate for Packed<T> {
    fn validate(&self) {
        assert_eq!(self.values.len(), storage!(self.size));
    }
}

size!(Packed, size, size);

impl<T: Element> Packed<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, format: Format) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        new!(rows, format, vec![T::zero(); storage!(rows)])
    }
}

impl<T: Element> Matrix for Packed<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    fn transpose(&self) -> Self {
        let &Packed { size, format, .. } = self;
        let lower = format == Format::Lower;
        let mut matrix = Packed::new(size, format.flip());
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
        Packed::new(size, Format::Lower)
    }
}

impl Format {
    /// Return the other format.
    #[inline]
    pub fn flip(&self) -> Self {
        match *self {
            Format::Lower => Format::Upper,
            Format::Upper => Format::Lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use Matrix;
    use storage::packed::{Format, Packed};

    #[test]
    fn nonzeros() {
        let matrix = new!(4, Format::Lower, vec![
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        assert_eq!(matrix.nonzeros(), 7);
    }

    #[test]
    fn transpose_lower() {
        let matrix = new!(4, Format::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Format::Upper, vec![
            1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0,
        ]));
    }

    #[test]
    fn transpose_upper() {
        let matrix = new!(4, Format::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Format::Lower, vec![
            1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 8.0, 6.0, 9.0, 10.0,
        ]));
    }
}
