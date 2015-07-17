//! The diagonal storage.
//!
//! The storage is suitable for diagonal matrices.

use std::ops::{Deref, DerefMut};

use compressed::Format;
use {Banded, Compressed, Conventional, Element, Matrix, Size};

/// A diagonal matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagonal<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub values: Vec<T>,
}

macro_rules! debug_validate(
    ($matrix:ident) => (debug_assert!(
        $matrix.values.len() == min!($matrix.rows, $matrix.columns)
    ));
);

size!(Diagonal);

impl<T: Element> Diagonal<T> {
    /// Create a matrix from a slice.
    pub fn from_slice<S: Size>(values: &[T], size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), min!(rows, columns));
        Diagonal { rows: rows, columns: columns, values: values.to_vec() }
    }

    /// Create a matrix from a vector.
    pub fn from_vec<S: Size>(values: Vec<T>, size: S) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), min!(rows, columns));
        Diagonal { rows: rows, columns: columns, values: values }
    }
}

impl<T: Element> Matrix for Diagonal<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        self.clone()
    }

    fn zero<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        Diagonal { rows: rows, columns: columns, values: vec![T::zero(); min!(rows, columns)] }
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Banded<T> {
    #[inline]
    fn from(matrix: &'l Diagonal<T>) -> Self {
        matrix.clone().into()
    }
}

impl<T: Element> From<Diagonal<T>> for Banded<T> {
    fn from(matrix: Diagonal<T>) -> Self {
        debug_validate!(matrix);
        Banded {
            rows: matrix.rows,
            columns: matrix.columns,
            superdiagonals: 0,
            subdiagonals: 0,
            values: {
                let mut values = matrix.values;
                for _ in matrix.rows..matrix.columns {
                    values.push(T::zero());
                }
                values
            },
        }
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Compressed<T> {
    #[inline]
    fn from(matrix: &'l Diagonal<T>) -> Self {
        matrix.clone().into()
    }
}

impl<T: Element> From<Diagonal<T>> for Compressed<T> {
    #[inline]
    fn from(matrix: Diagonal<T>) -> Self {
        debug_validate!(matrix);
        let Diagonal { rows, columns, values } = matrix;
        let nonzeros = values.len();
        Compressed {
            rows: rows,
            columns: columns,
            nonzeros: nonzeros,
            values: values,
            format: Format::Column,
            indices: (0..nonzeros).collect(),
            offsets: (0..(columns + 1)).map(|i| if i < nonzeros { i } else { nonzeros }).collect(),
        }
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Conventional<T> {
    fn from(matrix: &Diagonal<T>) -> Self {
        debug_validate!(matrix);

        let &Diagonal { rows, columns, ref values } = matrix;

        let mut conventional = Conventional::new((rows, columns));
        for i in 0..min!(rows, columns) {
            conventional.values[i * rows + i] = values[i];
        }

        conventional
    }
}

impl<T: Element> From<Diagonal<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Diagonal<T>) -> Self {
        (&matrix).into()
    }
}

impl<T: Element> Into<Vec<T>> for Diagonal<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}

impl<T: Element> Deref for Diagonal<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.values.deref()
    }
}

impl<T: Element> DerefMut for Diagonal<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use compressed::Format;
    use {Banded, Compressed, Conventional, Diagonal, Matrix};

    macro_rules! new(
        ($rows:expr, $columns:expr, $values:expr) => (
            Diagonal { rows: $rows, columns: $columns, values: $values }
        );
    );

    #[test]
    fn nonzeros() {
        let matrix = Diagonal::from_vec(vec![1.0, 2.0, 0.0, 3.0], 4);
        assert_eq!(matrix.nonzeros(), 3);
    }

    #[test]
    fn into_banded_tall() {
        let matrix = new!(5, 3, vec![1.0, 2.0, 3.0]);
        let matrix: Banded<_> = matrix.into();
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_banded_wide() {
        let matrix = new!(3, 5, vec![1.0, 2.0, 3.0]);
        let matrix: Banded<_> = matrix.into();
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn into_compressed_tall() {
        let matrix = new!(5, 3, vec![1.0, 2.0, 0.0]);

        let matrix: Compressed<_> = matrix.into();

        assert_eq!(matrix, Compressed {
            rows: 5, columns: 3, nonzeros: 3, format: Format::Column, values: vec![1.0, 2.0, 0.0],
            indices: vec![0, 1, 2], offsets: vec![0, 1, 2, 3]
        });
    }

    #[test]
    fn into_compressed_wide() {
        let matrix = new!(3, 5, vec![1.0, 0.0, 3.0]);

        let matrix: Compressed<_> = matrix.into();

        assert_eq!(matrix, Compressed {
            rows: 3, columns: 5, nonzeros: 3, format: Format::Column, values: vec![1.0, 0.0, 3.0],
            indices: vec![0, 1, 2], offsets: vec![0, 1, 2, 3, 3, 3]
        });
    }

    #[test]
    fn into_conventional() {
        let matrix = new!(3, 5, vec![1.0, 2.0, 3.0]);

        let matrix: Conventional<_> = matrix.into();

        assert_eq!(matrix, Conventional::from_vec(vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], (3, 5)));
    }
}
