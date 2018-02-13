//! The banded format.
//!
//! The format is suitable for matrices with a small number of superdiagonals
//! and/or subdiagonals relative to the smallest dimension. The format is
//! compatible with the [one][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node124.html
//! [2]: http://www.netlib.org/lapack

use std::iter;

use {Element, Matrix, Size};

/// A banded matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Banded<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The number of superdiagonals.
    pub superdiagonals: usize,
    /// The number of subdiagonals.
    pub subdiagonals: usize,
    /// The values of the diagonal elements stored as a `(superdiagonals + 1 +
    /// subdiagonals) × columns` matrix such that the first row corresponds to
    /// the uppermost superdiagonal whereas the last row corresponds to the
    /// lowest supdiagonal.
    pub values: Vec<T>,
}

macro_rules! new(
    ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr, $values:expr) => (
        Banded {
            rows: $rows,
            columns: $columns,
            superdiagonals: $superdiagonals,
            subdiagonals: $subdiagonals,
            values: $values,
        }
    );
);

macro_rules! max_difference(
    ($limit:expr, $left:expr, $right:expr) => ({
        let (limit, left, right) = ($limit, $left, $right);
        if left < limit + right { limit } else { left - right }
    });
);

macro_rules! row_start(
    ($rows:expr, $superdiagonals:expr, $column:expr) => (
        max_difference!(0, $column, $superdiagonals)
    );
    ($matrix:ident, $column:expr) => (
        row_start!($matrix.rows, $matrix.superdiagonals, $column)
    );
);

macro_rules! row_finish(
    ($rows:expr, $subdiagonals:expr, $column:expr) => (
        min!($rows, $column + $subdiagonals + 1)
    );
    ($matrix:ident, $column:expr) => (
        row_finish!($matrix.rows, $matrix.subdiagonals, $column)
    );
);

macro_rules! row_range(
    ($rows:expr, $superdiagonals:expr, $subdiagonals:expr, $column:expr) => (
        row_start!($rows, $superdiagonals, $column)..row_finish!($rows, $subdiagonals, $column)
    );
);

mod convert;
mod operation;

/// A sparse iterator.
pub struct Iterator<'l, T: 'l + Element> {
    matrix: &'l Banded<T>,
    column: usize,
    start: usize,
    finish: usize,
}

#[cfg(debug_assertions)]
impl<T: Element> ::format::Validate for Banded<T> {
    fn validate(&self) {
        assert_eq!(self.values.len(), self.diagonals() * self.columns);
    }
}

size!(Banded);

impl<T: Element> Banded<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, superdiagonals: usize, subdiagonals: usize) -> Self {
        let (rows, columns) = size.dimensions();
        let values = vec![T::zero(); (superdiagonals + 1 + subdiagonals) * columns];
        new!(rows, columns, superdiagonals, subdiagonals, values)
    }

    /// Return the number of diagonals.
    #[inline]
    pub fn diagonals(&self) -> usize {
        self.superdiagonals + 1 + self.subdiagonals
    }

    /// Return a sparse iterator.
    #[inline]
    pub fn iter<'l>(&'l self) -> Iterator<'l, T> {
        Iterator::new(self)
    }
}

impl<T: Element> Matrix for Banded<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.iter().fold(
            0,
            |sum, (_, _, &value)| if value.is_zero() { sum } else { sum + 1 },
        )
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Banded::new(size, 0, 0)
    }
}

impl<'l, T: Element> Iterator<'l, T> {
    fn new(matrix: &'l Banded<T>) -> Iterator<'l, T> {
        Iterator {
            matrix: matrix,
            column: 0,
            start: row_start!(matrix, 0),
            finish: row_finish!(matrix, 0),
        }
    }
}

impl<'l, T: Element> iter::Iterator for Iterator<'l, T> {
    type Item = (usize, usize, &'l T);

    fn next(&mut self) -> Option<Self::Item> {
        let &mut Iterator {
            matrix,
            ref mut column,
            ref mut start,
            ref mut finish,
        } = self;
        while *column < matrix.columns {
            if *start >= *finish {
                *column += 1;
                *start = row_start!(matrix, *column);
                *finish = row_finish!(matrix, *column);
                continue;
            }
            let i = *start;
            let k = matrix.superdiagonals + i - *column;
            *start += 1;
            return Some((i, *column, &matrix.values[*column * matrix.diagonals() + k]));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn nonzeros() {
        let matrix = new!(
            7,
            4,
            2,
            2,
            matrix![
                7.0,  7.0,  3.0,  7.0;
                7.0,  2.0,  6.0, 11.0;
                1.0,  5.0, 10.0,  0.0;
                4.0,  9.0,  0.0, 16.0;
                8.0, 12.0, 15.0, 17.0;
            ]
        );
        assert_eq!(matrix.nonzeros(), 17 - 2);
    }

    #[test]
    fn iter_tall() {
        let matrix = new!(
            7,
            4,
            2,
            2,
            matrix![
                0.0,  0.0,  3.0,  7.0;
                0.0,  2.0,  6.0, 11.0;
                1.0,  5.0, 10.0, 14.0;
                4.0,  9.0, 13.0, 16.0;
                8.0, 12.0, 15.0, 17.0;
            ]
        );

        let result = matrix.iter().map(|(i, _, _)| i).collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5]
        );

        let result = matrix.iter().map(|(_, j, _)| j).collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        );

        let result = matrix
            .iter()
            .map(|(_, _, &value)| value)
            .collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![
                1.0, 4.0, 8.0, 2.0, 5.0, 9.0, 12.0, 3.0, 6.0, 10.0, 13.0, 15.0, 7.0, 11.0, 14.0,
                16.0, 17.0,
            ]
        );
    }

    #[test]
    fn iter_wide() {
        let matrix = new!(
            4,
            7,
            2,
            2,
            matrix![
                0.0,  0.0,  3.0,  7.0, 12.0, 17.0, 0.0;
                0.0,  2.0,  6.0, 11.0, 16.0,  0.0, 0.0;
                1.0,  5.0, 10.0, 15.0,  0.0,  0.0, 0.0;
                4.0,  9.0, 14.0,  0.0,  0.0,  0.0, 0.0;
                8.0, 13.0,  0.0,  0.0,  0.0,  0.0, 0.0;
            ]
        );

        let result = matrix.iter().map(|(i, _, _)| i).collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]
        );

        let result = matrix.iter().map(|(_, j, _)| j).collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
        );

        let result = matrix
            .iter()
            .map(|(_, _, &value)| value)
            .collect::<Vec<_>>();
        assert_eq!(
            &result,
            &vec![
                1.0, 4.0, 8.0, 2.0, 5.0, 9.0, 13.0, 3.0, 6.0, 10.0, 14.0, 7.0, 11.0, 15.0, 12.0,
                16.0, 17.0,
            ]
        );
    }
}
