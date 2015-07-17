//! The banded storage.
//!
//! The storage is suitable for matrices with a small number of superdiagonals
//! and/or subdiagonals relative to the smallest dimension. Data are stored in
//! the [format][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node124.html
//! [2]: http://www.netlib.org/lapack

use std::iter;

use {Conventional, Diagonal, Element, Matrix, Size};

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

/// A sparse iterator.
pub struct Iterator<'l, T: 'l + Element> {
    matrix: &'l Banded<T>,
    column: usize,
    start: usize,
    finish: usize,
}

macro_rules! debug_validate(
    ($matrix:ident) => (debug_assert!(
        $matrix.values.len() == $matrix.diagonals() * $matrix.columns
    ));
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

size!(Banded);

impl<T: Element> Banded<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, superdiagonals: usize, subdiagonals: usize) -> Self {
        let (rows, columns) = size.dimensions();
        Banded {
            rows: rows,
            columns: columns,
            superdiagonals: superdiagonals,
            subdiagonals: subdiagonals,
            values: vec![T::zero(); (superdiagonals + 1 + subdiagonals) * columns],
        }
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
        self.iter().fold(0, |sum, (_, _, &value)| if value.is_zero() { sum } else { sum + 1 })
    }

    fn transpose(&self) -> Self {
        let &Banded { rows, columns, superdiagonals, subdiagonals, .. } = self;
        let diagonals = self.diagonals();

        let mut matrix = Banded::new((columns, rows), subdiagonals, superdiagonals);
        for j in 0..columns {
            for i in row_range!(rows, superdiagonals, subdiagonals, j) {
                let k = superdiagonals + i - j;
                let l = subdiagonals + j - i;
                matrix.values[i * diagonals + l] = self.values[j * diagonals + k];
            }
        }

        matrix
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Banded::new(size, 0, 0)
    }
}

impl<'l, T: Element> From<&'l Banded<T>> for Conventional<T> {
    fn from(matrix: &'l Banded<T>) -> Self {
        debug_validate!(matrix);

        let &Banded { rows, columns, superdiagonals, subdiagonals, ref values } = matrix;
        let diagonals = matrix.diagonals();

        let mut matrix = Conventional::new((rows, columns));
        for j in 0..columns {
            for i in row_range!(rows, superdiagonals, subdiagonals, j) {
                let k = superdiagonals + i - j;
                matrix.values[j * rows + i] = values[j * diagonals + k];
            }
        }

        matrix
    }
}

impl<T: Element> From<Banded<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Banded<T>) -> Self {
        (&matrix).into()
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
        let &mut Iterator { matrix, ref mut column, ref mut start, ref mut finish } = self;
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
    use {Banded, Conventional, Diagonal, Matrix};

    macro_rules! new(
        ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr, $values:expr) => (
            Banded { rows: $rows, columns: $columns, superdiagonals: $superdiagonals,
                     subdiagonals: $subdiagonals, values: $values }
        );
    );

    #[test]
    fn nonzeros() {
        let matrix = new!(7, 4, 2, 2, vec![
            7.0,  7.0,  1.0,  4.0,  8.0,
            7.0,  2.0,  5.0,  9.0, 12.0,
            3.0,  6.0, 10.0,  0.0, 15.0,
            7.0, 11.0,  0.0, 16.0, 17.0,
        ]);
        assert_eq!(matrix.nonzeros(), 17 - 2);
    }

    #[test]
    fn transpose() {
        let matrix = new!(4, 8, 3, 1, vec![
             0.0,  0.0,  0.0,  1.0,  5.0,
             0.0,  0.0,  2.0,  6.0, 10.0,
             0.0,  3.0,  7.0, 11.0, 15.0,
             4.0,  8.0, 12.0, 16.0,  0.0,
             9.0, 13.0, 17.0,  0.0,  0.0,
            14.0, 18.0,  0.0,  0.0,  0.0,
            19.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(8, 4, 1, 3, vec![
             0.0,  1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,  9.0,
            10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0,
        ]));
    }

    #[test]
    fn iter_tall() {
        let matrix = new!(7, 4, 2, 2, vec![
            0.0,  0.0,  1.0,  4.0,  8.0,
            0.0,  2.0,  5.0,  9.0, 12.0,
            3.0,  6.0, 10.0, 13.0, 15.0,
            7.0, 11.0, 14.0, 16.0, 17.0,
        ]);

        let result = matrix.iter().map(|(i, _, _)| i).collect::<Vec<_>>();
        assert_eq!(&result, &vec![0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5]);

        let result = matrix.iter().map(|(_, j, _)| j).collect::<Vec<_>>();
        assert_eq!(&result, &vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]);

        let result = matrix.iter().map(|(_, _, &value)| value).collect::<Vec<_>>();
        assert_eq!(&result, &vec![
            1.0, 4.0, 8.0,
            2.0, 5.0, 9.0, 12.0,
            3.0, 6.0, 10.0, 13.0, 15.0,
            7.0, 11.0, 14.0, 16.0, 17.0,
        ]);
    }

    #[test]
    fn iter_wide() {
        let matrix = new!(4, 7, 2, 2, vec![
             0.0,  0.0,  1.0,  4.0,  8.0,
             0.0,  2.0,  5.0,  9.0, 13.0,
             3.0,  6.0, 10.0, 14.0,  0.0,
             7.0, 11.0, 15.0,  0.0,  0.0,
            12.0, 16.0,  0.0,  0.0,  0.0,
            17.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,
        ]);

        let result = matrix.iter().map(|(i, _, _)| i).collect::<Vec<_>>();
        assert_eq!(&result, &vec![0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]);

        let result = matrix.iter().map(|(_, j, _)| j).collect::<Vec<_>>();
        assert_eq!(&result, &vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]);

        let result = matrix.iter().map(|(_, _, &value)| value).collect::<Vec<_>>();
        assert_eq!(&result, &vec![
            1.0, 4.0, 8.0,
            2.0, 5.0, 9.0, 13.0,
            3.0, 6.0, 10.0, 14.0,
            7.0, 11.0, 15.0,
            12.0, 16.0,
            17.0,
        ]);
    }

    #[test]
    fn from_diagonal_tall() {
        let matrix = Banded::from(Diagonal::from_vec(vec![1.0, 2.0, 3.0], (5, 3)));
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_diagonal_wide() {
        let matrix = Banded::from(Diagonal::from_vec(vec![1.0, 2.0, 3.0], (3, 5)));
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn into_conventional_tall() {
        let matrix = new!(7, 4, 2, 2, vec![
            0.0,  0.0,  1.0,  4.0,  8.0,
            0.0,  2.0,  5.0,  9.0, 12.0,
            3.0,  6.0, 10.0, 13.0, 15.0,
            7.0, 11.0, 14.0, 16.0, 17.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &[
            1.0, 4.0,  8.0,  0.0,  0.0,  0.0, 0.0,
            2.0, 5.0,  9.0, 12.0,  0.0,  0.0, 0.0,
            3.0, 6.0, 10.0, 13.0, 15.0,  0.0, 0.0,
            0.0, 7.0, 11.0, 14.0, 16.0, 17.0, 0.0,
        ]);
    }

    #[test]
    fn into_conventional_wide() {
        let matrix = new!(4, 7, 2, 2, vec![
             0.0,  0.0,  1.0,  4.0,  8.0,
             0.0,  2.0,  5.0,  9.0, 13.0,
             3.0,  6.0, 10.0, 14.0,  0.0,
             7.0, 11.0, 15.0,  0.0,  0.0,
            12.0, 16.0,  0.0,  0.0,  0.0,
            17.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &[
            1.0, 4.0,  8.0,  0.0,
            2.0, 5.0,  9.0, 13.0,
            3.0, 6.0, 10.0, 14.0,
            0.0, 7.0, 11.0, 15.0,
            0.0, 0.0, 12.0, 16.0,
            0.0, 0.0,  0.0, 17.0,
            0.0, 0.0,  0.0,  0.0,
        ]);
    }
}
