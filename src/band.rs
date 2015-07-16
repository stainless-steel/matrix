//! Band matrices.
//!
//! The storage is suitable for matrices with a small number of superdiagonals
//! and/or subdiagonals relative to the smallest dimension. Data are stored in
//! the [format][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node124.html
//! [2]: http://www.netlib.org/lapack

use std::iter;

use {Dense, Element, Matrix, Size, Sparse};

/// A band matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Band<T: Element> {
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

/// A sparse iterator of a band matrix.
pub struct Iterator<'l, T: 'l + Element> {
    matrix: &'l Band<T>,
    taken: usize,
}

macro_rules! debug_valid(
    ($matrix:ident) => (debug_assert!(
        $matrix.values.len() == $matrix.diagonals() * $matrix.columns
    ));
);

size!(Band);

impl<T: Element> Band<T> {
    /// Return the number of diagonals.
    #[inline]
    pub fn diagonals(&self) -> usize {
        self.superdiagonals + 1 + self.subdiagonals
    }

    /// Return a sparse iterator.
    #[inline]
    pub fn iter<'l>(&'l self) -> Iterator<'l, T> {
        Iterator { matrix: self, taken: 0 }
    }
}

impl<T: Element> Matrix for Band<T> {
    type Element = T;

    fn zero<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        Band {
            rows: rows,
            columns: columns,
            superdiagonals: 0,
            subdiagonals: 0,
            values: vec![T::zero(); min!(rows, columns)],
        }
    }
}

impl<T: Element> Sparse for Band<T> {
    fn nonzeros(&self) -> usize {
        let &Band { rows, columns, superdiagonals, subdiagonals, .. } = self;
        let mut count = 0;
        for k in 0..(superdiagonals + 1) {
            count += min!(columns - k, rows);
        }
        for k in 1..(subdiagonals + 1) {
            count += min!(columns, rows - k);
        }
        count
    }
}

impl<'l, T: Element> From<&'l Band<T>> for Dense<T> {
    fn from(matrix: &'l Band<T>) -> Self {
        debug_valid!(matrix);

        let &Band { rows, columns, superdiagonals, subdiagonals, ref values } = matrix;
        let diagonals = matrix.diagonals();

        let mut matrix = Dense {
            rows: rows,
            columns: columns,
            values: vec![T::zero(); rows * columns],
        };

        for k in 0..(superdiagonals + 1) {
            for i in 0..min!(columns - k, rows) {
                let j = i + k;
                matrix.values[j * rows + i] = values[j * diagonals + superdiagonals - k];
            }
        }
        for k in 1..(subdiagonals + 1) {
            for i in k..min!(columns + k, rows) {
                let j = i - k;
                matrix.values[j * rows + i] = values[j * diagonals + superdiagonals + k];
            }
        }

        matrix
    }
}

impl<T: Element> From<Band<T>> for Dense<T> {
    fn from(matrix: Band<T>) -> Self {
        (&matrix).into()
    }
}

impl<'l, T: Element> iter::Iterator for Iterator<'l, T> {
    type Item = (usize, usize, &'l T);

    fn next(&mut self) -> Option<Self::Item> {
        let &mut Iterator { matrix, ref mut taken } = self;
        let (diagonals, storage) = (matrix.diagonals(), matrix.values.len());
        while *taken < storage {
            let k = *taken;
            let j = k / diagonals;
            let i = (j + k % diagonals) as isize - matrix.superdiagonals as isize;
            *taken += 1;
            if i >= 0 && i < matrix.rows as isize {
                return Some((i as usize, j, &matrix.values[k]));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use {Band, Dense, Sparse};

    macro_rules! new(
        ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr, $values:expr) => (
            Band { rows: $rows, columns: $columns, superdiagonals: $superdiagonals,
                   subdiagonals: $subdiagonals, values: $values }
        );
        ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr) => (
            new!($rows, $columns, $superdiagonals, $subdiagonals,
                 vec![0.0; ($superdiagonals + 1 + $subdiagonals) * $columns])
        );
    );

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
    fn into_dense_tall() {
        let matrix = new!(7, 4, 2, 2, vec![
            0.0,  0.0,  1.0,  4.0,  8.0,
            0.0,  2.0,  5.0,  9.0, 12.0,
            3.0,  6.0, 10.0, 13.0, 15.0,
            7.0, 11.0, 14.0, 16.0, 17.0,
        ]);

        let matrix: Dense<_> = matrix.into();

        assert_eq!(&*matrix, &[
            1.0, 4.0,  8.0,  0.0,  0.0,  0.0, 0.0,
            2.0, 5.0,  9.0, 12.0,  0.0,  0.0, 0.0,
            3.0, 6.0, 10.0, 13.0, 15.0,  0.0, 0.0,
            0.0, 7.0, 11.0, 14.0, 16.0, 17.0, 0.0,
        ]);
    }

    #[test]
    fn into_dense_wide() {
        let matrix = new!(4, 7, 2, 2, vec![
             0.0,  0.0,  1.0,  4.0,  8.0,
             0.0,  2.0,  5.0,  9.0, 13.0,
             3.0,  6.0, 10.0, 14.0,  0.0,
             7.0, 11.0, 15.0,  0.0,  0.0,
            12.0, 16.0,  0.0,  0.0,  0.0,
            17.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,
        ]);

        let matrix: Dense<_> = matrix.into();

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

    #[test]
    fn nonzeros() {
        assert_eq!(new!(4, 4, 0, 0).nonzeros(), 4);
        assert_eq!(new!(4, 4, 1, 0).nonzeros(), 7);
        assert_eq!(new!(4, 4, 2, 0).nonzeros(), 9);
        assert_eq!(new!(4, 4, 0, 1).nonzeros(), 7);
        assert_eq!(new!(4, 4, 0, 2).nonzeros(), 9);

        assert_eq!(new!(4, 5, 0, 0).nonzeros(), 4);
        assert_eq!(new!(4, 5, 1, 0).nonzeros(), 8);
        assert_eq!(new!(4, 5, 2, 0).nonzeros(), 11);
        assert_eq!(new!(4, 5, 0, 1).nonzeros(), 7);
        assert_eq!(new!(4, 5, 0, 2).nonzeros(), 9);

        assert_eq!(new!(5, 4, 0, 0).nonzeros(), 4);
        assert_eq!(new!(5, 4, 1, 0).nonzeros(), 7);
        assert_eq!(new!(5, 4, 2, 0).nonzeros(), 9);
        assert_eq!(new!(5, 4, 0, 1).nonzeros(), 8);
        assert_eq!(new!(5, 4, 0, 2).nonzeros(), 11);
    }
}
