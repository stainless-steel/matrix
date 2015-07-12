use {Dense, Element, Sparse};

macro_rules! min(
    ($left:expr, $right:expr) => ({
        let (left, right) = ($left, $right);
        if left < right { left } else { right }
    });
);

/// A band matrix.
///
/// The storage is suitable for matrices with a small number of superdiagonals
/// and/or subdiagonals relative to the smallest dimension. Data are stored in
/// the [format][1] adopted by [LAPACK][2].
///
/// [1]: http://www.netlib.org/lapack/lug/node124.html
/// [2]: http://www.netlib.org/lapack
#[derive(Clone, Debug)]
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
    pub data: Vec<T>,
}

matrix!(Band);

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

impl<T: Element> From<Band<T>> for Dense<T> {
    fn from(band: Band<T>) -> Dense<T> {
        let Band { rows, columns, superdiagonals, subdiagonals, ref data } = band;

        let diagonals = superdiagonals + 1 + subdiagonals;
        debug_assert_eq!(data.len(), diagonals * columns);

        let mut dense = Dense {
            rows: rows,
            columns: columns,
            data: vec![T::zero(); rows * columns],
        };

        for k in 0..(superdiagonals + 1) {
            for i in 0..min!(columns - k, rows) {
                let j = i + k;
                dense.data[j * rows + i] = data[j * diagonals + superdiagonals - k];
            }
        }
        for k in 1..(subdiagonals + 1) {
            for i in k..min!(columns + k, rows) {
                let j = i - k;
                dense.data[j * rows + i] = data[j * diagonals + superdiagonals + k];
            }
        }

        dense
    }
}

#[cfg(test)]
mod tests {
    use {Band, Dense, Sparse};

    macro_rules! new(
        ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr, $data:expr) => (
            Band {
                rows: $rows,
                columns: $columns,
                superdiagonals: $superdiagonals,
                subdiagonals: $subdiagonals,
                data: $data,
            }
        );
        ($rows:expr, $columns:expr, $superdiagonals:expr, $subdiagonals:expr) => (
            new!($rows, $columns, $superdiagonals, $subdiagonals,
                 vec![0.0; ($superdiagonals + 1 + $subdiagonals) * $columns])
        );
    );

    #[test]
    fn into_tall_dense() {
        let band = new!(7, 4, 2, 2, vec![
            0.0,  0.0,  1.0,  4.0,  8.0,
            0.0,  2.0,  5.0,  9.0, 12.0,
            3.0,  6.0, 10.0, 13.0, 15.0,
            7.0, 11.0, 14.0, 16.0, 17.0,
        ]);

        let dense: Dense<f64> = band.into();

        assert_eq!(&dense[..], &[
            1.0, 4.0,  8.0,  0.0,  0.0,  0.0, 0.0,
            2.0, 5.0,  9.0, 12.0,  0.0,  0.0, 0.0,
            3.0, 6.0, 10.0, 13.0, 15.0,  0.0, 0.0,
            0.0, 7.0, 11.0, 14.0, 16.0, 17.0, 0.0,
        ]);
    }

    #[test]
    fn into_wide_dense() {
        let band = new!(4, 7, 2, 2, vec![
             0.0,  0.0,  1.0,  4.0,  8.0,
             0.0,  2.0,  5.0,  9.0, 13.0,
             3.0,  6.0, 10.0, 14.0,  0.0,
             7.0, 11.0, 15.0,  0.0,  0.0,
            12.0, 16.0,  0.0,  0.0,  0.0,
            17.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,
        ]);

        let dense: Dense<f64> = band.into();

        assert_eq!(&dense[..], &[
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
