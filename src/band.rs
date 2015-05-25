//! Band matrices.
///
/// Matrices are stored in the [format][1] adopted by [LAPACK][2].
///
/// [1]: http://www.netlib.org/lapack/lug/node124.html
/// [2]: http://www.netlib.org/lapack

use dense;

/// A band matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The number of superdiagonals.
    pub superdiagonals: usize,
    /// The number of subdiagonals.
    pub subdiagonals: usize,
    /// The values of the diagonal elements.
    pub values: Vec<f64>,
}

impl From<Matrix> for dense::Matrix {
    fn from(matrix: Matrix) -> dense::Matrix {
        let Matrix { rows, columns, superdiagonals, subdiagonals, ref values } = matrix;

        let diagonals = superdiagonals + 1 + subdiagonals;
        debug_assert_eq!(values.len(), diagonals * columns);

        let mut dense = dense::Matrix {
            rows: rows,
            columns: columns,
            values: vec![0.0; rows * columns],
        };

        for k in 1..(superdiagonals + 1) {
            for j in k..columns {
                let i = j - k;
                if i >= rows { break; }
                dense.values[j * rows + i] = values[j * diagonals + superdiagonals - k];
            }
        }
        for i in 0..columns {
            if i >= rows || i >= columns { break; }
            dense.values[i * rows + i] = values[i * diagonals + superdiagonals];
        }
        for k in 1..(subdiagonals + 1) {
            for j in 0..columns {
                let i = j + k;
                if i >= rows { break; }
                dense.values[j * rows + i] = values[j * diagonals + superdiagonals + k];
            }
        }

        dense
    }
}

#[cfg(test)]
mod tests {
    use {assert, dense};

    #[test]
    fn into_tall_dense() {
        let matrix = super::Matrix {
            rows: 7,
            columns: 4,
            superdiagonals: 2,
            subdiagonals: 2,
            values: vec![
                0.0,  0.0,  1.0,  4.0,  8.0,
                0.0,  2.0,  5.0,  9.0, 12.0,
                3.0,  6.0, 10.0, 13.0, 15.0,
                7.0, 11.0, 14.0, 16.0, 17.0,
            ],
        };

        let matrix: dense::Matrix = matrix.into();

        assert::equal(&matrix[..], &vec![
            1.0, 4.0,  8.0,  0.0,  0.0,  0.0, 0.0,
            2.0, 5.0,  9.0, 12.0,  0.0,  0.0, 0.0,
            3.0, 6.0, 10.0, 13.0, 15.0,  0.0, 0.0,
            0.0, 7.0, 11.0, 14.0, 16.0, 17.0, 0.0,
        ]);
    }

    #[test]
    fn into_wide_dense() {
        let matrix = super::Matrix {
            rows: 4,
            columns: 7,
            superdiagonals: 2,
            subdiagonals: 2,
            values: vec![
                 0.0,  0.0,  1.0,  4.0,  8.0,
                 0.0,  2.0,  5.0,  9.0, 13.0,
                 3.0,  6.0, 10.0, 14.0,  0.0,
                 7.0, 11.0, 15.0,  0.0,  0.0,
                12.0, 16.0,  0.0,  0.0,  0.0,
                17.0,  0.0,  0.0,  0.0,  0.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,
            ],
        };

        let matrix: dense::Matrix = matrix.into();

        assert::equal(&matrix[..], &vec![
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
