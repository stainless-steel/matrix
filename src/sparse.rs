//! Sparse matrices.
//!
//! Two formats of sparse storage are currently supported:
//!
//! * the [compressed-row][1] format,
//! * the [compressed-column][2] format, and
//! * the [compressed-diagonal][1] format.
//!
//! [1]: http://netlib.org/linalg/html_templates/node91.html
//! [2]: http://netlib.org/linalg/html_templates/node92.html
//! [3]: http://netlib.org/linalg/html_templates/node94.html

use std::convert::Into;

use {generic, dense};

/// A sparse matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The actual data.
    pub data: Data,
}

/// Data of a sparse matrix.
#[derive(Debug)]
pub enum Data {
    /// Data stored using the compressed-row format.
    Row(Dimension),
    /// Data stored using the compressed-column format.
    Column(Dimension),
    /// Data stored using the compressed-diagonal format.
    Diagonal(Diagonal),
}

/// Data stored in the compressed-column or compressed-row format.
#[derive(Debug)]
pub struct Dimension {
    /// The number of nonzero elements.
    pub nonzeros: usize,
    /// The values of the nonzero elements.
    pub values: Vec<f64>,
    /// The indices of columns (rows) the nonzero elements.
    pub indices: Vec<usize>,
    /// The offsets of columns (rows) such that the values and indices of the `i`th column (row)
    /// are stored starting from `values[j]` and `indices[j]`, respectively, where `j =
    /// offsets[i]`. The vector has one additional element, which is always equal to `nonzeros`,
    /// that is, `offsets[columns] = nonzeros` (`offsets[rows] = nonzeros`).
    pub offsets: Vec<usize>,
}

/// Data stored in the compressed-diagonal format.
#[derive(Debug)]
pub struct Diagonal {
    /// The number of subdiagonals.
    pub subdiagonals: usize,
    /// The number of superdiagonals.
    pub superdiagonals: usize,
    /// The values of the diagonals. The head elements of subdiagonals and tail elements of
    /// superdiagonals are not used but should be present.
    pub values: Vec<f64>,
}

impl Matrix {
}

impl generic::Matrix for Matrix {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn columns(&self) -> usize {
        self.columns
    }
}

impl Into<Data> for Matrix {
    #[inline]
    fn into(self) -> Data {
        self.data
    }
}

impl From<Matrix> for dense::Matrix {
    fn from(sparse: Matrix) -> dense::Matrix {
        let (rows, columns) = (sparse.rows, sparse.columns);

        let mut dense = dense::Matrix {
            rows: rows,
            columns: columns,
            data: vec![0.0; rows * columns],
        };

        match sparse.data {
            Data::Row(Dimension { nonzeros, ref values, ref indices, ref offsets }) => {
                debug_assert_eq!(values.len(), nonzeros);
                debug_assert_eq!(indices.len(), nonzeros);
                debug_assert_eq!(offsets.len(), rows + 1);
                for i in 0..rows {
                    for k in offsets[i]..offsets[i + 1] {
                        let j = indices[k];
                        dense.data[j * rows + i] = values[k];
                    }
                }
            },
            Data::Column(Dimension { nonzeros, ref values, ref indices, ref offsets }) => {
                debug_assert_eq!(values.len(), nonzeros);
                debug_assert_eq!(indices.len(), nonzeros);
                debug_assert_eq!(offsets.len(), columns + 1);
                for j in 0..columns {
                    for k in offsets[j]..offsets[j + 1] {
                        let i = indices[k];
                        dense.data[j * rows + i] = values[k];
                    }
                }
            },
            Data::Diagonal(Diagonal { subdiagonals: l, superdiagonals: u, ref values }) => {
                let diagonals = l + 1 + u;
                debug_assert_eq!(values.len(), diagonals * columns);
                for k in 1..(u + 1) {
                    for j in k..columns {
                        let i = j - k;
                        if i >= rows { break; }
                        dense.data[j * rows + i] = values[j * diagonals + u - k];
                    }
                }
                for i in 0..columns {
                    if i >= rows || i >= columns { break; }
                    dense.data[i * rows + i] = values[i * diagonals + u];
                }
                for k in 1..(l + 1) {
                    for j in 0..columns {
                        let i = j + k;
                        if i >= rows { break; }
                        dense.data[j * rows + i] = values[j * diagonals + u + k];
                    }
                }
            },
        }

        dense
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use super::{Matrix, Data, Dimension, Diagonal};

    #[test]
    fn column_into() {
        use dense;

        let matrix = Matrix {
            rows: 5,
            columns: 3,
            data: Data::Column(Dimension {
                nonzeros: 3,
                values: vec![1.0, 2.0, 3.0],
                indices: vec![0, 1, 2],
                offsets: vec![0, 1, 2, 3],
            }),
        };

        let matrix: dense::Matrix = matrix.into();

        assert::equal(&matrix[..], &vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn diagonal_into() {
        use dense;

        let matrix = Matrix {
            rows: 7,
            columns: 4,
            data: Data::Diagonal(Diagonal {
                subdiagonals: 2,
                superdiagonals: 2,
                values: vec![
                    0.0,  0.0,  1.0,  4.0,  8.0,
                    0.0,  2.0,  5.0,  9.0, 12.0,
                    3.0,  6.0, 10.0, 13.0, 15.0,
                    7.0, 11.0, 14.0, 16.0, 17.0,
                ],
            }),
        };

        let matrix: dense::Matrix = matrix.into();

        assert::equal(&matrix[..], &vec![
            1.0, 4.0,  8.0,  0.0,  0.0,  0.0, 0.0,
            2.0, 5.0,  9.0, 12.0,  0.0,  0.0, 0.0,
            3.0, 6.0, 10.0, 13.0, 15.0,  0.0, 0.0,
            0.0, 7.0, 11.0, 14.0, 16.0, 17.0, 0.0,
        ]);

        let matrix = Matrix {
            rows: 4,
            columns: 7,
            data: Data::Diagonal(Diagonal {
                subdiagonals: 2,
                superdiagonals: 2,
                values: vec![
                     0.0,  0.0,  1.0,  4.0,  8.0,
                     0.0,  2.0,  5.0,  9.0, 13.0,
                     3.0,  6.0, 10.0, 14.0,  0.0,
                     7.0, 11.0, 15.0,  0.0,  0.0,
                    12.0, 16.0,  0.0,  0.0,  0.0,
                    17.0,  0.0,  0.0,  0.0,  0.0,
                     0.0,  0.0,  0.0,  0.0,  0.0,
                ],
            }),
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
