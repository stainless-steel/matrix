//! Sparse matrices.
//!
//! Two formats of sparse storage are currently supported:
//!
//! * the [compressed-row][1] format and
//! * the [compressed-column][2] format.
//!
//! [1]: http://netlib.org/linalg/html_templates/node91.html
//! [2]: http://netlib.org/linalg/html_templates/node92.html

use std::convert::Into;

use {generic, dense};

/// A sparse matrix.
#[derive(Debug)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The number of nonzero elements.
    pub nonzeros: usize,
    /// The actual data.
    pub data: Data,
}

/// Data of a sparse matrix.
#[derive(Debug)]
pub enum Data {
    /// Data stored using the compressed-row format.
    CompressedRow(CompressedDimension),
    /// Data stored using the compressed-column format.
    CompressedColumn(CompressedDimension),
}

/// Data stored in the compressed-column or compressed-row format.
#[derive(Debug)]
pub struct CompressedDimension {
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

impl Matrix {
    #[inline]
    pub fn nonzeros(&self) -> usize {
        self.nonzeros
    }
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
        let (rows, columns, nonzeros) = (sparse.rows, sparse.columns, sparse.nonzeros);

        let mut dense = dense::Matrix {
            rows: rows,
            columns: columns,
            data: vec![0.0; rows * columns],
        };

        if nonzeros == 0 {
            return dense;
        }

        match sparse.data {
            Data::CompressedRow(ref data) => {
                debug_assert_eq!(data.values.len(), nonzeros);
                debug_assert_eq!(data.indices.len(), nonzeros);
                debug_assert_eq!(data.offsets.len(), rows + 1);
                for i in 0..rows {
                    for k in data.offsets[i]..data.offsets[i + 1] {
                        let j = data.indices[k];
                        dense.data[j * rows + i] = data.values[k];
                    }
                }
            },
            Data::CompressedColumn(ref data) => {
                debug_assert_eq!(data.values.len(), nonzeros);
                debug_assert_eq!(data.indices.len(), nonzeros);
                debug_assert_eq!(data.offsets.len(), columns + 1);
                for j in 0..columns {
                    for k in data.offsets[j]..data.offsets[j + 1] {
                        let i = data.indices[k];
                        dense.data[j * rows + i] = data.values[k];
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

    use super::{Matrix, Data, CompressedDimension};

    #[test]
    fn into() {
        use dense;

        let matrix = Matrix {
            rows: 5,
            columns: 3,
            nonzeros: 3,
            data: Data::CompressedColumn(CompressedDimension {
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
}
