/// A sparse matrix.
#[derive(Debug)]
pub enum Sparse {
    /// A sparse matrix stored using the [compressed-column format][1].
    ///
    /// [1]: http://netlib.org/linalg/html_templates/node92.html
    CompressedColumn(CompressedData),

    /// A sparse matrix stored using the [compressed-row format][1].
    ///
    /// [1]: http://netlib.org/linalg/html_templates/node91.html
    CompressedRow(CompressedData),
}

/// A sparse storage in the compressed-column or compressed-row format.
#[derive(Debug)]
pub struct CompressedData {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The number of nonzero elements.
    pub nonzeros: usize,
    /// The values of the nonzero elements.
    pub values: Vec<f64>,
    /// The indices of columns or rows the nonzero elements.
    pub indices: Vec<usize>,
    /// The offsets of columns (rows) such that the values and indices of the `i`th column (row)
    /// are stored starting from `values[j]` and `indices[j]`, respectively, where `j =
    /// offsets[i]`. The vector has one additional element, which is always equal to `nonzeros`,
    /// that is, `offsets[columns] = nonzeros` (`offsets[rows] = nonzeros`).
    pub offsets: Vec<usize>,
}
