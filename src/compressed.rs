use {Dense, Element, Major, Sparse};

/// A compressed matrix.
///
/// The storage is suitable for generic sparse matrices. Data are stored in one
/// of the following formats:
///
/// * the [compressed-column][1] format or
/// * the [compressed-row][2] format.
///
/// [1]: http://netlib.org/linalg/html_templates/node92.html
/// [2]: http://netlib.org/linalg/html_templates/node91.html
#[derive(Clone, Debug, PartialEq)]
pub struct Compressed<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The number of nonzero elements.
    pub nonzeros: usize,
    /// The storage format.
    pub format: Major,
    /// The values of the nonzero elements.
    pub data: Vec<T>,
    /// The indices of rows for `Major::Column` or columns for `Major::Row` of
    /// the nonzero elements.
    pub indices: Vec<usize>,
    /// The offsets of columns for `Major::Column` or rows for `Major::Row` such
    /// that the values and indices of the `i`th column for `Major::Column` or
    /// the `i`th row for `Major::Row` are stored starting from `data[j]` and
    /// `indices[j]`, respectively, where `j = offsets[i]`. The vector has one
    /// additional element, which is always equal to `nonzeros`.
    pub offsets: Vec<usize>,
}

matrix!(Compressed);

impl<T: Element> Sparse for Compressed<T> {
    #[inline]
    fn nonzeros(&self) -> usize {
        self.nonzeros
    }
}

impl<'l, T: Element> From<&'l Dense<T>> for Compressed<T> {
    fn from(dense: &'l Dense<T>) -> Compressed<T> {
        let mut data = vec![];
        let mut indices = vec![];
        let mut offsets = vec![];

        let mut k = 0;
        let zero = T::zero();
        for _ in 0..dense.columns {
            offsets.push(data.len());
            for i in 0..dense.rows {
                if dense.data[k] != zero {
                    data.push(dense.data[k]);
                    indices.push(i);
                }
                k += 1;
            }
        }
        offsets.push(data.len());

        Compressed {
            rows: dense.rows,
            columns: dense.columns,
            nonzeros: data.len(),
            format: Major::Column,
            data: data,
            indices: indices,
            offsets: offsets,
        }
    }
}

impl< T: Element> From<Dense<T>> for Compressed<T> {
    #[inline]
    fn from(dense: Dense<T>) -> Compressed<T> {
        (&dense).into()
    }
}

impl<T: Element> From<Compressed<T>> for Dense<T> {
    fn from(compressed: Compressed<T>) -> Dense<T> {
        let Compressed {
            rows, columns, nonzeros, format, ref data, ref indices, ref offsets
        } = compressed;

        debug_assert_eq!(data.len(), nonzeros);
        debug_assert_eq!(indices.len(), nonzeros);

        let mut dense = Dense {
            rows: rows,
            columns: columns,
            data: vec![T::zero(); rows * columns],
        };

        match format {
            Major::Row => {
                debug_assert_eq!(offsets.len(), rows + 1);
                for i in 0..rows {
                    for k in offsets[i]..offsets[i + 1] {
                        let j = indices[k];
                        dense.data[j * rows + i] = data[k];
                    }
                }
            },
            Major::Column => {
                debug_assert_eq!(offsets.len(), columns + 1);
                for j in 0..columns {
                    for k in offsets[j]..offsets[j + 1] {
                        let i = indices[k];
                        dense.data[j * rows + i] = data[k];
                    }
                }
            },
        }

        dense
    }
}

#[cfg(test)]
mod tests {
    use {Compressed, Dense, Major, Make, Shape};

    #[test]
    fn from_dense() {
        let dense = Dense::make(&[
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 4.0,
        ][..], Shape::Rectangular(5, 3));

        let compressed: Compressed<f64> = (&dense).into();

        assert_eq!(compressed.rows, 5);
        assert_eq!(compressed.columns, 3);
        assert_eq!(compressed.nonzeros, 4);
        assert_eq!(compressed.format, Major::Column);
        assert_eq!(&compressed.data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&compressed.indices, &[1, 3, 4, 4]);
        assert_eq!(&compressed.offsets, &[0, 1, 3, 4]);

        assert_eq!(dense, compressed.into());
    }

    #[test]
    fn into_dense() {
        let compressed = Compressed {
            rows: 5,
            columns: 3,
            nonzeros: 3,
            format: Major::Column,
            data: vec![1.0, 2.0, 3.0],
            indices: vec![0, 1, 2],
            offsets: vec![0, 1, 2, 3],
        };

        let dense: Dense<f64> = compressed.into();

        assert_eq!(&dense[..], &[
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
    }
}
