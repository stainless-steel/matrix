use {Dense, Diagonal, Element, Major, Sparse};

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
    pub values: Vec<T>,
    /// The indices of rows for `Major::Column` or columns for `Major::Row` of
    /// the nonzero elements.
    pub indices: Vec<usize>,
    /// The offsets of columns for `Major::Column` or rows for `Major::Row` such
    /// that the values and indices of the `i`th column for `Major::Column` or
    /// the `i`th row for `Major::Row` are stored starting from `values[j]` and
    /// `indices[j]`, respectively, where `j = offsets[i]`. The vector has one
    /// additional element, which is always equal to `nonzeros`.
    pub offsets: Vec<usize>,
}

impl<T: Element> Compressed<T> {
    /// Resize the matrix.
    pub fn resize(&mut self, rows: usize, columns: usize) {
        self.retain(|i, j, _| i < rows && j < columns);
        let (from, into) = match self.format {
            Major::Column => (self.columns, columns),
            Major::Row => (self.rows, rows),
        };
        if from > into {
            self.offsets.truncate(into + 1);
        } else if from < into {
            self.offsets.extend(vec![self.nonzeros; into - from]);
        }
        self.columns = columns;
        self.rows = rows;
    }

    /// Retain only those elements that satisfy a condition.
    pub fn retain<F>(&mut self, mut condition: F) where F: FnMut(usize, usize, &T) -> bool {
        let mut i = 0;
        let mut k = 0;
        while k < self.indices.len() {
            match self.format {
                Major::Column => {
                    while i < self.columns && self.offsets[i + 1] <= k {
                        i += 1;
                    }
                    if condition(self.indices[k], i, &self.values[k]) {
                        k += 1;
                        continue;
                    }
                },
                Major::Row => {
                    while i < self.rows && self.offsets[i + 1] <= k {
                        i += 1;
                    }
                    if condition(i, self.indices[k], &self.values[k]) {
                        k += 1;
                        continue;
                    }
                },
            }
            self.nonzeros -= 1;
            self.values.remove(k);
            self.indices.remove(k);
            for offset in self.offsets.iter_mut().rev() {
                if k >= *offset {
                    break;
                }
                *offset -= 1;
            }
        }
    }
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
        let mut values = vec![];
        let mut indices = vec![];
        let mut offsets = vec![];

        let mut k = 0;
        let zero = T::zero();
        for _ in 0..dense.columns {
            offsets.push(values.len());
            for i in 0..dense.rows {
                if dense.values[k] != zero {
                    values.push(dense.values[k]);
                    indices.push(i);
                }
                k += 1;
            }
        }
        offsets.push(values.len());

        Compressed {
            rows: dense.rows,
            columns: dense.columns,
            nonzeros: values.len(),
            format: Major::Column,
            values: values,
            indices: indices,
            offsets: offsets,
        }
    }
}

impl<T: Element> From<Dense<T>> for Compressed<T> {
    #[inline]
    fn from(dense: Dense<T>) -> Compressed<T> {
        (&dense).into()
    }
}

impl<'l, T: Element> From<&'l Compressed<T>> for Dense<T> {
    fn from(compressed: &'l Compressed<T>) -> Dense<T> {
        let &Compressed {
            rows, columns, nonzeros, format, ref values, ref indices, ref offsets
        } = compressed;

        debug_assert_eq!(values.len(), nonzeros);
        debug_assert_eq!(indices.len(), nonzeros);

        let mut dense = Dense {
            rows: rows,
            columns: columns,
            values: vec![T::zero(); rows * columns],
        };

        match format {
            Major::Row => {
                debug_assert_eq!(offsets.len(), rows + 1);
                for i in 0..rows {
                    for k in offsets[i]..offsets[i + 1] {
                        dense.values[indices[k] * rows + i] = values[k];
                    }
                }
            },
            Major::Column => {
                debug_assert_eq!(offsets.len(), columns + 1);
                for j in 0..columns {
                    for k in offsets[j]..offsets[j + 1] {
                        dense.values[j * rows + indices[k]] = values[k];
                    }
                }
            },
        }

        dense
    }
}

impl<T: Element> From<Compressed<T>> for Dense<T> {
    fn from(compressed: Compressed<T>) -> Dense<T> {
        (&compressed).into()
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Compressed<T> {
    #[inline]
    fn from(diagonal: &'l Diagonal<T>) -> Compressed<T> {
        diagonal.clone().into()
    }
}

impl<T: Element> From<Diagonal<T>> for Compressed<T> {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Compressed<T> {
        let Diagonal { rows, columns, values } = diagonal;
        let nonzeros = values.len();
        debug_assert_eq!(nonzeros, min!(rows, columns));
        Compressed {
            rows: rows,
            columns: columns,
            nonzeros: nonzeros,
            values: values,
            format: Major::Column,
            indices: (0..nonzeros).collect(),
            offsets: (0..(nonzeros + 1)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use {Compressed, Dense, Diagonal, Major, Make, Shape};

    macro_rules! new(
        ($rows:expr, $columns:expr, $nonzeros:expr, $format:expr,
         $values:expr, $indices:expr, $offsets:expr) => (
            Compressed { rows: $rows, columns: $columns, nonzeros: $nonzeros, format: $format,
                         values: $values, indices: $indices, offsets: $offsets }
        );
    );

    #[test]
    fn from_dense() {
        let dense = Dense::make(&[
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 4.0,
        ][..], Shape::Rectangular(5, 3));

        let compressed: Compressed<_> = (&dense).into();

        assert_eq!(compressed, new!(5, 3, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                    vec![1, 3, 4, 4], vec![0, 1, 3, 4]));

        assert_eq!(dense, compressed.into());
    }

    #[test]
    fn into_dense() {
        let compressed = new!(5, 3, 3, Major::Column, vec![1.0, 2.0, 3.0],
                              vec![0, 1, 2], vec![0, 1, 2, 3]);

        let dense: Dense<_> = compressed.into();

        assert_eq!(&dense[..], &[
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn from_diagonal() {
        let diagonal = Diagonal { rows: 5, columns: 3, values: vec![1.0, 2.0, 0.0] };

        let compressed: Compressed<_> = diagonal.into();

        assert_eq!(compressed, new!(5, 3, 3, Major::Column, vec![1.0, 2.0, 0.0],
                                    vec![0, 1, 2], vec![0, 1, 2, 3]));
    }

    #[test]
    fn resize_fewer_columns() {
        let mut compressed = new!(5, 7, 5, Major::Column, vec![1.0, 2.0, 3.0, 4.0, 5.0],
                                  vec![1, 0, 3, 1, 4], vec![0, 0, 0, 1, 2, 2, 3, 5]);

        compressed.resize(5, 5);
        assert_eq!(compressed, new!(5, 5, 2, Major::Column, vec![1.0, 2.0],
                               vec![1, 0], vec![0, 0, 0, 1, 2, 2]));

        compressed.resize(5, 3);
        assert_eq!(compressed, new!(5, 3, 1, Major::Column, vec![1.0],
                               vec![1], vec![0, 0, 0, 1]));

        compressed.resize(5, 1);
        assert_eq!(compressed, new!(5, 1, 0, Major::Column, vec![],
                               vec![], vec![0, 0]));
    }

    #[test]
    fn resize_fewer_rows() {
        let mut compressed = new!(5, 7, 5, Major::Column, vec![1.0, 2.0, 3.0, 4.0, 5.0],
                                  vec![1, 0, 3, 1, 4], vec![0, 0, 0, 1, 2, 2, 3, 5]);

        compressed.resize(3, 7);
        assert_eq!(compressed, new!(3, 7, 3, Major::Column, vec![1.0, 2.0, 4.0],
                                    vec![1, 0, 1], vec![0, 0, 0, 1, 2, 2, 2, 3]));

        compressed.resize(1, 7);
        assert_eq!(compressed, new!(1, 7, 1, Major::Column, vec![2.0],
                                    vec![0], vec![0, 0, 0, 0, 1, 1, 1, 1]));
    }

    #[test]
    fn resize_more_columns() {
        let mut compressed = new!(5, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                  vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]);

        compressed.resize(5, 9);
        assert_eq!(compressed, new!(5, 9, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                               vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4, 4, 4]));

        compressed.resize(5, 11);
        assert_eq!(compressed, new!(5, 11, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                               vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 4]));
    }

    #[test]
    fn resize_more_rows() {
        let mut compressed = new!(5, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                  vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]);

        compressed.resize(7, 7);
        assert_eq!(compressed, new!(7, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                               vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]));

        compressed.resize(9, 7);
        assert_eq!(compressed, new!(9, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                               vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]));
    }
}
