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

macro_rules! debug_valid(
    ($matrix:ident) => (debug_assert!(
        $matrix.nonzeros == $matrix.values.len() &&
        $matrix.nonzeros == $matrix.indices.len() &&
        match $matrix.format {
            Major::Column => $matrix.columns + 1 == $matrix.offsets.len(),
            Major::Row => $matrix.rows + 1 == $matrix.offsets.len(),
        }
    ));
);

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
    fn from(matrix: &'l Dense<T>) -> Compressed<T> {
        let mut values = vec![];
        let mut indices = vec![];
        let mut offsets = vec![];

        let mut k = 0;
        let zero = T::zero();
        for _ in 0..matrix.columns {
            offsets.push(values.len());
            for i in 0..matrix.rows {
                if matrix.values[k] != zero {
                    values.push(matrix.values[k]);
                    indices.push(i);
                }
                k += 1;
            }
        }
        offsets.push(values.len());

        Compressed {
            rows: matrix.rows,
            columns: matrix.columns,
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
    fn from(matrix: Dense<T>) -> Compressed<T> {
        (&matrix).into()
    }
}

impl<'l, T: Element> From<&'l Compressed<T>> for Dense<T> {
    fn from(matrix: &'l Compressed<T>) -> Dense<T> {
        debug_valid!(matrix);

        let &Compressed {
            rows, columns, format, ref values, ref indices, ref offsets, ..
        } = matrix;

        let mut matrix = Dense {
            rows: rows,
            columns: columns,
            values: vec![T::zero(); rows * columns],
        };

        match format {
            Major::Row => {
                for i in 0..rows {
                    for k in offsets[i]..offsets[i + 1] {
                        matrix.values[indices[k] * rows + i] = values[k];
                    }
                }
            },
            Major::Column => {
                for j in 0..columns {
                    for k in offsets[j]..offsets[j + 1] {
                        matrix.values[j * rows + indices[k]] = values[k];
                    }
                }
            },
        }

        matrix
    }
}

impl<T: Element> From<Compressed<T>> for Dense<T> {
    fn from(matrix: Compressed<T>) -> Dense<T> {
        (&matrix).into()
    }
}

#[cfg(test)]
mod tests {
    use {Compressed, Dense, Major, Shape};

    macro_rules! new(
        ($rows:expr, $columns:expr, $nonzeros:expr, $format:expr,
         $values:expr, $indices:expr, $offsets:expr) => (
            Compressed { rows: $rows, columns: $columns, nonzeros: $nonzeros, format: $format,
                         values: $values, indices: $indices, offsets: $offsets }
        );
    );

    #[test]
    fn from_dense() {
        let matrix = Dense::from_vec(vec![
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 4.0,
        ], Shape::Rectangular(5, 3));

        let matrix: Compressed<_> = matrix.into();

        assert_eq!(matrix, new!(5, 3, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                vec![1, 3, 4, 4], vec![0, 1, 3, 4]));
    }

    #[test]
    fn into_dense() {
        let matrix = new!(5, 3, 3, Major::Column, vec![1.0, 2.0, 3.0],
                          vec![0, 1, 2], vec![0, 1, 2, 3]);

        let matrix: Dense<_> = matrix.into();

        assert_eq!(&matrix[..], &[
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn resize_fewer_columns() {
        let mut matrix = new!(5, 7, 5, Major::Column, vec![1.0, 2.0, 3.0, 4.0, 5.0],
                              vec![1, 0, 3, 1, 4], vec![0, 0, 0, 1, 2, 2, 3, 5]);

        matrix.resize(5, 5);
        assert_eq!(matrix, new!(5, 5, 2, Major::Column, vec![1.0, 2.0],
                                vec![1, 0], vec![0, 0, 0, 1, 2, 2]));

        matrix.resize(5, 3);
        assert_eq!(matrix, new!(5, 3, 1, Major::Column, vec![1.0],
                                vec![1], vec![0, 0, 0, 1]));

        matrix.resize(5, 1);
        assert_eq!(matrix, new!(5, 1, 0, Major::Column, vec![],
                                vec![], vec![0, 0]));
    }

    #[test]
    fn resize_fewer_rows() {
        let mut matrix = new!(5, 7, 5, Major::Column, vec![1.0, 2.0, 3.0, 4.0, 5.0],
                              vec![1, 0, 3, 1, 4], vec![0, 0, 0, 1, 2, 2, 3, 5]);

        matrix.resize(3, 7);
        assert_eq!(matrix, new!(3, 7, 3, Major::Column, vec![1.0, 2.0, 4.0],
                                vec![1, 0, 1], vec![0, 0, 0, 1, 2, 2, 2, 3]));

        matrix.resize(1, 7);
        assert_eq!(matrix, new!(1, 7, 1, Major::Column, vec![2.0],
                                vec![0], vec![0, 0, 0, 0, 1, 1, 1, 1]));
    }

    #[test]
    fn resize_more_columns() {
        let mut matrix = new!(5, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                              vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]);

        matrix.resize(5, 9);
        assert_eq!(matrix, new!(5, 9, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4, 4, 4]));

        matrix.resize(5, 11);
        assert_eq!(matrix, new!(5, 11, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 4]));
    }

    #[test]
    fn resize_more_rows() {
        let mut matrix = new!(5, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                              vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]);

        matrix.resize(7, 7);
        assert_eq!(matrix, new!(7, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]));

        matrix.resize(9, 7);
        assert_eq!(matrix, new!(9, 7, 4, Major::Column, vec![1.0, 2.0, 3.0, 4.0],
                                vec![1, 1, 3, 4], vec![0, 0, 0, 1, 2, 2, 3, 4]));
    }
}
