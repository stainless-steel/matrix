//! Triangular matrices.
//!
//! Apart from triangular matrices, the storage is suitable for symmetric and
//! Hermitian matrices. Data are stored in the [format][1] adopted by
//! [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node123.html
//! [2]: http://www.netlib.org/lapack

use {Dense, Element, Matrix, Size};

/// A triangular matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Triangular<T: Element> {
    /// The number of rows or columns.
    pub size: usize,
    /// The storage format.
    pub format: Format,
    /// The values stored in the column-major order.
    pub values: Vec<T>,
}

/// A format of a triangular matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// The lower-triangular format.
    Lower,
    /// The upper-triangular format.
    Upper,
}

macro_rules! debug_valid(
    ($matrix:ident) => (debug_assert!(
        $matrix.values.len() == $matrix.size * ($matrix.size + 1) / 2
    ));
);

size!(Triangular, size, size);

impl<T: Element> Matrix for Triangular<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        let zero = T::zero();
        self.values.iter().fold(0, |sum, &value| if value != zero { sum + 1 } else { sum })
    }

    fn zero<S: Size>(size: S) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        Triangular {
            size: rows,
            format: Format::Lower,
            values: vec![T::zero(); rows * (rows + 1) / 2],
        }
    }
}

impl<'l, T: Element> From<&'l Triangular<T>> for Dense<T> {
    fn from(matrix: &'l Triangular<T>) -> Self {
        debug_valid!(matrix);

        let &Triangular { size, format, ref values } = matrix;

        let mut matrix = Dense {
            rows: size,
            columns: size,
            values: vec![T::zero(); size * size],
        };

        match format {
            Format::Lower => {
                let mut k = 0;
                for j in 0..size {
                    for i in j..size {
                        matrix.values[j * size + i] = values[k];
                        k += 1;
                    }
                }
            },
            Format::Upper => {
                let mut k = 0;
                for j in 0..size {
                    for i in 0..(j + 1) {
                        matrix.values[j * size + i] = values[k];
                        k += 1;
                    }
                }
            },
        }

        matrix
    }
}

impl<T: Element> From<Triangular<T>> for Dense<T> {
    fn from(matrix: Triangular<T>) -> Self {
        (&matrix).into()
    }
}

#[cfg(test)]
mod tests {
    use triangular::Format;
    use {Dense, Matrix, Triangular};

    #[test]
    fn nonzeros() {
        let matrix = Triangular {
            size: 4,
            format: Format::Lower,
            values: vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0],
        };
        assert_eq!(matrix.nonzeros(), 7);
    }

    #[test]
    fn into_dense_lower() {
        let matrix = Triangular {
            size: 4,
            format: Format::Lower,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let matrix: Dense<_> = matrix.into();

        assert_eq!(&*matrix, &[
            1.0, 2.0, 3.0,  4.0,
            0.0, 5.0, 6.0,  7.0,
            0.0, 0.0, 8.0,  9.0,
            0.0, 0.0, 0.0, 10.0,
        ]);
    }

    #[test]
    fn into_dense_upper() {
        let matrix = Triangular {
            size: 4,
            format: Format::Upper,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let matrix: Dense<_> = matrix.into();

        assert_eq!(&*matrix, &[
            1.0, 0.0, 0.0,  0.0,
            2.0, 3.0, 0.0,  0.0,
            4.0, 5.0, 6.0,  0.0,
            7.0, 8.0, 9.0, 10.0,
        ]);
    }
}
