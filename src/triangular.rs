use {Dense, Element, Sparse};

/// A triangular matrix.
///
/// Apart from triangular matrices, the storage is suitable for symmetric and
/// Hermitian matrices. Data are stored in the [format][1] adopted by
/// [LAPACK][2].
///
/// [1]: http://www.netlib.org/lapack/lug/node123.html
/// [2]: http://www.netlib.org/lapack
#[derive(Clone, Debug)]
pub struct Triangular<T: Element> {
    /// The number of rows or columns.
    pub size: usize,
    /// The storage format.
    pub format: TriangularFormat,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

/// The storage format of a triangular matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TriangularFormat {
    /// The lower triangular format.
    Lower,
    /// The upper triangular format.
    Upper,
}

matrix!(Triangular, size, size);

impl<T: Element> Sparse for Triangular<T> {
    #[inline]
    fn nonzeros(&self) -> usize {
        self.size * (self.size + 1) / 2
    }
}

impl<T: Element> From<Triangular<T>> for Dense<T> {
    fn from(triangular: Triangular<T>) -> Dense<T> {
        let Triangular { size, format, ref data } = triangular;

        debug_assert_eq!(data.len(), size * (size + 1) / 2);

        let mut dense = Dense {
            rows: size,
            columns: size,
            data: vec![T::zero(); size * size],
        };

        match format {
            TriangularFormat::Lower => {
                let mut k = 0;
                for j in 0..size {
                    for i in j..size {
                        dense.data[j * size + i] = data[k];
                        k += 1;
                    }
                }
            },
            TriangularFormat::Upper => {
                let mut k = 0;
                for j in 0..size {
                    for i in 0..(j + 1) {
                        dense.data[j * size + i] = data[k];
                        k += 1;
                    }
                }
            },
        }

        dense
    }
}

#[cfg(test)]
mod tests {
    use {Dense, Triangular, TriangularFormat};

    #[test]
    fn into_lower_dense() {
        let triangular = Triangular {
            size: 4,
            format: TriangularFormat::Lower,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let dense: Dense<f64> = triangular.into();

        assert_eq!(&dense[..], &[
            1.0, 2.0, 3.0,  4.0,
            0.0, 5.0, 6.0,  7.0,
            0.0, 0.0, 8.0,  9.0,
            0.0, 0.0, 0.0, 10.0,
        ]);
    }

    #[test]
    fn into_upper_dense() {
        let triangular = Triangular {
            size: 4,
            format: TriangularFormat::Upper,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let dense: Dense<f64> = triangular.into();

        assert_eq!(&dense[..], &[
            1.0, 0.0, 0.0,  0.0,
            2.0, 3.0, 0.0,  0.0,
            4.0, 5.0, 6.0,  0.0,
            7.0, 8.0, 9.0, 10.0,
        ]);
    }
}
