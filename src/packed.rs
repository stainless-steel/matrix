use {DenseMatrix, Element};

/// A packed matrix.
///
/// Data are stored in the [format][1] adopted by [LAPACK][2].
///
/// [1]: http://www.netlib.org/lapack/lug/node123.html
/// [2]: http://www.netlib.org/lapack
#[derive(Clone, Debug)]
pub struct PackedMatrix<T> {
    /// The number of rows or columns.
    pub size: usize,
    /// The storage format.
    pub format: PackedFormat,
    /// The data stored in the column-major order.
    pub data: Vec<T>,
}

/// The storage format of a packed matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedFormat {
    /// The lower triangular format.
    Lower,
    /// The upper triangular format.
    Upper,
}

impl<T> From<PackedMatrix<T>> for DenseMatrix<T> where T: Element {
    fn from(packed: PackedMatrix<T>) -> DenseMatrix<T> {
        let PackedMatrix { size, format, ref data } = packed;

        debug_assert_eq!(data.len(), size * (size + 1) / 2);

        let mut dense = DenseMatrix {
            rows: size,
            columns: size,
            data: vec![Element::zero(); size * size],
        };

        match format {
            PackedFormat::Lower => {
                let mut k = 0;
                for j in 0..size {
                    for i in j..size {
                        dense.data[j * size + i] = data[k];
                        k += 1;
                    }
                }
            },
            PackedFormat::Upper => {
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
    use {DenseMatrix, PackedMatrix, PackedFormat};

    #[test]
    fn into_lower_dense() {
        let packed = PackedMatrix {
            size: 4,
            format: PackedFormat::Lower,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let dense: DenseMatrix<f64> = packed.into();

        assert_eq!(&dense[..], &[
            1.0, 2.0, 3.0,  4.0,
            0.0, 5.0, 6.0,  7.0,
            0.0, 0.0, 8.0,  9.0,
            0.0, 0.0, 0.0, 10.0,
        ]);
    }

    #[test]
    fn into_upper_dense() {
        let packed = PackedMatrix {
            size: 4,
            format: PackedFormat::Upper,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        let dense: DenseMatrix<f64> = packed.into();

        assert_eq!(&dense[..], &[
            1.0, 0.0, 0.0,  0.0,
            2.0, 3.0, 0.0,  0.0,
            4.0, 5.0, 6.0,  0.0,
            7.0, 8.0, 9.0, 10.0,
        ]);
    }
}
