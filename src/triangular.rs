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

macro_rules! arithmetic(
    ($count:expr, $first:expr, $last:expr) => (
        $count * ($first + $last) / 2
    );
);

macro_rules! storage(
    ($size:expr) => (arithmetic!($size, 1, $size))
);

macro_rules! debug_valid(
    ($matrix:ident) => (debug_assert!(
        $matrix.values.len() == storage!($matrix.size)
    ));
);

size!(Triangular, size, size);

impl<T: Element> Triangular<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, format: Format) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        Triangular { size: rows, format: format, values: vec![T::zero(); storage!(rows)] }
    }
}

impl<T: Element> Matrix for Triangular<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values.iter().fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    fn transpose(&mut self) {
        let &mut Triangular { size, format, .. } = self;
        let lower = format == Format::Lower;
        let mut matrix = Triangular::new(size, format.flip());
        let mut k = 0;
        for j in 0..size {
            for i in j..size {
                if lower {
                    matrix.values[arithmetic!(i, 1, i) + j] = self.values[k];
                } else {
                    matrix.values[k] = self.values[arithmetic!(i, 1, i) + j];
                }
                k += 1;
            }
        }
        *self = matrix;
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Triangular::new(size, Format::Lower)
    }
}

impl<'l, T: Element> From<&'l Triangular<T>> for Dense<T> {
    fn from(matrix: &'l Triangular<T>) -> Self {
        debug_valid!(matrix);

        let &Triangular { size, format, ref values } = matrix;

        let mut matrix = Dense::new(size);
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

impl Format {
    /// Return the other format.
    #[inline]
    pub fn flip(&self) -> Self {
        match *self {
            Format::Lower => Format::Upper,
            Format::Upper => Format::Lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use triangular::Format;
    use {Dense, Matrix, Triangular};

    macro_rules! new(
        ($size:expr, $format:expr, $values:expr) => (
            Triangular { size: $size, format: $format, values: $values }
        );
    );

    #[test]
    fn nonzeros() {
        let matrix = new!(4, Format::Lower, vec![
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        assert_eq!(matrix.nonzeros(), 7);
    }

    #[test]
    fn transpose_lower() {
        let mut matrix = new!(4, Format::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        matrix.transpose();
        assert_eq!(matrix, new!(4, Format::Upper, vec![
            1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0,
        ]));
    }

    #[test]
    fn transpose_upper() {
        let mut matrix = new!(4, Format::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);
        matrix.transpose();
        assert_eq!(matrix, new!(4, Format::Lower, vec![
            1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 8.0, 6.0, 9.0, 10.0,
        ]));
    }

    #[test]
    fn into_dense_lower() {
        let matrix = new!(4, Format::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

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
        let matrix = new!(4, Format::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix: Dense<_> = matrix.into();

        assert_eq!(&*matrix, &[
            1.0, 0.0, 0.0,  0.0,
            2.0, 3.0, 0.0,  0.0,
            4.0, 5.0, 6.0,  0.0,
            7.0, 8.0, 9.0, 10.0,
        ]);
    }
}
