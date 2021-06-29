//! The conventional format.
//!
//! The format is suitable for dense matrices.

use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;
use std::fmt;

use {Element, Matrix, Position, Size};

/// A conventional matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Conventional<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values stored in the column-major order.
    pub values: Vec<T>,
}

macro_rules! new(
    ($rows:expr, $columns:expr, $values:expr) => (
        Conventional {
            rows: $rows,
            columns: $columns,
            values: $values,
        }
    );
);

mod convert;
mod decomposition;
mod operation;

size!(Conventional);

impl<T: Element> Conventional<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        new!(rows, columns, vec![T::zero(); rows * columns])
    }

    /// Create a matrix from a slice.
    pub fn from_slice<S: Size>(size: S, values: &[T]) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), rows * columns);
        new!(rows, columns, values.to_vec())
    }

    /// Create a matrix from a vector.
    pub fn from_vec<S: Size>(size: S, values: Vec<T>) -> Self {
        let (rows, columns) = size.dimensions();
        debug_assert_eq!(values.len(), rows * columns);
        new!(rows, columns, values)
    }

    /// Create a matrix with uninitialized elements.
    pub unsafe fn with_uninitialized<S: Size>(size: S) -> Self {
        let (rows, columns) = size.dimensions();
        new!(rows, columns, buffer!(rows * columns))
    }

    /// Zero out the content.
    ///
    /// The function should only be used when it is safe to overwrite `T` with
    /// zero bytes.
    #[inline]
    pub unsafe fn erase(&mut self) {
        ptr::write_bytes(self.values.as_mut_ptr(), 0, self.values.len())
    }

    /// Resize.
    pub fn resize<S: Size>(&mut self, size: S) {
        let (rows, columns) = size.dimensions();
        if self.rows == rows {
            if self.columns > columns {
                self.values.truncate(rows * columns);
            } else {
                self.values
                    .extend(vec![T::zero(); rows * (columns - self.columns)]);
            }
            self.columns = columns;
        } else {
            let mut matrix = Conventional::zero(size);
            let rows = min!(self.rows, rows);
            let columns = min!(self.columns, columns);
            for j in 0..columns {
                for i in 0..rows {
                    matrix[(i, j)] = self[(i, j)];
                }
            }
            *self = matrix;
        }
    }
}

impl<T: Element> Matrix for Conventional<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values
            .iter()
            .fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Conventional::new(size)
    }
}

impl<T: Element, P: Position> Index<P> for Conventional<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: P) -> &Self::Output {
        let (i, j) = index.coordinates();
        &self.values[j * self.rows + i]
    }
}

impl<T: Element, P: Position> IndexMut<P> for Conventional<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: P) -> &mut Self::Output {
        let (i, j) = index.coordinates();
        &mut self.values[j * self.rows + i]
    }
}

impl<T: Element> Deref for Conventional<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.values.deref()
    }
}

impl<T: Element> DerefMut for Conventional<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.deref_mut()
    }
}

impl<T: Element> fmt::Display for Conventional<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[\n")?;
        for i in 0..self.rows {
            write!(f, "\t")?;
            for j in 0..self.columns {
                write!(f, "{}", self.values[i + j * self.rows])?;
                if j == self.columns-1 {
                    write!(f, ";")?;
                } else {
                    write!(f, ",\t")?;
                }
            }
            write!(f, "\n")?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn erase() {
        let mut matrix = Conventional::from_vec(10, vec![42.0; 10 * 10]);
        unsafe { matrix.erase() };
        assert!(matrix.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn resize_fewer_columns() {
        let mut matrix = Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        matrix.resize((2, 2));
        assert_eq!(
            matrix,
            Conventional::from_vec(
                (2, 2),
                matrix![
                    1.0, 2.0;
                    4.0, 5.0;
                ],
            )
        );
    }

    #[test]
    fn resize_fewer_rows() {
        let mut matrix = Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        matrix.resize((1, 3));
        assert_eq!(
            matrix,
            Conventional::from_vec(
                (1, 3),
                matrix![
                    1.0, 2.0, 3.0;
                ],
            )
        );
    }

    #[test]
    fn resize_more_columns() {
        let mut matrix = Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        matrix.resize((2, 4));
        assert_eq!(
            matrix,
            Conventional::from_vec(
                (2, 4),
                matrix![
                    1.0, 2.0, 3.0, 0.0;
                    4.0, 5.0, 6.0, 0.0;
                ],
            )
        );
    }

    #[test]
    fn resize_more_rows() {
        let mut matrix = Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        matrix.resize((3, 3));
        assert_eq!(
            matrix,
            Conventional::from_vec(
                (3, 3),
                matrix![
                    1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                    0.0, 0.0, 0.0;
                ],
            )
        );
    }

    #[test]
    fn resize_more_columns_rows() {
        let mut matrix = Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        matrix.resize((3, 4));
        assert_eq!(
            matrix,
            Conventional::from_vec(
                (3, 4),
                matrix![
                    1.0, 2.0, 3.0, 0.0;
                    4.0, 5.0, 6.0, 0.0;
                    0.0, 0.0, 0.0, 0.0;
                ],
            )
        );
    }

    #[test]
    fn nonzeros() {
        let matrix = Conventional::from_vec(2, vec![1.0, 2.0, 3.0, 0.0]);
        assert_eq!(matrix.nonzeros(), 3);
    }
}
