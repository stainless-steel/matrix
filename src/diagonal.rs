use std::ops::{Deref, DerefMut};

use {Band, Dense, Element, Make, Shape, Sparse};

/// A diagonal matrix.
///
/// The storage is suitable for generic diagonal matrices.
#[derive(Clone, Debug)]
pub struct Diagonal<T: Element> {
    /// The number of rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The values of the diagonal elements.
    pub data: Vec<T>,
}

matrix!(Diagonal);

impl<'l, T: Element> Make<&'l [T]> for Diagonal<T> {
    fn make(data: &'l [T], shape: Shape) -> Self {
        let (rows, columns) = match shape {
            Shape::Square(size) => (size, size),
            Shape::Rectangular(rows, columns) => (rows, columns),
        };
        debug_assert_eq!(data.len(), min!(rows, columns));
        Diagonal { rows: rows, columns: columns, data: data.to_vec() }
    }
}

impl<T: Element> Make<Vec<T>> for Diagonal<T> {
    fn make(data: Vec<T>, shape: Shape) -> Self {
        let (rows, columns) = match shape {
            Shape::Square(size) => (size, size),
            Shape::Rectangular(rows, columns) => (rows, columns),
        };
        debug_assert_eq!(data.len(), min!(rows, columns));
        Diagonal { rows: rows, columns: columns, data: data }
    }
}

impl<T: Element> Sparse for Diagonal<T> {
    #[inline]
    fn nonzeros(&self) -> usize {
        if self.rows < self.columns { self.rows } else { self.columns }
    }
}

impl<T: Element> From<Diagonal<T>> for Band<T> {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Band<T> {
        Band {
            rows: diagonal.rows,
            columns: diagonal.columns,
            superdiagonals: 0,
            subdiagonals: 0,
            data: {
                let mut data = diagonal.data;
                for _ in diagonal.rows..diagonal.columns {
                    data.push(T::zero());
                }
                data
            },
        }
    }
}

impl<T: Element> From<Diagonal<T>> for Dense<T> {
    #[inline]
    fn from(diagonal: Diagonal<T>) -> Dense<T> {
        <Diagonal<T> as Into<Band<T>>>::into(diagonal).into()
    }
}

impl<T: Element> Deref for Diagonal<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

impl<T: Element> DerefMut for Diagonal<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use {Band, Diagonal};

    #[test]
    fn into_tall_band() {
        let diagonal = Diagonal {
            rows: 5,
            columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };

        let band: Band<f64> = diagonal.into();

        assert_eq!(&band.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_wide_band() {
        let diagonal = Diagonal {
            rows: 3,
            columns: 5,
            data: vec![1.0, 2.0, 3.0],
        };

        let band: Band<f64> = diagonal.into();

        assert_eq!(&band.data, &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }
}
