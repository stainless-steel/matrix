use Element;
use format::{Banded, Conventional, Diagonal};

impl<'l, T: Element> From<&'l Banded<T>> for Conventional<T> {
    fn from(matrix: &'l Banded<T>) -> Self {
        let &Banded {
            rows,
            columns,
            superdiagonals,
            subdiagonals,
            ref values,
        } = validate!(matrix);
        let diagonals = matrix.diagonals();
        let mut matrix = Conventional::new((rows, columns));
        for j in 0..columns {
            for i in row_range!(rows, superdiagonals, subdiagonals, j) {
                let k = superdiagonals + i - j;
                matrix.values[j * rows + i] = values[j * diagonals + k];
            }
        }
        matrix
    }
}

impl<T: Element> From<Banded<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Banded<T>) -> Self {
        (&matrix).into()
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Banded<T> {
    #[inline]
    fn from(matrix: &'l Diagonal<T>) -> Self {
        matrix.clone().into()
    }
}

impl<T: Element> From<Diagonal<T>> for Banded<T> {
    fn from(matrix: Diagonal<T>) -> Self {
        let Diagonal {
            rows,
            columns,
            mut values,
        } = validate!(matrix);
        for _ in rows..columns {
            values.push(T::zero());
        }
        new!(rows, columns, 0, 0, values)
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn from_diagonal_tall() {
        let matrix = Banded::from(Diagonal::from_vec((5, 3), vec![1.0, 2.0, 3.0]));
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_diagonal_wide() {
        let matrix = Banded::from(Diagonal::from_vec((3, 5), vec![1.0, 2.0, 3.0]));
        assert_eq!(&matrix.values, &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn into_conventional_tall() {
        let matrix = new!(
            7,
            4,
            2,
            2,
            matrix![
                0.0,  0.0,  3.0,  7.0;
                0.0,  2.0,  6.0, 11.0;
                1.0,  5.0, 10.0, 14.0;
                4.0,  9.0, 13.0, 16.0;
                8.0, 12.0, 15.0, 17.0;
            ]
        );

        let matrix = Conventional::from(matrix);

        assert_eq!(
            &*matrix,
            &*matrix![
                1.0,  2.0,  3.0,  0.0;
                4.0,  5.0,  6.0,  7.0;
                8.0,  9.0, 10.0, 11.0;
                0.0, 12.0, 13.0, 14.0;
                0.0,  0.0, 15.0, 16.0;
                0.0,  0.0,  0.0, 17.0;
                0.0,  0.0,  0.0,  0.0;
            ]
        );
    }

    #[test]
    fn into_conventional_wide() {
        let matrix = new!(
            4,
            7,
            2,
            2,
            matrix![
                0.0,  0.0,  3.0,  7.0, 12.0, 17.0, 0.0;
                0.0,  2.0,  6.0, 11.0, 16.0,  0.0, 0.0;
                1.0,  5.0, 10.0, 15.0,  0.0,  0.0, 0.0;
                4.0,  9.0, 14.0,  0.0,  0.0,  0.0, 0.0;
                8.0, 13.0,  0.0,  0.0,  0.0,  0.0, 0.0;
            ]
        );

        let matrix = Conventional::from(matrix);

        assert_eq!(
            &*matrix,
            &*matrix![
                1.0,  2.0,  3.0,  0.0,  0.0,  0.0, 0.0;
                4.0,  5.0,  6.0,  7.0,  0.0,  0.0, 0.0;
                8.0,  9.0, 10.0, 11.0, 12.0,  0.0, 0.0;
                0.0, 13.0, 14.0, 15.0, 16.0, 17.0, 0.0;
            ]
        );
    }
}
