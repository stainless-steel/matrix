use format::compressed::Variant;
use format::{Compressed, Conventional, Diagonal};
use {Element, Size};

impl<'l, T: Element> From<&'l Conventional<T>> for Compressed<T> {
    fn from(conventional: &'l Conventional<T>) -> Self {
        let (rows, columns) = conventional.dimensions();
        let mut matrix = Compressed::new((rows, columns), Variant::Column);
        for (k, &value) in conventional.values.iter().enumerate() {
            if !value.is_zero() {
                matrix.set((k % rows, k / rows), value);
            }
        }
        matrix
    }
}

impl<T: Element> From<Conventional<T>> for Compressed<T> {
    #[inline]
    fn from(matrix: Conventional<T>) -> Self {
        (&matrix).into()
    }
}

impl<'l, T: Element> From<&'l Compressed<T>> for Conventional<T> {
    fn from(matrix: &'l Compressed<T>) -> Self {
        let &Compressed {
            rows,
            columns,
            variant,
            ref values,
            ref indices,
            ref offsets,
            ..
        } = validate!(matrix);
        let mut matrix = Conventional::new((rows, columns));
        match variant {
            Variant::Row => {
                for i in 0..rows {
                    for k in offsets[i]..offsets[i + 1] {
                        matrix.values[indices[k] * rows + i] = values[k];
                    }
                }
            }
            Variant::Column => {
                for j in 0..columns {
                    for k in offsets[j]..offsets[j + 1] {
                        matrix.values[j * rows + indices[k]] = values[k];
                    }
                }
            }
        }
        matrix
    }
}

impl<T: Element> From<Compressed<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Compressed<T>) -> Self {
        (&matrix).into()
    }
}

impl<'l, T: Element> From<&'l Diagonal<T>> for Compressed<T> {
    #[inline]
    fn from(matrix: &'l Diagonal<T>) -> Self {
        matrix.clone().into()
    }
}

impl<T: Element> From<Diagonal<T>> for Compressed<T> {
    fn from(matrix: Diagonal<T>) -> Self {
        let Diagonal {
            rows,
            columns,
            values,
        } = validate!(matrix);
        let nonzeros = values.len();
        let indices = (0..nonzeros).collect();
        let offsets = (0..(columns + 1))
            .map(|i| if i < nonzeros { i } else { nonzeros })
            .collect();
        new!(
            rows,
            columns,
            nonzeros,
            Variant::Column,
            values,
            indices,
            offsets
        )
    }
}

#[cfg(test)]
mod tests {
    use format::compressed::Variant;
    use prelude::*;

    #[test]
    fn from_conventional() {
        let matrix = Conventional::from_vec(
            (5, 3),
            matrix![
                0.0, 0.0, 0.0;
                1.0, 0.0, 0.0;
                0.0, 0.0, 0.0;
                0.0, 2.0, 0.0;
                0.0, 3.0, 4.0;
            ],
        );
        let matrix = Compressed::from(matrix);
        assert_eq!(
            matrix,
            new!(
                5,
                3,
                4,
                Variant::Column,
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1, 3, 4, 4],
                vec![0, 1, 3, 4]
            )
        );
    }

    #[test]
    fn from_diagonal_tall() {
        let matrix = Compressed::from(Diagonal::from_vec((5, 3), vec![1.0, 2.0, 0.0]));
        assert_eq!(
            matrix,
            new!(
                5,
                3,
                3,
                Variant::Column,
                vec![1.0, 2.0, 0.0],
                vec![0, 1, 2],
                vec![0, 1, 2, 3]
            )
        );
    }

    #[test]
    fn from_diagonal_wide() {
        let matrix = Compressed::from(Diagonal::from_vec((3, 5), vec![1.0, 0.0, 3.0]));
        assert_eq!(
            matrix,
            new!(
                3,
                5,
                3,
                Variant::Column,
                vec![1.0, 0.0, 3.0],
                vec![0, 1, 2],
                vec![0, 1, 2, 3, 3, 3]
            )
        );
    }

    #[test]
    fn into_conventional() {
        let matrix = new!(
            5,
            3,
            3,
            Variant::Column,
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 2],
            vec![0, 1, 2, 3]
        );
        let matrix = Conventional::from(matrix);
        assert_eq!(
            &*matrix,
            &*matrix![
                1.0, 0.0, 0.0;
                0.0, 2.0, 0.0;
                0.0, 0.0, 3.0;
                0.0, 0.0, 0.0;
                0.0, 0.0, 0.0;
            ]
        );
    }
}
