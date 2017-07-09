use Element;
use format::Banded;
use operation::Transpose;

impl<T: Element> Transpose for Banded<T> {
    fn transpose(&self) -> Self {
        let &Banded {
            rows,
            columns,
            superdiagonals,
            subdiagonals,
            ..
        } = self;
        let diagonals = self.diagonals();
        let mut matrix = Banded::new((columns, rows), subdiagonals, superdiagonals);
        for j in 0..columns {
            for i in row_range!(rows, superdiagonals, subdiagonals, j) {
                let k = superdiagonals + i - j;
                let l = subdiagonals + j - i;
                matrix.values[i * diagonals + l] = self.values[j * diagonals + k];
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn transpose() {
        let matrix = new!(
            4,
            8,
            3,
            1,
            matrix![
                0.0,  0.0,  0.0,  4.0,  9.0, 14.0, 19.0, 0.0;
                0.0,  0.0,  3.0,  8.0, 13.0, 18.0,  0.0, 0.0;
                0.0,  2.0,  7.0, 12.0, 17.0,  0.0,  0.0, 0.0;
                1.0,  6.0, 11.0, 16.0,  0.0,  0.0,  0.0, 0.0;
                5.0, 10.0, 15.0,  0.0,  0.0,  0.0,  0.0, 0.0;
            ]
        );

        let matrix = matrix.transpose();

        assert_eq!(
            matrix,
            new!(
                8,
                4,
                1,
                3,
                matrix![
                    0.0, 5.0, 10.0, 15.0;
                    1.0, 6.0, 11.0, 16.0;
                    2.0, 7.0, 12.0, 17.0;
                    3.0, 8.0, 13.0, 18.0;
                    4.0, 9.0, 14.0, 19.0;
                ]
            )
        );
    }
}
