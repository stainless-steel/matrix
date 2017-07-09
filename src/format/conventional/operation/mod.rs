use format::{Conventional, Diagonal};
use operation::{MultiplySelf, Transpose};
use {Element, Number};

#[cfg(feature = "acceleration")]
mod acceleration;

impl<T> MultiplySelf<Diagonal<T>> for Conventional<T>
where
    T: Element + Number,
{
    #[inline]
    fn multiply_self(&mut self, right: &Diagonal<T>) {
        let (rows, insides, columns) = (self.rows, self.columns, right.columns);
        debug_assert_eq!(insides, right.rows);
        self.resize((rows, columns));
        for j in 0..insides {
            let factor = right[j];
            for i in 0..rows {
                self[(i, j)] = factor * self[(i, j)];
            }
        }
    }
}

impl<T: Element> Transpose for Conventional<T> {
    fn transpose(&self) -> Self {
        let (rows, columns) = (self.rows, self.columns);
        let mut matrix = Conventional::new((columns, rows));
        for i in 0..rows {
            for j in 0..columns {
                matrix.values[i * columns + j] = self.values[j * rows + i];
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn multiply_self() {
        let mut matrix = Conventional::from_vec(
            (3, 2),
            matrix![
                1.0, 4.0;
                2.0, 5.0;
                3.0, 6.0;
            ],
        );
        let right = Diagonal::from_vec((2, 4), vec![2.0, 3.0]);
        matrix.multiply_self(&right);
        assert_eq!(
            &*matrix,
            &*matrix![
                2.0, 12.0, 0.0, 0.0;
                4.0, 15.0, 0.0, 0.0;
                6.0, 18.0, 0.0, 0.0;
            ]
        );
    }

    #[test]
    fn transpose() {
        let matrix = Conventional::from_vec(
            (3, 2),
            matrix![
                1.0, 4.0;
                2.0, 5.0;
                3.0, 6.0;
            ],
        );
        assert_eq!(
            matrix.transpose(),
            Conventional::from_vec(
                (2, 3),
                matrix![
                    1.0, 2.0, 3.0;
                    4.0, 5.0, 6.0;
                ],
            )
        );
    }
}
