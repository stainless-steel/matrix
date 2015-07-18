use Element;
use format::Conventional;
use operation::Transpose;

#[cfg(feature = "acceleration")]
mod acceleration;

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
    fn transpose() {
        let matrix = Conventional::from_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix = matrix.transpose();
        assert_eq!(matrix, Conventional::from_vec((2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }
}
