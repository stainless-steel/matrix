use Element;
use format::{Conventional, Diagonal};

impl<'l, T: Element> From<&'l Diagonal<T>> for Conventional<T> {
    fn from(matrix: &Diagonal<T>) -> Self {
        let &Diagonal {
            rows,
            columns,
            ref values,
        } = validate!(matrix);
        let mut conventional = Conventional::new((rows, columns));
        for i in 0..min!(rows, columns) {
            conventional.values[i * rows + i] = values[i];
        }
        conventional
    }
}

impl<T: Element> From<Diagonal<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Diagonal<T>) -> Self {
        (&matrix).into()
    }
}

impl<T: Element> Into<Vec<T>> for Diagonal<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    #[test]
    fn into_conventional() {
        let matrix = Conventional::from(new!(3, 5, vec![1.0, 2.0, 3.0]));

        assert_eq!(
            matrix,
            Conventional::from_vec(
                (3, 5),
                matrix![
                    1.0, 0.0, 0.0, 0.0, 0.0;
                    0.0, 2.0, 0.0, 0.0, 0.0;
                    0.0, 0.0, 3.0, 0.0, 0.0;
                ],
            )
        );
    }
}
