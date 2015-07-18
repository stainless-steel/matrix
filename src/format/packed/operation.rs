use Element;
use format::Packed;
use format::packed::Variant;
use operation::Transpose;

impl<T: Element> Transpose for Packed<T> {
    fn transpose(&self) -> Self {
        let &Packed { size, variant, .. } = self;
        let lower = variant == Variant::Lower;
        let mut matrix = Packed::new(size, variant.flip());
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
        matrix
    }
}

#[cfg(test)]
mod tests {
    use format::packed::Variant;
    use prelude::*;

    #[test]
    fn transpose_lower() {
        let matrix = new!(4, Variant::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Variant::Upper, vec![
            1.0, 2.0, 5.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 10.0,
        ]));
    }

    #[test]
    fn transpose_upper() {
        let matrix = new!(4, Variant::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(4, Variant::Lower, vec![
            1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 8.0, 6.0, 9.0, 10.0,
        ]));
    }
}
