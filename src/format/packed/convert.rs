use Element;
use format::packed::Variant;
use format::{Conventional, Packed};

impl<'l, T: Element> From<&'l Packed<T>> for Conventional<T> {
    fn from(matrix: &'l Packed<T>) -> Self {
        let &Packed { size, variant, ref values } = validate!(matrix);

        let mut matrix = Conventional::new(size);
        match variant {
            Variant::Lower => {
                let mut k = 0;
                for j in 0..size {
                    for i in j..size {
                        matrix.values[j * size + i] = values[k];
                        k += 1;
                    }
                }
            },
            Variant::Upper => {
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

impl<T: Element> From<Packed<T>> for Conventional<T> {
    #[inline]
    fn from(matrix: Packed<T>) -> Self {
        (&matrix).into()
    }
}

#[cfg(test)]
mod tests {
    use format::packed::Variant;
    use prelude::*;

    #[test]
    fn into_conventional_lower() {
        let matrix = new!(4, Variant::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &*matrix![
            1.0, 0.0, 0.0,  0.0;
            2.0, 5.0, 0.0,  0.0;
            3.0, 6.0, 8.0,  0.0;
            4.0, 7.0, 9.0, 10.0;
        ]);
    }

    #[test]
    fn into_conventional_upper() {
        let matrix = new!(4, Variant::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &*matrix![
            1.0, 2.0, 4.0,  7.0;
            0.0, 3.0, 5.0,  8.0;
            0.0, 0.0, 6.0,  9.0;
            0.0, 0.0, 0.0, 10.0;
        ]);
    }
}
