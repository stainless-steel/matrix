use Element;
use storage::Conventional;
use storage::packed::{Format, Packed};

impl<'l, T: Element> From<&'l Packed<T>> for Conventional<T> {
    fn from(matrix: &'l Packed<T>) -> Self {
        let &Packed { size, format, ref values } = validate!(matrix);

        let mut matrix = Conventional::new(size);
        match format {
            Format::Lower => {
                let mut k = 0;
                for j in 0..size {
                    for i in j..size {
                        matrix.values[j * size + i] = values[k];
                        k += 1;
                    }
                }
            },
            Format::Upper => {
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
    use storage::Conventional;
    use storage::packed::{Format, Packed};

    #[test]
    fn into_conventional_lower() {
        let matrix = new!(4, Format::Lower, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &[
            1.0, 2.0, 3.0,  4.0,
            0.0, 5.0, 6.0,  7.0,
            0.0, 0.0, 8.0,  9.0,
            0.0, 0.0, 0.0, 10.0,
        ]);
    }

    #[test]
    fn into_conventional_upper() {
        let matrix = new!(4, Format::Upper, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ]);

        let matrix = Conventional::from(matrix);

        assert_eq!(&*matrix, &[
            1.0, 0.0, 0.0,  0.0,
            2.0, 3.0, 0.0,  0.0,
            4.0, 5.0, 6.0,  0.0,
            7.0, 8.0, 9.0, 10.0,
        ]);
    }
}
