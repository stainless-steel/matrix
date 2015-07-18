use format::{Compressed, Conventional, Diagonal};
use operation::{Multiply, MultiplyInto, MultiplySelf, Transpose};
use {Element, Number};

impl<T> Multiply<Diagonal<T>, Compressed<T>> for Compressed<T> where T: Element + Number {
    #[inline]
    fn multiply(&self, right: &Diagonal<T>) -> Self {
        let mut result = self.clone();
        result.multiply_self(right);
        result
    }
}

impl<'l, T> MultiplyInto<[T], [T]> for Compressed<T> where T: Element + Number {
    #[inline]
    fn multiply_into(&self, right: &[T], result: &mut [T]) {
        let (m, p) = (self.rows, self.columns);
        let n = right.len() / p;
        multiply_matrix_left(self, right, result, m, p, n)
    }
}

impl<'l, T> MultiplyInto<Compressed<T>, [T]> for Conventional<T>
    where T: Element + Number
{
    #[inline]
    fn multiply_into(&self, right: &Compressed<T>, result: &mut [T]) {
        let (m, p, n) = (self.rows, self.columns, right.columns);
        multiply_matrix_right(&self.values, right, result, m, p, n)
    }
}

impl<T> MultiplySelf<Diagonal<T>> for Compressed<T> where T: Element + Number {
    #[inline]
    fn multiply_self(&mut self, right: &Diagonal<T>) {
        let (m, n) = (self.rows, right.columns);
        debug_assert_eq!(self.columns, right.rows);
        self.resize((m, n));
        for (_, j, value) in self.iter_mut() {
            *value = *value * right[j];
        }
    }
}

impl<T: Element> Transpose for Compressed<T> {
    fn transpose(&self) -> Self {
        let &Compressed { rows, columns, nonzeros, variant, .. } = self;
        let mut matrix = Compressed::with_capacity((columns, rows), variant, nonzeros);
        for (i, j, &value) in self.iter() {
            matrix.set((j, i), value);
        }
        matrix
    }
}

fn multiply_matrix_left<T>(a: &Compressed<T>, b: &[T], c: &mut [T], m: usize, p: usize, n: usize)
    where T: Element + Number
{
    debug_assert_eq!(a.rows * a.columns, m * p);
    debug_assert_eq!(b.len(), p * n);
    debug_assert_eq!(c.len(), m * n);
    let (mut k, mut l) = (0, 0);
    for _ in 0..n {
        multiply_vector_left(a, &b[k..(k + p)], &mut c[l..(l + m)], p);
        k += p;
        l += m;
    }
}

#[inline(always)]
fn multiply_vector_left<T>(a: &Compressed<T>, b: &[T], c: &mut [T], p: usize)
    where T: Element + Number
{
    let &Compressed { ref values, ref indices, ref offsets, .. } = a;
    for j in 0..p {
        for k in offsets[j]..offsets[j + 1] {
            let current = c[indices[k]];
            c[indices[k]] = current + values[k] * b[j];
        }
    }
}

fn multiply_matrix_right<T>(a: &[T], b: &Compressed<T>, c: &mut [T], m: usize, p: usize, n: usize)
    where T: Element + Number
{
    debug_assert_eq!(a.len(), m * p);
    debug_assert_eq!(b.rows * b.columns, p * n);
    debug_assert_eq!(c.len(), m * n);
    let mut k = 0;
    for j in 0..n {
        multiply_vector_right(a, b, &mut c[k..(k + m)], m, j);
        k += m;
    }
}

#[inline(always)]
fn multiply_vector_right<T>(a: &[T], b: &Compressed<T>, c: &mut [T], m: usize, j: usize)
    where T: Element + Number
{
    let &Compressed { ref values, ref indices, ref offsets, .. } = b;
    for k in offsets[j]..offsets[j + 1] {
        for i in 0..m {
            c[i] = c[i] + values[k] * a[indices[k] * m + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use format::compressed::Variant;

    #[test]
    fn multiply_self() {
        let mut matrix = new!(3, 2, 3, Variant::Column, vec![1.0, 2.0, 3.0],
                              vec![1, 0, 2], vec![0, 1, 3]);

        let right = Diagonal::from_vec((2, 4), vec![4.0, 5.0]);

        matrix.multiply_self(&right);

        assert_eq!(matrix, new!(3, 4, 3, Variant::Column, vec![4.0, 10.0, 15.0],
                                vec![1, 0, 2], vec![0, 1, 3, 3, 3]));
    }

    #[test]
    fn multiply_into_left() {
        let matrix = Compressed::from(Conventional::from_vec((4, 3), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0,
        ]));

        let right = Conventional::from_vec((3, 2), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);

        let mut result = Conventional::from_vec((4, 2), vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]);

        matrix.multiply_into(&right, &mut result);

        assert_eq!(&result.values, &vec![
            24.0, 24.0, 22.0, 18.0,
            54.0, 57.0, 55.0, 48.0,
        ]);
    }

    #[test]
    fn multiply_into_right() {
        let matrix = Conventional::from_vec((4, 3), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0,
        ]);

        let right = Compressed::from(Conventional::from_vec((3, 2), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]));

        let mut result = Conventional::from_vec((4, 2), vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]);

        matrix.multiply_into(&right, &mut result);

        assert_eq!(&result.values, &vec![
            24.0, 24.0, 22.0, 18.0,
            54.0, 57.0, 55.0, 48.0,
        ]);
    }

    #[test]
    fn transpose() {
        let matrix = new!(5, 7, 5, Variant::Column, vec![1.0, 2.0, 3.0, 4.0, 5.0],
                          vec![1, 0, 3, 1, 4], vec![0, 0, 0, 1, 2, 2, 3, 5]);

        let matrix = matrix.transpose();

        assert_eq!(matrix, new!(7, 5, 5, Variant::Column, vec![2.0, 1.0, 4.0, 3.0, 5.0],
                                vec![3, 2, 6, 5, 6], vec![0, 1, 3, 3, 4, 5]));
    }
}
