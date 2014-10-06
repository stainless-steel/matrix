#![feature(macro_rules)]

extern crate blas;

#[inline]
pub fn multiply(a: *const f64, b: *const f64, c: *mut f64, m: uint, p: uint, n: uint) {
    if n == 1 {
        blas::dgemv(blas::NORMAL, m as i32, p as i32, 1.0, a, m as i32, b, 1, 0.0, c, 1);
    } else {
        blas::dgemm(blas::NORMAL, blas::NORMAL, m as i32, n as i32, p as i32,
            1.0, a, m as i32, b, p as i32, 0.0, c, m as i32);
    }
}

#[inline]
pub fn multiply_add(a: *const f64, b: *const f64, c: *const f64, d: *mut f64,
    m: uint, p: uint, n: uint) {

    if c != (d as *const f64) {
        unsafe {
            std::ptr::copy_nonoverlapping_memory(d, c, m * n);
        }
    }

    if n == 1 {
        blas::dgemv(blas::NORMAL, m as i32, p as i32, 1.0, a, m as i32, b, 1, 1.0, d, 1);
    } else {
        blas::dgemm(blas::NORMAL, blas::NORMAL, m as i32, n as i32, p as i32,
            1.0, a, m as i32, b, p as i32, 1.0, d, m as i32);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;

    macro_rules! assert_equal(
        ($given:expr , $expected:expr) => ({
            assert_eq!($given.len(), $expected.len());
            for i in range(0u, $given.len()) {
                assert_eq!($given[i], $expected[i]);
            }
        });
    )

    #[test]
    fn multiply() {
        let (m, p, n) = (2, 4, 1);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0, 0.0];

        super::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, p, n);

        let expected_c = vec![50.0, 60.0];
        assert_equal!(c, expected_c);
    }

    #[test]
    fn multiply_add() {
        let (m, p, n) = (2, 3, 4);

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut d = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        super::multiply_add(a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_mut_ptr(), m, p, n);

        let expected_c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_equal!(c, expected_c);

        let expected_d = vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0];
        assert_equal!(d, expected_d);
    }

    #[bench]
    fn multiply_matrix_matrix(bench: &mut Bencher) {
        let m = 100;

        let a = Vec::from_elem(m * m, 1.0);
        let b = Vec::from_elem(m * m, 1.0);
        let mut c = Vec::from_elem(m * m, 1.0);

        bench.iter(|| {
            super::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, m, m)
        })
    }

    #[bench]
    fn multiply_matrix_vector(bench: &mut Bencher) {
        let m = 100;

        let a = Vec::from_elem(m * m, 1.0);
        let b = Vec::from_elem(m * 1, 1.0);
        let mut c = Vec::from_elem(m * 1, 1.0);

        bench.iter(|| {
            super::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, m, 1)
        })
    }
}
