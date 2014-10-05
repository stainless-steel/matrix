#![feature(macro_rules)]
#![feature(unsafe_destructor)]

extern crate alloc;
extern crate blas;
extern crate core;

use alloc::heap;
use core::raw;
use std::mem;
use std::ptr;

#[unsafe_no_drop_flag]
pub struct Matrix<T> {
    rows: uint,
    cols: uint,
    ptr: *mut T,
}

impl<T> Matrix<T> {
    #[inline]
    pub fn new(rows: uint, cols: uint) -> Matrix<T> {
        let ptr = unsafe {
            heap::allocate(rows * cols * mem::size_of::<T>(),
                mem::min_align_of::<T>())
        };
        Matrix { rows: rows, cols: cols, ptr: ptr as *mut T }
    }

    #[inline]
    pub fn rows(&self) -> uint {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> uint {
        self.cols
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe {
            mem::transmute(raw::Slice {
                data: self.as_mut_ptr() as *const T,
                len: self.len(),
            })
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Matrix<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            heap::deallocate(self.ptr as *mut u8,
                self.len() * mem::size_of::<T>(),
                mem::min_align_of::<T>())
        }
    }
}

impl<T> Index<uint, T> for Matrix<T> {
    #[inline]
    fn index<'a>(&'a self, index: &uint) -> &'a T {
        &self.as_slice()[*index]
    }
}

impl<T> Slice<T> for Matrix<T> {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe {
            mem::transmute(raw::Slice {
                data: self.as_ptr(),
                len: self.len(),
            })
        }
    }
}

impl<T> Collection for Matrix<T> {
    #[inline]
    fn len(&self) -> uint {
        self.rows * self.cols
    }
}

#[inline]
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], m: uint, p: uint, n: uint) {
    if n == 1 {
        blas::dgemv(blas::NORMAL, m as i32, p as i32, 1.0, a, m as i32, b, 1,
            0.0, c, 1);
    } else {
        blas::dgemm(blas::NORMAL, blas::NORMAL, m as i32, n as i32, p as i32,
            1.0, a, m as i32, b, p as i32, 0.0, c, m as i32);
    }
}

#[inline]
pub fn multiply_add(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64], m: uint, p: uint, n: uint) {
    if c.as_ptr() != d.as_ptr() {
        unsafe {
            ptr::copy_nonoverlapping_memory(d.as_mut_ptr(), c.as_ptr(), d.len());
        }
    }

    if n == 1 {
        blas::dgemv(blas::NORMAL, m as i32, p as i32, 1.0, a, m as i32, b, 1,
            1.0, d, 1);
    } else {
        blas::dgemm(blas::NORMAL, blas::NORMAL, m as i32, n as i32, p as i32,
            1.0, a, m as i32, b, p as i32, 1.0, d, m as i32);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use super::{Matrix, multiply, multiply_add};

    macro_rules! assert_equal(
        ($given:expr , $expected:expr) => ({
            assert_eq!($given.len(), $expected.len());
            for i in range(0u, $given.len()) {
                assert_eq!($given[i], $expected[i]);
            }
        });
    )

    #[test]
    fn test_new() {
        let matrix: Matrix<f64> = Matrix::new(100, 100);
        assert!(matrix.len() == 10_000);
    }

    #[test]
    fn test_multiply_matrix_vector() {
        let (m, p, n) = (2, 4, 1);

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0, 0.0];

        multiply(a, b, c, m, p, n);

        let expected_c = [50.0, 60.0];
        assert_equal!(c, expected_c);
    }

    #[test]
    fn test_multiply_add() {
        let (m, p, n) = (2, 3, 4);

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        multiply_add(a, b, c, d, m, p, n);

        let expected_c = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_equal!(c, expected_c);

        let expected_d = [23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0];
        assert_equal!(d, expected_d);
    }

    #[bench]
    fn bench_new(bench: &mut Bencher) {
        bench.iter(|| {
            let matrix: Matrix<f64> = Matrix::new(100, 100);
            assert!(matrix.len() == 10_000);
        })
    }

    #[bench]
    fn bench_multiply_matrix_matrix(bench: &mut Bencher) {
        let (m, p, n) = (1000 + 1, 1000 + 2, 1000 + 3);

        let a = Matrix::new(m, p);
        let b = Matrix::new(p, n);
        let mut c = Matrix::new(m, n);

        let sa = a.as_slice();
        let sb = b.as_slice();
        let sc = c.as_mut_slice();

        bench.iter(|| {
            multiply(sa, sb, sc, m, p, n)
        })
    }

    #[bench]
    fn bench_multiply_matrix_vector(bench: &mut Bencher) {
        let m = 1000;

        let a = Matrix::new(m, m);
        let b = Matrix::new(m, 1);
        let mut c = Matrix::new(m, m);

        let sa = a.as_slice();
        let sb = b.as_slice();
        let sc = c.as_mut_slice();

        bench.iter(|| {
            multiply(sa, sb, sc, m, m, 1)
        })
    }
}
