#![feature(unsafe_destructor)]

extern crate alloc;
extern crate core;

use alloc::heap;
use core::raw;
use std::mem;

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

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use super::Matrix;

    #[test]
    fn test_new() {
        let matrix: Matrix<f64> = Matrix::new(100, 100);
        assert!(matrix.len() == 10_000);
    }

    #[bench]
    fn bench_new(b: &mut Bencher) {
        b.iter(|| {
            let matrix: Matrix<f64> = Matrix::new(100, 100);
            assert!(matrix.len() == 10_000);
        })
    }
}
