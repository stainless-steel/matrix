extern crate blas;

#[inline]
pub fn multiply(a: *const f64, b: *const f64, c: *mut f64, m: uint, p: uint, n: uint) {
    if n == 1 {
        blas::dgemv('N' as i8, m as i32, p as i32, 1.0, a, m as i32, b, 1, 0.0, c, 1);
    } else {
        blas::dgemm('N' as i8, 'N' as i8, m as i32, n as i32, p as i32,
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
        blas::dgemv('N' as i8, m as i32, p as i32, 1.0, a, m as i32, b, 1, 1.0, d, 1);
    } else {
        blas::dgemm('N' as i8, 'N' as i8, m as i32, n as i32, p as i32,
            1.0, a, m as i32, b, p as i32, 1.0, d, m as i32);
    }
}
