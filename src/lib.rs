extern crate blas;
extern crate lapack;

#[inline]
pub fn multiply(a: *const f64, b: *const f64, c: *mut f64, m: uint, p: uint, n: uint) {
    if n == 1 {
        blas::dgemv('N' as i8, m as i32, p as i32, 1.0, a, m as i32, b, 1, 0.0, c, 1);
    } else {
        blas::dgemm('N' as i8, 'N' as i8, m as i32, n as i32, p as i32, 1.0, a,
                    m as i32, b, p as i32, 0.0, c, m as i32);
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
        blas::dgemm('N' as i8, 'N' as i8, m as i32, n as i32, p as i32, 1.0, a,
                    m as i32, b, p as i32, 1.0, d, m as i32);
    }
}

/// Performs the eigendecomposition of a symmetric m-by-m matrix and stores
/// the resulting eigenvectors and eigenvalues in an m-by-m matrix and
/// m-by-1 matrix, respectively.
///
/// https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices
pub fn sym_eig(a: *const f64, vecs: *mut f64, vals: *mut f64, m: uint)
    -> Result<(), int> {

    if a != (vecs as *const f64) {
        // NOTE: Only the upper triangular matrix is actually needed; however,
        // copying only that part might not be optimal for performance. Check!
        unsafe {
            std::ptr::copy_nonoverlapping_memory(vecs, a, m * m);
        }
    }

    // The size of the temporary array should have at least 3 * m - 1.
    let mut temp = Vec::from_elem(4 * m, 0.0);
    let mut flag = 0;

    lapack::dsyev('V' as i8, 'U' as i8, m as i32, vecs, m as i32, vals,
                  temp.as_mut_ptr(), 4 * m as i32, &mut flag);

    match flag {
        0 => Ok(()),
        _ => Err(flag as int),
    }
}
