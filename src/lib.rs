//! Algorithms for manipulating real matrices.

extern crate blas;
extern crate lapack;

/// Multiplies two matrices.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is stored in an `m`-by-`n` matrix `c`.
#[inline]
pub fn multiply(a: *const f64, b: *const f64, c: *mut f64, m: uint, p: uint, n: uint) {
    if n == 1 {
        blas::dgemv(b'N', m, p, 1.0, a, m, b, 1, 0.0, c, 1);
    } else {
        blas::dgemm(b'N', b'N', m, n, p, 1.0, a, m, b, p, 0.0, c, m);
    }
}

/// Multiplies two matrices and adds another matrix.
///
/// An `m`-by-`p` matrix `a` is multiplied by a `p`-by-`n` matrix `b`; the
/// result is summed up with an `m`-by-`n` matrix `c` and stored in an
/// `m`-by-`n` matrix `d`.
#[inline]
pub fn multiply_add(a: *const f64, b: *const f64, c: *const f64, d: *mut f64,
                    m: uint, p: uint, n: uint) {

    if c != (d as *const f64) {
        unsafe {
            std::ptr::copy_nonoverlapping_memory(d, c, m * n);
        }
    }

    if n == 1 {
        blas::dgemv(b'N', m, p, 1.0, a, m, b, 1, 1.0, d, 1);
    } else {
        blas::dgemm(b'N', b'N', m, n, p, 1.0, a, m, b, p, 1.0, d, m);
    }
}

/// Performs the eigendecomposition of a symmetric matrix.
///
/// A symmetric `m`-by-`m` matrix `a` is decomposed; the resulting eigenvectors
/// and eigenvalus are stored in an `m`-by-`m` matrix `vecs` and an `m`-element
/// vector `vals`, respectively.
pub fn sym_eig(a: *const f64, vecs: *mut f64, vals: *mut f64, m: uint) -> Result<(), int> {
    if a != (vecs as *const f64) {
        // NOTE: Only the upper triangular matrix is actually needed; however,
        // copying only that part might not be optimal for performance. Check!
        unsafe {
            std::ptr::copy_nonoverlapping_memory(vecs, a, m * m);
        }
    }

    // The size of the temporary array should be >= max(1, 3 * m - 1).
    // http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html
    let mut temp = Vec::from_elem(4 * m, 0.0);
    let mut flag = 0;

    lapack::dsyev(b'V', b'U', m, vecs, m, vals, temp.as_mut_ptr(), 4 * m, &mut flag);

    if flag == 0 { Ok(()) } else { Err(flag) }
}
