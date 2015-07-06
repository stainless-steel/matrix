//! Matrix storage schemes.

mod band;
mod compressed;
mod dense;
mod diagonal;
mod packed;

pub use band::BandMatrix;
pub use compressed::{CompressedMatrix, CompressedFormat};
pub use dense::DenseMatrix;
pub use diagonal::DiagonalMatrix;
pub use packed::{PackedMatrix, PackedFormat};

/// An element of a matrix.
pub trait Element: Copy {
    /// Return the zero element.
    fn zero() -> Self;
}

macro_rules! implement(
    ($kind:ty, $zero:expr) => (
        impl Element for $kind {
            #[inline(always)]
            fn zero() -> Self {
                $zero
            }
        }
    );
);

implement!(u8, 0);
implement!(u16, 0);
implement!(u32, 0);
implement!(u64, 0);

implement!(i8, 0);
implement!(i16, 0);
implement!(i32, 0);
implement!(i64, 0);

implement!(f32, 0.0);
implement!(f64, 0.0);

implement!(isize, 0);
implement!(usize, 0);
