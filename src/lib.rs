//! Matrix storage schemes.

mod band;
mod compressed;
mod dense;
mod diagonal;
mod packed;

pub use band::Band;
pub use compressed::{Compressed, CompressedFormat};
pub use dense::Dense;
pub use diagonal::Diagonal;
pub use packed::{Packed, PackedFormat};

/// An element of a matrix.
pub trait Element: Copy {
    /// Return the zero element.
    fn zero() -> Self;
}

macro_rules! element(
    ($kind:ty, $zero:expr) => (
        impl Element for $kind {
            #[inline(always)]
            fn zero() -> Self {
                $zero
            }
        }
    );
    ($kind:ty) => (
        element!($kind, 0);
    );
);

element!(u8);
element!(u16);
element!(u32);
element!(u64);

element!(i8);
element!(i16);
element!(i32);
element!(i64);

element!(f32, 0.0);
element!(f64, 0.0);

element!(isize);
element!(usize);
