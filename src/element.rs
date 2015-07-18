#[cfg(feature = "complex")]
use complex::{c32, c64};

/// An element.
pub trait Element: Copy + PartialEq {
    /// Check if the element is zero.
    fn is_zero(&self) -> bool;

    /// Return the zero element.
    fn zero() -> Self;
}

macro_rules! implement(
    ($name:ty, $zero:expr) => (
        impl Element for $name {
            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == $zero
            }

            #[inline(always)]
            fn zero() -> Self {
                $zero
            }
        }
    );
    ($name:ty) => (
        implement!($name, 0);
    );
);

implement!(u8);
implement!(u16);
implement!(u32);
implement!(u64);

implement!(i8);
implement!(i16);
implement!(i32);
implement!(i64);

implement!(f32, 0.0);
implement!(f64, 0.0);

implement!(isize);
implement!(usize);

#[cfg(feature = "complex")]
implement!(c32, c32(0.0, 0.0));

#[cfg(feature = "complex")]
implement!(c64, c64(0.0, 0.0));
