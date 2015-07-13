#[cfg(feature = "complex")]
use complex::{c32, c64};

/// A matrix element.
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

#[cfg(feature = "complex")]
element!(c32, c32(0.0, 0.0));

#[cfg(feature = "complex")]
element!(c64, c64(0.0, 0.0));
