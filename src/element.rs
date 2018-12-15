use {c32, c64};

/// An element.
pub trait Element: Copy + PartialEq {
    /// Check if the element is zero.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }

    /// Return the zero element.
    fn zero() -> Self;
}

macro_rules! implement(
    ($name:ty, $zero:expr) => (
        impl Element for $name {
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

implement!(bool, false);
implement!(f32, 0.0);
implement!(f64, 0.0);

implement!(isize);
implement!(usize);

implement!(c32, c32::new(0.0, 0.0));
implement!(c64, c64::new(0.0, 0.0));
