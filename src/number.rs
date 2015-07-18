#[cfg(feature = "complex")]
use complex::{c32, c64};

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A number.
pub trait Number: Add<Output=Self> +
                  Div<Output=Self> +
                  Mul<Output=Self> +
                  Neg<Output=Self> +
                  Sub<Output=Self> +
                  Copy + PartialEq {
}

macro_rules! implement(
    ($name:ty) => (
        impl Number for $name {
        }
    );
);

implement!(f32);
implement!(f64);

#[cfg(feature = "complex")]
implement!(c32);

#[cfg(feature = "complex")]
implement!(c64);
