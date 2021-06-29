//! The packed format.
//!
//! The format is suitable for symmetric, Hermitian, and triangular matrices.
//! The format is compatible with the [one][1] adopted by [LAPACK][2].
//!
//! [1]: http://www.netlib.org/lapack/lug/node123.html
//! [2]: http://www.netlib.org/lapack

use std::fmt;
use {Element, Matrix, Size};

/// A packed matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Packed<T: Element> {
    /// The number of rows or columns.
    pub size: usize,
    /// The format variant.
    pub variant: Variant,
    /// The values of the lower triangle when `variant = Lower` or upper
    /// triangle when `variant = Upper` stored by columns.
    pub values: Vec<T>,
}

macro_rules! new(
    ($size:expr, $variant:expr, $values:expr) => (
        Packed { size: $size, variant: $variant, values: $values }
    );
);

macro_rules! arithmetic(
    ($count:expr, $first:expr, $last:expr) => (
        $count * ($first + $last) / 2
    );
);

macro_rules! storage(
    ($size:expr) => (arithmetic!($size, 1, $size))
);

mod convert;
mod operation;

/// A variant of a packed matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    /// The lower-triangular variant.
    Lower,
    /// The upper-triangular variant.
    Upper,
}

#[cfg(debug_assertions)]
impl<T: Element> ::format::Validate for Packed<T> {
    fn validate(&self) {
        assert_eq!(self.values.len(), storage!(self.size));
    }
}

size!(Packed, size, size);

impl<T: Element> Packed<T> {
    /// Create a zero matrix.
    pub fn new<S: Size>(size: S, variant: Variant) -> Self {
        let (rows, _columns) = size.dimensions();
        debug_assert!(rows == _columns);
        new!(rows, variant, vec![T::zero(); storage!(rows)])
    }
}

impl<T: Element> Matrix for Packed<T> {
    type Element = T;

    fn nonzeros(&self) -> usize {
        self.values
            .iter()
            .fold(0, |sum, &value| if value.is_zero() { sum } else { sum + 1 })
    }

    #[inline]
    fn zero<S: Size>(size: S) -> Self {
        Packed::new(size, Variant::Lower)
    }
}

impl<T: Element> fmt::Display for Packed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[\n")?;
        for i in 0..self.size {
            write!(f, "\t")?;
            for j in 0..self.size {
                match self.variant {
                    Variant::Upper => {
                        if i <= j {
                            write!(f, "{}", self.values[(2+(j+1))*j/2-(j-i)])?;
                        } else {
                            write!(f, "{}", 0.0)?;
                        }
                    }
                    Variant::Lower => {
                        if i >= j {
                            write!(f, "{}", self.values[(i-j)+(self.size+(self.size-j+1))*j/2])?;
                        } else {
                            write!(f, "{}", 0.0)?;
                        }
                    }
                }

                if j == self.size-1 {
                    write!(f, ";")?;
                } else {
                    write!(f, ",\t")?;
                }
            }
            write!(f, "\n")?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl Variant {
    /// Return the other variant.
    #[inline]
    pub fn flip(&self) -> Self {
        match *self {
            Variant::Lower => Variant::Upper,
            Variant::Upper => Variant::Lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use format::packed::Variant;
    use prelude::*;

    #[test]
    fn nonzeros() {
        let matrix = new!(
            4,
            Variant::Lower,
            vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0]
        );
        assert_eq!(matrix.nonzeros(), 7);
    }
}
