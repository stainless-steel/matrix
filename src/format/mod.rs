//! Storage formats.

#[cfg(debug_assertions)]
trait Validate {
    fn validate(&self);
}

macro_rules! buffer(
    ($capacity:expr) => ({
        let capacity = $capacity as usize;
        let mut buffer = Vec::with_capacity(capacity);
        buffer.set_len(capacity);
        buffer
    });
);

macro_rules! min(
    ($left:expr, $right:expr) => ({
        let (left, right) = ($left, $right);
        if left > right { right } else { left }
    });
);

macro_rules! size(
    ($kind:ident, $rows:ident, $columns:ident) => (
        impl<T: ::Element> ::Size for $kind<T> {
            #[inline(always)]
            fn rows(&self) -> usize {
                self.$rows
            }

            #[inline(always)]
            fn columns(&self) -> usize {
                self.$columns
            }
        }
    );
    ($kind:ident) => (
        size!($kind, rows, columns);
    );
);

#[cfg(debug_assertions)]
macro_rules! validate(
    ($matrix:expr) => ({
        use ::format::Validate;
        let matrix = $matrix;
        matrix.validate();
        matrix
    });
);

#[cfg(not(debug_assertions))]
macro_rules! validate(
    ($matrix:expr) => ($matrix);
);

pub mod banded;
pub mod compressed;
pub mod conventional;
pub mod diagonal;
pub mod packed;

pub use self::banded::Banded;
pub use self::compressed::Compressed;
pub use self::conventional::Conventional;
pub use self::diagonal::Diagonal;
pub use self::packed::Packed;
