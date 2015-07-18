//! Storage schemes.

#[cfg(debug_assertions)]
trait Validate {
    fn validate(&self);
}

#[cfg(debug_assertions)]
macro_rules! validate(
    ($matrix:expr) => ({
        use ::storage::Validate;
        let matrix = $matrix;
        matrix.validate();
        matrix
    });
);

#[cfg(not(debug_assertions))]
macro_rules! validate(
    ($matrix:expr) => ($matrix);
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

macro_rules! min(
    ($left:expr, $right:expr) => ({
        let (left, right) = ($left, $right);
        if left > right { right } else { left }
    });
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
