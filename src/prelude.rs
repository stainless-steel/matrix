//! Reexports of modules, traits, and types.

pub use Element;
pub use Matrix;
pub use Position;
pub use Size;

pub use algebra::MultiplyInto;
pub use algebra::MultiplySelf;
pub use algebra::MultiplyThat;

pub use format::banded;
pub use format::compressed;
pub use format::conventional;
pub use format::diagonal;
pub use format::packed;

pub use format::banded::Banded;
pub use format::compressed::Compressed;
pub use format::conventional::Conventional;
pub use format::diagonal::Diagonal;
pub use format::packed::Packed;