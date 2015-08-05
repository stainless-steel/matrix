//! Reexports of modules, traits, and types.

pub use Element;
pub use Matrix;
pub use Position;
pub use Size;

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

pub use operation::Multiply;
pub use operation::MultiplyInto;
pub use operation::MultiplySelf;
pub use operation::ScaleSelf;
pub use operation::Transpose;

pub use decomposition::SymmetricEigen;
