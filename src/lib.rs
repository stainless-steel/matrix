//! Matrix storage schemes.

extern crate num;

mod band;
mod compressed;
mod dense;
mod diagonal;
mod packed;

pub use band::BandMatrix;
pub use compressed::{CompressedMatrix, CompressedFormat};
pub use dense::DenseMatrix;
pub use diagonal::DiagonalMatrix;
pub use packed::{PackedMatrix, PackedFormat};
