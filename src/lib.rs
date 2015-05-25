//! Matrix storage schemes.

#[cfg(test)]
extern crate assert;

extern crate num;

mod band;
mod compressed;
mod dense;
mod diagonal;
mod packed;

pub use band::Band;
pub use compressed::{Compressed, CompressedFormat};
pub use dense::Dense;
pub use diagonal::Diagonal;
pub use packed::{Packed, PackedFormat};
