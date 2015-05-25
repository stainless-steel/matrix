//! Matrix storage schemes.

#[cfg(test)]
extern crate assert;

extern crate num;

mod band;
mod compressed;
mod dense;
mod diagonal;
mod packed;

pub use band::Matrix as Band;
pub use compressed::Matrix as Compressed;
pub use compressed::Format as CompressedFormat;
pub use dense::Matrix as Dense;
pub use diagonal::Matrix as Diagonal;
pub use packed::Matrix as Packed;
pub use packed::Format as PackedFormat;
