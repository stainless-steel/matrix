//! Matrix storage schemes.

#[cfg(test)]
extern crate assert;

extern crate num;

pub mod band;
pub mod compressed;
pub mod dense;
pub mod diagonal;
pub mod packed;
