# Matrix [![Package][package-img]][package-url] [![Documentation][documentation-img]][documentation-url] [![Build][build-img]][build-url]

The package provides a matrix laboratory.

## Example

```rust
#[macro_use]
extern crate matrix;

use matrix::prelude::*;

let mut sparse = Compressed::zero((2, 4));
sparse.set((0, 0), 42.0);
sparse.set((1, 3), 69.0);

let dense = Conventional::from(&sparse);
assert_eq!(
    &*dense,
    &*matrix![
        42.0, 0.0, 0.0,  0.0;
         0.0, 0.0, 0.0, 69.0;
    ]
);
```

## Contribution

Your contribution is highly appreciated. Do not hesitate to open an issue or a
pull request. Note that any contribution submitted for inclusion in the project
will be licensed according to the terms given in [LICENSE.md](LICENSE.md).

[build-img]: https://travis-ci.org/stainless-steel/matrix.svg?branch=master
[build-url]: https://travis-ci.org/stainless-steel/matrix
[documentation-img]: https://docs.rs/matrix/badge.svg
[documentation-url]: https://docs.rs/matrix
[package-img]: https://img.shields.io/crates/v/matrix.svg
[package-url]: https://crates.io/crates/matrix
