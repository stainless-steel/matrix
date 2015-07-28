# Matrix [![Version][version-img]][version-url] [![Status][status-img]][status-url]

The package provides a matrix laboratory.

## [Documentation][docs]

## Example

```rust
#[macro_use]
extern crate matrix;

use matrix::prelude::*;

let mut sparse = Compressed::new((2, 4), compressed::Variant::Column);
sparse.set((0, 0), 42.0);
sparse.set((1, 3), 69.0);

let dense = Conventional::from(&sparse);
assert_eq!(&*dense, &*matrix![
    42.0, 0.0, 0.0,  0.0;
     0.0, 0.0, 0.0, 69.0;
]);
```

## Contributing

1. Fork the project.
2. Implement your idea.
3. Open a pull request.

[version-img]: http://stainless-steel.github.io/images/crates.svg
[version-url]: https://crates.io/crates/matrix
[status-img]: https://travis-ci.org/stainless-steel/matrix.svg?branch=master
[status-url]: https://travis-ci.org/stainless-steel/matrix
[docs]: https://stainless-steel.github.io/matrix
