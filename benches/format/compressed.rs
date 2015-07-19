use random::{self, Source};
use test::Bencher;

use matrix::format::compressed::Variant;
use matrix::format::{Compressed, Conventional};
use matrix::operation::{MultiplyInto};

#[bench] fn multiply_left_0010(b: &mut Bencher) { multiply_left(10, 10, b); }
#[bench] fn multiply_left_0100(b: &mut Bencher) { multiply_left(100, 100, b); }
#[bench] fn multiply_left_1000(b: &mut Bencher) { multiply_left(1000, 1000, b); }

#[bench] fn multiply_left_0010_dense(b: &mut Bencher) { multiply_left_dense(10, 10, b); }
#[bench] fn multiply_left_0100_dense(b: &mut Bencher) { multiply_left_dense(100, 100, b); }
#[bench] fn multiply_left_1000_dense(b: &mut Bencher) { multiply_left_dense(1000, 1000, b); }

#[bench] fn multiply_right_0010(b: &mut Bencher) { multiply_right(10, 10, b); }
#[bench] fn multiply_right_0100(b: &mut Bencher) { multiply_right(100, 100, b); }
#[bench] fn multiply_right_1000(b: &mut Bencher) { multiply_right(1000, 1000, b); }

#[bench] fn multiply_right_0010_dense(b: &mut Bencher) { multiply_right_dense(10, 10, b); }
#[bench] fn multiply_right_0100_dense(b: &mut Bencher) { multiply_right_dense(100, 100, b); }
#[bench] fn multiply_right_1000_dense(b: &mut Bencher) { multiply_right_dense(1000, 1000, b); }

fn multiply_left(size: usize, nonzeros: usize, bencher: &mut Bencher) {
    let left = make_compressed(size, size, nonzeros);
    let right = make_conventional(size, size);
    let mut result = Conventional::new(size);
    bencher.iter(|| left.multiply_into(&right, &mut result));
}

fn multiply_left_dense(size: usize, nonzeros: usize, bencher: &mut Bencher) {
    let left = Conventional::from(make_compressed(size, size, nonzeros));
    let right = make_conventional(size, size);
    let mut result = Conventional::new(size);
    bencher.iter(|| left.multiply_into(&right, &mut result));
}

fn multiply_right(size: usize, nonzeros: usize, bencher: &mut Bencher) {
    let left = make_conventional(size, size);
    let right = make_compressed(size, size, nonzeros);
    let mut result = Conventional::new(size);
    bencher.iter(|| left.multiply_into(&right, &mut result));
}

fn multiply_right_dense(size: usize, nonzeros: usize, bencher: &mut Bencher) {
    let left = make_conventional(size, size);
    let right = Conventional::from(&make_compressed(size, size, nonzeros));
    let mut result = Conventional::new(size);
    bencher.iter(|| left.multiply_into(&right, &mut result));
}

fn make_compressed(rows: usize, columns: usize, nonzeros: usize) -> Compressed<f64> {
    let mut source = random::default().seed([42, 69]);
    let mut matrix = Compressed::with_capacity((rows, columns), Variant::Column, nonzeros);
    for _ in 0..nonzeros {
        let i = source.read::<u64>() as usize % rows;
        let j = source.read::<u64>() as usize % columns;
        matrix.set((i, j), source.read::<f64>());
    }
    matrix
}

fn make_conventional(rows: usize, columns: usize) -> Conventional<f64> {
    let mut source = random::default().seed([69, 42]);
    let values = source.iter().take(rows * columns).collect::<Vec<f64>>();
    Conventional::from_vec((rows, columns), values)
}
