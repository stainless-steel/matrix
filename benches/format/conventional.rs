use random::{self, Source};
use test::Bencher;

use matrix::decomposition::SymmetricEigen;
use matrix::format::Conventional;

#[bench] fn symmetric_eigen_0010(b: &mut Bencher) { symmetric_eigen(10, b); }
#[bench] fn symmetric_eigen_0100(b: &mut Bencher) { symmetric_eigen(100, b); }
#[bench] fn symmetric_eigen_1000(b: &mut Bencher) { symmetric_eigen(1000, b); }

fn symmetric_eigen(size: usize, bencher: &mut Bencher) {
    let matrix = make_symmetric(size);
    bencher.iter(|| matrix.decompose());
}

fn make_symmetric(size: usize) -> Conventional<f64> {
    let mut source = random::default().seed([69, 42]);
    let values = source.iter().take(size * size).collect::<Vec<f64>>();
    Conventional::from_vec(size, values)
}
