#![feature(macro_rules)]

extern crate matrix;

macro_rules! assert_equal(
    ($given:expr , $expected:expr) => ({
        assert_eq!($given.len(), $expected.len());
        for i in range(0u, $given.len()) {
            assert_eq!($given[i], $expected[i]);
        }
    });
)

macro_rules! assert_almost_equal(
    ($given:expr, $expected:expr) => ({
        assert_eq!($given.len(), $expected.len());
        for i in range(0u, $given.len()) {
            assert!(::std::num::abs($given[i] - $expected[i]) < 1e-8);
        }
    });
)

#[test]
fn multiply() {
    let (m, p, n) = (2, 4, 1);

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0, 0.0];

    matrix::multiply(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, p, n);

    let expected_c = vec![50.0, 60.0];
    assert_equal!(c, expected_c);
}

#[test]
fn multiply_add() {
    let (m, p, n) = (2, 3, 4);

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut d = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    matrix::multiply_add(a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_mut_ptr(), m, p, n);

    let expected_c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_equal!(c, expected_c);

    let expected_d = vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0];
    assert_equal!(d, expected_d);
}

#[test]
fn sym_eig() {
    let m = 5;

    let a = vec![
        0.814723686393179,
        0.097540404999410,
        0.157613081677548,
        0.141886338627215,
        0.655740699156587,
        0.097540404999410,
        0.278498218867048,
        0.970592781760616,
        0.421761282626275,
        0.035711678574190,
        0.157613081677548,
        0.970592781760616,
        0.957166948242946,
        0.915735525189067,
        0.849129305868777,
        0.141886338627215,
        0.421761282626275,
        0.915735525189067,
        0.792207329559554,
        0.933993247757551,
        0.655740699156587,
        0.035711678574190,
        0.849129305868777,
        0.933993247757551,
        0.678735154857773,
    ];

    let mut vecs = Vec::from_elem(m * m, 0.0);
    let mut vals = Vec::from_elem(m, 0.0);

    assert!(matrix::sym_eig(a.as_ptr(), vecs.as_mut_ptr(), vals.as_mut_ptr(), m).is_ok());

    let expected_vecs = vec![
         0.200767588469279,
        -0.613521879994358,
         0.529492579537623,
         0.161735212201923,
        -0.526082320114459,
        -0.241005628008408,
        -0.272281143378657,
         0.443280672960843,
        -0.675165120368165,
         0.464148221418878,
         0.509762909240926,
         0.555609456752178,
         0.244072927029371,
        -0.492754485897426,
        -0.359251069377747,
        -0.766321363493223,
         0.386556170387878,
         0.341170928524320,
         0.084643789583352,
        -0.373849864790357,
         0.233456648876442,
         0.302202482503382,
         0.589211894835079,
         0.517708631263932,
         0.488854547655902,
    ];

    let expected_vals = vec![
        -0.671640666831794,
        -0.230366398529950,
         0.397221322493687,
         0.999582068576074,
         3.026535012212483,
    ];

    assert_almost_equal!(vecs, expected_vecs);
    assert_almost_equal!(vals, expected_vals);
}
