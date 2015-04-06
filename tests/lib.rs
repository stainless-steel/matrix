extern crate assert;
extern crate matrix;

#[test]
fn multiply() {
    let (m, p, n) = (2, 4, 1);

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0, 0.0];

    matrix::multiply(&a, &b, &mut c, m, p, n);

    let expected_c = vec![50.0, 60.0];
    assert::equal(&c, &expected_c);
}

#[test]
fn multiply_add() {
    let (m, p, n) = (2, 3, 4);

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut d = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    matrix::multiply_add(&a, &b, &c, &mut d, m, p, n);

    let expected_c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert::equal(&c, &expected_c);

    let expected_d = vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0];
    assert::equal(&d, &expected_d);
}
