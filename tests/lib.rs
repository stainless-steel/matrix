#![allow(non_snake_case)]

extern crate assert;
extern crate matrix;

#[test]
fn multiply() {
    let (m, p, n) = (2, 4, 1);

    let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let B = vec![1.0, 2.0, 3.0, 4.0];
    let mut C = vec![0.0, 0.0];

    matrix::multiply(&A, &B, &mut C, m, p, n);

    assert::equal(&C, &vec![50.0, 60.0]);
}

#[test]
fn multiply_add() {
    let (m, p, n) = (2, 3, 4);

    let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let B = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let C = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut D = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    matrix::multiply_add(&A, &B, &C, &mut D, m, p, n);

    assert::equal(&C, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert::equal(&D, &vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0]);
}
