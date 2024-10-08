use crate::concurrent::{decode_concurrent, encode_concurrent, fit_concurrent};
use crate::core::{decode, encode, fit};
use std::sync::Once;
use std::collections::HashMap;

static INIT: Once = Once::new();

fn init_env_logger() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

#[test]
fn test_fit() {
    init_env_logger();
    let units = vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
    let (encoded_units, merges) = fit(units, 10);

    let expected_encoded_units = vec![6, 7, 8, 9, 9, 5];
    assert_eq!(expected_encoded_units, encoded_units);

    let expected_merges: HashMap<(i32, i32), i32> = [
        ((0, 1), 6),
        ((8, 4), 9),
        ((7, 3), 8),
        ((6, 2), 7),
    ].iter().cloned().collect();
    assert_eq!(expected_merges, merges);
}

#[test]
fn test_core() {
    init_env_logger();
    let units = vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
    let (_encoded_units, merges) = fit(units, 10);
    let units_to_encode = vec![0, 1, 0, 1, 2, 3, 4, 5];
    let units_to_encode_copy = units_to_encode.clone();
    let encoded = encode(units_to_encode, &merges);
    let decoded = decode(encoded, &merges);
    assert_eq!(units_to_encode_copy, decoded);
}

#[test]
fn test_fit_concurrent() {
    init_env_logger();
    let units_list = vec![
        vec![0, 1, 0, 1, 2, 0, 1, 2, 3],
        vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
    ];
    let (encoded_units, merges) = fit_concurrent(units_list, 10);

    let expected_encoded_units = vec![vec![6, 7, 8], vec![9, 9, 5]];
    assert_eq!(expected_encoded_units, encoded_units);

    let expected_merges: HashMap<(i32, i32), i32> = [
        ((0, 1), 6),
        ((8, 4), 9),
        ((7, 3), 8),
        ((6, 2), 7),
    ].iter().cloned().collect();
    assert_eq!(expected_merges, merges);
}

#[test]
fn test_concurrent() {
    init_env_logger();
    let units_list = vec![
        vec![0, 1, 0, 1, 2, 0, 1, 2, 3],
        vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
    ];
    let (_encoded_units, merges) = fit_concurrent(units_list, 10);

    let units_list_to_encode = vec![vec![0, 1, 0, 1, 2, 3, 4, 5], vec![0, 1, 2, 0, 1, 2, 3]];
    let units_list_to_encode_copy = units_list_to_encode.clone();
    let encoded = encode_concurrent(units_list_to_encode, &merges);
    let decoded = decode_concurrent(encoded, &merges);

    assert_eq!(units_list_to_encode_copy, decoded)
}
