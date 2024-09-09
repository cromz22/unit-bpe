use crate::concurrent::{decode_concurrent, encode_concurrent, fit_concurrent};
use crate::core::{decode, encode, fit};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn fit_py(units: Vec<i32>, target_vocab_size: usize) -> (Vec<i32>, Vec<((i32, i32), i32)>) {
    let (units, merges) = fit(units, target_vocab_size);

    (
        units,
        merges.into_iter().collect::<Vec<((i32, i32), i32)>>(),
    )
}

#[pyfunction]
pub fn fit_concurrent_py(
    units_list: Vec<Vec<i32>>,
    target_vocab_size: usize,
) -> (Vec<Vec<i32>>, Vec<((i32, i32), i32)>) {
    let (units_list, merges) = fit_concurrent(units_list, target_vocab_size);

    (
        units_list,
        merges.into_iter().collect::<Vec<((i32, i32), i32)>>(),
    )
}

#[pyfunction]
pub fn encode_py(units: Vec<i32>, merges: Vec<((i32, i32), i32)>) -> Vec<i32> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    encode(units, &merges_map)
}

#[pyfunction]
pub fn encode_concurrent_py(
    units_list: Vec<Vec<i32>>,
    merges: Vec<((i32, i32), i32)>,
) -> Vec<Vec<i32>> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    encode_concurrent(units_list, &merges_map)
}

#[pyfunction]
pub fn decode_py(units: Vec<i32>, merges: Vec<((i32, i32), i32)>) -> Vec<i32> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    decode(units, &merges_map)
}

#[pyfunction]
pub fn decode_concurrent_py(
    units_list: Vec<Vec<i32>>,
    merges: Vec<((i32, i32), i32)>,
) -> Vec<Vec<i32>> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    decode_concurrent(units_list, &merges_map)
}

#[pymodule]
fn unit_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_concurrent_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_concurrent_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_concurrent_py, m)?)?;
    Ok(())
}
