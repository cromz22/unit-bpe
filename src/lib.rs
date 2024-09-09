use dashmap::{DashMap, DashSet};
use log::{debug, info};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

fn get_counts(units: &[i32]) -> HashMap<(i32, i32), i32> {
    let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

    for pair in units.windows(2) {
        let pair_tuple = (pair[0], pair[1]);
        *counts.entry(pair_tuple).or_insert(0) += 1;
    }

    counts
}

fn get_counts_concurrent(units_list: &[Vec<i32>]) -> HashMap<(i32, i32), i32> {
    let global_counts = DashMap::new();

    units_list.par_iter().for_each(|units| {
        let local_counts = get_counts(units);
        for (pair, count) in local_counts {
            *global_counts.entry(pair).or_insert(0) += count;
        }
    });

    global_counts.into_iter().collect()
}

fn merge(units: &[i32], pair: &(i32, i32), idx: i32) -> Vec<i32> {
    let mut new_units = Vec::new();
    let mut i = 0;
    while i < units.len() {
        if i < units.len() - 1 && (units[i], units[i + 1]) == *pair {
            new_units.push(idx);
            i += 2;
        } else {
            new_units.push(units[i]);
            i += 1;
        }
    }

    new_units
}

fn merge_concurrent(units_list: &[Vec<i32>], pair: &(i32, i32), idx: i32) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| merge(units, pair, idx))
        .collect()
}

pub fn fit(mut units: Vec<i32>, target_vocab_size: usize) -> (Vec<i32>, HashMap<(i32, i32), i32>) {
    let mut merges = HashMap::new();
    let initial_vocab_size = units.iter().cloned().collect::<HashSet<_>>().len();
    let mut max_idx = *units.iter().max().unwrap();

    if target_vocab_size <= initial_vocab_size {
        panic!(
            "Target vocab size ({}) must be greater than the initial vocab size ({}).",
            target_vocab_size, initial_vocab_size
        );
    }

    let num_merges = target_vocab_size - initial_vocab_size;
    info!("Performing {} merges.", num_merges);
    debug!("Initial units: {:?}", units);

    for i in 0..num_merges {
        let counts = get_counts(&units);
        if counts.is_empty() {
            info!("No pairs to merge.");
            break;
        }
        let top_pair = counts.iter().max_by_key(|(_, &v)| v).unwrap().0;
        let new_idx = max_idx + 1;
        units = merge(&units, top_pair, new_idx);
        merges.insert(*top_pair, new_idx);
        info!(
            "Merge {}/{}: {:?} -> {}",
            i + 1,
            num_merges,
            top_pair,
            new_idx,
        );
        debug!("Units: {:?}", units);

        max_idx = new_idx;
    }

    (units, merges)
}

#[pyfunction]
pub fn fit_py(units: Vec<i32>, target_vocab_size: usize) -> (Vec<i32>, Vec<((i32, i32), i32)>) {
    let (units, merges) = fit(units, target_vocab_size);

    (
        units,
        merges.into_iter().collect::<Vec<((i32, i32), i32)>>(),
    )
}

pub fn fit_concurrent(
    mut units_list: Vec<Vec<i32>>,
    target_vocab_size: usize,
) -> (Vec<Vec<i32>>, HashMap<(i32, i32), i32>) {
    let unique_units = DashSet::new();
    let max_idx = units_list
        .par_iter()
        .flat_map(|units| units.par_iter().cloned())
        .inspect(|&unit| {
            unique_units.insert(unit);
        })
        .max()
        .unwrap();

    let initial_vocab_size = unique_units.len();
    if target_vocab_size <= initial_vocab_size {
        panic!(
            "Target vocab size ({}) must be greater than the initial vocab size ({}).",
            target_vocab_size, initial_vocab_size
        );
    }

    let num_merges = target_vocab_size - initial_vocab_size;
    info!("Performing {} merges.", num_merges);
    debug!("Initial units: {:?}", units_list);

    let merges = DashMap::new();
    let mut current_max_idx = max_idx;

    for i in 0..num_merges {
        let counts = get_counts_concurrent(&units_list);
        if counts.is_empty() {
            info!("No pairs to merge.");
            break;
        }
        let top_pair = counts.iter().max_by_key(|(_, &v)| v).unwrap().0;
        let new_idx = current_max_idx + 1;
        units_list = merge_concurrent(&units_list, top_pair, new_idx);
        merges.insert(*top_pair, new_idx);
        info!(
            "Merge {}/{}: {:?} -> {}",
            i + 1,
            num_merges,
            top_pair,
            new_idx
        );
        debug!("Units: {:?}", units_list);

        current_max_idx = new_idx;
    }

    (units_list, merges.into_iter().collect())
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

pub fn encode(mut units: Vec<i32>, merges: &HashMap<(i32, i32), i32>) -> Vec<i32> {
    while units.len() >= 2 {
        let counts = get_counts(&units);
        let pair_to_merge = counts
            .keys()
            .min_by_key(|pair| merges.get(pair).unwrap_or(&i32::MAX))
            .unwrap();
        if !merges.contains_key(pair_to_merge) {
            break;
        }
        let idx = merges[pair_to_merge];
        units = merge(&units, pair_to_merge, idx);
    }
    units
}

#[pyfunction]
pub fn encode_py(
    units: Vec<i32>,
    merges: Vec<((i32, i32), i32)>,
) -> Vec<i32> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    encode(units, &merges_map)
}

pub fn encode_concurrent(
    units_list: Vec<Vec<i32>>,
    merges: &HashMap<(i32, i32), i32>,
) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| encode(units.clone(), merges))
        .collect()
}

#[pyfunction]
pub fn encode_concurrent_py(
    units_list: Vec<Vec<i32>>,
    merges: Vec<((i32, i32), i32)>,
) -> Vec<Vec<i32>> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    encode_concurrent(units_list, &merges_map)
}

pub fn decode(units: Vec<i32>, merges: &HashMap<(i32, i32), i32>) -> Vec<i32> {
    let swapped_merges: HashMap<i32, (i32, i32)> = merges.iter().map(|(k, v)| (*v, *k)).collect();
    let mut decoded_units = units.clone();

    loop {
        let mut has_replacement = false;
        let mut new_units = Vec::new();
        let mut i = 0;

        while i < decoded_units.len() {
            if let Some(&(a, b)) = swapped_merges.get(&decoded_units[i]) {
                new_units.push(a);
                new_units.push(b);
                has_replacement = true;
            } else {
                new_units.push(decoded_units[i]);
            }
            i += 1;
        }

        if !has_replacement {
            break;
        }

        decoded_units = new_units;
    }

    decoded_units
}

#[pyfunction]
pub fn decode_py(
    units: Vec<i32>,
    merges: Vec<((i32, i32), i32)>,
) -> Vec<i32> {
    let merges_map: HashMap<(i32, i32), i32> = merges.iter().cloned().collect();

    decode(units, &merges_map)
}

pub fn decode_concurrent(
    units_list: Vec<Vec<i32>>,
    merges: &HashMap<(i32, i32), i32>,
) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| decode(units.clone(), merges))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_env_logger() {
        INIT.call_once(|| {
            env_logger::init();
        });
    }

    #[test]
    fn test_fit_encode_decode() {
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
}
