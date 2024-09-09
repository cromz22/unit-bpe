use crate::core::{decode, encode, get_counts, merge};
use dashmap::{DashMap, DashSet};
use log::{debug, info};
use rayon::prelude::*;
use std::collections::HashMap;

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

fn merge_concurrent(units_list: &[Vec<i32>], pair: &(i32, i32), idx: i32) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| merge(units, pair, idx))
        .collect()
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

pub fn encode_concurrent(
    units_list: Vec<Vec<i32>>,
    merges: &HashMap<(i32, i32), i32>,
) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| encode(units.clone(), merges))
        .collect()
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
