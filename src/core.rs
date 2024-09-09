use log::{debug, info};
use std::collections::{HashMap, HashSet};

pub fn get_counts(units: &[i32]) -> HashMap<(i32, i32), i32> {
    let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

    for pair in units.windows(2) {
        let pair_tuple = (pair[0], pair[1]);
        *counts.entry(pair_tuple).or_insert(0) += 1;
    }

    counts
}

pub fn merge(units: &[i32], pair: &(i32, i32), idx: i32) -> Vec<i32> {
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
