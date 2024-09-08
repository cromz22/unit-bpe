use std::collections::{HashMap, HashSet};

fn get_counts(units: &[i32]) -> HashMap<(i32, i32), i32> {
    let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

    for pair in units.windows(2) {
        let pair_tuple = (pair[0], pair[1]);
        *counts.entry(pair_tuple).or_insert(0) += 1;
    }

    counts
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

fn fit(mut units: Vec<i32>, target_vocab_size: usize) -> (Vec<i32>, HashMap<(i32, i32), i32>) {
    let mut merges = HashMap::new();
    let initial_vocab_size = units.iter().cloned().collect::<HashSet<_>>().len();
    let mut max_idx = *units.iter().max().unwrap();

    if target_vocab_size <= initial_vocab_size {
        let error_message = format!(
            "Target vocab size ({}) must be greater than the initial vocab size ({}).",
            target_vocab_size, initial_vocab_size
        );
        panic!("{}", error_message);
    }

    let num_merges = target_vocab_size - initial_vocab_size;
    println!("Performing {} merges. Units: {:?}", num_merges, units);

    for i in 0..num_merges {
        let counts = get_counts(&units);
        if counts.is_empty() {
            println!("No more pairs to merge.");
            break;
        }
        let top_pair = counts.iter().max_by_key(|(_, &v)| v).unwrap().0;
        let new_idx = max_idx + 1;
        units = merge(&units, top_pair, new_idx);
        merges.insert(*top_pair, new_idx);
        println!(
            "Merge {}/{}: {:?} -> {}; Units: {:?}",
            i + 1,
            num_merges,
            top_pair,
            new_idx,
            units
        );

        max_idx = new_idx;
    }

    (units, merges)
}

fn encode(mut units: Vec<i32>, merges: &HashMap<(i32, i32), i32>) -> Vec<i32> {
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

fn decode(units: Vec<i32>, merges: &HashMap<(i32, i32), i32>) -> Vec<i32> {
    let swapped_merges: HashMap<i32, (i32, i32)> = merges.iter().map(|(k, v)| (*v, *k)).collect();
    let swapped_merges_keys = &swapped_merges.keys().cloned().collect();

    let mut units_set: HashSet<i32> = units.iter().cloned().collect();
    let mut decoded_units = units.clone();

    while !units_set.is_disjoint(swapped_merges_keys) {
        let mut new_units = Vec::new();
        let mut i = 0;

        while i < decoded_units.len() {
            if let Some(&(a, b)) = swapped_merges.get(&decoded_units[i]) {
                new_units.push(a);
                new_units.push(b);
            } else {
                new_units.push(decoded_units[i]);
            }
            i += 1;
        }

        decoded_units = new_units;
        units_set = decoded_units.iter().cloned().collect();
    }

    decoded_units
}

fn main() {
    let units = vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
    let (_encoded_units, merges) = fit(units, 10);
    let units_to_encode = vec![0, 1, 0, 1, 2, 3, 4, 5];
    let encoded = encode(units_to_encode, &merges);
    println!("{:?}", encoded);
    let decoded = decode(encoded, &merges);
    println!("{:?}", decoded)
}
