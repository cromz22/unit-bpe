use crate::core::{decode, encode};
use log::{debug, info};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use crate::{Pair, Position};

pub fn fit_concurrent(
    units_list: Vec<Vec<i32>>,
    target_vocab_size: usize,
) -> (Vec<Vec<i32>>, HashMap<Pair, i32>) {
    // Compute unique units and max index without shared mutable state
    let (unique_units, max_idx) = units_list
        .par_iter()
        .map(|units| {
            let max_unit = units.iter().cloned().max().unwrap_or_default();
            let unique_units: HashSet<i32> = units.iter().cloned().collect();
            (unique_units, max_unit)
        })
        .reduce(
            || (HashSet::new(), i32::MIN),
            |(mut acc_set, acc_max), (set, max)| {
                acc_set.extend(set);
                (acc_set, acc_max.max(max))
            },
        );

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

    let mut merges: HashMap<Pair, i32> = HashMap::new();
    let mut current_max_idx = max_idx;

    // Shared counts and positions protected by Mutex for thread safety
    let counts = Arc::new(Mutex::new(HashMap::<Pair, i32>::new()));
    let pair_positions = Arc::new(Mutex::new(HashMap::<Pair, Vec<Position>>::new()));

    // Wrap each sequence in a Mutex and share units_list across threads with Arc
    let units_list = units_list.into_iter().map(Mutex::new).collect::<Vec<_>>();
    let units_list = Arc::new(units_list);

    // Initialize counts and positions
    units_list
        .par_iter()
        .enumerate()
        .for_each(|(seq_idx, sequence_mutex)| {
            let sequence = sequence_mutex.lock().unwrap();
            let mut local_counts = HashMap::<Pair, i32>::new();
            let mut local_positions = HashMap::<Pair, Vec<Position>>::new();

            for (pos, pair) in sequence.windows(2).enumerate() {
                let pair_tuple = (pair[0], pair[1]);

                // Update local counts
                *local_counts.entry(pair_tuple).or_insert(0) += 1;

                // Update local positions
                local_positions
                    .entry(pair_tuple)
                    .or_insert_with(Vec::new)
                    .push((seq_idx, pos));
            }

            // Merge local counts and positions into global ones
            {
                let mut counts_lock = counts.lock().unwrap();
                for (pair, count) in local_counts {
                    *counts_lock.entry(pair).or_insert(0) += count;
                }
            }
            {
                let mut positions_lock = pair_positions.lock().unwrap();
                for (pair, positions) in local_positions {
                    positions_lock
                        .entry(pair)
                        .or_insert_with(Vec::new)
                        .extend(positions);
                }
            }
        });

    for i in 0..num_merges {
        // Get the top pair
        let top_pair = {
            let counts_lock = counts.lock().unwrap();
            if counts_lock.is_empty() {
                info!("No pairs to merge.");
                break;
            }
            *counts_lock.iter().max_by_key(|&(_, &v)| v).unwrap().0
        };

        let new_idx = current_max_idx + 1;
        merges.insert(top_pair, new_idx);

        info!(
            "Merge {}/{}: {:?} -> {}",
            i + 1,
            num_merges,
            top_pair,
            new_idx
        );

        // Positions where the top_pair occurs
        let positions_to_update = {
            let mut positions_lock = pair_positions.lock().unwrap();
            positions_lock.remove(&top_pair).unwrap_or_default()
        };

        let units_list = Arc::clone(&units_list);
        let counts = Arc::clone(&counts);
        let pair_positions = Arc::clone(&pair_positions);

        // Update sequences and adjust counts and positions
        positions_to_update
            .par_iter()
            .for_each(|&(seq_idx, pos)| {
                // Lock the sequence for updating
                let sequence_mutex = &units_list[seq_idx];
                let mut sequence = sequence_mutex.lock().unwrap();

                // Skip if position is out of bounds (due to previous merges)
                if pos >= sequence.len() - 1 {
                    return;
                }

                // Merge the pair into new_idx
                sequence[pos] = new_idx;
                sequence.remove(pos + 1);

                // Update counts and positions
                let mut counts_lock = counts.lock().unwrap();
                let mut positions_lock = pair_positions.lock().unwrap();

                // Decrease count of old left pair
                if pos > 0 {
                    let left_pair = (sequence[pos - 1], top_pair.0);
                    if let Some(count) = counts_lock.get_mut(&left_pair) {
                        *count -= 1;
                        if *count == 0 {
                            counts_lock.remove(&left_pair);
                            positions_lock.remove(&left_pair);
                        }
                    }
                }

                // Decrease count of old right pair
                if pos < sequence.len() - 1 {
                    let right_pair = (top_pair.1, sequence[pos + 1]);
                    if let Some(count) = counts_lock.get_mut(&right_pair) {
                        *count -= 1;
                        if *count == 0 {
                            counts_lock.remove(&right_pair);
                            positions_lock.remove(&right_pair);
                        }
                    }
                }

                // Increase count of new left pair
                if pos > 0 {
                    let new_left_pair = (sequence[pos - 1], new_idx);
                    *counts_lock.entry(new_left_pair).or_insert(0) += 1;
                    positions_lock
                        .entry(new_left_pair)
                        .or_insert_with(Vec::new)
                        .push((seq_idx, pos - 1));
                }

                // Increase count of new right pair
                if pos < sequence.len() - 1 {
                    let new_right_pair = (new_idx, sequence[pos + 1]);
                    *counts_lock.entry(new_right_pair).or_insert(0) += 1;
                    positions_lock
                        .entry(new_right_pair)
                        .or_insert_with(Vec::new)
                        .push((seq_idx, pos));
                }
            });

        current_max_idx = new_idx;
    }

    // Collect the updated units_list
    let units_list = Arc::try_unwrap(units_list)
        .expect("Arc has multiple strong references")
        .into_iter()
        .map(|mutex| mutex.into_inner().unwrap())
        .collect();

    (units_list, merges)
}

pub fn encode_concurrent(
    units_list: Vec<Vec<i32>>,
    merges: &HashMap<Pair, i32>,
) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| encode(units.clone(), merges))
        .collect()
}

pub fn decode_concurrent(
    units_list: Vec<Vec<i32>>,
    merges: &HashMap<Pair, i32>,
) -> Vec<Vec<i32>> {
    units_list
        .par_iter()
        .map(|units| decode(units.clone(), merges))
        .collect()
}
