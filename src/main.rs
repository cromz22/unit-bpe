use std::collections::HashMap;

fn get_counts(units: &[i32]) -> HashMap<(i32, i32), i32> {
    let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

    for pair in units.windows(2) {
        let pair_tuple = (pair[0], pair[1]);
        *counts.entry(pair_tuple).or_insert(0) += 1;
    }

    counts
}

fn merge(units: &[i32], pair: (i32, i32), idx: i32) -> Vec<i32> {
    let mut new_units = Vec::new();
    let mut i = 0;
    while i < units.len() {
        if i < units.len() - 1 && (units[i], units[i + 1]) == pair {
            new_units.push(idx);
            i += 2;
        } else {
            new_units.push(units[i]);
            i += 1;
        }
    }

    new_units
}

fn main() {
    let units = vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
    let counts = get_counts(&units);
    for ((k1, k2), value) in &counts {
        println!("{k1}, {k2}: {value}");
    }
    let new_units = merge(&units, (0, 1), 6);
    for unit in new_units {
        println!("{unit}");
    }
}
