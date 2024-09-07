use std::collections::HashMap;

fn get_counts(units: Vec<i32>) -> HashMap<(i32, i32), i32> {
    let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

    for pair in units.windows(2) {
        let pair_tuple = (pair[0], pair[1]);
        *counts.entry(pair_tuple).or_insert(0) += 1;
    }

    counts
}

fn main() {
    let units = vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
    let counts = get_counts(units);
    for ((k1, k2), value) in &counts {
        println!("{k1}, {k2}: {value}");
    }
}
