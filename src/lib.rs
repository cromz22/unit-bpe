type Pair = (i32, i32);
type Position = (usize, usize);

mod concurrent;
mod core;
mod python_bindings;

#[cfg(test)]
mod test;
