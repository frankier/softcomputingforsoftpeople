use rand::Rng;
use std::mem::swap;

pub fn gen_two_points<R: Rng>(rng: &mut R, start: usize, end: usize) -> (usize, usize) {
    let mut xover_p1 = rng.gen_range(start, end);
    let mut xover_p2 = rng.gen_range(start, end);
    if xover_p1 > xover_p2 {
        swap(&mut xover_p1, &mut xover_p2);
    }
    (xover_p1, xover_p2)
}
