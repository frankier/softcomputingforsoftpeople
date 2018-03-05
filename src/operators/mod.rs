pub mod real;
pub mod binary;

use individual::{Stats, Individual, State};
use rand::Rng;
use rand::seq::sample_slice;
use std::fmt::Debug;

pub fn select_2way_tournament<R, S, SS, F>(
        mut rng: &mut R,
        breeding_pool: &mut Vec<S>,
        population: &[Individual<S, SS>],
        pool_size: usize,
        prob_select: f32,
        verbosity: u64)
            where
                R: Rng,
                F: PartialOrd + Debug,
                S: State<Fitness=F> + Clone,
                SS: Stats<Fitness=F> {
    breeding_pool.clear();
    for i in 0..pool_size {
        let mut competitors = sample_slice(&mut rng, &population, 2);
        competitors.sort_unstable_by_key(|ind| ind.stats.fitness());
        if verbosity >= 2 {
            println!("#{} Tournament between {:?} and {:?}",
                     i,
                     competitors[1].state,
                     competitors[0].state);
        }
        let win_chance: f32 = rng.gen();
        breeding_pool.push(if win_chance < prob_select {
                               if verbosity >= 2 {
                                   println!("{:?} wins due to higher fitness",
                                            competitors[0].state);
                               }
                               competitors[1].state
                           } else {
                               if verbosity >= 2 {
                                   println!("{:?} wins despite lower fitness",
                                            competitors[1].state);
                               }
                               competitors[0].state
                           });
    }
}
