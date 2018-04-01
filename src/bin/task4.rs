extern crate softcomputingforsoftpeople as sc;
extern crate ordered_float;
extern crate nalgebra as na;

use std::collections::BTreeSet;
use na::{VectorN, Vector2, Dim, DefaultAllocator, U2};
use na::allocator::Allocator;
use sc::individual::Stats;
use sc::individual::multiobj::MultipleFitnessStats;
use sc::algorithms::nsga2::{DomCrowd, dom_crowd};
use std::f32;

/*pub trait Fitnesses<D>
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D> {
    fn fitness(&self) -> VectorN<f32, D>;
}*/

//#[derive(Copy, Clone, Debug)]
//struct Gene();

#[derive(Copy, Clone, Debug)]
pub struct FitnessesRankStats<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    pub fitness: VectorN<f32, D>,
    pub domination: DomCrowd,
}

impl<D> Stats for FitnessesRankStats<D>
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    type Fitness = VectorN<f32, D>;
    type CompetitionFitness = DomCrowd;

    fn new(fitness: VectorN<f32, D>) -> FitnessesRankStats<D> {
        FitnessesRankStats {
            fitness,
            domination: DomCrowd::new(),
        }
    }

    fn fitness(&self) -> DomCrowd {
        self.domination
    }
}

impl<D> MultipleFitnessStats<D> for FitnessesRankStats<D>
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    fn fitnesses(&self) -> &VectorN<f32, D> {
        &self.fitness
    }
}

fn main() {
    let mut individuals: Vec<_> = [(0.0, 1.0), (1.0, 0.0), (2.0, 1.5), (1.5, 3.0), (3.0, 1.6),
                                   (4.0, 3.5), (4.5, 3.1), (5.0, 2.5), (6.0, 5.0), (5.5, 7.0),
                                   (4.2, 6.0), (3.3, 6.5)]
            .into_iter()
            .map(|&(x, y)| FitnessesRankStats::new(Vector2::<f32>::new(x, y)))
            .collect();
    let (dcs, fronts) = dom_crowd(individuals.as_slice());
    for (i, &dc) in dcs.iter().enumerate() {
        individuals[i].domination = dc;
    }
    for (i, front) in fronts.iter().enumerate() {
        let front1: BTreeSet<_> = front.iter().map(|p| p + 1).collect();
        println!("front {}: {:?}", i, front1);
    }
    let mut numbered_individuals: Vec<_> = individuals.iter().enumerate().collect();
    numbered_individuals.sort_by_key(|&(_, idv)| idv.domination);
    for (idx, individual) in numbered_individuals {
        println!("individual {}: ({}, {}) rank: {} crowding: {}",
                 idx + 1,
                 individual.fitness.x,
                 individual.fitness.y,
                 individual.domination.rank,
                 individual.domination.crowding);
    }
}
