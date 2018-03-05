extern crate ordered_float;

use ordered_float::NotNaN;
use std::fmt::Debug;

pub trait GetFitness {
    type Fitness: PartialOrd + Debug;

    fn fitness(&self) -> Self::Fitness;
}

pub trait State: GetFitness + Copy + Clone + Debug {}

pub trait Stats: Copy + Clone {
    type Fitness: PartialOrd + Debug;
    type CompetitionFitness: Ord + Debug;

    fn new(fitness: Self::Fitness) -> Self;

    fn fitness(&self) -> Self::CompetitionFitness;
}

#[derive(Copy, Clone)]
pub struct Individual<S: State, SS: Stats> {
    pub state: S,
    pub stats: SS,
}

impl<F, S, SS> Individual<S, SS>
        where F: PartialOrd + Debug,
              S: State<Fitness=F>,
              SS: Stats<Fitness=F> {
    pub fn new(state: S) -> Individual<S, SS> {
        let stats = SS::new(state.fitness());
        Individual { state, stats }
    }
}

#[derive(Copy, Clone)]
pub struct OnlyFitnessStats {
    pub fitness: f32,
}

impl Stats for OnlyFitnessStats {
    type Fitness = f32;
    type CompetitionFitness = NotNaN<f32>;

    fn new(fitness: f32) -> OnlyFitnessStats {
        OnlyFitnessStats {
            fitness
        }
    }

    fn fitness(&self) -> NotNaN<f32> {
        NotNaN::new(self.fitness).unwrap()
    }
}

pub mod real {
    use na::{VectorN, Dim, DefaultAllocator, Real};
    use na::allocator::Allocator;

    pub trait RealGene<N, D>: ::State + Sized
        where N: Real,
              D: Dim,
              DefaultAllocator: Allocator<N, D>,
    {
        fn from_vec(VectorN<N, D>) -> Self;
        fn get_vec(&self) -> &VectorN<N, D>;
        fn get_mut_vec(&mut self) -> &mut VectorN<N, D>;
    }
}

/*#[derive(Copy, Clone)]
pub struct PSOIndividual<G: Gene> {
    pub gene: G,
    pub vel: G,
    pub pbest: G,
    pub fitness: f32,
}*/
