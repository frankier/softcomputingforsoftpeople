use ordered_float::NotNaN;
use std::fmt::Debug;

pub trait GetFitness {
    type Fitness: PartialOrd + Debug;

    fn fitness(&self) -> Self::Fitness;
}

pub trait State: GetFitness + Clone + Debug {}

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
          S: State<Fitness = F>,
          SS: Stats<Fitness = F>
{
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
        OnlyFitnessStats { fitness }
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
              DefaultAllocator: Allocator<N, D>
    {
        fn from_vec(VectorN<N, D>) -> Self;
        fn get_vec(&self) -> &VectorN<N, D>;
        fn get_mut_vec(&mut self) -> &mut VectorN<N, D>;
    }
}

pub mod order {
    use indexmap::IndexSet;

    pub trait OrderGene<N>: ::State + Sized {
        // XXX: Bad: from_order shouldn't need &self, but it does since the Gene might currently be
        // carrying around extra information to get its fitness. This is just bad design and needs
        // to be fixed.
        fn from_order(&self, IndexSet<N>) -> Self;
        fn get_order(&self) -> &IndexSet<N>;
        fn get_mut_order(&mut self) -> &mut IndexSet<N>;
    }
}

pub mod multiobj {
    use na::{VectorN, Dim, DefaultAllocator};
    use na::allocator::Allocator;
    use ord_subset::OrdVar;

    pub trait MultipleFitnessStats<D>
        where D: Dim,
              DefaultAllocator: Allocator<f32, D>,
              <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy {
        fn fitnesses(&self) -> &VectorN<f32, D>;
    }

    #[derive(Copy, Clone, Debug)]
    pub struct OnlyFitnessesStats<D>
        where D: Dim + Copy,
              DefaultAllocator: Allocator<f32, D>,
              <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
    {
        pub fitness: VectorN<f32, D>,
    }

    impl<D> ::Stats for OnlyFitnessesStats<D>
        where D: Dim,
              DefaultAllocator: Allocator<f32, D>,
              <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
    {
        type Fitness = VectorN<f32, D>;
        type CompetitionFitness = OrdVar<VectorN<f32, D>>;

        fn new(fitness: VectorN<f32, D>) -> OnlyFitnessesStats<D> {
            OnlyFitnessesStats { fitness }
        }

        fn fitness(&self) -> OrdVar<VectorN<f32, D>> {
            OrdVar::<VectorN<f32, D>>::new_unchecked(self.fitness)
        }
    }

    impl<D> MultipleFitnessStats<D> for OnlyFitnessesStats<D>
        where D: Dim,
              DefaultAllocator: Allocator<f32, D>,
              <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
    {
        fn fitnesses(&self) -> &VectorN<f32, D> {
            &self.fitness
        }
    }
}

/*#[derive(Copy, Clone)]
pub struct PSOIndividual<G: Gene> {
    pub gene: G,
    pub vel: G,
    pub pbest: G,
    pub fitness: f32,
}*/
