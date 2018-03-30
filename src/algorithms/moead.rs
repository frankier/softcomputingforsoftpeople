use rand::Rng;
use na::{VectorN, Vector2, Dim, DimName, DefaultAllocator, U2, distance, DMatrix, abs};
use na::geometry::Point;
use na::allocator::Allocator;
use ordered_float::NotNaN;
use std::collections::BinaryHeap;
use std::ops::Range;
use rand::seq::sample_indices;

use individual::{Stats, Individual, State};
use individual::real::RealGene;
use individual::order::OrderGene;
use individual::multiobj::MultipleFitnessStats;
use operators::real::Crossover as RealCrossover;
use operators::order::Crossover as OrderCrossover;

// XXX: To implement for higher dimensions see
// "Sampling Uniformly from the Unit Simplex"
// http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
// and literature on n-dimensional space filling curves
pub trait WeightVector<'a, D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    type NeighborIterator: Iterator<Item = usize>;
    type RandNeighborIterator: Iterator<Item = usize>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator;

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator;

    fn get_lambda(&self) -> &[VectorN<f32, D>];
}

pub struct UniformWeightVector2D {
    lambda: Vec<Vector2<f32>>,
    t: u32,
}

impl UniformWeightVector2D {
    pub fn new(num: u32, t: u32) -> UniformWeightVector2D {
        assert!(t <= num);
        let n = num as f32;
        let n1 = n - 1.0;
        let lambda: Vec<_> = (0..num)
            .into_iter()
            .map(|i| {
                     let i_f = i as f32;
                     Vector2::new(i_f / n1, (n - i_f) / n1)
                 })
            .collect();

        UniformWeightVector2D { t, lambda }
    }

    pub fn neighbor_bounds(from: usize, t: usize, len: usize) -> (usize, usize) {
        let ht = t / 2;
        let start = if ht > from { 0 } else { from - ht };
        if start + t <= len {
            (start, (start + t))
        } else {
            (len - t, len)
        }
    }
}

impl<'a> WeightVector<'a, U2> for UniformWeightVector2D {
    type NeighborIterator = Range<usize>;
    type RandNeighborIterator = Box<Iterator<Item = usize>>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator {
        let (start, end) =
            UniformWeightVector2D::neighbor_bounds(from, self.t as usize, self.lambda.len());
        start..end
    }

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator {
        let (start, end) =
            UniformWeightVector2D::neighbor_bounds(from, self.t as usize, self.lambda.len());
        Box::new(sample_indices(rng, end - start, select)
                     .into_iter()
                     .map(move |idx| start + idx))
    }

    fn get_lambda(&self) -> &[VectorN<f32, U2>] {
        &self.lambda
    }
}

pub struct ChosenWeightVector<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D> //<DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    lambda: Vec<VectorN<f32, D>>,
    t: usize,
    // This could be a 2D matrix
    b: DMatrix<usize>,
}

impl<D> ChosenWeightVector<D>
    where D: Dim + DimName + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    pub fn new(lambda: Vec<VectorN<f32, D>>, t: usize) -> ChosenWeightVector<D> {
        assert!(t >= 1);
        assert!(t <= lambda.len());

        // Matrix is column major, so neighbors should be
        let mut b = DMatrix::<usize>::from_element(t, lambda.len(), 0);
        let mut heap = BinaryHeap::with_capacity(t);
        for (i, l1) in lambda.iter().enumerate() {
            for (j, l2) in lambda.iter().enumerate() {
                let d = NotNaN::new(distance(&Point::<f32, D>::from_coordinates(l1.clone()),
                                             &Point::<f32, D>::from_coordinates(l2.clone())))
                        .unwrap();
                if heap.len() < t {
                    heap.push((d, j));
                } else if d < heap.peek().unwrap().0 {
                    *(heap.peek_mut().unwrap()) = (d, j);
                }
            }
            for (n_i, (_d, j)) in heap.drain().enumerate() {
                b[(i, n_i)] = j;
            }
        }

        ChosenWeightVector { t, lambda, b }
    }
}

impl<'a, D> WeightVector<'a, D> for ChosenWeightVector<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    // XXX: impl trait
    type NeighborIterator = Box<Iterator<Item = usize> + 'a>;
    type RandNeighborIterator = Box<Iterator<Item = usize> + 'a>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator {
        // XXX: column doesn't work because it is not guaranteed to produce a contiguous slice --
        // still, hopefully there will possibly be a neater way to do this in future
        let start = from * self.t;
        Box::new(self.b.as_slice()[start..(start + self.t)]
                     .iter()
                     .map(|&n| n))
    }

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator {
        Box::new(sample_indices(rng, self.t, select)
                     .into_iter()
                     .map(move |idx| self.b[(from, idx)]))
    }

    fn get_lambda(&self) -> &[VectorN<f32, D>] {
        &self.lambda
    }
}


pub fn moead_next_gen<'a, R, X, G, H, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: X,
        parents: usize,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        X: Fn(&mut R, &[G]) -> (G, G),
        G: State<Fitness=VectorN<f32, FitD>>,
        H: Fn(&mut G) + Sized,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    for idx in 0..individuals.len() {
        let parents: Vec<_> = weight_vec
            .rand_neighbors(rng, idx, parents)
            .map(|idx| individuals[idx].state.clone())
            .collect();
        // Reproduction
        let mut child_gene = {
            let children_genes = crossover(rng, parents.as_slice());
            let flip: bool = rng.gen();
            if flip {
                children_genes.1
            } else {
                children_genes.0
            }
        };
        // Improvement
        apply_heuristic(&mut child_gene);
        let child = Individual::<G, St>::new(child_gene);
        // Update of z_star
        for d in 0..z_star.len() {
            if child.stats.fitnesses()[d] < z_star[d] {
                z_star[d] = child.stats.fitnesses()[d];
            }
        }
        // Update of neighboring solutions
        for neighbor_idx in weight_vec.neighbors(idx) {
            let l_i = weight_vec.get_lambda()[neighbor_idx];
            let child_g = scalarizer.scalarize(&child.stats.fitnesses(), &l_i);
            let neighbor_g = scalarizer.scalarize(&individuals[neighbor_idx].stats.fitnesses(), &l_i);
            if child_g < neighbor_g {
                individuals[neighbor_idx] = child.clone();
            }
        }
    }
}

pub fn moead_next_gen_real<'a, R, G, CX, H, SolD, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: CX,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        G: RealGene<f32, SolD, Fitness=VectorN<f32, FitD>>,
        CX: RealCrossover<SolD, G>,
        H: Fn(&mut G) + Sized,
        SolD: Dim + Copy,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, SolD>,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    moead_next_gen(
        rng,
        |rng, parents| {
            crossover.crossover(rng, parents)
        },
        crossover.parents() as usize,
        apply_heuristic,
        scalarizer,
        individuals,
        weight_vec,
        z_star
    )
}

pub fn moead_next_gen_order<'a, R, G, CX, H, E, N, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: CX,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        G: OrderGene<N, Fitness=VectorN<f32, FitD>>,
        CX: OrderCrossover<G, N>,
        H: Fn(&mut G) + Sized,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    moead_next_gen(
        rng,
        |rng, parents| {
            crossover.crossover(rng, parents)
        },
        crossover.parents() as usize,
        apply_heuristic,
        scalarizer,
        individuals,
        weight_vec,
        z_star
    )
}

pub trait Scalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32;
}

#[derive(new)]
pub struct TchebycheffScalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D> //<DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    z_star: VectorN<f32, D>,
}

impl<D> Scalarizer<D> for TchebycheffScalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32 {
        fitness
            .iter()
            .zip(lambda.iter())
            .zip(self.z_star.iter())
            .map(|((f, l), z)| NotNaN::new(l * abs(&(f - z))).unwrap())
            .max()
            .unwrap()
            .into_inner()
    }
}

#[derive(new)]
pub struct WeightedSumScalarizer();

impl<D> Scalarizer<D> for WeightedSumScalarizer
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32 {
        fitness
            .iter()
            .zip(lambda.iter())
            .map(|(f, l)| l * f)
            .sum()
    }
}
