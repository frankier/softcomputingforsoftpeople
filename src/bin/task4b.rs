extern crate softcomputingforsoftpeople as sc;
extern crate ordered_float;
extern crate nalgebra as na;
extern crate itertools;
#[macro_use]
extern crate derive_new;
extern crate rand;
#[macro_use]
extern crate lazy_static;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate ord_subset;

use structopt::StructOpt;
use na::{VectorN, Vector2, Vector3, Dim, DimName, DefaultAllocator, U2, U3, U1, distance, DMatrix,
         abs};
use na::geometry::Point;
use na::allocator::Allocator;
use sc::individual::{Stats, Individual, GetFitness, State};
use ordered_float::NotNaN;
use std::f32;
use std::collections::BinaryHeap;
use std::ops::Range;
use itertools::Itertools;
use rand::seq::sample_indices;
use rand::Rng;
use sc::operators::real::{Crossover, UndxCrossover, LinearCrossover, BlendCrossover};
use sc::individual::real::RealGene;
use sc::utils::real::Hypercube;
use sc::utils::rand::{parse_seed, get_rng};
use ord_subset::OrdVar;

#[derive(Copy, Clone, Debug)]
struct Gene(Vector3<f32>);

impl RealGene<f32, U3> for Gene {
    fn from_vec(space: Vector3<f32>) -> Gene {
        Gene(space)
    }

    fn get_vec(&self) -> &Vector3<f32> {
        &self.0
    }

    fn get_mut_vec(&mut self) -> &mut Vector3<f32> {
        &mut self.0
    }
}

impl GetFitness for Gene {
    type Fitness = Vector2<f32>;

    fn fitness(&self) -> Vector2<f32> {
        let x = self.0;
        let f1 = (0..2)
            .map(|i| -10.0 * f32::exp(-0.2 * f32::sqrt(x[i] * x[i] + x[i + 1] * x[i + 1])))
            .sum();
        let f2 = (0..3).map(|i| x[i].powi(2) + 5.0 * x[i]).sum();
        Vector2::<f32>::new(f1, f2)
    }
}

impl State for Gene {}

#[derive(Copy, Clone, Debug)]
pub struct FitnessesStats<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    pub fitness: VectorN<f32, D>,
}

impl<D> Stats for FitnessesStats<D>
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    type Fitness = VectorN<f32, D>;
    type CompetitionFitness = OrdVar<VectorN<f32, D>>;

    fn new(fitness: VectorN<f32, D>) -> FitnessesStats<D> {
        FitnessesStats { fitness }
    }

    fn fitness(&self) -> OrdVar<VectorN<f32, D>> {
        OrdVar::<VectorN<f32, D>>::new_unchecked(self.fitness)
    }
}

trait Scalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32;
}

#[derive(new)]
struct TchebycheffScalarizer<D>
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
struct WeightedSumScalarizer();

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

type T4bIndividual = Individual<Gene, FitnessesStats<U2>>;

// XXX: To implement for higher dimensions see
// "Sampling Uniformly from the Unit Simplex"
// http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
// and literature on n-dimensional space filling curves
trait WeightVector<'a, D>
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

struct UniformWeightVector2D {
    lambda: Vec<Vector2<f32>>,
    t: u32,
}

impl UniformWeightVector2D {
    fn new(num: u32, t: u32) -> UniformWeightVector2D {
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

    fn neighbor_bounds(from: usize, t: usize, len: usize) -> (usize, usize) {
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

struct ChosenWeightVector<D>
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
    fn new(lambda: Vec<VectorN<f32, D>>, t: usize) -> ChosenWeightVector<D> {
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

fn print_fitnesses<S: Scalarizer<U2>>(scalarizer: S,
                                      individuals: &[T4bIndividual],
                                      lambda: &[VectorN<f32, U2>]) {
    for (idx, (individual, l_i)) in individuals.iter().zip(lambda).enumerate() {
        let scalarized = scalarizer.scalarize(&individual.stats.fitness, l_i);
        println!("#{}: {}", idx + 1, scalarized);
    }
}

fn moead_next_gen<'a, R, G, CX, H, SolD, FitD, S, WV>(
        rng: &mut R,
        xover: CX,
        apply_heuristic: &H,
        scalarizer: &S,
        individuals: &mut [Individual<G, FitnessesStats<FitD>>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        G: RealGene<f32, SolD, Fitness=VectorN<f32, FitD>>,
        CX: Crossover<SolD, G>,
        H: Fn(&mut G) + Sized,
        SolD: Dim + Copy,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, SolD>,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        S: Scalarizer<FitD>,
        WV: WeightVector<'a, FitD>
{
    for idx in 0..individuals.len() {
        let parents: Vec<_> = weight_vec
            .rand_neighbors(rng, idx, xover.parents() as usize)
            .map(|idx| individuals[idx].state)
            .collect();
        // Reproduction
        let mut child_gene = {
            let children_genes = xover.crossover(rng, parents.as_slice());
            let flip: bool = rng.gen();
            if flip {
                children_genes.1
            } else {
                children_genes.0
            }
        };
        // Improvement
        apply_heuristic(&mut child_gene);
        let child = Individual::<G, FitnessesStats<FitD>>::new(child_gene);
        // Update of z_star
        for d in 0..z_star.len() {
            if child.stats.fitness[d] < z_star[d] {
                z_star[d] = child.stats.fitness[d];
            }
        }
        // Update of neighboring solutions
        for neighbor_idx in weight_vec.neighbors(idx) {
            let l_i = weight_vec.get_lambda()[neighbor_idx];
            let child_g = scalarizer.scalarize(&child.stats.fitness, &l_i);
            let neighbor_g = scalarizer.scalarize(&individuals[neighbor_idx].stats.fitness, &l_i);
            if child_g < neighbor_g {
                individuals[neighbor_idx] = child;
            }
        }
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "real-ga")]
struct Opt {
    /// A flag, true if used in the command line.
    #[structopt(short = "v", long = "verbosity", help = "Verbosity", parse(from_occurrences))]
    verbosity: u64,

    /// An argument of type float, with a default value.
    #[structopt(short = "s", long = "seed", help = "Seed for the PRNG",
                parse(try_from_str = "parse_seed"))]
    seed: Option<[u32; 4]>,

    #[structopt(long = "xover-op", default_value = "linear",
                help = "The crossover operation linear | blend | udnx")]
    xover_op: String,

    #[structopt(long = "undx-xi", default_value = "0.5",
                help = "The value xi, the variance along the primary axis for undx")]
    xi: f32,

    #[structopt(long = "undx-eta", default_value = "0.5",
                help = "The value eta, the variance along the secondary axes for undx")]
    eta: f32,

    #[structopt(long = "blend-alpha", default_value = "0.5",
                help = "The value eta, the variance along the secondary axes for undx")]
    blend_alpha: f32,
}

lazy_static! {
    static ref HYPERCUBE: Hypercube<f32, U3, U1> = Hypercube::new(Vector3::<f32>::new(-5.0, -5.0, -5.0), Vector3::<f32>::new(5.0, 5.0, 5.0));
}

fn dispatch_xover<R: Rng, S: Scalarizer<U2>, H: Fn(&mut Gene) + Sized>(
        opt: &Opt,
        rng: &mut R,
        apply_heuristic: &H,
        scalarizer: &S,
        individuals: &mut [Individual<Gene, FitnessesStats<U2>>],
        weight_vec: &UniformWeightVector2D,
        z_star: &mut VectorN<f32, U2>)
{
    if opt.xover_op == "udnx" {
        moead_next_gen(rng,
                       UndxCrossover::new(opt.xi, opt.eta),
                       &apply_heuristic,
                       scalarizer,
                       individuals,
                       weight_vec,
                       z_star);
    } else if opt.xover_op == "linear" {
        moead_next_gen(rng,
                       LinearCrossover::new(),
                       &apply_heuristic,
                       scalarizer,
                       individuals,
                       weight_vec,
                       z_star);
    } else if opt.xover_op == "blend" {
        moead_next_gen(rng,
                       BlendCrossover::new(opt.blend_alpha),
                       &apply_heuristic,
                       scalarizer,
                       individuals,
                       weight_vec,
                       z_star);
    } else {
        panic!("Unknown xover operation");
    }
    println!("New z_star: ({}, {})", z_star.x, z_star.y);
    for (idx, ind) in individuals.iter().enumerate() {
        println!("#{} solution: ({}, {}, {}) fitnesses: ({}, {}) scalarized (prev z): {}",
                 idx,
                 ind.state.0.x,
                 ind.state.0.y,
                 ind.state.0.z,
                 ind.stats.fitness.x,
                 ind.stats.fitness.y,
                 scalarizer.scalarize(&ind.stats.fitness, &weight_vec.get_lambda()[idx]));
    }
}

fn main() {
    // Get options
    let opt = Opt::from_args();
    let mut rng = get_rng(opt.seed);

    // Init
    let individuals_x = [(-1.0074, -3.0188, 4.8305),
                         (0.2688, -0.1031, -1.9855),
                         (-0.8320, -1.6051, 2.0110),
                         (1.5686, 4.5163, 1.6634),
                         (1.2797, 4.2033, 0.3913),
                         (-2.0802, -4.4732, 1.9811),
                         (-0.6835, 2.3786, 1.6653),
                         (-4.8451, -2.3088, -3.2187),
                         (4.8406, -0.7716, -3.7199),
                         (-3.3283, 0.4787, 4.9908),
                         (-3.9378, 4.4274, -3.2888),
                         (-1.2759, -0.8226, -4.6740)];
    let weight_vec = UniformWeightVector2D::new(12, 4);
    let z_star = Vector2::<f32>::new(-20.0, -12.0);

    let individuals: Vec<_> = individuals_x
        .into_iter()
        .map(|&(x, y, z)| T4bIndividual::new(Gene(Vector3::<f32>::new(x, y, z))))
        .collect();

    // Print 4 nearest neighbors of each individual
    println!("Nearest neighbors according to weight vectors");
    for idx in 0..individuals.len() {
        println!("#{}: {}",
                 idx + 1,
                 weight_vec.neighbors(idx).map(|n| n + 1).join("; "));
    }

    // Evaluate initial fitness of individuals
    println!("\nTchebycheff fitnesses");
    print_fitnesses(TchebycheffScalarizer::new(z_star),
                    individuals.as_slice(),
                    weight_vec.get_lambda());
    println!("\nWeighted sum fitnesses");
    print_fitnesses(WeightedSumScalarizer::new(),
                    individuals.as_slice(),
                    weight_vec.get_lambda());

    // Get next generation
    let clamp_solution = |gene: &mut Gene| { (*gene).0 = HYPERCUBE.clamp(&gene.0); };
    println!("\nTchebycheff next generation");
    let mut tchebycheff_individuals = individuals.clone();
    let mut tchebycheff_z_star = z_star.clone();
    dispatch_xover(&opt,
                   &mut rng,
                   &clamp_solution,
                   &TchebycheffScalarizer::new(z_star),
                   tchebycheff_individuals.as_mut_slice(),
                   &weight_vec,
                   &mut tchebycheff_z_star);

    println!("\nWeighted sum next generation");
    let mut weight_sum_individuals = individuals.clone();
    let mut weight_sum_z_star = z_star.clone();
    dispatch_xover(&opt,
                   &mut rng,
                   &clamp_solution,
                   &WeightedSumScalarizer::new(),
                   weight_sum_individuals.as_mut_slice(),
                   &weight_vec,
                   &mut weight_sum_z_star);
}
