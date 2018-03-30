extern crate softcomputingforsoftpeople as sc;
extern crate ordered_float;
extern crate nalgebra as na;
extern crate itertools;
extern crate rand;
#[macro_use]
extern crate lazy_static;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate ord_subset;

use structopt::StructOpt;
use na::{VectorN, Vector2, Vector3, Dim, DefaultAllocator, U2, U3, U1};
use na::allocator::Allocator;
use sc::individual::{Stats, Individual, GetFitness, State};
use std::f32;
use itertools::Itertools;
use rand::Rng;
use sc::operators::real::{UndxCrossover, LinearCrossover, BlendCrossover};
use sc::individual::real::RealGene;
use sc::individual::multiobj::MultipleFitnessStats;
use sc::utils::real::Hypercube;
use sc::utils::rand::{parse_seed, get_rng};
use sc::scalarize::{Scalarizer, TchebycheffScalarizer, WeightedSumScalarizer, UniformWeightVector2D, WeightVector};
use sc::algorithms::moead::{moead_next_gen_real};
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

impl<D> MultipleFitnessStats<D> for FitnessesStats<D>
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    fn fitnesses(&self) -> &VectorN<f32, D> {
        &self.fitness
    }
}

type T4bIndividual = Individual<Gene, FitnessesStats<U2>>;

fn print_fitnesses<S: Scalarizer<U2>>(scalarizer: S,
                                      individuals: &[T4bIndividual],
                                      lambda: &[VectorN<f32, U2>]) {
    for (idx, (individual, l_i)) in individuals.iter().zip(lambda).enumerate() {
        let scalarized = scalarizer.scalarize(&individual.stats.fitness, l_i);
        println!("#{}: {}", idx + 1, scalarized);
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
        moead_next_gen_real(rng,
                       UndxCrossover::new(opt.xi, opt.eta),
                       &apply_heuristic,
                       scalarizer,
                       individuals,
                       weight_vec,
                       z_star);
    } else if opt.xover_op == "linear" {
        moead_next_gen_real(rng,
                       LinearCrossover::new(),
                       &apply_heuristic,
                       scalarizer,
                       individuals,
                       weight_vec,
                       z_star);
    } else if opt.xover_op == "blend" {
        moead_next_gen_real(rng,
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
