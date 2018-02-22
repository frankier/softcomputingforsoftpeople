extern crate rand;
extern crate ordered_float;
extern crate arrayvec;
extern crate num_iter;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate nalgebra as na;
extern crate softcomputingforsoftpeople as sc;
#[macro_use]
extern crate lazy_static;
extern crate alga;
extern crate num_traits;
extern crate itertools;
#[macro_use]
extern crate derive_new;

use structopt::StructOpt;
use rand::Rng;
use ordered_float::NotNaN;
use num_iter::range_step;
use std::fmt;
use std::f32::consts::PI;
use alga::general::Ring;
use alga::linear::FiniteDimInnerSpace;
use na::{Vector2, Real, DimName, Dim, U1, U2, DefaultAllocator, Scalar, Vector, VectorN, abs};
use na::storage::{Storage, Owned};
use na::allocator::Allocator;
use sc::{parse_seed, get_rng, gen_rand_pop, Individual, Fitness, OnlyFitnessStats, select_2way_tournament, Hypercube};
use rand::distributions::range::SampleRange;
use rand::distributions::{Normal, Sample, IndependentSample};
use num_traits::sign::Signed;
use num_traits::Float;
use itertools::Itertools;

static mut fitness_evals: u64 = 0;
static mut g1_evals: u64 = 0;

#[derive(Copy, Clone, Debug)]
struct Gene (
    Vector2<f32>,
);

trait RealGene<N, D>: sc::State + Sized
    where N: Real,
          D: Dim,
          DefaultAllocator: Allocator<N, D>,
{
    fn from_vec(VectorN<N, D>) -> Self;
    fn get_vec(&self) -> &VectorN<N, D>;
}

impl RealGene<f32, U2> for Gene {
    fn from_vec(space: Vector2<f32>) -> Gene {
        Gene(space)
    }

    fn get_vec(&self) -> &Vector2<f32> {
        &self.0
    }
}

impl sc::Fitness for Gene {
    fn fitness(&self) -> f32 {
        unsafe {
            fitness_evals += 1;
        }
        let v = self.get_vec();
        -(v[0] + v[1])
    }
}

trait Crossover<D, RG: RealGene<f32, D>>
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D> {
    fn parents(&self) -> u8;
    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG);
}

#[derive(new)]
struct LinearCrossover();

impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for LinearCrossover
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 { 2 }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.25.5297
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();
        let c1 = 0.5 * mummy_v + 0.5 * daddy_v;
        let c2 = -0.5 * mummy_v + 1.5 * daddy_v;
        let c3 = 1.5 * mummy_v - 0.5 * daddy_v;
        let c1g = RG::from_vec(c1);
        let c2g = RG::from_vec(c2);
        let c3g = RG::from_vec(c3);
        // XXX: Penalties not taken account of here
        let c1f = c1g.fitness();
        let c2f = c2g.fitness();
        let c3f = c2g.fitness();
        if c1f > c2f {
            if c2f > c3f {
                (c1g, c2g)
            } else {
                (c1g, c3g)
            }
        } else {
            if c1f > c3f {
                (c1g, c2g)
            } else {
                (c2g, c3g)
            }
        }
    }
}

#[derive(new)]
struct BlendCrossover {
    alpha: f32,
}

impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for BlendCrossover
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 { 2 }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // http://heuristic.kaist.ac.kr/cylee/xga/Paper%20Review/Papers/[P6]%20Real-Coded%20Genetic%20Algorithms.pdf
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();

        fn mk_child<R, D>(rng: &mut R, alpha: f32, mummy_v: &VectorN<f32, D>, daddy_v: &VectorN<f32, D>) -> VectorN<f32, D>
            where R: Rng,
                  D: Dim,
                  DefaultAllocator: Allocator<f32, D>, {
            let u: f32 = rng.gen();
            let gamma = (1.0 + 2.0 * alpha) * u - alpha;
            (1.0 - gamma) * mummy_v + gamma * daddy_v
        }

        (RG::from_vec(mk_child(rng, self.alpha, mummy_v, daddy_v)),
         RG::from_vec(mk_child(rng, self.alpha, mummy_v, daddy_v)))
    }
}

#[derive(new)]
struct UndxCrossover {
    xi_var: f32,
    eta_var: f32,
}


impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for UndxCrossover
        where
            D: Dim + DimName,
            DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 { 3 }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // https://pdfs.semanticscholar.org/d486/e294dff024ee600aa7a9b718528ccef532e1.pdf?_ga=2.108420555.853277512.1518964893-1720394428.1516112356&_gac=1.191070936.1518425497.CjwKCAiAk4XUBRB5EiwAHBLUMRdkvVUvd8tx7dT5MDBsj1X0BXp5ZRtj8nwe-UBne64B-cYfoy_OthoCBWYQAvD_BwE
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();
        let milkman_v = parents[2].get_vec();
        let x_p = 0.5 * (mummy_v + daddy_v);
        // primary search direction
        let d = daddy_v - mummy_v;
        // distance in seconary search direction (distance from line formed by d to milkman)
        let s = ((milkman_v - mummy_v) - (milkman_v - mummy_v).dot(&d) * &d).norm();
        // Now we seek an orthonormal basis forming a hyperplane orthongonal to the primary search
        // direction d.
        // We can use orthonormalize to run the Gram-Schmitt procedure to get an orthonormal basis from
        // any other basis. This basis will leave its first dimension untouched.
        // We know we can use the current basis as a starting point, but we must make sure to, deal
        // with the situation where e.g. d = [1, 0, 0]. In this case we must make sure not to add
        // [1, 0, 0] again, otherwise we won't end up with a basis. The code below ensures this.
        let mut basis = d.iter().enumerate().collect_vec();
        basis.sort_unstable_by_key(|&(i, x)| NotNaN::new(-abs(x)).unwrap());
        let mut basis = basis.iter().map(|&(i, x)| {
            let vec = VectorN::<f32, D>::from_fn(|r, _| {
                if r == i {
                    1.0
                } else {
                    0.0
                }
            });
            vec
        }).collect_vec();
        basis[0] = d.to_owned();
        let n = FiniteDimInnerSpace::orthonormalize(basis.as_mut_slice());
        // Now generate the two children
        let xi_dist = Normal::new(0.0, self.xi_var.sqrt().into());
        let eta_dist = Normal::new(0.0, self.eta_var.sqrt().into());

        fn mk_child<R, D>(rng: &mut R, x_p: &VectorN<f32, D>, xi_dist: &Normal, d: &VectorN<f32, D>, eta_dist: &Normal, e: &[VectorN<f32, D>], s: f32) -> VectorN<f32, D>
            where R: Rng,
                  D: Dim + DimName,
                  DefaultAllocator: Allocator<f32, D>, {
            let xi = xi_dist.ind_sample(rng) as f32;
            let primary = x_p + xi * d;
            let it = e.iter().map(|e_i| {
                let eta_i = eta_dist.ind_sample(rng) as f32;
                eta_i * e_i
            });
            // VectorN::<f32, D>::sum(it)
            let mut summed = VectorN::<f32, D>::zeros();
            for elem in it {
                summed += elem;
            }
            let secondary = s * summed;
            primary + secondary
        }

        (RG::from_vec(mk_child(rng, &x_p, &xi_dist, &d, &eta_dist, &basis[1..], s)),
         RG::from_vec(mk_child(rng, &x_p, &xi_dist, &d, &eta_dist, &basis[1..], s)))
    }
}

trait Mutation {
    fn mutate<R: Rng>(&self, rng: &mut R, parent: &mut Gene);
}

// F: Scalar + Ring + Signed + SampleRange + PartialOrd
#[derive(new)]
struct GlobalUniformMut {
    hypercube: Hypercube<f32, U2, U1>
}

// F: Scalar + Ring + Signed + SampleRange + PartialOrd
impl Mutation for GlobalUniformMut {
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut Gene) {
        parent.0 = self.hypercube.sample(rng);
    }
}

#[derive(new)]
struct LocalUniformMut {
    range: f32,
}

impl Mutation for LocalUniformMut {
    fn mutate<R: Rng>(&self, rng: &mut R, parent: &mut Gene) {
        parent.0 = parent.0.map(|xn| {
            xn + rng.gen_range(-0.5 * self.range, 0.5 * self.range)
        });
    }
}

#[derive(new)]
struct NonUniformMut {
    hypercube: Hypercube<f32, U2, U1>,
    b: f32,
    t: u32,
    t_max: u32,
}

impl Mutation for NonUniformMut {
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut Gene) {
        //parent.0 = parent.0.map(|x| {
            //let tau = ;
            //x + rng.gen_range(-0.5 * self.range, 0.5 * self.range)
        //});
    }
}

#[derive(new)]
struct NormalMut {
    var: u32,
}

impl Mutation for NormalMut {
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut Gene) {
        //parent.0 = parent.0.map(|x| {
            //let tau = ;
            //x + rng.gen_range(-0.5 * self.range, 0.5 * self.range)
        //});
    }
}

impl sc::State for Gene {}

fn print_individuals(individuals: &[Individual<Gene, OnlyFitnessStats>]) {
    for (idx, ind) in individuals.iter().enumerate() {
        println!("#{:?} {:?} fitness: {:?} inf: {:?}", idx, ind.state, ind.stats.fitness, infeasibility(&ind.state));
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "binary-coded-ga")]
struct Opt {
    /// A flag, true if used in the command line.
    #[structopt(short = "v", long = "verbosity", help = "Verbosity", parse(from_occurrences))]
    verbosity: u64,

    /// An argument of type float, with a default value.
    #[structopt(short = "s", long = "seed", help = "Seed for the PRNG",
                parse(try_from_str = "parse_seed"))]
    seed: Option<[u32; 4]>,

    #[structopt(short = "g", long = "generations", default_value = "2000",
                help = "The number of generations to run for")]
    generations: u64,

    #[structopt(short = "z", long = "pop-size", default_value = "20",
                help = "The population size")]
    pop_size: usize,

    #[structopt(short = "o", long = "pool-size", default_value = "20",
                help = "The pool size")]
    pool_size: usize,

    #[structopt(short = "c", long = "prob-select", default_value = "0.8",
                help = "The selection probability")]
    prob_select: f32,

    #[structopt(short = "x", long = "prob-xover", default_value = "0.8",
                help = "The crossover probability")]
    prob_xover: f32,

    #[structopt(short = "m", long = "prob-mut", default_value = "0.06",
                help = "The mutation probability")]
    prob_mut: f32,

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

    #[structopt(long = "mut-op", default_value = "local",
                help = "The mutation operation global | local")]
    mut_op: String,

    #[structopt(short = "r", long = "local-range", default_value = "1.0",
                help = "Range parameter for (uniform) local mutation")]
    local_range: f32,

    #[structopt(short = "t", long = "constraint-handling", default_value = "death",
                help = "The constraint handling technique: death | static | dynamic")]
    constraint_handling: String,

    #[structopt(short = "C", long = "dynamic-C", default_value = "0.5",
                help = "Parameter C for dynamic penalty")]
    C: f32,

    #[structopt(short = "a", long = "dynamic-alpha", default_value = "2.0",
                help = "Parameter alpha for dynamic penalty")]
    alpha: f32,

    #[structopt(short = "b", long = "dynamic-beta", default_value = "2.0",
                help = "Parameter beta for dynamic penalty")]
    beta: f32,
}

lazy_static! {
    static ref HYPERCUBE: Hypercube<f32, U2, U1> = Hypercube::new(Vector2::<f32>::new(-5.0, 0.0), Vector2::<f32>::new(10.0, 15.0));
}

fn g1(solution: &Gene) -> f32 {
    unsafe {
        g1_evals += 1;
    }
    (solution.0[1] - 5.1 * solution.0[0] * solution.0[0] / (4.0 * PI * PI) +
     5.0 * solution.0[0] / PI - 6.0).powf(2.0) +
    (10.0 - 10.0 / (8.0 * PI)) * f32::cos(solution.0[0]) + 9.0
}

fn g2(solution: &Gene) -> f32 {
    solution.0[1] + (solution.0[0] - 12.0) / 1.2
}

fn infeasibility(solution: &Gene) -> u8 {
    let f1 = g1(solution) <= 0.0;
    let f2 = g2(solution) <= 0.0;
    f1 as u8 + f2 as u8
}

fn run_ga<R: Rng, CX: Crossover<U2, Gene>, M: Mutation>(rng: &mut R, opt: &Opt, xover_op: CX, mutation_op: M, population: &mut Vec<Individual<Gene, OnlyFitnessStats>>) {
    let mut breeding_pool = Vec::<Gene>::with_capacity(opt.pool_size);
    // Iterate
    for gen in 0..opt.generations {
        if opt.verbosity >= 1 {
            println!("Generation {}", gen);
            print_individuals(population.as_slice());
        }
        // Selection by 2-way tournament
        select_2way_tournament(
            rng, &mut breeding_pool, population.as_slice(),
            opt.pool_size, opt.prob_select, opt.verbosity);
        // Crossover
        for i in range_step(0, opt.pop_size, 2) {
            let crossover_chance: f32 = rng.gen();
            if crossover_chance < opt.prob_xover {
                // XXX: This could be done with a trait rather than just giving all 3 at runtime
                let parents = [breeding_pool[i], breeding_pool[i + 1], breeding_pool[(i + 2) % opt.pop_size]];
                let mut son;
                let mut daughter;
                loop {
                    let children = xover_op.crossover(rng, &parents);
                    son = children.0;
                    daughter = children.1;
                    son.0 = HYPERCUBE.place_torus(&son.0);
                    daughter.0 = HYPERCUBE.place_torus(&daughter.0);
                    let son_feasible = infeasibility(&son) == 0;
                    let daughter_feasible = infeasibility(&daughter) == 0;
                    if opt.constraint_handling != "death" || (son_feasible && daughter_feasible) {
                        break;
                    }
                }
                population[i].state = son;
                population[i + 1].state = daughter;
                if opt.verbosity >= 2 {
                    println!("#{}: {:?} and #{}: {:?} crossover to produce {:?} and {:?}",
                             i,
                             breeding_pool[i],
                             i + 1,
                             breeding_pool[i + 1],
                             son,
                             daughter);
                }
            } else {
                population[i].state = breeding_pool[i];
                population[i + 1].state = breeding_pool[i + 1];
                if opt.verbosity >= 2 {
                    println!("#{}: {:?} and #{}: {:?} survive until the next generation",
                             i,
                             breeding_pool[i],
                             i + 1,
                             breeding_pool[i + 1]);
                }
            }
        }
        // Mutation
        for (i, individual) in population.iter_mut().enumerate() {
            let mutation_chance: f32 = rng.gen();
            if mutation_chance < opt.prob_mut {
                if opt.verbosity >= 2 {
                    println!("Mutating #{} {:?}", i, individual.state);
                }
                loop {
                    mutation_op.mutate(rng, &mut individual.state);
                    individual.state.0 = HYPERCUBE.place_torus(&individual.state.0);
                    if opt.constraint_handling != "death" || infeasibility(&individual.state) == 0 {
                        break;
                    }
                }
                if opt.verbosity >= 2 {
                    println!("into {:?}", individual.state);
                }
            }
        }
        // Evaluate fitness
        for ind in population.iter_mut() {
            if opt.constraint_handling == "static" {
                if infeasibility(&ind.state) == 0 {
                    ind.stats.fitness = ind.state.fitness();
                } else {
                    // Not exactly as stated since we're doing maximisation instead of minimisation
                    // but equivalent as far as I can see
                    const K: f32 = 1e9;
                    const m: f32 = 2.0;
                    ind.stats.fitness = -K * (infeasibility(&ind.state) as f32) / m;
                }
            } else if opt.constraint_handling == "dynamic" {
                let mut sum_g = 0.0;
                let g1_result = g1(&ind.state);
                if (g1_result <= 0.0) {
                    sum_g += abs(&g1_result).powf(opt.beta);
                }
                let g2_result = g2(&ind.state);
                if (g2_result <= 0.0) {
                    sum_g += abs(&g2_result).powf(opt.beta);
                }
                ind.stats.fitness = ind.state.fitness() - (opt.C * gen as f32).powf(opt.alpha) * sum_g;
            } else {
                ind.stats.fitness = ind.state.fitness();
            }
        }
    }
}

fn dispatch_mut<R: Rng, CX: Crossover<U2, Gene>>(rng: &mut R, opt: &Opt, xover_op: CX, population: &mut Vec<Individual<Gene, OnlyFitnessStats>>) {
    if opt.mut_op == "global" {
        run_ga(rng, opt, xover_op, GlobalUniformMut::new(HYPERCUBE.clone()), population);
    } else if opt.mut_op == "local" {
        run_ga(rng, opt, xover_op, LocalUniformMut::new(opt.local_range), population);
    } else {
        panic!("Unknown mutation operation");
    }
}

fn main() {
    let opt = Opt::from_args();
    let mut rng = get_rng(opt.seed);
    // Generate random population
    let mut population = gen_rand_pop(|| {
        loop {
            let individual = Gene(HYPERCUBE.sample(&mut rng));
            if opt.constraint_handling != "death" || infeasibility(&individual) == 0 {
                return individual;
            }
        }
    }, opt.pop_size);
    if opt.xover_op == "udnx" {
        dispatch_mut(&mut rng, &opt, UndxCrossover::new(opt.xi, opt.eta), &mut population);
    } else if opt.xover_op == "linear" {
        dispatch_mut(&mut rng, &opt, LinearCrossover::new(), &mut population);
    } else if opt.xover_op == "blend" {
        dispatch_mut(&mut rng, &opt, BlendCrossover::new(opt.blend_alpha), &mut population);
    } else {
        panic!("Unknown xover operation");
    };
    println!("Final results");
    population.sort_unstable_by_key(|ind| NotNaN::new(ind.stats.fitness).unwrap());
    print_individuals(population.as_slice());
    unsafe {
        println!("Fitness evals {}", fitness_evals);
        println!("g1 evals {}", g1_evals);
    }
}
