extern crate rand;
#[macro_use]
extern crate rand_derive;
extern crate ordered_float;
extern crate bitwise;
#[macro_use]
extern crate static_assertions;
extern crate num_iter;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate softcomputingforsoftpeople as sc;
extern crate safe_transmute;
extern crate hex_slice;

use structopt::StructOpt;
use rand::Rng;
use ordered_float::NotNaN;
use num_iter::range_step;
use std::fmt;
use sc::operators::select_2way_tournament;
use sc::operators::binary::{crossover, mutate_inplace};
use sc::individual::{Individual, GetFitness, OnlyFitnessStats, State};
use sc::utils::rand::{parse_seed, get_rng};
use sc::gen_rand_pop;
use safe_transmute::{PodTransmutable, guarded_transmute_to_bytes_pod};
use hex_slice::AsHex;

const POP_SIZE: usize = 10;
const POOL_SIZE: usize = 10;
const P_S: f32 = 0.8;
const P_C: f32 = 0.8;
const P_M: f32 = 0.06;

const_assert!(pop_size_even; POP_SIZE % 2 == 0);
const_assert!(pool_size_even; POOL_SIZE % 2 == 0);

#[derive(Rand, Copy, Clone, Debug)]
#[repr(C)]
struct Gene {
    x1: u8,
    x2: u8,
}

unsafe impl PodTransmutable for Gene {}

impl fmt::Display for Gene {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (x1f, x2f) = interpret_gene(self.to_owned());
        write!(f,
               "<Gene {:02x} x1: {:02x} = {} x2: {:02x} = {}>",
               guarded_transmute_to_bytes_pod(self).as_hex(),
               self.x1,
               x1f,
               self.x2,
               x2f)
    }
}

//#[derive(Copy, Clone)]
//struct Individual {
//gene: Gene,
//fitness: f32,
//}

fn fixed_to_float(x: u8) -> f32 {
    // This is slightly different from the task.
    // It covers the interval (1.0, 4 + 255/256)
    1.0 + (f32::from(x)) * 4.0 / 256.0
}

fn interpret_gene(gene: Gene) -> (f32, f32) {
    (fixed_to_float(gene.x1), fixed_to_float(gene.x2))
}

impl GetFitness for Gene {
    type Fitness = f32;

    fn fitness(&self) -> f32 {
        let (x1f, x2f) = interpret_gene(*self);
        -(x1f + x2f - 2.0 * x1f * x1f - x2f * x2f + x1f * x2f)
    }
}

impl State for Gene {}

fn print_individuals(individuals: &[Individual<Gene, OnlyFitnessStats>]) {
    for (idx, ind) in individuals.iter().enumerate() {
        println!("#{} {} fitness: {}", idx, ind.state, ind.stats.fitness);
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
}

fn main() {
    let opt = Opt::from_args();
    let mut rng = get_rng(opt.seed);
    // Generate random population
    let mut population = gen_rand_pop(|| rng.gen(), POP_SIZE);
    let mut breeding_pool = Vec::<Gene>::with_capacity(POOL_SIZE);
    // Iterate
    for gen in 0..50 {
        if opt.verbosity >= 1 {
            println!("Generation {}", gen);
            print_individuals(population.as_slice());
        }
        // Selection by 2-way tournament
        select_2way_tournament(&mut rng,
                               &mut breeding_pool,
                               &population,
                               POOL_SIZE,
                               P_S,
                               opt.verbosity);
        // Crossover
        for i in range_step(0, POP_SIZE, 2) {
            let crossover_chance: f32 = rng.gen();
            if crossover_chance < P_C {
                let crossover_point = rng.gen_range(1, 16);
                let (son, daughter) =
                    crossover(breeding_pool[i], breeding_pool[i + 1], crossover_point);
                population[i].state = son;
                population[i + 1].state = daughter;
                if opt.verbosity >= 2 {
                    println!("#{}: {} and #{}: {} crossover at {} to produce {} and {}",
                             i,
                             breeding_pool[i],
                             i + 1,
                             breeding_pool[i + 1],
                             crossover_point,
                             son,
                             daughter);
                }
            } else {
                population[i].state = breeding_pool[i];
                population[i + 1].state = breeding_pool[i + 1];
                if opt.verbosity >= 2 {
                    println!("#{}: {} and #{}: {} survive until the next generation",
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
            if mutation_chance < P_M {
                if opt.verbosity >= 2 {
                    println!("Mutating #{} {}", i, individual.state);
                }
                mutate_inplace(&mut rng, &mut individual.state);
                if opt.verbosity >= 2 {
                    println!("into {}", individual.state);
                }
            }
        }
        // Evaluate fitness
        for ind in population.iter_mut() {
            ind.stats.fitness = ind.state.fitness();
        }
    }
    println!("Final results");
    population.sort_unstable_by_key(|ind| NotNaN::new(ind.stats.fitness).unwrap());
    print_individuals(population.as_slice());
}
