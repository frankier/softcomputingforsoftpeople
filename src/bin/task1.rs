extern crate rand;
#[macro_use]
extern crate rand_derive;
extern crate ordered_float;
extern crate arrayvec;
extern crate bitwise;
#[macro_use]
extern crate static_assertions;
extern crate num_iter;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate softcomputingforsoftpeople;


use structopt::StructOpt;
use rand::Rng;
use rand::seq::sample_slice;
use ordered_float::NotNaN;
use bitwise::word::*;
use num_iter::range_step;
use std::fmt;
use softcomputingforsoftpeople::{parse_seed, get_rng, gen_rand_pop, Individual, Fitness, OnlyFitnessStats, select_2way_tournament};

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

impl fmt::Display for Gene {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (x1f, x2f) = interpret_gene(self.to_owned());
        write!(f,
               "<Gene {:04x} x1: {:02x} = {} x2: {:02x} = {}>",
               gene_bits(self.to_owned()),
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

impl softcomputingforsoftpeople::Fitness for Gene {
    fn fitness(&self) -> f32 {
        let (x1f, x2f) = interpret_gene(*self);
        -(x1f + x2f - 2.0 * x1f * x1f - x2f * x2f + x1f * x2f)
    }
}

impl softcomputingforsoftpeople::State for Gene {}

fn gene_bits(gene: Gene) -> u16 {
    unsafe { std::mem::transmute::<Gene, u16>(gene) }
}

fn gene_bits_mut(gene: &mut Gene) -> &mut u16 {
    unsafe { std::mem::transmute::<&mut Gene, &mut u16>(gene) }
}

/// Performs single-point crossover in place, replacing the parents with the children
fn crossover_inplace(mummy: &mut Gene, daddy: &mut Gene, bit: u8) {
    let mummys_bits = gene_bits_mut(mummy);
    let daddys_bits = gene_bits_mut(daddy);
    let mask = 0.set_bits_geq(bit);
    *daddys_bits ^= *mummys_bits & mask;
    *mummys_bits ^= *daddys_bits & mask;
    *daddys_bits ^= *mummys_bits & mask;
}

fn crossover(mut mummy: Gene, mut daddy: Gene, bit: u8) -> (Gene, Gene) {
    crossover_inplace(&mut mummy, &mut daddy, bit);
    (mummy, daddy)
}

fn mutate_inplace<R: Rng>(rng: &mut R, parent: &mut Gene) {
    let bits = gene_bits_mut(parent);
    let mut mask = 1;
    for _ in 0..16 {
        let flip_chance: f32 = rng.gen();
        if flip_chance < 1.0 / 16.0 {
            *bits ^= mask;
        }
        mask <<= 1;
    }
}

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
    let mut population = gen_rand_pop(|| {
        rng.gen()
    }, POP_SIZE);
    let mut breeding_pool = Vec::<Gene>::with_capacity(POOL_SIZE);
    // Iterate
    for gen in 0..50 {
        if opt.verbosity >= 1 {
            println!("Generation {}", gen);
            print_individuals(population.as_slice());
        }
        // Selection by 2-way tournament
        select_2way_tournament(
            &mut rng, &mut breeding_pool, &population,
            POOL_SIZE, P_S, opt.verbosity);
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
