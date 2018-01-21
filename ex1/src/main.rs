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


use structopt::StructOpt;
use rand::{Rng, XorShiftRng, SeedableRng, thread_rng};
use rand::seq::sample_slice;
use ordered_float::NotNaN;
use arrayvec::ArrayVec;
use bitwise::word::*;
use num_iter::range_step;
use std::num::ParseIntError;
use std::str::{from_utf8, Utf8Error};
use std::fmt;

const POP_SIZE: usize = 10;
const POOL_SIZE: usize = 10;
const P_S: f32 = 0.8;
const P_C: f32 = 0.8;
const P_M: f32 = 0.06;

const_assert!(pop_size_even; POP_SIZE % 2 == 0);
const_assert!(pool_size_even; POOL_SIZE % 2 == 0);

#[derive(Rand, Copy, Clone)]
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

#[derive(Copy, Clone)]
struct Individual {
    gene: Gene,
    fitness: f32,
}

fn rng_with_seed() -> (XorShiftRng, [u32; 4]) {
    let mut seed: [u32; 4] = thread_rng().gen();
    while seed == [0, 0, 0, 0] {
        seed = thread_rng().gen();
    }
    (XorShiftRng::from_seed(seed), seed)
}

fn fixed_to_float(x: u8) -> f32 {
    // This is slightly different from the task.
    // It covers the interval (1.0, 4 + 255/256)
    1.0 + (f32::from(x)) * 4.0 / 256.0
}

fn interpret_gene(gene: Gene) -> (f32, f32) {
    (fixed_to_float(gene.x1), fixed_to_float(gene.x2))
}

fn f_fitness(gene: Gene) -> f32 {
    let (x1f, x2f) = interpret_gene(gene);
    -(x1f + x2f - 2.0 * x1f * x1f - x2f * x2f + x1f * x2f)
}

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

fn print_individuals(individuals: &[Individual]) {
    for (idx, ind) in individuals.iter().enumerate() {
        println!("#{} {} fitness: {}", idx, ind.gene, ind.fitness);
    }
}

#[derive(Debug)]
enum ParseSeedError {
    IncorrectLength,
    HexParseError(ParseIntError),
    UnexpectedCharError(Utf8Error),
}

impl From<ParseIntError> for ParseSeedError {
    fn from(err: ParseIntError) -> Self {
        ParseSeedError::HexParseError(err)
    }
}

impl From<Utf8Error> for ParseSeedError {
    fn from(err: Utf8Error) -> Self {
        ParseSeedError::UnexpectedCharError(err)
    }
}

impl ToString for ParseSeedError {
    fn to_string(&self) -> String {
        match *self {
            ParseSeedError::IncorrectLength => {
                "Incorrect length of seed. Excepted 128 bits, which is 32 nibbles/characters."
                    .to_owned()
            }
            ParseSeedError::HexParseError(ref err) => {
                format!("Error parsing hexademical string {}", err.to_string()).to_owned()
            }
            ParseSeedError::UnexpectedCharError(ref err) => {
                format!("Unexpected wide character in hexademical string caused encoding error {}",
                        err.to_string())
                        .to_owned()
            }
        }
    }
}

fn parse_seed(src: &str) -> Result<[u32; 4], ParseSeedError> {
    let mut bits = src.as_bytes().chunks(8);
    if bits.len() != 4 {
        Err(ParseSeedError::IncorrectLength)
    } else {
        fn take_u32<'a, I: Iterator<Item = &'a [u8]>>(bits: &mut I) -> Result<u32, ParseSeedError> {
            Ok(u32::from_str_radix(from_utf8(bits.next().unwrap())?, 16)?)
        }
        Ok([take_u32(&mut bits)?,
            take_u32(&mut bits)?,
            take_u32(&mut bits)?,
            take_u32(&mut bits)?])
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "binary-coded-ga")]
struct Opt {
    /// A flag, true if used in the command line.
    #[structopt(short = "v", long = "verbosity", help = "Verbosity")]
    verbosity: u64,

    /// An argument of type float, with a default value.
    #[structopt(short = "s", long = "seed", help = "Seed for the PRNG",
                parse(try_from_str = "parse_seed"))]
    seed: Option<[u32; 4]>,
}

fn main() {
    let opt = Opt::from_args();
    let mut rng: XorShiftRng;
    if let Some(seed) = opt.seed {
        rng = rand::XorShiftRng::from_seed(seed);
    } else {
        let (new_rng, seed) = rng_with_seed();
        rng = new_rng;
        println!("This run can be replicated with -s {:08x}{:08x}{:08x}{:08x}",
                 seed[0],
                 seed[1],
                 seed[2],
                 seed[3]);
    }
    // Generate random population
    let mut population: ArrayVec<[Individual; POP_SIZE]> = (0..POP_SIZE)
        .map(|_| {
                 let gene: Gene = rng.gen();
                 let fitness = f_fitness(gene);
                 Individual { gene, fitness }
             })
        .collect();
    let mut breeding_pool = ArrayVec::<[Gene; POOL_SIZE]>::new();
    // Iterate
    for gen in 0..50 {
        if opt.verbosity >= 1 {
            println!("Generation {}", gen);
            print_individuals(population.as_slice());
        }
        // Selection by 2-way tournament
        breeding_pool.clear();
        for i in 0..POOL_SIZE {
            let mut competitors = sample_slice(&mut rng, &population, 2);
            competitors.sort_unstable_by_key(|ind| NotNaN::new(-ind.fitness).unwrap());
            if opt.verbosity >= 2 {
                println!("#{} Tournament between {} and {}",
                         i,
                         competitors[0].gene,
                         competitors[1].gene);
            }
            let win_chance: f32 = rng.gen();
            breeding_pool.push(if win_chance < P_S {
                                   if opt.verbosity >= 2 {
                                       println!("{} wins due to higher fitness",
                                                competitors[0].gene);
                                   }
                                   competitors[0].gene
                               } else {
                                   if opt.verbosity >= 2 {
                                       println!("{} wins despite lower fitness",
                                                competitors[1].gene);
                                   }
                                   competitors[1].gene
                               });
        }
        // Crossover
        for i in range_step(0, POP_SIZE, 2) {
            let crossover_chance: f32 = rng.gen();
            if crossover_chance < P_C {
                let crossover_point = rng.gen_range(1, 16);
                let (son, daughter) =
                    crossover(breeding_pool[i], breeding_pool[i + 1], crossover_point);
                population[i].gene = son;
                population[i + 1].gene = daughter;
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
                population[i].gene = breeding_pool[i];
                population[i + 1].gene = breeding_pool[i + 1];
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
                    println!("Mutating #{} {}", i, individual.gene);
                }
                mutate_inplace(&mut rng, &mut individual.gene);
                if opt.verbosity >= 2 {
                    println!("into {}", individual.gene);
                }
            }
        }
        // Evaluate fitness
        for ind in population.iter_mut() {
            ind.fitness = f_fitness(ind.gene);
        }
    }
    println!("Final results");
    population.sort_unstable_by_key(|ind| NotNaN::new(ind.fitness).unwrap());
    print_individuals(population.as_slice());
}
