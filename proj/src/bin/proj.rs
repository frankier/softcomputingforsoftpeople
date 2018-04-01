extern crate bincode;
extern crate nalgebra as na;
extern crate softcomputingforsoftpeople_proj as lib;
extern crate softcomputingforsoftpeople as sc;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate rand;
extern crate indexmap;
extern crate itertools;

use std::mem;
use std::cell::RefCell;
use std::fs::File;
use std::iter::FromIterator;
use bincode::{deserialize_from, serialize_into};
use structopt::StructOpt;
use indexmap::IndexSet;
use rand::Rng;
use std::collections::BTreeSet;
use itertools::Itertools;
use na::{VectorN, Vector4, U4, DimName, DefaultAllocator};
use na::allocator::Allocator;
use std::str::FromStr;

use sc::utils::rand::{parse_seed, get_rng};
use sc::utils::indexmap::{sample as sample_indexset};
use sc::scalarize::{Scalarizer, TchebycheffScalarizer};
use sc::algorithms::nsga2::non_dominated_sort;
use sc::algorithms::moead::multi_moead_next_gen;
use sc::individual::Individual;
use sc::individual::multiobj::MultipleFitnessStats;
use sc::operators::order::{
    IntersectRandAnchorDist, IntersectRandAnchorCrossover, PMX,
    Crossover as OrderCrossover};
use sc::scalarize::{sample_simplex, ChosenWeightVector, WeightVector, RescalingScalarizer};

//use corpus::;
use lib::individual::{Gene, TOO_SOON_FACTOR, ProjIndividual};
use lib::corpus::CorpusDump;
use lib::operators::{
    greedy_extend_rare_word, greedy_rand_extend_nov_movie,
    greedy_rand_extend_nov_all, greedy_extend_length, local_permutation_mutate,
    delete_mutate, repair};

fn apply_heuristic<R: Rng>(rng: &mut R, gene: &mut Gene) {
    // XXX: Should reuse a range object
    let randn = rng.gen_range(0, 6);
    match randn {
        0 => {
            greedy_rand_extend_nov_all(rng, gene, 1, 3, 100);
        }
        1 => {
            greedy_rand_extend_nov_movie(rng, gene, 1, 3, 100);
        }
        2 => {
            greedy_extend_rare_word(gene)
        }
        3 => {
            greedy_extend_length(gene)
        }
        4 => {
            local_permutation_mutate(rng, gene)
        }
        5 => {
            delete_mutate(rng, gene)
        }
        _ => {
            panic!("Impossible!");
        }
    }
    repair(gene);
}

fn crossover<'a, R: Rng>(rng: &mut R, gene: &[Gene<'a>]) -> (Gene<'a>, Gene<'a>) {
    let randn = rng.gen_range(0, 3);
    let mut children = match randn {
        0 => {
            let max_anchors = rng.gen_range(4, 16);
            IntersectRandAnchorDist::new(max_anchors).crossover(rng, gene)
        }
        1 => {
            let max_anchors = rng.gen_range(4, 16);
            IntersectRandAnchorCrossover::new(max_anchors).crossover(rng, gene)
        }
        2 => {
            PMX::new().crossover(rng, gene)
        }
        _ => {
            panic!("Impossible!");
        }
    };
    repair(&mut children.0);
    repair(&mut children.1);
    children
}

pub fn parse_vec<D>(src: &&str) -> VectorN<f32, D>
    where D: DimName,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    let bits = src.split(",").collect_vec();
    if bits.len() != D::try_to_usize().unwrap() {
        panic!("Wrong number of elements");
    } else {
        let mut vec = VectorN::<f32, D>::zeros();
        for (i, bit) in bits.iter().enumerate() {
            vec[i] = f32::from_str(bit).unwrap();
        }
        vec
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "pso")]
struct Opt {
    #[structopt(name = "OUTPUT", help = "Path to preprocessed corpus dump")]
    input: String,

    #[structopt(short = "v", long = "verbosity", help = "Verbosity", parse(from_occurrences))]
    verbosity: u64,

    #[structopt(short = "s", long = "seed", help = "Seed for the PRNG",
                parse(try_from_str = "parse_seed"))]
    seed: Option<[u32; 4]>,

    #[structopt(short = "g", long = "generations", default_value = "2000",
                help = "The number of generations to run for")]
    generations: u64,

    #[structopt(short = "z", long = "pop-size", default_value = "20",
                help = "The population size")]
    pop_size: usize,

    #[structopt(short = "i", long = "initial-gen-method", default_value = "rand",
                help = "The initial population generation method: rand | greedy | greedy_movie | order | length | mix")]
    initial_gen_method: String,

    #[structopt(short = "l", long = "initial-ind-length", default_value = "20",
                help = "The initial individual length")]
    initial_ind_length: usize,

    #[structopt(long = "scaling-input",
                help = "The file to read the scaling parameters from")]
    scaling_input: Option<String>,

    #[structopt(long = "scaling-output",
                help = "The file to write the scaling paramters to")]
    scaling_output: Option<String>,

    #[structopt(short = "t", long = "neighborhood-size", default_value = "3",
                help = "The neighborhood size")]
    t: usize,

    #[structopt(long = "z-star", help = "z-star", parse(from_str = "parse_vec"))]
    z_star: Option<Vector4<f32>>,

    #[structopt(long = "scale", help = "scale", parse(from_str = "parse_vec"))]
    scale: Option<Vector4<f32>>,

    #[structopt(long = "too-soon", help = "Too soon factor", default_value = "1.0")]
    too_soon: f32,

    #[structopt(long = "save-solutions",
                help = "The file to save the solutions to")]
    solutions_output: Option<String>,
}

fn rand_solution<'a, R>(rng: &mut R, corpus_dump: &'a CorpusDump, length: usize) -> Gene<'a>
    where R: Rng
{
    let sentences = &corpus_dump.stats.length_sentences;
    let rand_o = IndexSet::from_iter(sample_indexset(rng, sentences, length).cloned());

    Gene::new(corpus_dump, rand_o)
}

fn greedy_solution<'a, R>(rng: &mut R, corpus_dump: &'a CorpusDump, length: usize,
                          extend: fn(&mut R, &mut Gene, usize, usize, usize)) -> Gene<'a>
    where R: Rng
{
    let order = IndexSet::new();
    let mut gene = Gene::new(corpus_dump, order);
    gene.vocab = RefCell::new(Some(BTreeSet::new()));
    for _ in 0..length {
        extend(rng, &mut gene, 1, 3, 100);
    }
    gene
}

fn order_greedy_solution<'a>(corpus_dump: &'a CorpusDump, length: usize,
                             extend: fn(&mut Gene)) -> Gene<'a>
{
    let order = IndexSet::new();
    let mut gene = Gene::new(corpus_dump, order);
    gene.vocab = RefCell::new(Some(BTreeSet::new()));
    for _ in 0..length {
        extend(&mut gene);
    }
    gene
}

fn print_fitnesses<'a, S: Scalarizer<U4>>(scalarizer: &S,
                                          individuals: &[ProjIndividual<'a>],
                                          lambda: &[VectorN<f32, U4>]) {
    for (idx, (individual, l_i)) in individuals.iter().zip(lambda).enumerate() {
        let scalarized = scalarizer.scalarize(&individual.stats.fitness, l_i);
        println!("#{}: {} {:?} {:?}", idx + 1, scalarized, l_i, individual.stats.fitnesses());
    }
}

fn main() {
    let opt = Opt::from_args();
    let mut rng = get_rng(opt.seed);

    println!("Loading corpus");
    let buffer = File::open(&opt.input).unwrap();
    let corpus_dump: CorpusDump = deserialize_from(buffer).unwrap();
    println!("Loaded");

    let mut z_star;
    let base_scalarizer;
    let scalarizer;
    if opt.z_star.is_some() && opt.scale.is_some() {
        z_star = opt.z_star.unwrap();
        base_scalarizer = TchebycheffScalarizer::new(z_star);
        let scale = opt.scale.unwrap();
        scalarizer = RescalingScalarizer::new(scale, base_scalarizer);
    } else if let Some(ref scale_inf) = opt.scaling_input {
        let buffer = File::open(scale_inf).unwrap();
        let scaling_parameters: (Vector4<f32>, Vector4<f32>) = deserialize_from(buffer).unwrap();
        z_star = scaling_parameters.0;
        let scale = scaling_parameters.1;
        base_scalarizer = TchebycheffScalarizer::new(z_star);
        println!("Loaded z_star: {:?} scale: {:?}", z_star, scale);
        scalarizer = RescalingScalarizer::new(scale, base_scalarizer);
    } else {
        z_star = Vector4::<f32>::zeros();
        base_scalarizer = TchebycheffScalarizer::new(z_star);
        scalarizer = RescalingScalarizer::new(
            Vector4::<f32>::repeat(1.0),
            base_scalarizer);
    }

    unsafe {
        TOO_SOON_FACTOR = opt.too_soon;
    }
    //VectorN<f32, FitD>

    let lambda = sample_simplex(&mut rng, opt.pop_size);
    //println!("lambda {:?}", lambda);
    let weight_vec = ChosenWeightVector::new(lambda, opt.t);

    let mut population: Vec<ProjIndividual> = (0..opt.pop_size).map(|ind_idx| {
        Individual::new(
            match opt.initial_gen_method.as_str() {
                "rand" => {
                    rand_solution(&mut rng, &corpus_dump, opt.initial_ind_length)
                },
                "greedy" => {
                    greedy_solution(&mut rng, &corpus_dump, opt.initial_ind_length, greedy_rand_extend_nov_all)
                }
                "greedy_movie" => {
                    greedy_solution(&mut rng, &corpus_dump, opt.initial_ind_length, greedy_rand_extend_nov_movie)
                }
                "order" => {
                    order_greedy_solution(&corpus_dump, opt.initial_ind_length, greedy_extend_rare_word)
                }
                "order_length" => {
                    order_greedy_solution(&corpus_dump, opt.initial_ind_length, greedy_extend_length)
                }
                "mix" => {
                    match ind_idx % 5 {
                        0 => {
                            rand_solution(&mut rng, &corpus_dump, opt.initial_ind_length)
                        },
                        1 => {
                            greedy_solution(&mut rng, &corpus_dump, opt.initial_ind_length, greedy_rand_extend_nov_all)
                        },
                        2 => {
                            greedy_solution(&mut rng, &corpus_dump, opt.initial_ind_length, greedy_rand_extend_nov_movie)
                        },
                        3 => {
                            order_greedy_solution(&corpus_dump, opt.initial_ind_length, greedy_extend_rare_word)
                        },
                        4 => {
                            order_greedy_solution(&corpus_dump, opt.initial_ind_length, greedy_extend_length)
                        },
                        _ => {
                            panic!("impossibleu :-{O");
                        }
                    }
                }
                _ => {
                    panic!("bad initial gen method");
                }
            }
        )
    }).collect();
    
    let ind_stats = population.iter().map(|ind| ind.stats).collect_vec();
    let (_, ranks) = non_dominated_sort(ind_stats.as_slice());
    let mut solutions = ranks[0].iter().map(|&idx| {
        (population[idx].clone(), weight_vec.get_lambda()[idx])
    }).collect_vec();

    for gen in 0..opt.generations {
        println!("Generation {}", gen);
        print_fitnesses(&scalarizer, population.as_slice(), weight_vec.get_lambda());
        multi_moead_next_gen(
            &mut rng,
            &crossover,
            2,
            &apply_heuristic,
            &scalarizer,
            &mut population,
            &weight_vec,
            &mut z_star);

        let ind_stats = solutions.iter().map(|(ind, _)| ind.stats).chain(population.iter().map(|ind| ind.stats)).collect_vec();
        let (_, ranks) = non_dominated_sort(ind_stats.as_slice());
        let new_solutions = ranks[0].iter().map(|idx| {
            if *idx < solutions.len() {
                solutions[*idx].clone()
            } else {
                let adj_idx = *idx - solutions.len();
                (population[adj_idx].clone(), weight_vec.get_lambda()[adj_idx])
            }
        }).collect_vec();
        mem::replace(&mut solutions, new_solutions);
    }
    
    println!("Final generation");
    print_fitnesses(&scalarizer, population.as_slice(), weight_vec.get_lambda());

    {
        println!("Patero optimal archive");
        let (only_solutions, weights): (Vec<_>, Vec<_>) = solutions.iter().cloned().unzip();
        print_fitnesses(&scalarizer, only_solutions.as_slice(), weights.as_slice());
    }

    if let Some(scale_outf) = opt.scaling_output {
        let mut buffer = File::create(&scale_outf).unwrap();
        /*
        let scaling = population.iter()
            .map(|ind| ind.stats.fitnesses() - z_star)
            .fold(Vector5::<f32>::zeros(), |acc, dist| {
                acc.zip_map(&dist, |a, d| {
                    if a > d {
                        a
                    } else {
                        d
                    }
                })
            });
            */
        let scaling = population.iter()
            .map(|ind| ind.stats.fitnesses().abs())
            .fold(Vector4::<f32>::repeat(1.0), |acc, dist| {
                acc.zip_map(&dist, |a, d| {
                    if a > d {
                        a
                    } else {
                        d
                    }
                })
            });
            
        let scaled_z_star = z_star.component_div(&scaling);
        serialize_into(&mut buffer, &(scaled_z_star, scaling)).unwrap();
    }

    if let Some(solutions_outf) = opt.solutions_output {
        let mut buffer = File::create(&solutions_outf).unwrap();

        serialize_into(&mut buffer, &solutions.into_iter().map(|(s, w)| (s.state.solution, w)).collect_vec()).unwrap();
    }
}
