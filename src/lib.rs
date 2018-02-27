#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate rand;
extern crate ordered_float;
extern crate arrayvec;
extern crate nalgebra;
#[macro_use]
extern crate itertools;
extern crate alga;
extern crate num_traits;

use alga::general::Ring;
use num_traits::sign::Signed;
use std::fmt::Debug;
use std::str::{from_utf8, Utf8Error};
use std::num::ParseIntError;
use rand::{Rng, XorShiftRng, SeedableRng, thread_rng, Rand};
use ordered_float::NotNaN;
use rand::seq::sample_slice;
use nalgebra::{MatrixMN, wrap, DimName, DefaultAllocator, Scalar, abs};
use nalgebra::allocator::Allocator;
use rand::distributions::range::SampleRange;

//pub mod proj;

#[derive(Debug)]
pub enum ParseSeedError {
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

pub fn parse_seed(src: &str) -> Result<[u32; 4], ParseSeedError> {
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

pub fn rng_with_seed() -> (XorShiftRng, [u32; 4]) {
    let mut seed: [u32; 4] = thread_rng().gen();
    while seed == [0, 0, 0, 0] {
        seed = thread_rng().gen();
    }
    (XorShiftRng::from_seed(seed), seed)
}

pub fn get_rng(maybe_seed: Option<[u32; 4]>) -> XorShiftRng {
    if let Some(seed) = maybe_seed {
        XorShiftRng::from_seed(seed)
    } else {
        let (new_rng, seed) = rng_with_seed();
        println!("This run can be replicated with -s {:08x}{:08x}{:08x}{:08x}",
                 seed[0],
                 seed[1],
                 seed[2],
                 seed[3]);
        new_rng
    }
}

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
    fn new(state: S) -> Individual<S, SS> {
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

/*#[derive(Copy, Clone)]
pub struct PSOIndividual<G: Gene> {
    pub gene: G,
    pub vel: G,
    pub pbest: G,
    pub fitness: f32,
}*/

pub fn gen_rand_pop<F, S, SS, G>(mut gen: G, size: usize) -> Vec<Individual<S, SS>>
        where
            F: PartialOrd + Debug,
            S: State<Fitness=F>,
            SS: Stats<Fitness=F>,
            G: FnMut() -> S {
    return (0..size).map(|_| { Individual::new(gen()) }).collect();
}

pub fn select_2way_tournament<R, S, SS, F>(
        mut rng: &mut R,
        breeding_pool: &mut Vec<S>,
        population: &[Individual<S, SS>],
        pool_size: usize,
        prob_select: f32,
        verbosity: u64)
            where
                R: Rng,
                F: PartialOrd + Debug,
                S: State<Fitness=F> + Clone,
                SS: Stats<Fitness=F> {
    breeding_pool.clear();
    for i in 0..pool_size {
        let mut competitors = sample_slice(&mut rng, &population, 2);
        competitors.sort_unstable_by_key(|ind| ind.stats.fitness());
        if verbosity >= 2 {
            println!("#{} Tournament between {:?} and {:?}",
                     i,
                     competitors[1].state,
                     competitors[0].state);
        }
        let win_chance: f32 = rng.gen();
        breeding_pool.push(if win_chance < prob_select {
                               if verbosity >= 2 {
                                   println!("{:?} wins due to higher fitness",
                                            competitors[0].state);
                               }
                               competitors[1].state
                           } else {
                               if verbosity >= 2 {
                                   println!("{:?} wins despite lower fitness",
                                            competitors[1].state);
                               }
                               competitors[0].state
                           });
    }
}

#[derive(Clone)]
pub struct Hypercube<N: Scalar + Ring + Signed + SampleRange + PartialOrd, R: DimName, C: DimName>
        where DefaultAllocator: Allocator<N, R, C>
{
    min: MatrixMN<N, R, C>,
    max: MatrixMN<N, R, C>,
    range: MatrixMN<N, R, C>,
}

impl<N: Scalar + Ring + Signed + SampleRange + PartialOrd, R: DimName, C: DimName> Hypercube<N, R, C>
        where DefaultAllocator: Allocator<N, R, C>
{
    pub fn new(min: MatrixMN<N, R, C>, max: MatrixMN<N, R, C>) -> Hypercube<N, R, C> {
        assert!(min < max);
        let range = &max - &min;
        Hypercube { min, max, range: range }
    }

    pub fn sample<G: Rng>(&self, rng: &mut G) -> MatrixMN<N, R, C>
        where DefaultAllocator: Allocator<N, R, C>
    {
        self.min.zip_map(&self.max, |min_e, max_e| {
            rng.gen_range(min_e, max_e)
        })
    }

    //pub fn map
    //pub fn zip_map

    pub fn go_nearest_torus(&self, from: &MatrixMN<N, R, C>, to: &MatrixMN<N, R, C>) -> MatrixMN<N, R, C> {
        // That the nearest point will be the nearest in each dimension follows for any normed
        // vector space. This follows from the triangle inequality.
        let dir = MatrixMN::<N, R, C>::from_iterator(
            izip!(from.iter(), to.iter(), self.range.iter()).map(|(fd, td, rd)| {
                // Imagine hypercube boundaries at | and fd could be in either of
                // two positions:
                // td   |fd td fd|   td

                // First find the other td which would be nearest:
                let other_td;
                if td > fd {
                    other_td = *td - *rd;
                } else {
                    other_td = *td + *rd;
                }
                // Now find which of other td and td are nearest and return vector in direction of
                // closest
                let towards_td = *td - *fd;
                let towards_other_td = other_td - *fd;
                if abs(&towards_td) < abs(&towards_other_td) {
                    towards_td
                } else {
                    towards_other_td
                }
            })
        );
        //println!("from {:?} to {:?} is nearest at {:?} dir is {:?}", from, to, from + &dir, &dir);
        //nearest - from
        dir
    }

    pub fn place_torus(&self, point: &MatrixMN<N, R, C>) -> MatrixMN<N, R, C> {
        MatrixMN::<N, R, C>::from_iterator(
            izip!(point.iter(), self.min.iter(), self.max.iter()).map(|(pd, mind, maxd)| {
                wrap(*pd, *mind, *maxd)
            }))
    }
}
