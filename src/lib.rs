extern crate rand;

use std::str::{from_utf8, Utf8Error};
use std::num::ParseIntError;
use rand::{Rng, XorShiftRng, SeedableRng, thread_rng, Rand};

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

pub trait Fitness {
    fn fitness(&self) -> f32;
}

pub trait State: Fitness + ToString {}

pub trait Stats {
    fn new(fitness: f32) -> Self;
}

#[derive(Copy, Clone)]
pub struct Individual<S: State, SS: Stats> {
    pub state: S,
    pub stats: SS,
}

#[derive(Copy, Clone)]
pub struct OnlyFitnessStats {
    pub fitness: f32,
}

impl Stats for OnlyFitnessStats {
    fn new(fitness: f32) -> OnlyFitnessStats {
        OnlyFitnessStats {
            fitness
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

pub fn gen_rand_pop<R: Rng, S: State + Rand, SS: Stats>(rng: &mut R, size: usize) -> Vec<Individual<S, SS>> {
    return (0..size)
        .map(|_| {
                 let state: S = rng.gen();
                 let stats = Stats::new(state.fitness());
                 Individual { state, stats }
             })
        .collect();
}
