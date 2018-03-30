#![feature(swap_with_slice)]

extern crate safe_transmute;
extern crate rand;
extern crate ordered_float;
extern crate indexmap;
extern crate nalgebra as na;
#[macro_use]
extern crate itertools;
extern crate alga;
extern crate num_traits;
#[macro_use]
extern crate derive_new;
extern crate bitwise;

use std::fmt::Debug;

pub mod utils;
pub mod individual;
pub mod operators;
pub mod algorithms;
pub mod scalarize;

use individual::*;

pub fn gen_rand_pop<F, S, SS, G>(mut gen: G, size: usize) -> Vec<Individual<S, SS>>
    where F: PartialOrd + Debug,
          S: State<Fitness = F>,
          SS: Stats<Fitness = F>,
          G: FnMut() -> S
{
    return (0..size).map(|_| Individual::new(gen())).collect();
}
