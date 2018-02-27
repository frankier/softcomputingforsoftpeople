extern crate softcomputingforsoftpeople as sc;
extern crate ordered_float;
extern crate nalgebra as na;

use ordered_float::NotNaN;
use std::collections::BTreeSet;
use na::{VectorN, Vector2, Dim, DefaultAllocator, U2};
use na::allocator::Allocator;
use std::cmp::Ordering;
use sc::{Stats, Individual};
use std::f32;

/*pub trait Fitnesses<D>
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D> {
    fn fitness(&self) -> VectorN<f32, D>;
}*/

#[derive(Copy, Clone, Debug)]
struct Gene();

#[derive(Copy, Clone, Debug)]
pub struct FitnessesRankStats<D>
        where
            D: Dim + Copy,
            DefaultAllocator: Allocator<f32, D>,
            <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy {
    pub fitness: VectorN<f32, D>,
    pub domination: DomCrowd
}

#[derive(Copy, Clone, Debug, Eq)]
pub struct DomCrowd {
    pub rank: u32,
    pub crowding: NotNaN<f32>,
}

impl DomCrowd {
    fn new() -> DomCrowd {
        DomCrowd {
            rank: 0,
            crowding: NotNaN::new(0.0).unwrap(),
        }
    }
}

impl Ord for DomCrowd {
    fn cmp(&self, other: &DomCrowd) -> Ordering {
        self.rank.cmp(&other.rank).then(
            other.crowding.cmp(&self.crowding))
    }
}

impl PartialOrd for DomCrowd {
    fn partial_cmp(&self, other: &DomCrowd) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DomCrowd {
    fn eq(&self, other: &DomCrowd) -> bool {
        self.rank == other.rank &&
            self.crowding == other.crowding
    }
}

impl<D> Stats for FitnessesRankStats<D>
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D>,
            <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy {
    type Fitness = VectorN<f32, D>;
    type CompetitionFitness = DomCrowd;

    fn new(fitness: VectorN<f32, D>) -> FitnessesRankStats<D> {
        FitnessesRankStats {
            fitness,
            domination: DomCrowd::new(),
        }
    }

    fn fitness(&self) -> DomCrowd {
        self.domination
    }
}

type T4aIndividual = Individual<Gene, FitnessesRankStats<U2>>;

fn non_dominated_sort<D>(points: &mut [&mut FitnessesRankStats<D>]) -> Vec<BTreeSet<usize>>
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D>,
            <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy {
    let mut sets = Vec::<BTreeSet<usize>>::with_capacity(points.len());
    let mut dom_counts = Vec::<u32>::with_capacity(points.len());
    let mut fronts = Vec::<BTreeSet<usize>>::new();
    fronts.push(BTreeSet::new());
    for (p_i, p) in points.iter().enumerate() {
        sets.push(BTreeSet::new());
        dom_counts.push(0);
        for (q_i, q) in points.iter().enumerate() {
            if p.fitness < q.fitness {
                sets[p_i].insert(q_i);
            } else if q.fitness < p.fitness {
                dom_counts[p_i] += 1;
            }
        }
        if dom_counts[p_i] == 0 {
            fronts[0].insert(p_i);
        }
    }
    for &p_i in fronts[0].iter() {
        points[p_i].domination.rank = 0;
    }
    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut new_front = BTreeSet::<usize>::new();
        for &p_i in fronts[i].iter() {
            for &q_i in sets[p_i].iter() {
                dom_counts[q_i] -= 1;
                if dom_counts[q_i] == 0 {
                    points[q_i].domination.rank = (i + 1) as u32;
                    new_front.insert(q_i);
                }
            }
        }
        fronts.push(new_front);
        i += 1;
    }
    fronts.pop();
    fronts
}

fn crowding<D>(points: &mut [&mut FitnessesRankStats<D>])
        where
            D: Dim,
            DefaultAllocator: Allocator<f32, D>,
            <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy {
    let l = points.len();
    if l == 0 {
        return;
    }
    for i in 0..l {
        points[i].domination.crowding = NotNaN::new(0.0).unwrap();
    }
    for m in 0..points[0].fitness.len() {
        let mut indices: Vec<_> = (0..points.len()).collect();
        indices.sort_by_key(|&idx| {
            NotNaN::new(points[idx].fitness[m]).unwrap()
        });
        //println!("m: {:?} indices: {:?} points: {:?}", m, indices, points);
        let f_min = points[indices[0]].fitness[m];
        let f_max = points[indices[indices.len() - 1]].fitness[m];
        points[indices[0]].domination.crowding = NotNaN::new(f32::INFINITY).unwrap();
        points[indices[l - 1]].domination.crowding = NotNaN::new(f32::INFINITY).unwrap();
        for i in 1..(l - 1) {
            //println!("i + 1: {:?}; i - 1: {:?}", points[indices[i + 1]].fitness[m], points[indices[i - 1]].fitness[m]);
            //println!("f_max: {:?}; f_min: {:?}", f_max, f_min);
            points[indices[i]].domination.crowding +=
                NotNaN::new(
                    (points[indices[i + 1]].fitness[m] - points[indices[i - 1]].fitness[m]) /
                    (f_max - f_min)).unwrap();
        }
    }
}

fn main() {
    let mut individuals: Vec<_> = [
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.5),
        (1.5, 3.0),
        (3.0, 1.6),
        (4.0, 3.5),
        (4.5, 3.1),
        (5.0, 2.5),
        (6.0, 5.0),
        (5.5, 7.0),
        (4.2, 6.0),
        (3.3, 6.5),
    ].into_iter().map(|&(x, y)| {
        FitnessesRankStats::new(
            Vector2::<f32>::new(x, y),
        )
    }).collect();
    {
        let fronts;
        {
            let mut individual_refs: Vec<_> = individuals.iter_mut().map(|x| x).collect();
            fronts = non_dominated_sort(individual_refs.as_mut_slice());
            for (i, front) in fronts.iter().enumerate() {
                let front1: BTreeSet<_> = front.iter().map(|p| { p + 1 }).collect();
                println!("front {}: {:?}", i, front1);
            }
        }
        for front in fronts.iter() {
            let mut front_refs = Vec::with_capacity(front.len());
            // XXX: This is okay because the indices are unique so we don't get aliasing mutable
            // borrows -- the fact we are using a set proves this -- there should be a non-unsafe
            // way to do this.
            for &elem_i in front.iter() {
                front_refs.push(unsafe {&mut *(&mut individuals[elem_i] as *mut FitnessesRankStats<U2>)});
            }
            crowding(front_refs.as_mut_slice());
        }
    }
    let mut numbered_individuals: Vec<_> = individuals.iter().enumerate().collect();
    numbered_individuals.sort_by_key(|&(_, idv)| {
        idv.domination
    });
    for (idx, individual) in numbered_individuals {
        println!(
            "individual {}: ({}, {}) rank: {} crowding: {}",
            idx + 1,
            individual.fitness.x, individual.fitness.y,
            individual.domination.rank, individual.domination.crowding);
    }
}
