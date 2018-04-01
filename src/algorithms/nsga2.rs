use ordered_float::NotNaN;
use std::collections::BTreeSet;
use na::{Dim, DefaultAllocator};
use na::allocator::Allocator;
use std::cmp::Ordering;
use std::f32;

use individual::multiobj::MultipleFitnessStats;

type Ranks = Vec<BTreeSet<usize>>;

#[derive(Copy, Clone, Debug, Eq)]
pub struct DomCrowd {
    pub rank: u32,
    pub crowding: NotNaN<f32>,
}

impl DomCrowd {
    pub fn new() -> DomCrowd {
        DomCrowd {
            rank: 0,
            crowding: NotNaN::new(0.0).unwrap(),
        }
    }
}

impl Ord for DomCrowd {
    fn cmp(&self, other: &DomCrowd) -> Ordering {
        self.rank
            .cmp(&other.rank)
            .then(other.crowding.cmp(&self.crowding))
    }
}

impl PartialOrd for DomCrowd {
    fn partial_cmp(&self, other: &DomCrowd) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DomCrowd {
    fn eq(&self, other: &DomCrowd) -> bool {
        self.rank == other.rank && self.crowding == other.crowding
    }
}

pub fn non_dominated_sort<St, D>(points: &[St]) -> (Vec<u32>, Ranks)
    where St: MultipleFitnessStats<D>,
          D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    let mut sets = Vec::<BTreeSet<usize>>::with_capacity(points.len());
    let mut dom_counts = Vec::<u32>::with_capacity(points.len());
    let mut ranks = vec![0; points.len()];
    let mut fronts = Vec::<BTreeSet<usize>>::new();
    fronts.push(BTreeSet::new());
    for (p_i, p) in points.iter().enumerate() {
        sets.push(BTreeSet::new());
        dom_counts.push(0);
        for (q_i, q) in points.iter().enumerate() {
            let p_fit = &p.fitnesses();
            let q_fit = &q.fitnesses();
            if p_fit < q_fit {
                sets[p_i].insert(q_i);
            } else if q_fit < p_fit {
                dom_counts[p_i] += 1;
            }
        }
        if dom_counts[p_i] == 0 {
            fronts[0].insert(p_i);
        }
    }
    for &p_i in fronts[0].iter() {
        ranks[p_i] = 0;
    }
    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut new_front = BTreeSet::<usize>::new();
        for &p_i in fronts[i].iter() {
            for &q_i in sets[p_i].iter() {
                dom_counts[q_i] -= 1;
                if dom_counts[q_i] == 0 {
                    ranks[q_i] = (i + 1) as u32;
                    new_front.insert(q_i);
                }
            }
        }
        fronts.push(new_front);
        i += 1;
    }
    fronts.pop();
    (ranks, fronts)
}

pub fn crowding<St, D>(points: &[&St]) -> Vec<NotNaN<f32>>
    where St: MultipleFitnessStats<D>,
          D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    let l = points.len();
    if l == 0 {
        return vec![];
    }
    let mut crowdings = vec![NotNaN::new(0.0).unwrap(); l];
    for m in 0..points[0].fitnesses().len() {
        let mut indices: Vec<_> = (0..points.len()).collect();
        indices.sort_by_key(|&idx| NotNaN::new(points[idx].fitnesses()[m]).unwrap());
        //println!("m: {:?} indices: {:?} points: {:?}", m, indices, points);
        let f_min = points[indices[0]].fitnesses()[m];
        let f_max = points[indices[indices.len() - 1]].fitnesses()[m];
        crowdings[indices[0]] = NotNaN::new(f32::INFINITY).unwrap();
        crowdings[indices[l - 1]] = NotNaN::new(f32::INFINITY).unwrap();
        for i in 1..(l - 1) {
            //println!("i + 1: {:?}; i - 1: {:?}", points[indices[i + 1]].fitness[m], points[indices[i - 1]].fitness[m]);
            //println!("f_max: {:?}; f_min: {:?}", f_max, f_min);
            crowdings[indices[i]] +=
                NotNaN::new((points[indices[i + 1]].fitnesses()[m] -
                             points[indices[i - 1]].fitnesses()[m]) /
                            (f_max - f_min))
                        .unwrap();
        }
    }
    crowdings
}

pub fn dom_crowd<St, D>(individuals: &[St]) -> (Vec<DomCrowd>, Ranks)
    where St: MultipleFitnessStats<D>,
          D: Dim,
          DefaultAllocator: Allocator<f32, D>,
          <DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    let mut dom_crowds = vec![
        DomCrowd {
            rank: 0,
            crowding: NotNaN::new(0.0).unwrap(),
        }; individuals.len()];
    let (ranks, fronts) = non_dominated_sort(individuals);
    for (i, rank) in ranks.iter().enumerate() {
        dom_crowds[i].rank = *rank;
    }
    for front in fronts.iter() {
        let crowdings = {
            let mut front_refs: Vec<&St> = 
                front.iter().map(|elem_i| {
                    &individuals[*elem_i]
                }).collect();
            crowding(front_refs.as_slice())
        };
        //
        for (&i, crowding) in front.iter().zip(crowdings.iter()) {
            dom_crowds[i].crowding = *crowding;
        }
    }
    (dom_crowds, fronts)
}
