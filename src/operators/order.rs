use itertools::Itertools;
use rand::Rng;
use std::ops::Range;
use std::cmp::{max, min};
use indexmap::IndexSet;
use std::collections::btree_map::BTreeMap;
use operators::utils::gen_two_points;
use std::hash::Hash;

use individual::order::OrderGene;

pub trait IndexSetExt<T>
    where T: Hash + Eq
{
    /// Returns the index of the element `elem`.
    fn index_of(&self, elem: &T) -> Option<usize>;
}

impl<T> IndexSetExt<T> for IndexSet<T>
    where T: Hash + Eq
{
    fn index_of(&self, elem: &T) -> Option<usize> {
        self.get_full(elem).map(|(idx, _elem)| idx)
    }
}


///
fn index_set_slice<'a, T>(haystack: &'a IndexSet<T>, range: Range<usize>) -> impl Iterator<Item=&'a T> + 'a {
    range.filter_map(move |idx| {
        haystack.get_index(idx)
    })
}

pub trait Crossover<G, N>
    where G: OrderGene<N>
{
    fn parents(&self) -> u8;
    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[G]) -> (G, G);
}

/* These classic crossover types that have often been applied finding to TSP tours assume fixed
 * length genes */

// (*) https://link-springer-com.ezproxy.jyu.fi/article/10.1023%2FA%3A1006529012972
// (*) https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#For_ordered_chromosomes
// http://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf
// https://stackoverflow.com/questions/35861221/genetic-algorithms-how-to-do-crossover-on-ordered-collections-of-unique-element
// https://arxiv.org/pdf/1203.3097.pdf


#[derive(new)]
pub struct PMX();

impl<G, N> Crossover<G, N> for PMX
    where G: OrderGene<N>,
          N: Ord + Hash + Eq + Clone
{
    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[G]) -> (G, G) {
        fn pm_copy<N: Ord + Hash + Eq + Clone>(dest_order: &mut IndexSet<N>, range: Range<usize>, par_order: &IndexSet<N>, mapping: &BTreeMap<N, N>) {
            for idx in range {
                let par_e = par_order.get_index(idx).unwrap();
                if let Some(mapped_to) = mapping.get(par_e) {
                    dest_order.insert(mapped_to.clone());
                } else {
                    dest_order.insert(par_e.clone());
                }
            }
        }

        fn copy<N: Hash + Eq + Clone>(dest_order: &mut IndexSet<N>, range: Range<usize>, par_order: &IndexSet<N>) {
            for idx in range {
                let par_e = par_order.get_index(idx).unwrap();
                dest_order.insert(par_e.clone());
            }
        }

        let mummy_o = parents[0].get_order();
        let daddy_o = parents[1].get_order();
        let shorter_len = min(mummy_o.len(), daddy_o.len());
        let (xover_p1, xover_p2) = gen_two_points(rng, 0, shorter_len + 1);

        let mut daughter_o = IndexSet::with_capacity(mummy_o.len());
        let mut son_o = IndexSet::with_capacity(daddy_o.len());

        // Generate mappings
        let mut fwd_mapping: BTreeMap<N, N> = BTreeMap::new();
        let mut bwd_mapping: BTreeMap<N, N> = BTreeMap::new();

        for idx in xover_p1..xover_p2 {
            let mummy_e = mummy_o.get_index(idx).unwrap();
            let daddy_e = daddy_o.get_index(idx).unwrap();
            fwd_mapping.insert(mummy_e.clone(), daddy_e.clone());
            bwd_mapping.insert(daddy_e.clone(), mummy_e.clone());
        }

        pm_copy(&mut daughter_o, 0..xover_p1, mummy_o, &bwd_mapping);
        pm_copy(&mut son_o, 0..xover_p1, daddy_o, &fwd_mapping);
        copy(&mut daughter_o, xover_p1..xover_p1, daddy_o);
        copy(&mut son_o, xover_p1..xover_p1, mummy_o);
        pm_copy(&mut daughter_o, xover_p1..mummy_o.len(), mummy_o, &bwd_mapping);
        pm_copy(&mut son_o, xover_p1..daddy_o.len(), daddy_o, &fwd_mapping);

        let daughter = G::from_order(daughter_o);
        let son = G::from_order(son_o);
        (daughter, son)
    }
}

/* Anchor and distribute */
fn rand_anchors<R, N>(
        rng: &mut R, mummy_o: &IndexSet<N>, daddy_o: &IndexSet<N>,
        max_anchors: usize) -> (usize, Vec<N>)
    where R: Rng,
          N: Hash + Eq + Clone
{
    let mut potential_anchors: Vec<&N> = mummy_o.intersection(daddy_o).collect();
    let intersection_length = potential_anchors.len();
    rng.shuffle(potential_anchors.as_mut_slice());
    let mut anchors = Vec::with_capacity(potential_anchors.len());
    for anchor in potential_anchors {
        let mummy_idx = mummy_o.index_of(anchor).unwrap();
        let daddy_idx = daddy_o.index_of(anchor).unwrap();

        // anchors is kept ordered by mummy_o order
        // (and also daddy_o order as seen in a moment)
        let insert_position = anchors.binary_search_by_key(
            &mummy_idx,
            |other_anchor| mummy_o.index_of(other_anchor).unwrap()).unwrap_err();
        if let Some(mum_nearest_lt_e) = anchors.get(insert_position - 1) {
            let dad_nearest_lt_idx = daddy_o.index_of(mum_nearest_lt_e).unwrap();
            if dad_nearest_lt_idx > daddy_idx {
                continue;
            }
        }
        if let Some(mum_nearest_gt_e) = anchors.get(insert_position) {
            let dad_nearest_gt_idx = daddy_o.index_of(mum_nearest_gt_e).unwrap();
            if dad_nearest_gt_idx < daddy_idx {
                continue;
            }
        }
        anchors.insert(insert_position, anchor.clone());
        if anchors.len() >= max_anchors {
            break;
        }
    }
    (intersection_length, anchors)
}

/// Return positions of anchor nodes ready for creating intervals with .tuple_windows(...)
/// 
/// Say we have a index set [3, 5, 7, 42, 81, 99] and anchors {7, 81}, then this iterator will
/// return [0, 3, 5, 7]. That is it will return the indices after each anchor node while assuming
/// there is one anchor node before the beginning of the list and one after the end.
struct AnchorNodeIterator<'a, N: 'a> {
    anchors: &'a Vec<N>,
    index_set: &'a IndexSet<N>,
    idx: usize,
}

impl<'a, N> AnchorNodeIterator<'a, N> {
    fn new(anchors: &'a Vec<N>, index_set: &'a IndexSet<N>)
            -> AnchorNodeIterator<'a, N>
    {
        AnchorNodeIterator {
            anchors,
            index_set,
            idx: 0
        }
    }
}

impl<'a, N> Iterator for AnchorNodeIterator<'a, N>
        where N: Hash + Eq
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let result = 
            if self.idx == 0 {
                Some(0)
            } else if (self.idx as usize) <= self.anchors.len() {
                Some(self.index_set.index_of(&self.anchors[self.idx - 1]).unwrap() + 1)
            } else if (self.idx as usize) == self.anchors.len() + 1 {
                Some(self.index_set.len() + 1)
            } else {
                None
            };
        self.idx += 1;
        result
    }
}

#[derive(new)]
pub struct IntersectRandAnchorDist {
    max_anchors: usize
}

impl<G, N> Crossover<G, N> for IntersectRandAnchorDist
    where G: OrderGene<N>, 
          N: Hash + Eq + Ord + Clone
{
    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[G]) -> (G, G) {
        let mummy_o = parents[0].get_order();
        let daddy_o = parents[1].get_order();

        /* Get anchors */
        let (intersection_length, anchors) =
            rand_anchors(rng, mummy_o, daddy_o, self.max_anchors);

        /* Distribute */
        let anchor_interval_pairs = AnchorNodeIterator::new(&anchors, mummy_o)
            .zip(AnchorNodeIterator::new(&anchors, daddy_o));
        let child_len = mummy_o.len() + daddy_o.len() - intersection_length;
        let mut daughter_o = IndexSet::<N>::with_capacity(child_len);
        let mut son_o = IndexSet::<N>::with_capacity(child_len);
        for ((m_l, d_l), (m_r, d_r)) in anchor_interval_pairs.tuple_windows() {
            // Spread evenly by splitting into divisions of common divisor mum_gap * dad_gap and
            // then step through
            let mum_gap = m_r - m_l;
            let dad_gap = d_r - d_l;
            let divisions = mum_gap * dad_gap;
            let mut mum_divs = 0;
            let mut dad_divs = 0;
            let mut mum_idx = m_l;
            let mut dad_idx = d_l;
            // Place anchor element
            if m_l > 0 {
                let elem = mummy_o.get_index(m_l - 1).unwrap();
                daughter_o.insert(elem.clone());
                son_o.insert(elem.clone());
                mum_divs += dad_gap;
                dad_divs += mum_gap;
            }
            // Distribute remaining elements before the next anchor element
            while mum_divs < divisions && dad_divs < divisions {
                if mum_divs < dad_divs {
                    let elem = mummy_o.get_index(mum_idx).unwrap();
                    daughter_o.insert(elem.clone());
                    son_o.insert(elem.clone());
                    mum_divs += dad_gap;
                    mum_idx += 1;
                } else if dad_divs < mum_divs {
                    let elem = daddy_o.get_index(dad_idx).unwrap();
                    daughter_o.insert(elem.clone());
                    son_o.insert(elem.clone());
                    dad_divs += mum_gap;
                    dad_idx += 1;
                } else {
                    let mum_elem = mummy_o.get_index(mum_idx).unwrap();
                    let dad_elem = daddy_o.get_index(dad_idx).unwrap();
                    daughter_o.insert(mum_elem.clone());
                    daughter_o.insert(dad_elem.clone());
                    son_o.insert(dad_elem.clone());
                    son_o.insert(mum_elem.clone());
                    mum_divs += dad_gap;
                    dad_divs += mum_gap;
                    mum_idx += 1;
                    dad_idx += 1;
                }
            }
        }

        let daughter = G::from_order(daughter_o);
        let son = G::from_order(son_o);
        (daughter, son)
    }
}

#[derive(new)]
pub struct IntersectRandAnchorCrossover {
    max_anchors: usize
}

impl<G, N> Crossover<G, N> for IntersectRandAnchorCrossover
    where G: OrderGene<N>, 
          N: Hash + Eq + Ord + Clone
{

    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[G]) -> (G, G) {
        let mummy_o = parents[0].get_order();
        let daddy_o = parents[1].get_order();

        let longest_len = max(mummy_o.len(), daddy_o.len());

        /* Get anchors */
        let (_, anchors) =
            rand_anchors(rng, mummy_o, daddy_o, self.max_anchors);

        let mut daughter_o = IndexSet::<N>::with_capacity(longest_len);
        let mut son_o = IndexSet::<N>::with_capacity(longest_len);

        let anchor_interval_pairs = AnchorNodeIterator::new(&anchors, mummy_o)
            .zip(AnchorNodeIterator::new(&anchors, daddy_o));
        let mut swap = false;

        for ((m_l, d_l), (m_r, d_r)) in anchor_interval_pairs.tuple_windows() {
            let mum_slice = index_set_slice(mummy_o, m_l..m_r).cloned();
            let dad_slice = index_set_slice(daddy_o, d_l..d_r).cloned();
            if swap {
                daughter_o.extend(dad_slice);
                son_o.extend(mum_slice);
            } else {
                daughter_o.extend(mum_slice);
                son_o.extend(dad_slice);
            }
            swap = !swap;
        }

        let daughter = G::from_order(daughter_o);
        let son = G::from_order(son_o);
        (daughter, son)
    }
}

/*
#[derive(new)]
pub struct IntersectGreedyAnchorDist();

impl<G: OrderGene<N>, N> Crossover<G, N> for IntersectGreedyAnchorDist
{
    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[G]) -> (G, G) {
        let mummy_o = parents[0].get_order();
        let daddy_o = parents[1].get_order();
        let anchor = rng.gen_range(0, mummy_o.len() + 1);
    }
}
*/
