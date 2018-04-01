use indexmap::IndexSet;
use std::hash::Hash;
use std::ops::Range;
use rand::seq::sample_indices;
use rand::Rng;


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


/// Returns and iterator of a slice of an IndexSet
pub fn index_set_slice<'a, T>(haystack: &'a IndexSet<T>, range: Range<usize>)
        -> impl Iterator<Item=&'a T> + 'a
{
    range.filter_map(move |idx| {
        haystack.get_index(idx)
    })
}

pub fn sample<'a, R, T>(rng: &mut R, haystack: &'a IndexSet<T>, num: usize)
        -> impl Iterator<Item=&'a T> + 'a
    where R: Rng
{
    sample_indices(rng, haystack.len(), num)
        .into_iter()
        .filter_map(move |rand_idx| haystack.get_index(rand_idx))
}
