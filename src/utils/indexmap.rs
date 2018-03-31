use indexmap::IndexSet;
use std::hash::Hash;
use std::ops::Range;


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
