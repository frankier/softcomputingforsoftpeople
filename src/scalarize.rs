use rand::Rng;
use na::{VectorN, Vector2, Dim, DimName, DefaultAllocator, U2, distance, DMatrix, abs};
use na::geometry::Point;
use na::allocator::Allocator;
use ordered_float::NotNaN;
use std::collections::BinaryHeap;
use std::ops::Range;
use rand::seq::sample_indices;

// XXX: To implement for higher dimensions see
// "Sampling Uniformly from the Unit Simplex"
// http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
// and literature on n-dimensional space filling curves
pub trait WeightVector<'a, D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    type NeighborIterator: Iterator<Item = usize>;
    type RandNeighborIterator: Iterator<Item = usize>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator;

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator;

    fn get_lambda(&self) -> &[VectorN<f32, D>];
}

pub struct UniformWeightVector2D {
    lambda: Vec<Vector2<f32>>,
    t: u32,
}

impl UniformWeightVector2D {
    pub fn new(num: u32, t: u32) -> UniformWeightVector2D {
        assert!(t <= num);
        let n = num as f32;
        let n1 = n - 1.0;
        let lambda: Vec<_> = (0..num)
            .into_iter()
            .map(|i| {
                     let i_f = i as f32;
                     Vector2::new(i_f / n1, (n - i_f) / n1)
                 })
            .collect();

        UniformWeightVector2D { t, lambda }
    }

    pub fn neighbor_bounds(from: usize, t: usize, len: usize) -> (usize, usize) {
        let ht = t / 2;
        let start = if ht > from { 0 } else { from - ht };
        if start + t <= len {
            (start, (start + t))
        } else {
            (len - t, len)
        }
    }
}

impl<'a> WeightVector<'a, U2> for UniformWeightVector2D {
    type NeighborIterator = Range<usize>;
    type RandNeighborIterator = Box<Iterator<Item = usize>>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator {
        let (start, end) =
            UniformWeightVector2D::neighbor_bounds(from, self.t as usize, self.lambda.len());
        start..end
    }

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator {
        let (start, end) =
            UniformWeightVector2D::neighbor_bounds(from, self.t as usize, self.lambda.len());
        Box::new(sample_indices(rng, end - start, select)
                     .into_iter()
                     .map(move |idx| start + idx))
    }

    fn get_lambda(&self) -> &[VectorN<f32, U2>] {
        &self.lambda
    }
}

pub struct ChosenWeightVector<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D> //<DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    // The weights
    lambda: Vec<VectorN<f32, D>>,
    // The neighborhood size
    t: usize,
    // The neighborhood lookup
    b: DMatrix<usize>,
}

impl<D> ChosenWeightVector<D>
    where D: Dim + DimName + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    pub fn new(lambda: Vec<VectorN<f32, D>>, t: usize) -> ChosenWeightVector<D> {
        assert!(t >= 1);
        assert!(t <= lambda.len());

        // Matrix is column major, so neighbors should be
        let mut b = DMatrix::<usize>::from_element(t, lambda.len(), 0);
        let mut heap = BinaryHeap::with_capacity(t);
        for (i, l1) in lambda.iter().enumerate() {
            for (j, l2) in lambda.iter().enumerate() {
                let d = NotNaN::new(distance(&Point::<f32, D>::from_coordinates(l1.clone()),
                                             &Point::<f32, D>::from_coordinates(l2.clone())))
                        .unwrap();
                if heap.len() < t {
                    heap.push((d, j));
                } else if d < heap.peek().unwrap().0 {
                    *(heap.peek_mut().unwrap()) = (d, j);
                }
            }
            for (n_i, (_d, j)) in heap.drain().enumerate() {
                b[(i, n_i)] = j;
            }
        }

        ChosenWeightVector { t, lambda, b }
    }
}

impl<'a, D> WeightVector<'a, D> for ChosenWeightVector<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    // XXX: impl trait
    type NeighborIterator = Box<Iterator<Item = usize> + 'a>;
    type RandNeighborIterator = Box<Iterator<Item = usize> + 'a>;

    fn neighbors(&'a self, from: usize) -> Self::NeighborIterator {
        // XXX: column doesn't work because it is not guaranteed to produce a contiguous slice --
        // still, hopefully there will possibly be a neater way to do this in future
        let start = from * self.t;
        Box::new(self.b.as_slice()[start..(start + self.t)]
                     .iter()
                     .map(|&n| n))
    }

    fn rand_neighbors<R: Rng>(&'a self,
                              rng: &mut R,
                              from: usize,
                              select: usize)
                              -> Self::RandNeighborIterator {
        Box::new(sample_indices(rng, self.t, select)
                     .into_iter()
                     .map(move |idx| self.b[(from, idx)]))
    }

    fn get_lambda(&self) -> &[VectorN<f32, D>] {
        &self.lambda
    }
}

pub trait Scalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32;
}

#[derive(new)]
pub struct TchebycheffScalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D> //<DefaultAllocator as Allocator<f32, D>>::Buffer: Copy
{
    z_star: VectorN<f32, D>,
}

impl<D> Scalarizer<D> for TchebycheffScalarizer<D>
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32 {
        fitness
            .iter()
            .zip(lambda.iter())
            .zip(self.z_star.iter())
            .map(|((f, l), z)| NotNaN::new(l * abs(&(f - z))).unwrap())
            .max()
            .unwrap()
            .into_inner()
    }
}

#[derive(new)]
pub struct WeightedSumScalarizer();

impl<D> Scalarizer<D> for WeightedSumScalarizer
    where D: Dim + Copy,
          DefaultAllocator: Allocator<f32, D>
{
    fn scalarize(&self, fitness: &VectorN<f32, D>, lambda: &VectorN<f32, D>) -> f32 {
        fitness
            .iter()
            .zip(lambda.iter())
            .map(|(f, l)| l * f)
            .sum()
    }
}
