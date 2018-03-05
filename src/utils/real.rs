extern crate nalgebra as na;

use na::{MatrixMN, wrap, DimName, DefaultAllocator, Scalar, abs, clamp};
use alga::general::Ring;
use num_traits::sign::Signed;
use na::allocator::Allocator;
use rand::distributions::range::SampleRange;
use rand::Rng;

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
            })
        )
    }

    pub fn clamp(&self, point: &MatrixMN<N, R, C>) -> MatrixMN<N, R, C> {
        MatrixMN::<N, R, C>::from_iterator(
            izip!(point.iter(), self.min.iter(), self.max.iter()).map(|(pd, mind, maxd)| {
                clamp(*pd, *mind, *maxd)
            })
        )
    }
}
