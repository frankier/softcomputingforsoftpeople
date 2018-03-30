use individual::real::RealGene;
use na::{VectorN, Dim, DimName, DefaultAllocator, abs, U1};
use na::allocator::Allocator;
use rand::Rng;
use ordered_float::NotNaN;
use alga::linear::FiniteDimInnerSpace;
use utils::real::Hypercube;
use rand::distributions::{Normal, IndependentSample};

pub trait Crossover<D, RG>
    where RG: RealGene<f32, D>,
          D: Dim,
          DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8;
    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG);
}

#[derive(new)]
pub struct LinearCrossover();

impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for LinearCrossover
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, _rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.25.5297
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();
        let c1 = 0.5 * mummy_v + 0.5 * daddy_v;
        let c2 = -0.5 * mummy_v + 1.5 * daddy_v;
        let c3 = 1.5 * mummy_v - 0.5 * daddy_v;
        let c1g = RG::from_vec(c1);
        let c2g = RG::from_vec(c2);
        let c3g = RG::from_vec(c3);
        // XXX: Penalties not taken account of here
        let c1f = c1g.fitness();
        let c2f = c2g.fitness();
        let c3f = c2g.fitness();
        if c1f > c2f {
            if c2f > c3f { (c1g, c2g) } else { (c1g, c3g) }
        } else {
            if c1f > c3f { (c1g, c2g) } else { (c2g, c3g) }
        }
    }
}

#[derive(new)]
pub struct BlendCrossover {
    alpha: f32,
}

impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for BlendCrossover
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 {
        2
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // http://heuristic.kaist.ac.kr/cylee/xga/Paper%20Review/Papers/[P6]%20Real-Coded%20Genetic%20Algorithms.pdf
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();

        fn mk_child<R, D>(rng: &mut R,
                          alpha: f32,
                          mummy_v: &VectorN<f32, D>,
                          daddy_v: &VectorN<f32, D>)
                          -> VectorN<f32, D>
            where R: Rng,
                  D: Dim,
                  DefaultAllocator: Allocator<f32, D>
        {
            let u: f32 = rng.gen();
            let gamma = (1.0 + 2.0 * alpha) * u - alpha;
            (1.0 - gamma) * mummy_v + gamma * daddy_v
        }

        (RG::from_vec(mk_child(rng, self.alpha, mummy_v, daddy_v)),
         RG::from_vec(mk_child(rng, self.alpha, mummy_v, daddy_v)))
    }
}

#[derive(new)]
pub struct UndxCrossover {
    xi_var: f32,
    eta_var: f32,
}


impl<D, RG: RealGene<f32, D>> Crossover<D, RG> for UndxCrossover
    where D: Dim + DimName,
          DefaultAllocator: Allocator<f32, D>
{
    fn parents(&self) -> u8 {
        3
    }

    fn crossover<R: Rng>(&self, rng: &mut R, parents: &[RG]) -> (RG, RG) {
        // https://pdfs.semanticscholar.org/d486/e294dff024ee600aa7a9b718528ccef532e1.pdf?_ga=2.108420555.853277512.1518964893-1720394428.1516112356&_gac=1.191070936.1518425497.CjwKCAiAk4XUBRB5EiwAHBLUMRdkvVUvd8tx7dT5MDBsj1X0BXp5ZRtj8nwe-UBne64B-cYfoy_OthoCBWYQAvD_BwE
        let mummy_v = parents[0].get_vec();
        let daddy_v = parents[1].get_vec();
        let milkman_v = parents[2].get_vec();
        let x_p = 0.5 * (mummy_v + daddy_v);
        // primary search direction
        let d = daddy_v - mummy_v;
        // distance in seconary search direction (distance from line formed by d to milkman)
        let s = ((milkman_v - mummy_v) - (milkman_v - mummy_v).dot(&d) * &d).norm();
        // Now we seek an orthonormal basis forming a hyperplane orthongonal to the primary search
        // direction d.
        // We can use orthonormalize to run the Gram-Schmitt procedure to get an orthonormal basis from
        // any other basis. This basis will leave its first dimension untouched.
        // We know we can use the current basis as a starting point, but we must make sure to, deal
        // with the situation where e.g. d = [1, 0, 0]. In this case we must make sure not to add
        // [1, 0, 0] again, otherwise we won't end up with a basis. The code below ensures this.
        let mut basis: Vec<_> = d.iter().enumerate().collect();
        basis.sort_unstable_by_key(|&(_i, x)| NotNaN::new(-abs(x)).unwrap());
        let mut basis: Vec<_> = basis
            .iter()
            .map(|&(i, _x)| {
                     let vec = VectorN::<f32, D>::from_fn(|r, _| if r == i { 1.0 } else { 0.0 });
                     vec
                 })
            .collect();
        basis[0] = d.to_owned();
        let _n = FiniteDimInnerSpace::orthonormalize(basis.as_mut_slice());
        // Now generate the two children
        let xi_dist = Normal::new(0.0, self.xi_var.sqrt().into());
        let eta_dist = Normal::new(0.0, self.eta_var.sqrt().into());

        fn mk_child<R, D>(rng: &mut R,
                          x_p: &VectorN<f32, D>,
                          xi_dist: &Normal,
                          d: &VectorN<f32, D>,
                          eta_dist: &Normal,
                          e: &[VectorN<f32, D>],
                          s: f32)
                          -> VectorN<f32, D>
            where R: Rng,
                  D: Dim + DimName,
                  DefaultAllocator: Allocator<f32, D>
        {
            let xi = xi_dist.ind_sample(rng) as f32;
            let primary = x_p + xi * d;
            let it = e.iter()
                .map(|e_i| {
                         let eta_i = eta_dist.ind_sample(rng) as f32;
                         eta_i * e_i
                     });
            // VectorN::<f32, D>::sum(it)
            let mut summed = VectorN::<f32, D>::zeros();
            for elem in it {
                summed += elem;
            }
            let secondary = s * summed;
            primary + secondary
        }

        (RG::from_vec(mk_child(rng, &x_p, &xi_dist, &d, &eta_dist, &basis[1..], s)),
         RG::from_vec(mk_child(rng, &x_p, &xi_dist, &d, &eta_dist, &basis[1..], s)))
    }
}

pub trait Mutation<D, RG: RealGene<f32, D>>
    where D: Dim,
          DefaultAllocator: Allocator<f32, D>
{
    fn mutate<R: Rng>(&self, rng: &mut R, parent: &mut RG);
}

// F: Scalar + Ring + Signed + SampleRange + PartialOrd
#[derive(new)]
pub struct GlobalUniformMut<D: DimName>
    where DefaultAllocator: Allocator<f32, D, U1>
{
    hypercube: Hypercube<f32, D, U1>,
}

// F: Scalar + Ring + Signed + SampleRange + PartialOrd
impl<D, RG> Mutation<D, RG> for GlobalUniformMut<D>
    where D: Dim + DimName,
          RG: RealGene<f32, D>,
          DefaultAllocator: Allocator<f32, D>
{
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut RG) {
        let mut_vec = parent.get_mut_vec();
        *mut_vec = self.hypercube.sample(rng);
    }
}

#[derive(new)]
pub struct LocalUniformMut {
    range: f32,
}

impl<D, RG> Mutation<D, RG> for LocalUniformMut
    where D: Dim,
          RG: RealGene<f32, D>,
          DefaultAllocator: Allocator<f32, D>
{
    fn mutate<R: Rng>(&self, rng: &mut R, parent: &mut RG) {
        let parent_v = parent.get_mut_vec();
        *parent_v = parent_v.map(|xn| xn + rng.gen_range(-0.5 * self.range, 0.5 * self.range));
    }
}
/*

#[derive(new)]
struct NonUniformMut {
    hypercube: Hypercube<f32, U2, U1>,
    b: f32,
    t: u32,
    t_max: u32,
}

impl Mutation for NonUniformMut {
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut Gene) {
        //parent.0 = parent.0.map(|x| {
            //let tau = ;
            //x + rng.gen_range(-0.5 * self.range, 0.5 * self.range)
        //});
    }
}

#[derive(new)]
struct NormalMut {
    var: u32,
}

impl Mutation for NormalMut {
    fn mutate<X: Rng>(&self, rng: &mut X, parent: &mut Gene) {
        //parent.0 = parent.0.map(|x| {
            //let tau = ;
            //x + rng.gen_range(-0.5 * self.range, 0.5 * self.range)
        //});
    }
}
*/
