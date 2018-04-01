use rand::Rng;
use na::{VectorN, Dim, DefaultAllocator};
use na::allocator::Allocator;

use individual::{Stats, Individual, State};
use individual::real::RealGene;
use individual::order::OrderGene;
use individual::multiobj::MultipleFitnessStats;
use operators::real::Crossover as RealCrossover;
use operators::order::Crossover as OrderCrossover;
use scalarize::{Scalarizer, WeightVector};

fn moead_add_child<'a, Sc, G, St, WV, FitD>(
        idx: usize,
        child: &Individual<G, St>,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        G: State<Fitness=VectorN<f32, FitD>>,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    // Update of z_star
    let fitnesses = child.stats.fitnesses();
    for d in 0..z_star.len() {
        if fitnesses[d] < z_star[d] {
            z_star[d] = fitnesses[d];
        }
    }
    // Update of neighboring solutions
    for neighbor_idx in weight_vec.neighbors(idx) {
        let l_i = weight_vec.get_lambda()[neighbor_idx];
        let child_g = scalarizer.scalarize(&fitnesses, &l_i);
        let neighbor_g = scalarizer.scalarize(&individuals[neighbor_idx].stats.fitnesses(), &l_i);
        if child_g < neighbor_g {
            individuals[neighbor_idx] = child.clone();
        }
    }
}

pub fn multi_moead_next_gen<'a, R, X, G, H, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: X,
        parents: usize,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        X: Fn(&mut R, &[G]) -> (G, G),
        G: State<Fitness=VectorN<f32, FitD>>,
        H: Fn(&mut R, &mut G) + Sized,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    for idx in 0..individuals.len() {
        let parents: Vec<_> = weight_vec
            .rand_neighbors(rng, idx, parents)
            .map(|idx| individuals[idx].state.clone())
            .collect();
        let mut children = Vec::with_capacity(4);

        // Reproduction
        let mut children_genes = crossover(rng, parents.as_slice());
        children.push(Individual::<G, St>::new(children_genes.0));
        children.push(Individual::<G, St>::new(children_genes.1));
        let mut daughter_gene = children[0].state.clone();
        let mut son_gene = children[1].state.clone();

        // Improvement
        apply_heuristic(rng, &mut daughter_gene);
        apply_heuristic(rng, &mut son_gene);
        children.push(Individual::<G, St>::new(daughter_gene));
        children.push(Individual::<G, St>::new(son_gene));

        // Add children
        for child in children {
            moead_add_child(idx, &child, scalarizer, individuals, weight_vec, z_star);
        }
    }
}

pub fn moead_next_gen<'a, R, X, G, H, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: X,
        parents: usize,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        X: Fn(&mut R, &[G]) -> (G, G),
        G: State<Fitness=VectorN<f32, FitD>>,
        H: Fn(&mut G) + Sized,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    for idx in 0..individuals.len() {
        let parents: Vec<_> = weight_vec
            .rand_neighbors(rng, idx, parents)
            .map(|idx| individuals[idx].state.clone())
            .collect();
        // Reproduction
        let mut child_gene = {
            let children_genes = crossover(rng, parents.as_slice());
            let flip: bool = rng.gen();
            if flip {
                children_genes.1
            } else {
                children_genes.0
            }
        };
        // Improvement
        apply_heuristic(&mut child_gene);
        let child = Individual::<G, St>::new(child_gene);
        moead_add_child(idx, &child, scalarizer, individuals, weight_vec, z_star);
    }
}

pub fn moead_next_gen_real<'a, R, G, CX, H, SolD, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: &CX,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        G: RealGene<f32, SolD, Fitness=VectorN<f32, FitD>>,
        CX: RealCrossover<SolD, G>,
        H: Fn(&mut G) + Sized,
        SolD: Dim + Copy,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, SolD>,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    moead_next_gen(
        rng,
        |rng, parents| {
            crossover.crossover(rng, parents)
        },
        crossover.parents() as usize,
        apply_heuristic,
        scalarizer,
        individuals,
        weight_vec,
        z_star
    )
}

pub fn moead_next_gen_order<'a, R, G, CX, H, N, FitD, Sc, St, WV>(
        rng: &mut R,
        crossover: &CX,
        apply_heuristic: &H,
        scalarizer: &Sc,
        individuals: &mut [Individual<G, St>],
        weight_vec: &'a WV,
        z_star: &mut VectorN<f32, FitD>)
    where
        R: Rng,
        G: OrderGene<N, Fitness=VectorN<f32, FitD>>,
        CX: OrderCrossover<G, N>,
        H: Fn(&mut G) + Sized,
        FitD: Dim + Copy,
        DefaultAllocator: Allocator<f32, FitD>,
        <DefaultAllocator as Allocator<f32, FitD>>::Buffer: Copy,
        Sc: Scalarizer<FitD>,
        St: MultipleFitnessStats<FitD> + Stats<Fitness=VectorN<f32, FitD>>,
        WV: WeightVector<'a, FitD>
{
    moead_next_gen(
        rng,
        |rng, parents| {
            crossover.crossover(rng, parents)
        },
        crossover.parents() as usize,
        apply_heuristic,
        scalarizer,
        individuals,
        weight_vec,
        z_star
    )
}
