extern crate softcomputingforsoftpeople as sc;
extern crate ordered_float;
extern crate rand;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate nalgebra as na;
#[macro_use]
extern crate lazy_static;
extern crate alga;
extern crate num_traits;
extern crate csv;

use ordered_float::NotNaN;
use std::fmt;
use std::f32;
use rand::Rng;
use structopt::StructOpt;
use na::{Vector2, norm, U1, U2};
use sc::utils::rand::{parse_seed, get_rng};
use sc::utils::real::Hypercube;
use sc::individual::{Individual, GetFitness, Stats};
use std::fs::File;

lazy_static! {
    static ref HYPERCUBE: Hypercube<f32, U2, U1> = Hypercube::new(Vector2::<f32>::new(-1.0, -1.0), Vector2::<f32>::new(2.0, 1.0));
}

#[derive(Copy, Clone, Debug)]
struct State {
    position: Vector2<f32>,
    velocity: Vector2<f32>,
}


impl State {
    fn rand<G: Rng>(rng: &mut G, hypercube: &Hypercube<f32, U2, U1>, v_max: f32) -> State {
        let v_rand: Vector2<f32> = Vector2::<f32>::from_fn(|_x, _y| rng.gen_range(-1.0, 1.0));
        let v_unit = v_rand.normalize();
        let v_mag = rng.gen_range(0.0, v_max);
        State {
            position: hypercube.sample(rng),
            velocity: v_unit * v_mag,
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "<State x1: {} x2: {} vx1: {} vx2: {}>",
               self.position[0],
               self.position[1],
               self.velocity[0],
               self.velocity[1])
    }
}

impl fmt::Display for PSOBest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "<PSOBest x1: {} x2: {} fitness: {}>",
               self.position[0],
               self.position[1],
               self.fitness)
    }
}

impl GetFitness for State {
    type Fitness = f32;

    fn fitness(&self) -> f32 {
        f32::cos(self.position[0]) * f32::sin(self.position[1]) -
        self.position[0] / (self.position[1] * self.position[1] + 1.0)
    }
}

impl sc::individual::State for State {}

#[derive(Copy, Clone)]
struct PSOBest {
    pub position: Vector2<f32>,
    pub fitness: f32,
}

impl PSOBest {
    fn is_valid(&self) -> bool {
        !self.position[0].is_nan() && !self.position[1].is_nan()
    }

    fn new_invalid() -> PSOBest {
        PSOBest {
            position: Vector2::new(f32::NAN, f32::NAN),
            fitness: f32::INFINITY,
        }
    }
}

#[derive(Copy, Clone)]
struct PSOStats {
    pub pbest: PSOBest,
    pub fitness: f32,
}

impl Stats for PSOStats {
    type Fitness = f32;
    type CompetitionFitness = NotNaN<f32>;

    fn new(fitness: f32) -> PSOStats {
        PSOStats {
            fitness,
            pbest: PSOBest {
                position: Vector2::new(f32::NAN, f32::NAN),
                fitness: f32::INFINITY,
            },
        }
    }

    fn fitness(&self) -> NotNaN<f32> {
        NotNaN::new(self.fitness).unwrap()
    }
}

type PSOIndividual = Individual<State, PSOStats>;

fn print_individuals(individuals: &[PSOIndividual]) {
    for (idx, ind) in individuals.iter().enumerate() {
        println!("#{} {} fitness: {} best: {}",
                 idx,
                 ind.state,
                 ind.stats.fitness,
                 ind.stats.pbest);
    }
}

trait Neighborhood {
    fn update(&mut self, idx: usize, best: &PSOBest);
    fn get_best(&self, idx: usize) -> &PSOBest;
    fn overall_best(&self) -> &PSOBest;
}

struct GlobalNeighborhood {
    best: PSOBest,
}

struct RingNeighborhood {
    bests: Vec<PSOBest>,
}

impl Neighborhood for GlobalNeighborhood {
    fn update(&mut self, _idx: usize, best: &PSOBest) {
        // Update best
        if best.fitness <= self.best.fitness {
            self.best = *best;
        }
    }

    fn get_best(&self, _idx: usize) -> &PSOBest {
        self.overall_best()
    }

    fn overall_best(&self) -> &PSOBest {
        assert!(self.best.is_valid());
        &self.best
    }
}

impl GlobalNeighborhood {
    fn new() -> GlobalNeighborhood {
        GlobalNeighborhood { best: PSOBest::new_invalid() }
    }
}

impl Neighborhood for RingNeighborhood {
    fn update(&mut self, idx: usize, best: &PSOBest) {
        // Update left best
        let left_idx = if idx == 0 {
            self.bests.len() - 1
        } else {
            idx - 1
        };
        if best.fitness <= self.bests[left_idx].fitness {
            self.bests[left_idx] = *best;
        }
        // Update right best
        let right_idx = (idx + 1) % self.bests.len();
        if best.fitness <= self.bests[right_idx].fitness {
            self.bests[right_idx] = *best;
        }
    }

    fn get_best(&self, idx: usize) -> &PSOBest {
        let best = &self.bests[idx];
        assert!(best.is_valid());
        best
    }

    fn overall_best(&self) -> &PSOBest {
        self.bests
            .iter()
            .max_by_key(|best| NotNaN::new(-best.fitness).unwrap())
            .unwrap()
    }
}

impl RingNeighborhood {
    fn new(pop_size: usize) -> RingNeighborhood {
        assert!(pop_size >= 1);
        RingNeighborhood { bests: (0..pop_size).map(|_| PSOBest::new_invalid()).collect() }
    }
}

struct PSOOpt {
    verbosity: u64,
    generations: u64,
    cognitive: f32,
    social: f32,
    inertia: f32,
    v_max: f32,
}

fn pso<G: Rng, N: Neighborhood>(rng: &mut G,
                                population: &mut [PSOIndividual],
                                neighborhood: &mut N,
                                opt: &PSOOpt) {
    // Iterate
    for gen in 0..opt.generations {
        if opt.verbosity >= 1 {
            println!("Generation {}", gen);
            print_individuals(population);
        }
        // Update pbest/neighborhood best
        for idx in 0..population.len() {
            // Update pbest
            if population[idx].stats.fitness <= population[idx].stats.pbest.fitness {
                population[idx].stats.pbest = PSOBest {
                    position: population[idx].state.position,
                    fitness: population[idx].stats.fitness,
                };
                neighborhood.update(idx, &population[idx].stats.pbest);
            }
        }
        for (idx, individual) in population.iter_mut().enumerate() {
            {
                // Update velocity
                let mut velocity = &mut individual.state.velocity;
                let mut position = &mut individual.state.position;
                assert!(individual.stats.pbest.is_valid());
                let pbestx = &individual.stats.pbest.position;
                let pvec = HYPERCUBE.go_nearest_torus(position, pbestx);
                let nbest = neighborhood.get_best(idx);
                let gvec = HYPERCUBE.go_nearest_torus(position, &nbest.position);
                //println!("Velocity before {:?} pvec: {:?} gvec: {:?}", velocity, pvec, gvec);
                *velocity = opt.inertia * *velocity + opt.cognitive * rng.next_f32() * pvec +
                            opt.social * rng.next_f32() * gvec;
                //println!("Velocity after {:?}", velocity);
                // Rescale to VMAX if neccesary
                let vmag = norm(velocity);
                if vmag > opt.v_max {
                    *velocity = velocity.normalize() * opt.v_max;
                }
                // Update position
                *position = *position + *velocity;
                // Place on torus
                *position = HYPERCUBE.place_torus(position);
            }
            // Evaluate fitness function
            individual.stats.fitness = individual.state.fitness();
        }
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "pso")]
struct Opt {
    #[structopt(short = "v", long = "verbosity", help = "Verbosity", parse(from_occurrences))]
    verbosity: u64,

    #[structopt(short = "s", long = "seed", help = "Seed for the PRNG",
                parse(try_from_str = "parse_seed"))]
    seed: Option<[u32; 4]>,

    #[structopt(short = "c", long = "cognitive", default_value = "2.0",
                help = "The acceleration constant towards the personal best, aka the cognitive weight")]
    cognitive: f32,

    #[structopt(short = "o", long = "social", default_value = "2.0",
                help = "The acceleration constant towards the global or neighborhood best, aka the social weight")]
    social: f32,

    #[structopt(short = "i", long = "intertia", default_value = "1.0",
                help = "The constant by which the previous velocity is multiplied by when updating, aka inertia")]
    inertia: f32,

    #[structopt(short = "k", long = "constriction",
                help = "The constriction factor")]
    constriction: Option<f32>,

    #[structopt(short = "r", long = "ring",
                help = "Use a ring neighborhood rather than global best")]
    ring: bool,

    #[structopt(short = "g", long = "generations", default_value = "2000",
                help = "The number of generations to run for")]
    generations: u64,

    #[structopt(short = "z", long = "pop-size", default_value = "20",
                help = "The population size")]
    pop_size: usize,

    #[structopt(short = "m", long = "v-max", default_value = "3.0",
                help = "The maximum velocity")]
    v_max: f32,

    #[structopt(long = "csv",
                help = "Output a CSV for plotting using R")]
    csv_output: Option<String>,
}


fn main() {
    let opt = Opt::from_args();
    let cognitive;
    let social;
    let inertia;
    if let Some(k) = opt.constriction {
        // The Particle Swarm-Explosion, Stability, and Convergence in a Multidimensional
        // Complex Space, Maurice Clerc and James Kennedy 2002
        let phi = opt.social + opt.inertia;
        let chi = if phi > 4.0 {
            2.0 * k / (phi - 2.0 + (phi * phi - 4.0 * phi).sqrt())
        } else {
            k.sqrt()
        };
        println!("Applying constriction with chi = {}", chi);
        cognitive = opt.cognitive * chi;
        social = opt.social * chi;
        inertia = opt.inertia * chi;
    } else {
        cognitive = opt.cognitive;
        social = opt.social;
        inertia = opt.inertia;
    }
    let mut rng = get_rng(opt.seed);
    // Generate random population
    let mut population: Vec<PSOIndividual> = (0..opt.pop_size)
        .map(|_| {
                 let state = State::rand(&mut rng, &HYPERCUBE, opt.v_max);
                 let stats = Stats::new(state.fitness());
                 Individual { state, stats }
             })
        .collect();
    let pso_opt = PSOOpt {
        verbosity: opt.verbosity,
        v_max: opt.v_max,
        generations: opt.generations,
        cognitive,
        social,
        inertia,
    };
    let overall_best;
    if opt.ring {
        let mut neighborhood: RingNeighborhood = RingNeighborhood::new(opt.pop_size);
        pso(&mut rng,
            population.as_mut_slice(),
            &mut neighborhood,
            &pso_opt);
        overall_best = neighborhood.overall_best().clone();
    } else {
        let mut neighborhood = GlobalNeighborhood::new();
        pso(&mut rng,
            population.as_mut_slice(),
            &mut neighborhood,
            &pso_opt);
        overall_best = neighborhood.overall_best().clone();
    }
    println!("Final results");
    println!("Overall best {}", overall_best);
    population.sort_unstable_by_key(|ind| NotNaN::new(-ind.stats.fitness).unwrap());
    print_individuals(population.as_slice());
    if let Some(csv_fn) = opt.csv_output {
        let mut wtr = csv::Writer::from_writer(File::create(csv_fn).unwrap());
        // Since we're writing records manually, we must explicitly write our
        // header record. A header record is written the same way that other
        // records are written.
        wtr.write_record(&["x1", "x2", "type"]).unwrap();
        for individual in population.iter() {
            wtr.serialize((individual.state.position[0], individual.state.position[1], 0))
                .unwrap();
            wtr.serialize((individual.stats.pbest.position[0],
                            individual.stats.pbest.position[1],
                            1))
                .unwrap();
        }
        wtr.serialize((overall_best.position[0], overall_best.position[1], 2))
            .unwrap();

        // A CSV writer maintains an internal buffer, so it's important
        // to flush the buffer when you're done.
        wtr.flush().unwrap();
    }
}
