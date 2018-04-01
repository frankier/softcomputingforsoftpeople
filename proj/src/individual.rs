use itertools::Itertools;
use std::cell::RefCell;
use std::collections::BTreeSet;
use na::{Vector4, U4};

use sc::utils::indexmap::IndexSetExt;
use sc::individual::{GetFitness, State, Individual};
use sc::individual::order::OrderGene;
use sc::individual::multiobj::OnlyFitnessesStats;
use corpus::{CorpusDump, SentenceOrder, WordId, WordOrder, SentenceId};

pub static mut TOO_SOON_FACTOR: f32 = 1.0;

#[derive(Clone, Debug)]
pub struct Gene<'a> {
    pub dump: &'a CorpusDump,
    pub solution: SentenceOrder,
    pub vocab: RefCell<Option<BTreeSet<WordId>>>
}

impl<'a> Gene<'a> {
    pub fn new(dump: &'a CorpusDump, solution: SentenceOrder) -> Gene<'a> {
        Gene { dump, solution, vocab: RefCell::new(None) }
    }
}

pub fn intros(gene: &Gene) -> (Vec<Vec<WordId>>, BTreeSet<WordId>) {
    let mut vocab = BTreeSet::new();
    let mut intros = Vec::with_capacity(gene.solution.len());
    for sentence in gene.solution.iter() {
        let sent_words = gene.dump.bank.sentences.get(&sentence).unwrap();
        let new_words = sent_words.difference(&vocab).cloned().collect_vec();
        vocab.extend(new_words.iter());
        intros.push(new_words);
    }
    (intros, vocab)
}

pub fn novelty(intros: &Vec<Vec<WordId>>) -> usize {
    intros.iter().map(|intro_words| {
        intro_words.len()
    }).sum()
}

pub fn infeasibility(intros: &Vec<Vec<WordId>>) -> u64 {
    intros.iter().fold(0, |acc, intro_words| {
        if intro_words.len() == 0 {
            acc + 1
        } else {
            acc
        }
    })
}

pub fn too_soon(word_ranks: &WordOrder, intros: &Vec<Vec<WordId>>) -> f32 {
    let mut vocab_size = 0;
    let mut penalty = 0.0;

    for intro in intros.iter() {
        vocab_size += intro.len();
        for word in intro.iter() {
            let word_rank = word_ranks.index_of(word).unwrap() as f32;
            let max_word_rank = (vocab_size as f32 * unsafe { TOO_SOON_FACTOR }) - 1.0;
            if word_rank > max_word_rank {
                let overage = word_rank - max_word_rank;
                penalty += overage * overage;
            }
        }
    }
    penalty as f32 / intros.len() as f32
}

pub fn movie_changes(gene: &Gene) -> u64 {
    if gene.solution.len() == 0 {
        return 0;
    }
    let mut sentence_iter = gene.solution.iter();
    let prev_sentence = sentence_iter.next().unwrap();
    let mut changes = 0;
    for sentence in sentence_iter {
        if prev_sentence.movie_id != sentence.movie_id {
            changes += 1;
        }
    }
    changes
}

impl<'a> OrderGene<SentenceId> for Gene<'a> {
    fn from_order(&self, order: SentenceOrder) -> Gene<'a> {
        Gene::new(
            self.dump,
            order,
        )
    }

    fn get_order(&self) -> &SentenceOrder {
        &self.solution
    }

    fn get_mut_order(&mut self) -> &mut SentenceOrder {
        &mut self.solution
    }
}

impl<'a> GetFitness for Gene<'a> {
    type Fitness = Vector4<f32>;

    fn fitness(&self) -> Vector4<f32> {
        let (int, vocab) = intros(&self);
        self.vocab.replace(Some(vocab));

        let nov = novelty(&int);
        //let infes = infeasibility(&int);
        let soon = too_soon(&self.dump.stats.word_ranks, &int);
        let changes = movie_changes(&self);
        let length = self.solution.len();
        // Algorithm does minimisation, so must adjust for this.
        Vector4::<f32>::new(nov as f32, soon, changes as f32, -(length as f32))
    }
}

impl<'a> State for Gene<'a> {}

pub type ProjIndividual<'a> = Individual<Gene<'a>, OnlyFitnessesStats<U4>>;
