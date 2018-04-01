use std::collections::{BTreeSet, HashSet};
use std::ops::Range;
use rand::distributions::{Range as RandRange, Sample};
use rand::Rng;
use itertools::Itertools;

use sc::utils::indexmap::{IndexSetExt, sample as sample_indexset};
use corpus::{SentenceOrder, SentenceId};
use individual::Gene;

fn it_count_between<I: Iterator>(iter: I, range: Range<usize>) -> Result<usize, usize> {
    let mut count = 0;
    for _ in iter {
        count += 1;
        if count >= range.end {
            return Err(count);
        }
    }
    if count < range.start {
        return Err(count);
    }
    Ok(count)
}

/*
fn try_extend_nov<I>(gene: &mut Gene, from: I, max_novelty: usize) -> bool
    where I: Iterator<Item=SentenceId>
{
    let mut borrowed_vocab = gene.vocab.borrow_mut();
    let vocab = borrowed_vocab.as_mut().unwrap();
    for sent_id in from {
        let words = &gene.dump.bank.sentences[&sent_id];
        {
            let new_words = words.difference(vocab);
            if !it_count_between(new_words, 1..(max_novelty + 1)).is_ok() {
                continue;
            }
        }
        gene.solution.insert(sent_id);
        vocab.extend(words.iter());
        return true;
    }
    false
}

fn greedy_rand_extend_nov_base<SF, I>(
        mut sample: SF, gene: &mut Gene, max_novelty: usize, max_tries: usize) -> bool
    where SF: FnMut(usize) -> I,
          I: Iterator<Item=SentenceId> + Sized
{
    const CHUNK_SIZE: usize = 8;
    let mut remaining_tries = max_tries;
    while remaining_tries > CHUNK_SIZE {
        let sampled_sentences = sample(CHUNK_SIZE);
        if try_extend_nov(gene, sampled_sentences, max_novelty) {
            return true
        }
        remaining_tries -= CHUNK_SIZE;
    }
    let sampled_sentences = sample(remaining_tries);
    if try_extend_nov(gene, sampled_sentences, max_novelty) {
        return true
    }
    false
}
*/

fn greedy_rand_extend_nov_base<R, G>(
        rng: &mut R, get_sentence: G, mut range: RandRange<usize>, gene: &mut Gene,
        max_novelty: usize, backoff_novelty: usize, max_tries: usize)
    where R: Rng,
          G: Fn(usize) -> SentenceId,
{
    assert!(max_tries > 0);
    assert!(max_novelty > 0);
    assert!(backoff_novelty >= max_novelty);

    let mut borrowed_vocab = gene.vocab.borrow_mut();
    let vocab = borrowed_vocab.as_mut().unwrap();
    let sentences = &gene.dump.bank.sentences;

    let mut remaining_tries = max_tries;
    let mut seen = HashSet::new();
    let mut min_novelty = backoff_novelty + 1;
    let mut best_solution = None;
    while remaining_tries > 0 {
        remaining_tries -= 1;

        // Pick a new random sentence
        let rand_idx = range.sample(rng);
        if seen.contains(&rand_idx) {
            continue;
        }
        seen.insert(rand_idx);
        let sent_id = get_sentence(rand_idx);
        let words = &sentences[&sent_id];

        // Check how many new words
        let count = {
            let new_words = words.difference(vocab);
            it_count_between(new_words, 1..(backoff_novelty + 1))
        };

        // Reject if over backoff
        if !count.is_ok() {
            // Ensure we have some bad `best solution' always
            if best_solution.is_none() {
                best_solution = Some(sent_id);
            }
            continue;
        }
        let count_num = count.unwrap();

        // Reject and save if over accept
        if count_num > max_novelty {
            if count_num < min_novelty {
                min_novelty = count_num;
                best_solution = Some(sent_id);
            }
            continue;
        }

        // Otherwise accept and terminate
        gene.solution.insert(sent_id);
        vocab.extend(words.iter());
        return;
    }
    let sent_id = best_solution.unwrap();
    gene.solution.insert(sent_id);
    let words = &sentences[&sent_id];
    vocab.extend(words.iter());
}

pub fn greedy_rand_extend_nov_all<R: Rng>(
        rng: &mut R, gene: &mut Gene,
        max_novelty: usize, backoff_novelty: usize, max_tries: usize)
{
    let sentences = &gene.dump.stats.length_sentences;
    greedy_rand_extend_nov_base(
        rng, |idx| *sentences.get_index(idx).unwrap(), RandRange::new(0, sentences.len()), gene,
        max_novelty, backoff_novelty, max_tries)
}

pub fn greedy_rand_extend_nov_movie<R: Rng>(
        rng: &mut R, gene: &mut Gene,
        max_novelty: usize, backoff_novelty: usize, max_tries: usize)
{
    let movie_id;
    if gene.solution.is_empty() {
        movie_id = sample_indexset(rng, &gene.dump.stats.length_sentences, 1).next().unwrap().movie_id
    } else {
        movie_id = gene.solution.get_index(gene.solution.len() - 1).unwrap().movie_id;
    }
    let sentences = gene.dump.stats.movie_sent_map[&movie_id].as_slice();
    greedy_rand_extend_nov_base(
        rng, |idx| {
            let sentence_num = sentences[idx];
            SentenceId { movie_id, sentence_num }
        }, RandRange::new(0, sentences.len()), gene,
        max_novelty, backoff_novelty, max_tries);
}

fn greedy_extend_from_order(gene: &mut Gene, order: &SentenceOrder) {
    for sent_id in order.iter() {
        if !gene.solution.contains(sent_id) {
            gene.solution.insert(sent_id.clone());
            return
        }
    }
}

pub fn greedy_extend_rare_word(gene: &mut Gene) {
    greedy_extend_from_order(gene, &gene.dump.stats.rare_word_sentences)
}

pub fn greedy_extend_length(gene: &mut Gene) {
    greedy_extend_from_order(gene, &gene.dump.stats.length_sentences)
}

pub fn local_permutation_mutate<R: Rng>(rng: &mut R, gene: &mut Gene) {
    const SWAP_CHANCE: f32 = 0.05;
    let solution_orig = gene.solution.clone();
    gene.solution.sort_by(|sent1, sent2| {
        let ordering = solution_orig.index_of(sent1).unwrap().cmp(&solution_orig.index_of(sent1).unwrap());
        let chance: f32 = rng.gen();
        if chance < SWAP_CHANCE {
            ordering.reverse()
        } else {
            ordering
        }
    })
}

pub fn delete_mutate<R: Rng>(rng: &mut R, gene: &mut Gene) {
    const DELETE_CHANCE: f32 = 0.05;
    gene.solution.retain(|sentence| {
        let chance: f32 = rng.gen();
        if chance < DELETE_CHANCE {
            false
        } else {
            true
        }
    });
}

pub fn repair(gene: &mut Gene) {
    // XXX: Ideally we would like to try moving the sentence that doesn't introduce any words
    // towards the beginning of the sentence {while it doesn't introduce any more
    // inadmissibilities} && {until its own inadmissibility is repaired} -- but IndexSet needs some
    // more methods first. (A common `intros' caching approach would help too.)
    let mut vocab = BTreeSet::new();
    let sentences = &gene.dump.bank.sentences;
    gene.solution.retain(|sentence| {
        let sent_words = sentences.get(&sentence).unwrap();
        let new_words = sent_words.difference(&vocab).cloned().collect_vec();
        if new_words.is_empty() {
            false
        } else {
            vocab.extend(new_words.iter());
            true
        }
    });
}
