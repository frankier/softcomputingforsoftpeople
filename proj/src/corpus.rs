use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Entry;

use indexmap::IndexSet;

pub type WordId = usize;

#[derive(Serialize, Deserialize, Debug)]
pub struct WordInfo {
    pub string: String,
    pub freq: u64,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash, Debug)]
pub struct SentenceId {
    pub movie_id: u32,
    pub sentence_num: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SentenceBank {
    pub sentences: BTreeMap<SentenceId, BTreeSet<WordId>>,
    pub vocab: Vec<WordInfo>,
    pub inv_vocab: BTreeMap<String, WordId>,
}

impl SentenceBank {
    pub fn new() -> SentenceBank {
        SentenceBank {
            sentences: BTreeMap::new(),
            vocab: Vec::new(),
            inv_vocab: BTreeMap::new(),
        }
    }

    pub fn add_word(&mut self, sentence_id: SentenceId, word: &str) {
        let mut word_id = 0;
        // XXX: It would seem to make sense to use the entry API here but then we have to copy the
        // string always. See: https://github.com/rust-lang/rfcs/pull/1769
        let mut found = false;
        if let Some(the_word_id) = self.inv_vocab.get(word) {
            word_id = *the_word_id;
            self.vocab.get_mut(word_id).unwrap().freq += 1;
            found = true;
        }
        if !found {
            word_id = self.vocab.len();
            self.inv_vocab.insert(word.to_owned(), word_id);
            self.vocab.push(WordInfo {
                string: word.to_owned(),
                freq: 1,
            });
        }
        match self.sentences.entry(sentence_id) {
            Entry::Vacant(entry) => {
                let mut sentence = BTreeSet::new();
                sentence.insert(word_id);
                entry.insert(sentence);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().insert(word_id);
            }
        };
    }
}

pub type SentenceOrder = IndexSet<SentenceId>;
pub type WordOrder = IndexSet<WordId>;
pub type MovieMap = BTreeMap<u32, Vec<u32>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct CorpusStats {
    pub length_sentences: SentenceOrder,
    pub rare_word_sentences: SentenceOrder,
    pub word_ranks: WordOrder,
    pub movie_sent_map: MovieMap,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CorpusDump {
    pub bank: SentenceBank,
    pub stats: CorpusStats,
}
