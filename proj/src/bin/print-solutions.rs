extern crate softcomputingforsoftpeople as sc;
extern crate softcomputingforsoftpeople_proj as lib;
extern crate nalgebra as na;
extern crate bincode;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate opus_parse;
extern crate indexmap;

use std::fs::File;
use std::collections::BTreeMap;

use structopt::StructOpt;
use bincode::deserialize_from;
use opus_parse::opensubtitles::walk;
use opus_parse::*;
use na::Vector4;

use sc::individual::multiobj::MultipleFitnessStats;
use lib::individual::{Gene, ProjIndividual};
use lib::corpus::{SentenceOrder, SentenceId, CorpusDump};

#[derive(StructOpt, Debug)]
#[structopt(name = "print-solutions")]
struct Opt {
    #[structopt(name = "OPENSUBTITLES_FI", help = "Path to OpenSubtitles 2018 dump")]
    input_dir: String,

    #[structopt(name = "SOLUTIONS", help = "Path to solutions dump")]
    solutions: String,

    #[structopt(name = "CORPUS_DUMP", help = "Path to output processed corpus dump")]
    corpus_dump: String,
}

fn main() {
    let opt = Opt::from_args();

    println!("Loading corpus");
    let buffer = File::open(&opt.corpus_dump).unwrap();
    let corpus_dump: CorpusDump = deserialize_from(buffer).unwrap();
    println!("Loaded");

    let buffer = File::open(&opt.solutions).unwrap();
    let (solutions, weight_vec): (Vec<SentenceOrder>, Vector4<f32>) = deserialize_from(buffer).unwrap();

    let mut sentences = BTreeMap::new();
    let mut all_sentences = SentenceOrder::new();
    for solution in solutions.iter() {
        all_sentences.extend(solution.iter());
    }
    let mut cur_sent = None;

    for (movie_id, subtitle_path) in walk(opt.input_dir) {
        let mut ss = OpusStream::from_path(&subtitle_path).unwrap();
        loop {
            match ss.next() {
                Ok(FlatStreamBit::StreamBit(bit)) => match bit {
                    StreamBit::SentDelim(SentDelim { id, delim_type: DelimType::Start }) => {
                        let sent_id = SentenceId {
                            movie_id: movie_id as u32,
                            sentence_num: id as u32,
                        };
                        if all_sentences.contains(&sent_id) {
                            cur_sent = Some((sent_id, vec![]))
                        }
                    }
                    StreamBit::SentDelim(SentDelim { id: _, delim_type: DelimType::End }) => {
                        if let Some((sent_id, sent)) = cur_sent {
                            sentences.insert(sent_id, sent);
                        }
                        cur_sent = None;
                    }
                    StreamBit::Word(Word { id: _, word }) => {
                        if let Some((_, ref mut sent)) = cur_sent {
                            sent.push(word);
                        }
                    }
                    _ => {}
                },
                Ok(FlatStreamBit::Meta(_meta)) => {
                    continue;
                }
                Ok(FlatStreamBit::EndStream) => {
                    break;
                }
                Err(e) => {
                    println!("\nSkipping {}: {}", subtitle_path.to_string_lossy(), e.description());
                    break;
                }
            }
        }
    }

    for (i, (solution, l_i)) in solutions.into_iter().zip(weight_vec.iter()).enumerate() {
        let individual = ProjIndividual::new(Gene::new(&corpus_dump, solution));
        println!("Solution #{} {} {}", i + 1, l_i, individual.stats.fitnesses());
        for (j, sentence) in individual.state.solution.iter().enumerate() {
            println!("{}: {}", j + 1, sentences[sentence].join(" "));
        }
    }
}
