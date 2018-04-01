extern crate serde;
extern crate opus_parse;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;
extern crate pbr;
extern crate indexmap;
extern crate bincode;
extern crate softcomputingforsoftpeople_proj as lib;

use std::fs::File;
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::io::BufReader;
use std::collections::BTreeMap;
use structopt::StructOpt;
use opus_parse::opensubtitles::walk;
use opus_parse::*;
use std::thread;
use std::io::*;
use pbr::ProgressBar;
use bincode::serialize_into;
use lib::corpus::*;
use lib::print_fitnesses;

enum Message {
    MovieBegin(u64),
    SentBegin(u64),
    StreamEnd
}

#[derive(StructOpt, Debug)]
#[structopt(name = "proj-parse")]
struct Opt {
    #[structopt(name = "OPENSUBTITLES_FI", help = "Path to OpenSubtitles 2018 dump")]
    input_dir: String,

    #[structopt(name = "OUTPUT", help = "Path to output processed corpus dump")]
    output: String,
}

fn main() {
    let opt = Opt::from_args();
    let finnpos = Command::new("ftb-label")
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .spawn()
      .expect("ftb-label must be in the path");

    let (tx, rx) = channel();
    let stdout = finnpos.stdout.unwrap();
    let mut stdin = finnpos.stdin.unwrap();
    let count = walk(&opt.input_dir).count();
    println!("Building sentence bank");
    let mut pb = ProgressBar::new(count as u64);
    pb.format("╢▌▌░╟");
    let joinable = thread::spawn(move || {
        let mut bank = SentenceBank::new();
        let mut line = String::new();
        let mut buf_reader = BufReader::with_capacity(64 * 1024, stdout);
        let mut movie_id = 0;
        loop {
            /*
            buf_reader.read_until(b'\n', &mut line).unwrap();
            let line_str = str::from_utf8(line.as_slice()).unwrap();
            println!("Got {}", line_str);
            */
            match rx.recv().unwrap() {
                Message::MovieBegin(mid) => {
                    movie_id = mid;
                    pb.inc();
                }
                Message::SentBegin(sid) => {
                    loop {
                        line.clear();
                        buf_reader.read_line(&mut line).unwrap();
                        line.pop();
                        if line.is_empty() {
                            break;
                        } else {
                            let lemma = line.split('\t').nth(2).unwrap();
                            bank.add_word(
                                SentenceId {
                                    movie_id: movie_id as u32,
                                    sentence_num: sid as u32,
                                },
                                lemma);
                        }
                    }
                }
                Message::StreamEnd => {
                    break;
                }
            }
        }
        pb.finish_println("Done building sentence bank");
        bank
    });
    for (movie_id, subtitle_path) in walk(opt.input_dir) {
        tx.send(Message::MovieBegin(movie_id)).unwrap();
        let mut ss = OpusStream::from_path(&subtitle_path).unwrap();
        loop {
            match ss.next() {
                Ok(FlatStreamBit::StreamBit(bit)) => match bit {
                    StreamBit::SentDelim(SentDelim { id, delim_type: DelimType::Start }) => {
                        tx.send(Message::SentBegin(id)).unwrap();
                    }
                    StreamBit::SentDelim(SentDelim { id: _, delim_type: DelimType::End }) => {
                        stdin.write(b"\n").unwrap();
                    }
                    StreamBit::Word(Word { id: _, word }) => {
                        stdin.write(word.as_bytes()).unwrap();
                        stdin.write(b"\n").unwrap();
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
    tx.send(Message::StreamEnd).unwrap();
    let bank = joinable.join().unwrap();
    println!("Building stats");
    let (length_sentences, rare_word_sentences, movie_sent_map): (SentenceOrder, SentenceOrder, MovieMap) = {
        let mut movie_sent_map: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let mut length_sentences: Vec<SentenceId> = bank.sentences.iter().map(|(sentence_id, _sentence_set)| {
            *sentence_id
        }).collect();
        for SentenceId { movie_id, sentence_num } in length_sentences.iter() {
            movie_sent_map
                .entry(*movie_id)
                .or_insert(Vec::new())
                .push(*sentence_num)
        }
        let mut rare_word_sentences = length_sentences.clone();
        length_sentences.sort_unstable_by_key(|sentence_id| {
            bank.sentences.get(sentence_id).unwrap().len()
        });
        rare_word_sentences.sort_unstable_by_key(|sentence_id| {
            bank.sentences.get(sentence_id).unwrap().iter().map(|word_id| {
                bank.vocab.get(*word_id).unwrap().freq
            }).min()
        });
        (length_sentences.into_iter().collect(),
         rare_word_sentences.into_iter().collect(),
         movie_sent_map)
    };
    let word_ranks: WordOrder = {
        let mut word_ids: Vec<WordId> = (0..bank.vocab.len()).collect();
        word_ids.sort_unstable_by_key(|&word_id| {
            bank.vocab[word_id].freq
        });
        word_ids.into_iter().collect()
    };
    let stats = CorpusStats {
        length_sentences,
        rare_word_sentences,
        word_ranks,
        movie_sent_map,
    };
    println!("Done building stats");
    println!("Saving");
    let mut buffer = File::create(&opt.output).unwrap();
    let dump = CorpusDump {
        bank,
        stats,
    };
    serialize_into(&mut buffer, &dump).unwrap();
    println!("Done");
}
