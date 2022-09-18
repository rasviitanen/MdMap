#![allow(warnings)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dashmap::DashMap;
use mdcollections::mdlist::MdList;
use mdcollections::MdMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
#[derive(Clone, Copy)]
enum Instruction {
    Insert(usize, usize),
    Get(usize),
}

pub fn small_key_space(c: &mut Criterion) {
    let mut rng = thread_rng();
    let insert_percentage = 20;
    let mut num = usize::MAX / 2;
    let mut operate_on_keys = [0; 500].map(|_| {
        num = num.wrapping_mul(17).wrapping_add(255);
        num
    });
    operate_on_keys.shuffle(&mut rng);
    let grow_range = operate_on_keys.clone();
    let n_ops = 1_000;
    let mut iter = 0;
    let dists = [
        // (
        //     "exponential",
        //     operate_on_keys.map(|_| {
        //         let r = std::f32::consts::E.powf(iter as f32 * 0.72 * 0.5);
        //         iter += 1;
        //         r as usize
        //     }),
        // ),
        ("uniform", operate_on_keys.map(|_| 1)),
    ];

    for (name, weights) in dists {
        let mut group = c.benchmark_group(format!("small_key_space::{name}"));
        println!("starting benchmark...");
        println!("");
        println!("[config]");
        println!(
            "operation distribution: {}% get; {}% insert",
            100 - insert_percentage,
            insert_percentage,
        );
        println!("operationg on: {:?}", operate_on_keys);
        println!("executing: {} ops", n_ops);
        println!("key distribution: `{name}`");
        println!("key distribution summary:");
        println!("------------------------------------------------------");
        println!("|            key           |    ops     | percentage |");
        println!("======================================================");
        // let total_weight = weights.iter().sum::<usize>();
        // operate_on_keys
        //     .iter()
        //     .zip(weights.iter())
        //     .for_each(|(key, weight)| {
        //         println!(
        //             "| {:>24} | ~{:>5} ops |   {:>7.4}% |",
        //             key,
        //             ((*weight as f64 / total_weight as f64) * n_ops as f64) as usize,
        //             (*weight as f64 / total_weight as f64) * 100.0,
        //         );
        //     });
        println!("-----------------------------------------------------\n");

        let mut rng = thread_rng();
        let dist = WeightedIndex::new(weights).unwrap();

        let key_range = (0..n_ops)
            .into_iter()
            .map(|i| {
                if rng.gen_ratio(insert_percentage, 100) {
                    Instruction::Insert(operate_on_keys[dist.sample(&mut rng)], i)
                } else {
                    Instruction::Get(operate_on_keys[dist.sample(&mut rng)])
                }
            })
            .collect::<Vec<_>>();

        let collection = MdList::<usize, usize>::new();
        for i in grow_range.clone() {
            collection.insert(i, i);
        }
        group.bench_function("MdList", |b| {
            b.iter(|| {
                key_range
                    .par_iter()
                    .copied()
                    .for_each(|instruction| match instruction {
                        Instruction::Insert(k, v) => {
                            collection.insert(black_box(k), v);
                        }
                        Instruction::Get(k) => {
                            collection.get(black_box(k));
                        }
                    });
            })
        });
        drop(collection);

        let collection = DashMap::new();
        for i in grow_range.clone() {
            collection.insert(i, i);
        }
        group.bench_function("DashMap", |b| {
            b.iter(|| {
                key_range
                    .par_iter()
                    .copied()
                    .for_each(|instruction| match instruction {
                        Instruction::Insert(k, v) => {
                            collection.insert(black_box(k), v);
                        }
                        Instruction::Get(k) => {
                            collection.get(black_box(&k));
                        }
                    });
            })
        });
        drop(collection);

        let collection = std::sync::Mutex::new(std::collections::HashMap::new());
        for i in grow_range.clone() {
            collection.lock().unwrap().insert(i, i);
        }
        group.bench_function("Mutex<HashMap>", |b| {
            b.iter(|| {
                key_range
                    .par_iter()
                    .copied()
                    .for_each(|instruction| match instruction {
                        Instruction::Insert(k, v) => {
                            collection.lock().unwrap().insert(black_box(k), v);
                        }
                        Instruction::Get(k) => {
                            collection.lock().unwrap().get(black_box(&k));
                        }
                    });
            })
        });
        drop(collection);

        let collection = std::sync::RwLock::new(std::collections::HashMap::new());
        for i in grow_range.clone() {
            collection.write().unwrap().insert(i, i);
        }
        group.bench_function("RwLock<HashMap>", |b| {
            b.iter(|| {
                key_range
                    .par_iter()
                    .copied()
                    .for_each(|instruction| match instruction {
                        Instruction::Insert(k, v) => {
                            collection.write().unwrap().insert(black_box(k), v);
                        }
                        Instruction::Get(k) => {
                            collection.read().unwrap().get(black_box(&k));
                        }
                    });
            })
        });
        drop(collection);
    }
}

criterion_group!(benches, small_key_space,);
criterion_main!(benches);
