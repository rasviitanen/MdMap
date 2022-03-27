#![allow(warnings)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dashmap::DashMap;
use mdmap::MdMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Instruction {
    Insert(usize, usize),
    Get(usize),
}

pub fn small_key_space_high_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_key_space_high_contention");
    let insert_percentage = 8;
    let grow_range = 0..1_000;
    let operate_on_keys = [100, 200, 300, 400, 500, 600, 700];
    let weights = [74, 12, 6, 3, 3, 1, 1];
    assert_eq!(weights.iter().sum::<usize>(), 100);
    let n_ops = 1_000;

    println!("starting benchmark...");
    println!("");
    println!("[config]");
    println!("map size: {} entries", grow_range.end);
    println!(
        "operation distribution: {}% insert; {}% get",
        insert_percentage,
        100 - insert_percentage
    );
    println!("operationg on: {:?}", operate_on_keys);
    operate_on_keys
        .iter()
        .zip(weights.iter())
        .for_each(|(key, weight)| {
            println!("\t{}: {}%", key, weight);
        });
    println!("executing {} ops", n_ops);
    println!("--");

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

    let collection = MdMap::new();
    for i in grow_range.clone() {
        collection.insert(i, i);
    }
    group.bench_function("MdMap", |b| {
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

criterion_group!(benches, small_key_space_high_contention,);
criterion_main!(benches);
