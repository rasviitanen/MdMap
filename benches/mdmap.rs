#![allow(warnings)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dashmap::DashMap;
use mdmap::{FakeHashBuilder, MdMap};
use rand::distributions::Standard;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Instruction {
    Insert,
    Get,
}

pub fn small_key_space_high_contention(c: &mut Criterion) {
    let mut dist = c.benchmark_group("small_key_space_high_contention");
    let insert_percentage = 5;
    let grow_range = 0..200;
    let selection_range = 0..5;
    let n_ops = 25_000;

    let mut rng = thread_rng();

    let key_range = selection_range
        .cycle()
        .into_iter()
        .take(n_ops)
        .map(|i| {
            (i, {
                if rng.gen_ratio(insert_percentage, 100) {
                    Instruction::Insert
                } else {
                    Instruction::Get
                }
            })
        })
        .collect::<Vec<_>>();

    let collection = MdMap::new();
    for i in grow_range.clone() {
        collection.insert(i, i);
    }
    dist.bench_function("MdMap", |b| {
        b.iter(|| {
            key_range
                .par_iter()
                .copied()
                .for_each(|(i, instruction)| match instruction {
                    Instruction::Insert => {
                        collection.insert(black_box(i), i);
                    }
                    Instruction::Get => {
                        collection.get(black_box(&i));
                    }
                });
        })
    });
    drop(collection);

    let collection = DashMap::new();
    for i in grow_range.clone() {
        collection.insert(i, i);
    }
    dist.bench_function("DashMap", |b| {
        b.iter(|| {
            key_range
                .par_iter()
                .copied()
                .for_each(|(i, instruction)| match instruction {
                    Instruction::Insert => {
                        collection.insert(black_box(i), i);
                    }
                    Instruction::Get => {
                        collection.get(black_box(&i));
                    }
                });
        })
    });
    drop(collection);
}

pub fn medium_key_space_high_contention(c: &mut Criterion) {
    let mut dist = c.benchmark_group("medium_key_space_high_contention");
    let insert_percentage = 5;
    let grow_range = 0..2000;
    let selection_range = 0..5;
    let n_ops = 25_000;

    let mut rng = thread_rng();

    let key_range = selection_range
        .cycle()
        .into_iter()
        .take(n_ops)
        .map(|i| {
            (i, {
                if rng.gen_ratio(insert_percentage, 100) {
                    Instruction::Insert
                } else {
                    Instruction::Get
                }
            })
        })
        .collect::<Vec<_>>();

    let collection = MdMap::new();
    for i in grow_range.clone() {
        collection.insert(i, i);
    }
    dist.bench_function("MdMap", |b| {
        b.iter(|| {
            key_range
                .par_iter()
                .copied()
                .for_each(|(i, instruction)| match instruction {
                    Instruction::Insert => {
                        collection.insert(black_box(i), i);
                    }
                    Instruction::Get => {
                        collection.get(black_box(&i));
                    }
                });
        })
    });
    drop(collection);

    let collection = DashMap::new();
    for i in grow_range.clone() {
        collection.insert(i, i);
    }
    dist.bench_function("DashMap", |b| {
        b.iter(|| {
            key_range
                .par_iter()
                .copied()
                .for_each(|(i, instruction)| match instruction {
                    Instruction::Insert => {
                        collection.insert(black_box(i), i);
                    }
                    Instruction::Get => {
                        collection.get(black_box(&i));
                    }
                });
        })
    });
    drop(collection);
}

criterion_group!(
    benches,
    small_key_space_high_contention,
    medium_key_space_high_contention
);
criterion_main!(benches);
