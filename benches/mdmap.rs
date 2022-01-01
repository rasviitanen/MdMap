use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dashmap::DashMap;
use mdmap::MdMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

pub fn get(c: &mut Criterion) {
    let mut get = c.benchmark_group("get");

    let collection = MdMap::new();
    (1..5_000).into_par_iter().for_each(|i: usize| {
        collection.insert(black_box(i), i);
    });

    get.bench_function("MdMap", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                assert!(collection.get(black_box(i)).is_some());
            });
        })
    });
    drop(collection);

    let collection = Mutex::new(HashMap::new());
    (1..5_000).into_par_iter().for_each(|i: usize| {
        collection.lock().unwrap().insert(black_box(i), i);
    });

    get.bench_function("Mutex<HashMap>", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                assert!(collection.lock().unwrap().get(black_box(&i)).is_some());
            });
        })
    });
    drop(collection);

    let collection = DashMap::new();
    (1..5_000).into_par_iter().for_each(|i: usize| {
        collection.insert(black_box(i), i);
    });

    get.bench_function("DashMap", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                assert!(collection.get(black_box(&i)).is_some());
            });
        })
    });
    drop(collection);
}

pub fn insert(c: &mut Criterion) {
    let mut insert = c.benchmark_group("insert");

    let collection = MdMap::new();
    insert.bench_function("MdMap", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                collection.insert(black_box(i), i);
            });
        })
    });
    drop(collection);

    let collection = Mutex::new(HashMap::new());
    insert.bench_function("Mutex<HashMap>", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                collection.lock().unwrap().insert(black_box(i), i);
            });
        })
    });
    drop(collection);

    let collection = DashMap::new();
    insert.bench_function("DashMap", |b| {
        b.iter(|| {
            (1..5_000).into_par_iter().for_each(|i: usize| {
                collection.insert(black_box(i), i);
            });
        })
    });
    drop(collection);
}

criterion_group!(benches, insert, get);
criterion_main!(benches);
