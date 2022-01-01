use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dashmap::DashMap;
use mdmap::{FakeHashBuilder, MdMap};
use rand::distributions::Standard;
use rand::prelude::*;
use rayon::prelude::*;

pub fn get(c: &mut Criterion) {
    let mut get = c.benchmark_group("get");
    let mut rng = thread_rng();
    let key_range = (1..4_000)
        .into_iter()
        .map(|_| rng.sample::<usize, _>(Standard))
        .collect::<Vec<_>>();

    let collection = MdMap::new();
    key_range.par_iter().copied().for_each(|i: usize| {
        collection.insert(black_box(i), i);
    });

    get.bench_function("MdMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|i: usize| {
                assert!(collection.get(black_box(&i)).is_some());
            });
        })
    });
    drop(collection);

    // let collection = Mutex::new(HashMap::new());
    // key_range.par_iter().copied().for_each(|i: usize| {
    //     collection.lock().unwrap().insert(black_box(i), i);
    // });

    // get.bench_function("Mutex<HashMap>", |b| {
    //     b.iter(|| {
    //         key_range.par_iter().copied().for_each(|i: usize| {
    //             assert!(collection.lock().unwrap().get(black_box(&i)).is_some());
    //         });
    //     })
    // });
    // drop(collection);

    let collection = DashMap::new();
    key_range.par_iter().copied().for_each(|i: usize| {
        collection.insert(black_box(i), i);
    });

    get.bench_function("DashMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|i: usize| {
                assert!(collection.get(black_box(&i)).is_some());
            });
        })
    });
    drop(collection);
}

pub fn insert(c: &mut Criterion) {
    let mut insert = c.benchmark_group("insert");

    let mut rng = thread_rng();
    let key_range = (1..10_000)
        .into_iter()
        .map(|_| rng.sample::<usize, _>(Standard))
        .collect::<Vec<_>>();

    let collection = MdMap::new();
    insert.bench_function("MdMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|i: usize| {
                collection.insert(black_box(i), i);
            });
        })
    });
    drop(collection);

    // let collection = Mutex::new(HashMap::new());
    // insert.bench_function("Mutex<HashMap>", |b| {
    //     b.iter(|| {
    //         key_range.par_iter().copied().for_each(|i: usize| {
    //             collection.lock().unwrap().insert(black_box(i), i);
    //         });
    //     })
    // });
    // drop(collection);

    let collection = DashMap::new();
    insert.bench_function("DashMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|i: usize| {
                collection.insert(black_box(i), i);
            });
        })
    });
    drop(collection);
}

pub fn dist(c: &mut Criterion) {
    let mut dist = c.benchmark_group("dist");
    let insert_percentage = 20;
    #[derive(Clone, Copy)]
    enum Instruction {
        Insert,
        Get,
    }

    let mut rng = thread_rng();

    let key_range = (1..1_000)
        .into_iter()
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
    dist.bench_function("MdMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|(i, instruction)| {
                for i in i..i + 25 {
                    match instruction {
                        Instruction::Insert => {
                            collection.insert(black_box(i), i);
                        }
                        Instruction::Get => {
                            collection.get(black_box(&i));
                        }
                    }
                }
            });
        })
    });
    drop(collection);

    let collection = DashMap::new();
    dist.bench_function("DashMap", |b| {
        b.iter(|| {
            key_range.par_iter().copied().for_each(|(i, instruction)| {
                for i in i..i + 25 {
                    match instruction {
                        Instruction::Insert => {
                            collection.insert(black_box(i), i);
                        }
                        Instruction::Get => {
                            collection.get(black_box(&i));
                        }
                    }
                }
            });
        })
    });
    drop(collection);
}

criterion_group!(benches, dist);
criterion_main!(benches);
