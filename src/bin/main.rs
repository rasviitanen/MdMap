#![allow(warnings)]
use dashmap::DashMap;
use dhat::{Dhat, DhatAlloc};
use mdmap::{FakeHashBuilder, MdMap};
use rayon::prelude::*;

// #[global_allocator]
// static ALLOCATOR: DhatAlloc = DhatAlloc;

fn main() {
    // let _dhat = Dhat::start_heap_profiling();
    for t in 1..5 {
        let range = 1000 * 10_usize.pow(t);
        // let map = DashMap::<usize, usize>::new();
        let map = MdMap::<usize, usize, _>::with_hasher(FakeHashBuilder::default());

        let now = std::time::Instant::now();
        (1..range).into_iter().for_each(|i: usize| {
            map.insert(i, i);
        });
        let time = now.elapsed();
        println!(
            "Insert: {:>8} t={}ms ({}ns/item)",
            range,
            time.as_millis(),
            time.as_nanos() / range as u128
        );

        let now = std::time::Instant::now();
        (1..range).into_iter().for_each(|i: usize| {
            assert!(map.get(&i).is_some());
        });
        let time = now.elapsed();
        println!(
            "Get: {:>11} t={}us ({}ns/item)",
            range,
            time.as_micros(),
            time.as_nanos() / range as u128
        );
        std::mem::forget(map);

        // dbg!(&map);
        // dbg!(&map);
    }

    // (1..30).into_par_iter().for_each(|i| {
    // for _ in 0..100 {
    //     let mut tasks = Vec::new();
    //     for _ in 0..5 {
    //         let list = list.clone();
    //         let jh = std::thread::spawn(move || {
    //             list.insert(1, 1);
    //         });
    //         tasks.push(jh);
    //     }
    //     for task in tasks {
    //         let _ = task.join();
    //     }
    // }
    // });

    // drop(list);
}
