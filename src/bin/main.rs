use dhat::{Dhat, DhatAlloc};
use mdmap::MdMap;

#[global_allocator]
static ALLOCATOR: DhatAlloc = DhatAlloc;

fn main() {
    let _dhat = Dhat::start_heap_profiling();
    let list = std::sync::Arc::new(MdMap::<usize>::new());

    // (1..30).into_par_iter().for_each(|i| {
    for _ in 0..100 {
        let mut tasks = Vec::new();
        for _ in 0..5 {
            let list = list.clone();
            let jh = std::thread::spawn(move || {
                list.insert(1, 1);
            });
            tasks.push(jh);
        }
        for task in tasks {
            let _ = task.join();
        }
    }
    // });

    drop(list);
}
