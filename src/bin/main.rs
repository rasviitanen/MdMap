use mdcollections::MdList;

fn main() {
    // let guard = crossbeam_epoch::pin();
    let list = std::sync::Arc::new(MdList::<(), 16, 16>::new());
    // let mut jhs = Vec::new();
    // for _ in 0..100 {
    //     let list = list.clone();
    // let jh = std::thread::spawn(move || {
    for i in 1..25 {
        list.insert(i, ());
    }
    // });
    // jhs.push(jh);
    // }

    // for jh in jhs {
    //     jh.join().unwrap();
    // }

    drop(list);
    crossbeam_epoch::pin().flush();
    // std::mem::forget(list);
}
