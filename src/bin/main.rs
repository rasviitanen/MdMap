use mdcollections::MdList;

fn main() {
    // let guard = crossbeam_epoch::pin();
    let list = MdList::<usize, 16, 16>::new();
    for i in 1..1_000 {
        list.insert(i, i);
    }
    // std::mem::forget(list);
}
