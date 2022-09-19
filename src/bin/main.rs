use mdcollections::mdlist::MdList;

fn main() {
    let list = MdList::<usize, usize, 16>::new();
    list.insert(1, 1);
    list.insert(2, 2);
    list.insert(3, 3);
    drop(list);
}
