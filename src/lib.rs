#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]
use crossbeam_epoch::{self as epoch, Atomic, Guard, Shared};
use std::mem::{self, MaybeUninit};
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

#[inline]
fn set_adpinv(p: usize) -> usize {
    p | 0x1
}
#[inline]
fn clr_adpinv(p: usize) -> usize {
    p & !0x1
}
#[inline]
fn is_adpinv(p: usize) -> bool {
    p & 0x1 != 0
}

#[inline]
fn set_delinv(p: usize) -> usize {
    p | 0x2
}
#[inline]
fn clr_delinv(p: usize) -> usize {
    p & !0x2
}
#[inline]
fn is_delinv(p: usize) -> bool {
    p & 0x2 != 0
}

#[inline]
fn clr_invalid(p: usize) -> usize {
    p & !0x3
}
#[inline]
fn is_invalid(p: usize) -> bool {
    p & 0x3 != 0
}

#[derive(Debug)]
pub enum LocatePredStatus {
    Found,
    LogicallyRemoved,
}

const DIMENSION: usize = 16;
const MASK: [usize; DIMENSION] = [
    0x3 << 30,
    0x3 << 28,
    0x3 << 26,
    0x3 << 24,
    0x3 << 22,
    0x3 << 20,
    0x3 << 18,
    0x3 << 16,
    0x3 << 14,
    0x3 << 12,
    0x3 << 10,
    0x3 << 8,
    0x3 << 6,
    0x3 << 4,
    0x3 << 2,
    0x3,
];

/// An entry in the adjacency list.
/// It is guaranteed to live as long as the Guard
/// that is used to get the entry.
pub struct Entry<'t, 'g, T> {
    pub node: &'g MdNode<T>,
    _parent: &'t Inner<T>,
    _guard: &'g Guard,
}

impl<'t, 'g, T> Entry<'t, 'g, T> {
    fn coord(&self) -> [usize; DIMENSION] {
        self.node.coord
    }
}

impl<'t, 'g, T> std::ops::Deref for Entry<'t, 'g, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.node.val.assume_init_ref() }
    }
}

pub struct Iter<'t, 'g, T> {
    parent: &'t Inner<T>,
    guard: &'g Guard,
    stack: Vec<&'t Atomic<MdNode<T>>>,
    current: Option<&'t MdNode<T>>,
    returned_prematurely: bool,
}

impl<'t: 'g, 'g, T: 't> Iterator for Iter<'t, 'g, T> {
    type Item = Entry<'t, 'g, T>;

    fn next(&mut self) -> Option<Entry<'t, 'g, T>> {
        unsafe {
            let guard = &*(self.guard as *const _);

            if self.returned_prematurely {
                self.returned_prematurely = false;
                for d in 0..DIMENSION {
                    let child = &self.current.unwrap().children[d];
                    if !child.load(SeqCst, guard).is_null() {
                        self.stack.push(child);
                    }
                }
            }

            while let Some(node) = self.stack.pop().map(|n| n.load(SeqCst, guard)) {
                if node.is_null() || is_delinv(node.tag()) {
                    continue;
                }

                let node = node.as_ref().unwrap();
                self.current = Some(node);

                // Skip the root node
                if node.coord != [0; DIMENSION] {
                    self.returned_prematurely = true;
                    return Some(Entry {
                        node,
                        _parent: self.parent,
                        _guard: self.guard,
                    });
                }

                for d in 0..DIMENSION {
                    let child = &node.children[d];
                    if !child.load(SeqCst, guard).is_null() {
                        self.stack.push(child);
                    }
                }
            }

            None
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct MdDesc<T> {
    dim: usize,
    pred_dim: usize,
    curr: Atomic<MdNode<T>>,
}

// I don't think we should activate this,
// curr should be freed elsewhere
// impl<T: std::fmt::Debug> std::ops::Drop for MdDesc<T> {
//     fn drop(&mut self) {
//         unsafe {
//             if !self.curr.load(Relaxed, epoch::unprotected()).is_null() {
//                 drop(mem::replace(&mut self.curr, Atomic::null()).into_owned());
//             }
//         }
//     }
// }

/// A node in the `MDList`
/// Marked `repr(C)` to improve cache locality
#[repr(C)]
pub struct MdNode<T> {
    coord: [usize; DIMENSION],
    val: MaybeUninit<T>,
    pending: Atomic<MdDesc<T>>,
    children: [Atomic<Self>; DIMENSION],
}

impl<T> Default for MdNode<T> {
    fn default() -> Self {
        Self::new_uninit(0)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for MdNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);

            f.debug_struct("MdNode")
                .field(
                    "coord",
                    &self
                        .coord
                        .iter()
                        .map(ToString::to_string)
                        .collect::<String>(),
                )
                .field("val", self.val.assume_init_ref())
                .field("pending", &self.pending.load(Relaxed, guard).as_ref())
                .field(
                    "children",
                    &self
                        .children
                        .iter()
                        .filter_map(|x| x.load(Relaxed, guard).as_ref())
                        .collect::<Vec<_>>(),
                )
                .finish()
        }
    }
}

impl<T> std::ops::Drop for MdNode<T> {
    fn drop(&mut self) {
        unsafe {
            if self.coord != [0; 16] {
                std::ptr::drop_in_place(self.val.as_mut_ptr());
            }
            if !self.pending.load(Relaxed, epoch::unprotected()).is_null() {
                drop(mem::replace(&mut self.pending, Atomic::null()).into_owned());
            }

            let children = mem::take(&mut self.children);
            for child in children {
                let child_ref = child.load(Relaxed, epoch::unprotected());
                if !child_ref.is_null() && !is_invalid(child_ref.tag()) {
                    drop(child.into_owned());
                }
            }
        }
    }
}

impl<T> MdNode<T> {
    pub fn new(key: usize, val: T) -> Self {
        Self {
            val: MaybeUninit::new(val),
            coord: Inner::<T>::key_to_coord(key),
            pending: Atomic::null(),
            children: Default::default(),
        }
    }

    pub fn new_uninit(key: usize) -> Self {
        Self {
            val: MaybeUninit::uninit(),
            coord: Inner::<T>::key_to_coord(key),
            pending: Atomic::null(),
            children: Default::default(),
        }
    }

    pub fn with_coord(coord: [usize; DIMENSION], val: T) -> Self {
        Self {
            val: MaybeUninit::new(val),
            coord,
            pending: Atomic::null(),
            children: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct Inner<T> {
    head: Atomic<MdNode<T>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Inner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &epoch::pin();

            f.debug_struct("MdList")
                .field("head", &self.head.load(SeqCst, guard).as_ref().unwrap())
                .finish()
        }
    }
}

impl<T> std::ops::Drop for Inner<T> {
    fn drop(&mut self) {
        unsafe {
            let head = self.head.load(Relaxed, epoch::unprotected()).deref_mut();

            if !head.pending.load(Relaxed, epoch::unprotected()).is_null() {
                drop(mem::replace(&mut head.pending, Atomic::null()).into_owned());
            }

            let children = mem::take(&mut head.children);

            drop(mem::replace(&mut self.head, Atomic::null()).into_owned());

            for child in children {
                let child_ref = child.load(Relaxed, epoch::unprotected());
                if !child_ref.is_null() {
                    drop(child.into_owned());
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum InsertStatus {
    Inserted,
    AlreadyExists,
    Blocked,
}

impl<'g, T> Inner<T> {
    pub fn new() -> Self {
        Self {
            head: Atomic::new(MdNode::new_uninit(0)),
        }
    }

    pub fn iter<'t>(&'t self, guard: &'g Guard) -> Iter<'t, 'g, T> {
        Iter {
            parent: self,
            stack: vec![&self.head],
            guard,
            current: None,
            returned_prematurely: false,
        }
    }

    pub fn head(&self) -> &Atomic<MdNode<T>> {
        &self.head
    }

    pub unsafe fn pop_front(&self, guard: &Guard) -> Option<T> {
        let mut stack = vec![&self.head];

        while let Some(next) = stack.pop() {
            let mut next = next.load(SeqCst, guard);
            if let Some(next_ref) = next.as_ref() {
                for d in 0..DIMENSION {
                    let child = &next_ref.children[d];
                    let mut loaded_child = child.load(SeqCst, guard);
                    if !loaded_child.is_null() && !is_delinv(loaded_child.tag()) {
                        return Self::remove(&mut next, &mut loaded_child, d, DIMENSION, guard);
                    }
                    stack.push(child);
                }
            }
        }

        None
    }
    pub unsafe fn get(
        &self,
        key: usize,
        guard: &Guard,
    ) -> Result<Entry<'_, 'g, T>, impl std::fmt::Debug> {
        // Rebind lifetime to self
        let guard = &*(guard as *const _);

        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let curr = &mut self.head.load(SeqCst, guard);
        let mut dim = 0;
        let mut pred_dim = 0;
        if let LocatePredStatus::Found =
            Self::locate_pred(&coord, pred, curr, &mut dim, &mut pred_dim, guard)
        {
            if dim == DIMENSION {
                if let Some(curr_ref) = curr.as_ref() {
                    Ok(Entry {
                        node: curr_ref,
                        _parent: self,
                        _guard: guard,
                    })
                } else {
                    Err("Node was found, but it was NULL")
                }
            } else {
                Err("Node not found")
            }
        } else {
            Err("Node was found, but was logically removed")
        }
    }

    pub unsafe fn insert<'a>(
        &self,
        new_node: &Atomic<MdNode<T>>,
        pred: &mut Shared<'a, MdNode<T>>,
        curr: &mut Shared<'a, MdNode<T>>,
        dim: &mut usize,
        pred_dim: &mut usize,
        guard: &'a Guard,
    ) -> InsertStatus {
        let pred_ref = pred.as_ref().unwrap(); // Safe unwrap
        let pred_child_atomic = &pred_ref.children[*pred_dim];
        let pred_child = pred_child_atomic.load(SeqCst, guard);

        if *dim == DIMENSION && !is_delinv(pred_child.tag()) {
            return InsertStatus::AlreadyExists;
        }

        let expected = if is_delinv(pred_child.tag()) {
            if *dim == DIMENSION - 1 {
                *dim = DIMENSION;
            }
            curr.with_tag(set_delinv(curr.tag()))
        } else {
            *curr
        };

        if pred_child == expected {
            let mut new_node = new_node.load(Relaxed, epoch::unprotected());
            let new_node_val = new_node.deref_mut();
            let desc = Self::fill_new_node(new_node_val, expected, dim, pred_dim, guard);

            if pred_ref.children[*pred_dim]
                .compare_and_set(expected, new_node, SeqCst, guard)
                .is_ok()
            {
                let desc = desc.load(SeqCst, guard);
                if !desc.is_null() {
                    if let Some(curr_ref) = curr.as_ref() {
                        let pending = curr_ref.pending.load(SeqCst, guard);
                        if !pending.is_null() {
                            Self::finish_inserting(curr_ref, pending, guard);
                        }
                    }

                    Self::finish_inserting(&*new_node_val, desc, guard);
                }
                return InsertStatus::Inserted;
            }
            // drop(desc.into_owned());
        }

        if is_adpinv(pred_child.tag()) {
            *pred = Shared::null();
            *curr = self.head.load(SeqCst, guard);
            *dim = 0;
            *pred_dim = 0;
        } else if pred_child.with_tag(0) != *curr {
            *curr = *pred;
            *dim = *pred_dim;
        }

        if let Some(new_node_ref) = new_node.load(SeqCst, guard).as_ref() {
            if !new_node_ref.pending.load(SeqCst, guard).is_null() {
                new_node_ref.pending.store(Shared::null(), SeqCst);
            }
        }

        InsertStatus::Blocked
    }

    pub unsafe fn remove<'t>(
        pred: &mut Shared<'t, MdNode<T>>,
        curr: &mut Shared<'t, MdNode<T>>,
        pred_dim: usize,
        dim: usize,
        guard: &Guard,
    ) -> Option<T> {
        if dim == DIMENSION {
            let pred_child = &pred.as_ref().unwrap().children[pred_dim];
            if pred_child.load(SeqCst, guard) == *curr
                && pred_child
                    .compare_and_set(*curr, curr.with_tag(set_delinv(curr.tag())), SeqCst, guard)
                    .is_ok()
            {
                return Some(
                    std::mem::replace(&mut curr.deref_mut().val, MaybeUninit::uninit())
                        .assume_init(),
                );
            }
        }

        None
    }

    pub unsafe fn contains(&self, key: usize, guard: &'g Guard) -> bool {
        // Rebind lifetime to self
        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let curr = &mut self.head.load(SeqCst, guard);
        let mut dim = 0;
        let mut pred_dim = 0;

        Self::locate_pred(&coord, pred, curr, &mut dim, &mut pred_dim, guard);

        dim == DIMENSION
    }

    /// Computes the 16th root of a given key
    #[inline]
    fn key_to_coord(key: usize) -> [usize; DIMENSION] {
        // let mut coords = [0; DIMENSION];
        // for i in 0..DIMENSION {
        //     coords[i] = (key & MASK[i]) >> (30 - (i << 1));
        // }
        // coords

        // The above code is 83 assebly instructions, this is 63.
        // We mainly abvoid movups instructions
        // Does it improve performance signficantly? ¯\_(ツ)_/¯, probably not
        [
            (key & MASK[0]) >> 30,
            (key & MASK[1]) >> (30 - (1 << 1)),
            (key & MASK[2]) >> (30 - (2 << 1)),
            (key & MASK[3]) >> (30 - (3 << 1)),
            (key & MASK[4]) >> (30 - (4 << 1)),
            (key & MASK[5]) >> (30 - (5 << 1)),
            (key & MASK[6]) >> (30 - (6 << 1)),
            (key & MASK[7]) >> (30 - (7 << 1)),
            (key & MASK[8]) >> (30 - (8 << 1)),
            (key & MASK[9]) >> (30 - (9 << 1)),
            (key & MASK[10]) >> (30 - (10 << 1)),
            (key & MASK[11]) >> (30 - (11 << 1)),
            (key & MASK[12]) >> (30 - (12 << 1)),
            (key & MASK[13]) >> (30 - (13 << 1)),
            (key & MASK[14]) >> (30 - (14 << 1)),
            (key & MASK[15]),
        ]
    }

    #[inline]
    unsafe fn locate_pred<'t>(
        coord: &[usize; DIMENSION],
        pred: &mut Shared<'t, MdNode<T>>,
        curr: &mut Shared<'t, MdNode<T>>,
        dim: &mut usize,
        pred_dim: &mut usize,
        guard: &'t Guard,
    ) -> LocatePredStatus {
        let mut status = LocatePredStatus::Found;
        // Locate the proper position to insert
        // tranverses list from low dim to high dim
        while *dim < DIMENSION {
            // Locate predecessor and successor
            while let Some(curr_ref) = curr.as_ref() {
                if coord[*dim] > curr_ref.coord[*dim] {
                    *pred_dim = *dim;
                    *pred = *curr;

                    let pending = curr_ref.pending.load(SeqCst, guard);
                    if let Some(pending_ref) = pending.as_ref() {
                        if *dim >= pending_ref.pred_dim && *dim <= pending_ref.dim {
                            Self::finish_inserting(curr_ref, pending, guard);
                        }
                    }

                    let child = curr_ref.children[*dim].load(SeqCst, guard);
                    if is_delinv(child.tag()) {
                        status = LocatePredStatus::LogicallyRemoved;
                    };
                    *curr = child.with_tag(clr_invalid(child.tag()));
                } else {
                    break;
                }
            }

            // No successor has greater coord at this dimension
            // The position after pred is the insertion position

            match curr.as_ref() {
                None => break,
                Some(curr) if coord[*dim] < curr.coord[*dim] => {
                    break;
                }
                _ => {
                    // continue to search in the next dimension
                    // if coord[dim] of new_node overlaps with that of curr node
                    // dim only increases if two coords are exactly the same
                    *dim += 1;
                }
            }
        }

        status
    }

    #[inline]
    fn fill_new_node<'a>(
        new_node: &mut MdNode<T>,
        curr: Shared<'a, MdNode<T>>,
        dim: &mut usize,
        pred_dim: &mut usize,
        guard: &Guard,
    ) -> Atomic<MdDesc<T>> {
        let desc = if *pred_dim != *dim {
            // descriptor to instruct other insertion task to help migrate the children
            let curr_untagged = Atomic::null();
            curr_untagged.store(curr.with_tag(clr_delinv(curr.tag())), Relaxed);
            Atomic::new(MdDesc {
                curr: curr_untagged,
                pred_dim: *pred_dim,
                dim: *dim,
            })
        } else {
            Atomic::null()
        };

        for i in 0..*pred_dim {
            new_node.children[i].store(
                // Shared::null().with_tag(0x1),
                new_node.children[i].load(SeqCst, guard).with_tag(0x1),
                SeqCst,
            );
        }

        if *dim < DIMENSION {
            // unsafe {
            //     if !new_node.children[*dim].load(SeqCst, guard).is_null() {
            //         mem::replace(&mut new_node.children[*dim], Atomic::null()).into_owned();
            //     }
            // }
            new_node.children[*dim].store(curr, SeqCst);
        }

        new_node.pending.store(desc.load(SeqCst, guard), SeqCst);

        desc
    }

    #[inline]
    pub unsafe fn finish_inserting(n: &MdNode<T>, desc: Shared<'_, MdDesc<T>>, guard: &Guard) {
        let desc_ref = desc.as_ref().unwrap(); // Safe unwrap
        let pred_dim = desc_ref.pred_dim;
        let dim = desc_ref.dim;
        let curr = &desc_ref.curr;

        for i in pred_dim..dim {
            let child = &curr.load(SeqCst, guard).as_ref().unwrap().children[i];
            let child = child.fetch_or(0x1, SeqCst, guard);
            let child = child.with_tag(clr_adpinv(child.tag()));

            if !child.is_null() {
                if n.children[i].load(Relaxed, guard).is_null() {
                    let _ = n.children[i].compare_and_set(Shared::null(), child, SeqCst, guard);
                }
            } else {
            }
        }

        if n.pending.load(SeqCst, guard) == desc {
            if let Ok(p) = n
                .pending
                .compare_and_set(desc, Shared::null(), SeqCst, guard)
            {
                if !p.is_null() {
                    drop(p.into_owned());
                }
            }
        }
    }
}

#[derive(Default, Debug)]
pub struct MdMap<T> {
    list: Inner<T>,
}

impl<T> MdMap<T> {
    pub fn new() -> Self {
        Self { list: Inner::new() }
    }

    #[inline]
    pub fn insert(&self, key: usize, value: T) {
        let coord = Inner::<T>::key_to_coord(key);
        let new_node = Atomic::new(MdNode::with_coord(coord, value));
        let dim = &mut 0;
        let pred_dim = &mut 0;

        unsafe {
            let guard = &epoch::pin();

            let md_current = &mut self.list.head().load(Relaxed, guard);
            let md_pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();

            let mut first = true;
            loop {
                Inner::<T>::locate_pred(&coord, md_pred, md_current, dim, pred_dim, guard);
                if !first {
                    if let Some(current) = md_current.as_ref() {
                        if current.coord == coord {
                            break;
                        }
                    }
                } else {
                    first = false;
                }

                match self
                    .list
                    .insert(&new_node, md_pred, md_current, dim, pred_dim, guard)
                {
                    InsertStatus::Inserted => break,
                    InsertStatus::AlreadyExists => {
                        drop(new_node.into_owned());
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn iter(&'_ self) -> Iter<'_, '_, T> {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.iter(guard)
        }
    }

    pub fn pop_front(&self) -> Option<T> {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.pop_front(guard)
        }
    }

    pub fn get(&self, key: usize) -> Option<Entry<'_, '_, T>> {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.get(key, guard).ok()
        }
    }

    pub fn contains(&self, key: usize) -> bool {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.contains(key, guard)
        }
    }

    pub fn remove(&self, key: usize) -> Option<T> {
        unsafe {
            let dim = &mut 0;
            let pred_dim = &mut 0;
            let guard = &*(&epoch::pin() as *const _);

            let md_current = &mut self.list.head().load(SeqCst, guard);
            let md_pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();

            let coord = Inner::<T>::key_to_coord(key);
            Inner::<T>::locate_pred(&coord, md_pred, md_current, dim, pred_dim, guard);
            Inner::<T>::remove(md_pred, md_current, *pred_dim, *dim, guard)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let list = MdMap::new();
        list.insert(1, 10);
        assert_eq!(*list.get(1).unwrap(), 10);
    }

    #[test]
    fn test_remove() {
        let list = MdMap::new();
        list.insert(1, 10);
        assert_eq!(*list.get(1).unwrap(), 10);
        assert_eq!(list.remove(1), Some(10));
        assert!(list.get(1).is_none());
    }

    // not supported for now
    // #[test]
    // fn test_update() {
    //     let list = MdList::new();
    //     list.insert(1, 10);
    //     list.insert(1, 20);

    //     assert_eq!(*list.get(1).unwrap(), 20);
    // }

    #[test]
    fn test_multiple() {
        let list = MdMap::new();
        for i in 1..100 {
            list.insert(i, i);
        }

        for i in 1..100 {
            assert_eq!(*list.get(i).unwrap(), i);
        }

        for i in 1..100 {
            list.remove(i);
        }

        for i in 1..100 {
            assert!(list.get(i).is_none());
        }
    }

    #[test]
    fn test_is_sorted() {
        let list = MdMap::new();
        for i in [1, 3, 4, 7, 2, 5, 6] {
            list.insert(i, i * 10);
        }

        assert_eq!(
            list.iter().map(|x| *x).collect::<Vec<_>>(),
            [10, 20, 30, 40, 50, 60, 70]
        );
    }

    #[test]
    fn test_parallel() {
        use rayon::prelude::*;
        let mdlist = MdMap::new();
        let md_ref = &mdlist;
        for _ in 0..1000 {
            (1..300).into_par_iter().for_each(|i| {
                md_ref.insert(i, i);
            });

            (1..300).into_iter().for_each(|i| {
                assert!(md_ref.contains(i));
            });
        }
        drop(mdlist);
    }

    #[test]
    fn test_pop_front() {
        let list = MdMap::new();
        list.insert(1, 10);
        list.insert(3, 30);
        list.insert(2, 20);
        assert_eq!(list.pop_front(), Some(10));
        assert_eq!(list.pop_front(), Some(20));
        assert_eq!(list.pop_front(), Some(30));
    }
}
