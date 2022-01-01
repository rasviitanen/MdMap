#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

use crossbeam_epoch::{self as epoch, Atomic, Guard, Shared};
use epoch::{CompareAndSetError, Owned};
use std::borrow::{Borrow, BorrowMut};
use std::collections::hash_map::RandomState;
use std::hash::Hasher;
use std::ops::{BitXor, Deref};
use std::simd::Simd;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    hash::{BuildHasher, Hash},
    mem::{self},
};

const DIMENSION: usize = 16;
const BASIS: usize = 16;
const NULL: Atomic<()> = Atomic::null();

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

pub struct Entry<'t, 'g, T> {
    pub node: &'g MdNode<T>,
    _parent: &'t Inner<T>,
    _guard: &'g Guard,
}

impl<'t, 'g, T> Entry<'t, 'g, T> {
    fn coord(&self) -> [u8; DIMENSION] {
        self.node.coord
    }
}

impl<'t, 'g, T> std::ops::Deref for Entry<'t, 'g, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.node.val.as_ref().unwrap()
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

#[derive(Debug)]
pub struct MdDesc<T> {
    dim: usize,
    pred_dim: usize,
    curr: Atomic<MdNode<T>>,
}

pub struct MdNode<T> {
    coord: [u8; DIMENSION],
    val: Option<T>,
    pending: Atomic<MdDesc<T>>,
    children: [Atomic<Self>; DIMENSION],
}

impl<T> Default for MdNode<T> {
    fn default() -> Self {
        Self::new_uninit(0, 0)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for MdNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &&epoch::pin();

            f.debug_struct("MdNode")
                .field(
                    "coord",
                    &self
                        .coord
                        .iter()
                        .map(ToString::to_string)
                        .collect::<String>(),
                )
                .field("val", &self.val)
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
            if !self.pending.load(Relaxed, epoch::unprotected()).is_null() {
                drop(mem::replace(&mut self.pending, Atomic::null()).into_owned());
            }

            let children = mem::replace(&mut self.children, Self::NULL_CHILDREN);
            for child in children.into_iter() {
                let child_ref = child.load(Relaxed, epoch::unprotected());
                if !child_ref.is_null() && !is_invalid(child_ref.tag()) {
                    drop(child.into_owned());
                }
            }
        }
    }
}

impl<T> MdNode<T> {
    const NULL_CHILDREN: [Atomic<Self>; DIMENSION] =
        unsafe { std::mem::transmute([NULL; DIMENSION]) };
    pub fn new(key: usize, val: T, dim: usize) -> Self {
        Self {
            val: Some(val),
            coord: Inner::<T>::key_to_coord(key),
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }

    pub fn new_uninit(key: usize, dim: usize) -> Self {
        Self {
            val: None,
            coord: Inner::<T>::key_to_coord(key),
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }

    pub fn with_coord(coord: [u8; DIMENSION], val: T, dim: usize) -> Self {
        Self {
            val: Some(val),
            coord,
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }
}

#[derive(Default)]
pub struct Inner<T> {
    head: Atomic<MdNode<T>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Inner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &epoch::pin();

            f.debug_struct("MdList")
                .field("head", &self.head.load(Relaxed, guard).as_ref().unwrap())
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

            let children = mem::replace(&mut head.children, MdNode::<T>::NULL_CHILDREN);

            drop(mem::replace(&mut self.head, Atomic::null()).into_owned());

            for child in children.into_iter() {
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
        let head = Atomic::new(MdNode::new_uninit(0, 0));
        Self { head }
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

    pub unsafe fn get(&self, key: usize, guard: &Guard) -> Option<Entry<'_, 'g, T>> {
        // Rebind lifetime to self
        let guard = &*(guard as *const _);

        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let ad = &mut Shared::null();
        let curr = &mut self.head.load(Relaxed, guard);
        let mut dim = 0;
        let mut pred_dim = 0;
        Self::locate_pred(&coord, ad, pred, curr, &mut dim, &mut pred_dim, guard);
        if dim == DIMENSION {
            let curr_ref = curr.as_ref()?;
            if curr_ref.val.is_none() {
                return None;
            }

            return Some(Entry {
                node: curr_ref,
                _parent: self,
                _guard: guard,
            });
        }

        None
    }

    pub unsafe fn delete(&self, key: usize) -> Option<T> {
        let guard = &epoch::pin();
        let coord = Inner::<T>::key_to_coord(key);
        loop {
            let dim = &mut 0;
            let pred_dim = &mut 0;
            let ad = &mut Shared::null();
            let curr = &mut self.head.load(Relaxed, guard);
            let pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();
            Self::locate_pred(&coord, ad, pred, curr, dim, pred_dim, guard);

            if *dim != DIMENSION {
                // Could not find node to delete
                return None;
            }

            let marked = curr.with_tag(set_delinv(curr.tag()));
            if let Some(pred_ref) = pred.as_ref() {
                let child = pred_ref
                    .children
                    .get_unchecked(*pred_dim)
                    .load(SeqCst, guard);
                // makr node for deletion
                let new_child = match pred_ref
                    .children
                    .get_unchecked(*pred_dim)
                    .compare_and_set(*curr, marked, SeqCst, guard)
                {
                    Ok(new) => new,
                    Err(CompareAndSetError { current, .. }) => current,
                };

                if child.with_tag(clr_invalid(child.tag())) == *curr {
                    if !is_invalid(child.tag()) {
                        let res = curr.deref_mut().val.take();
                        return res;
                    } else if is_delinv(child.tag()) {
                        return None;
                    }
                }
            }
        }
    }

    #[inline(never)]
    pub unsafe fn insert(&self, key: usize, val: T) -> InsertStatus {
        let guard = &epoch::pin();
        let coord = Inner::<T>::key_to_coord(key);
        let mut new_node = Owned::new(MdNode::with_coord(coord, val, 0));

        loop {
            let dim = &mut 0;
            let pred_dim = &mut 0;
            let curr = &mut self.head.load(Relaxed, guard);
            let pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();
            let ad: &mut Shared<'_, MdDesc<T>> = &mut Shared::null();

            Self::locate_pred(&coord, ad, pred, curr, dim, pred_dim, guard);

            if *dim == DIMENSION {
                return InsertStatus::AlreadyExists;
            }

            if *dim != *pred_dim {
                if let Some(curr_ref) = curr.as_ref() {
                    let pending = curr_ref.pending.load(SeqCst, guard);
                    *ad = pending;
                    if !pending.is_null() {
                        Self::finish_inserting(curr_ref, pending, guard);
                    }
                }
            }

            if let Some(pred_ref) = pred.as_ref() {
                let child = pred_ref
                    .children
                    .get_unchecked(*pred_dim)
                    .load(Relaxed, guard);
                if is_delinv(child.tag()) {
                    *curr = curr.with_tag(set_delinv(curr.tag()));
                    if *dim == DIMENSION - 1 {
                        *dim = DIMENSION;
                    }
                }

                let desc = Self::fill_new_node(new_node.borrow_mut(), *curr, dim, pred_dim, guard);

                match pred_ref
                    .children
                    .get_unchecked(*pred_dim)
                    .compare_and_set(*curr, new_node, SeqCst, guard)
                {
                    Ok(mut new_node) => {
                        let desc = desc.load(Relaxed, guard);
                        if !desc.is_null() {
                            Self::finish_inserting(new_node.deref_mut(), desc, guard);
                        }

                        return InsertStatus::Inserted;
                    }
                    Err(err) => {
                        new_node = err.new;
                        if !desc.load(SeqCst, guard).is_null() {
                            drop(desc.into_owned());
                        }
                    }
                }
            }
        }
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
                return curr.deref_mut().val.take();
            }
        }

        None
    }

    pub unsafe fn contains(&self, key: usize, guard: &'g Guard) -> bool {
        // Rebind lifetime to self
        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let ad = &mut Shared::null();
        let curr = &mut self.head.load(SeqCst, guard);
        let mut dim = 0;
        let mut pred_dim = 0;

        Self::locate_pred(&coord, ad, pred, curr, &mut dim, &mut pred_dim, guard);

        dim == DIMENSION
    }

    // #[inline]
    // fn key_to_coord(key: usize) -> [u8; DIMENSION] {
    //     let v = key.to_le_bytes();

    //     let q = Simd::from_array([
    //         v[7], v[7], v[6], v[6], v[5], v[5], v[4], v[4], v[3], v[3], v[2], v[2], v[1], v[1],
    //         v[0], v[0],
    //     ]);
    //     const MASK: Simd<u8, 16> = Simd::from_array([
    //         0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F,
    //         0xF0, 0x0F,
    //     ]);
    //     (q & MASK).to_array()
    // }

    #[inline]
    fn key_to_coord(key: usize) -> [u8; DIMENSION] {
        let mut quotient = key;
        let mut coord = [0u8; DIMENSION];

        for d in (0..DIMENSION).rev() {
            coord[d] = (quotient % BASIS) as u8;
            quotient = quotient / BASIS;
        }

        coord
    }

    #[inline(never)]
    unsafe fn locate_pred<'t>(
        coord: &[u8; DIMENSION],
        ad: &mut Shared<'_, MdDesc<T>>,
        pred: &mut Shared<'t, MdNode<T>>,
        curr: &mut Shared<'t, MdNode<T>>,
        dim: &mut usize,
        pred_dim: &mut usize,
        guard: &'t Guard,
    ) {
        let guard = &*(guard as *const _);
        let mut pd = *pred_dim;
        let mut cd = *dim;
        let mut pred_new = Shared::null();

        while cd < DIMENSION {
            let coord_base = coord.get_unchecked(cd);
            while let Some(curr_ref) = curr.as_ref() {
                if coord_base > curr_ref.coord.get_unchecked(cd) {
                    pd = cd;
                    pred_new = *curr;

                    *ad = curr_ref.pending.load(Relaxed, guard);
                    if let Some(ad_ref) = ad.as_ref() {
                        if (ad_ref.pred_dim..=ad_ref.dim).contains(&pd) {
                            Self::finish_inserting(curr_ref, *ad, guard);
                        }
                    }

                    let child = curr_ref
                        .children
                        .get_unchecked(cd)
                        .load(Ordering::Acquire, guard);
                    *curr = child;
                    if curr.is_null() {
                        break;
                    }
                } else {
                    break;
                }
            }

            if let Some(curr) = curr.as_ref() {
                if coord.get_unchecked(cd) >= curr.coord.get_unchecked(cd) {
                    cd += 1;
                    continue;
                }
            }

            break;
        }

        *dim = cd;
        *pred = pred_new;
        *curr = curr.with_tag(0x0);
        *pred_dim = pd;
    }

    #[inline(never)]
    fn fill_new_node<'a>(
        new_node: &mut MdNode<T>,
        curr: Shared<'a, MdNode<T>>,
        dim: &mut usize,
        pred_dim: &mut usize,
        guard: &Guard,
    ) -> Atomic<MdDesc<T>> {
        let desc = if *pred_dim != *dim {
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
                Shared::null().with_tag(0x1),
                // new_node.children[i].load(Relaxed, guard).with_tag(0x1),
                Relaxed,
            );
        }

        if *dim < DIMENSION {
            unsafe {
                new_node.children.get_unchecked(*dim).store(curr, Relaxed);
            }
        }

        new_node.pending.store(desc.load(Relaxed, guard), SeqCst);

        desc
    }

    pub unsafe fn finish_inserting(n: &MdNode<T>, desc: Shared<'_, MdDesc<T>>, guard: &Guard) {
        let desc_ref = desc.deref(); // Safe unwrap
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
            }
        }

        if n.pending.load(Relaxed, guard) == desc {
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
pub struct MdMap<K, V, S = RandomState> {
    list: Inner<V>,
    hasher: S,
    phantom_data: std::marker::PhantomData<K>,
}

impl<K: Hash, T> MdMap<K, T> {
    pub fn new() -> Self {
        Self {
            list: Inner::new(),
            hasher: Default::default(),
            phantom_data: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FakeHasher(u64);

impl Hasher for FakeHasher {
    #[inline]
    fn finish(&self) -> u64 {
        if self.0 == 0 {
            u64::MAX
        } else {
            self.0
        }
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() == 4 {
            self.0 = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 0, 0, 0, 0]);
        } else {
            self.0 = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
        }
    }
}

#[derive(Default, Clone)]
pub struct FakeHashBuilder;

impl BuildHasher for FakeHashBuilder {
    type Hasher = FakeHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FakeHasher(0)
    }
}

impl<K: Hash, T, S: BuildHasher> MdMap<K, T, S> {
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            list: Inner::new(),
            hasher,
            phantom_data: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn hash_usize<V: Hash>(&self, item: &V) -> usize {
        let mut hasher = self.hasher.build_hasher();
        item.hash(&mut hasher);
        hasher.finish() as usize
    }

    pub fn iter(&'_ self) -> Iter<'_, '_, T> {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.iter(guard)
        }
    }

    pub fn insert(&self, key: K, value: T) -> InsertStatus {
        let key = self.hash_usize(&key);
        unsafe { self.list.insert(key, value) }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Entry<'_, '_, T>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        unsafe {
            let guard = &epoch::unprotected();
            self.list.get(key, guard)
        }
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            self.list.contains(key, guard)
        }
    }

    pub fn remove(&self, key: K) -> Option<T> {
        let key = self.hash_usize(&key);
        unsafe { self.list.delete(key) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_insert() {
        let list = MdMap::new();
        list.insert(0, 10);
        assert_eq!(*list.get(&0).unwrap(), 10);
    }

    #[test]
    fn test_remove() {
        let list = MdMap::new();
        list.insert(1, 10);
        assert_eq!(*list.get(&1).unwrap(), 10);
        assert_eq!(list.remove(1), Some(10));
        assert!(list.get(&1).is_none());
    }

    #[test]
    fn test_multiple() {
        let list = MdMap::new();
        for i in 0..100 {
            list.insert(i, i);
        }
        dbg!(&list);

        for i in 0..100 {
            assert_eq!(list.get(&i).map(|v| *v), Some(i), "key: {}", i);
        }

        for i in 0..100 {
            assert_eq!(list.remove(i), Some(i), "key: {}", i);
        }
    }

    #[test]
    fn test_parallel() {
        use rayon::prelude::*;
        let mdlist = MdMap::new();
        let md_ref = &mdlist;
        (1..100).into_par_iter().for_each(|_| {
            for i in 1..1000 {
                md_ref.insert(i, i);
            }
        });

        (1..100).into_iter().for_each(|_| {
            for i in 1..1000 {
                assert!(md_ref.contains(&i));
            }
        });
        drop(mdlist);
    }

    #[test]
    fn test_key_to_coord() {
        for i in 0..(1 << 8) {
            let v = Inner::<usize>::key_to_coord(i)
                .into_iter()
                .map(|s| s.to_string())
                .collect::<String>();
            dbg!(v);
        }
    }
}
