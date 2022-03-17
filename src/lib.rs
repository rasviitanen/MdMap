#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

//! [`MdMap`] is a lock-free data structure with a map-like interface.
//!
//! Items in [`MdMap`] are stored in a multi-dimensional linked list.
//! This makes it possible to achieve logarithmic search performance
//! while allowing many threads to operate on the list in parallel.
//! An effect of the multi-dimensional list is that keys are sorted,
//! which makes this suitable for things like priority queues.

use crossbeam_epoch::{self as epoch, Atomic, Guard, Shared};
use crossbeam_utils::CachePadded;
use epoch::{CompareAndSetError, Owned};
use std::{
    borrow::{Borrow, BorrowMut},
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
    mem::{self, MaybeUninit},
    ops::Range,
    ptr,
    sync::atomic::{
        AtomicUsize,
        Ordering::{self, Relaxed, SeqCst},
    },
};

const DIMENSION: usize = 16;
const BASIS: usize = 16;
#[allow(clippy::declare_interior_mutable_const)]
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

pub struct Entry<'t, 'g, T> {
    node: &'g MdNode<T>,
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
                    let child = &self.current.unwrap().children.get_unchecked(d);
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
                // FIXME: we should mark this node
                // and include it if set
                if node.coord != [0; DIMENSION] {
                    self.returned_prematurely = true;
                    return Some(Entry {
                        node,
                        _parent: self.parent,
                        _guard: self.guard,
                    });
                }

                for d in 0..DIMENSION {
                    let child = &node.children.get_unchecked(d);
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
struct MdDesc<T> {
    location: Location,
    curr: Atomic<MdNode<T>>,
}

struct MdNode<T> {
    coord: [u8; DIMENSION],
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
            let guard = &epoch::pin();

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
                        .filter_map(|x| {
                            let ptr = x.load(Relaxed, guard);
                            ptr.as_ref().map(|x| (ptr.tag(), x))
                        })
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

            ptr::drop_in_place(self.val.as_mut_ptr());

            let children = mem::replace(&mut self.children, Self::NULL_CHILDREN);
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
    #[allow(clippy::declare_interior_mutable_const)]
    const NULL_CHILDREN: [Atomic<Self>; DIMENSION] = unsafe { mem::transmute([NULL; DIMENSION]) };

    pub fn new(key: usize, val: T) -> Self {
        Self {
            coord: Inner::<T>::key_to_coord(key),
            val: MaybeUninit::new(val),
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }

    #[must_use = "must be initialized"]
    pub fn new_uninit(key: usize) -> Self {
        Self {
            coord: Inner::<T>::key_to_coord(key),
            val: MaybeUninit::uninit(),
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }

    pub fn with_coord(coord: [u8; DIMENSION], val: T) -> Self {
        Self {
            coord,
            val: MaybeUninit::new(val),
            pending: Atomic::null(),
            children: Self::NULL_CHILDREN,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Location {
    pub pd: usize,
    pub cd: usize,
}

impl Location {
    #[inline]
    fn exists(&self) -> bool {
        self.cd == DIMENSION
    }

    #[inline]
    fn try_bump_to_max(&mut self) {
        if self.cd == DIMENSION - 1 {
            self.cd = DIMENSION;
        }
    }

    #[inline]
    fn is_conflict(&self) -> bool {
        self.cd == self.pd
    }

    #[inline]
    fn mark_dimension_as_done(&mut self) {
        self.pd = self.cd;
    }

    #[inline]
    fn goto_next_dimension(&mut self) {
        self.cd += 1;
    }

    #[inline]
    fn current_coord(&self, coord: &[u8; DIMENSION]) -> u8 {
        unsafe { *coord.get_unchecked(self.cd) }
    }

    #[inline]
    fn prev_selection(&self) -> Range<usize> {
        self.pd..self.cd
    }

    #[inline]
    fn curr_selection_contains(&self, dim: usize) -> bool {
        self.pd <= dim && dim <= self.cd
    }
}

#[derive(Default)]
struct Inner<T> {
    head: CachePadded<Atomic<MdNode<T>>>,
    len: CachePadded<AtomicUsize>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Inner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &epoch::pin();

            f.debug_struct("MdList")
                .field("head", &self.head.load(Relaxed, guard).as_ref().unwrap())
                .field("len", &self.len.load(Relaxed))
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

            drop(
                mem::replace(&mut self.head, CachePadded::new(Atomic::null()))
                    .into_inner()
                    .into_owned(),
            );

            for child in children {
                let child_ref = child.load(Relaxed, epoch::unprotected());
                if !child_ref.is_null() {
                    drop(child.into_owned());
                }
            }
        }
    }
}

impl<'g, T> Inner<T> {
    #[must_use]
    pub fn new() -> Self {
        let head = Atomic::new(MdNode::new_uninit(0));
        Self {
            head: CachePadded::new(head),
            len: Default::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        // FIXME: handle over/underflow here...
        self.len.load(Ordering::Relaxed)
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

    pub unsafe fn get(&self, key: usize) -> Option<Entry<'_, 'g, T>> {
        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let ad = &mut Shared::null();
        let curr = &mut self.head.load(Relaxed, epoch::unprotected());
        let guard = &*(&epoch::pin() as *const _);
        let (_, location) = Self::locate_pred(&coord, ad, pred, curr, guard);
        if location.exists() {
            if is_invalid(curr.tag()) {
                return None;
            }

            let curr_ref = curr.as_ref()?;
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
            let ad = &mut Shared::null();
            let curr = &mut self.head.load(Relaxed, epoch::unprotected());
            let pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();
            let (_, location) = Self::locate_pred(&coord, ad, pred, curr, guard);

            if !location.exists() {
                // Could not find node to delete
                return None;
            }

            let marked = curr.with_tag(set_delinv(curr.tag()));
            if let Some(pred_ref) = pred.as_ref() {
                let child = pred_ref
                    .children
                    .get_unchecked(location.pd)
                    .load(SeqCst, guard);

                // mark node for deletion
                let _new_child = match pred_ref
                    .children
                    .get_unchecked(location.pd)
                    .compare_and_set(*curr, marked, SeqCst, guard)
                {
                    Ok(new) => new,
                    Err(CompareAndSetError { current, .. }) => current,
                };

                if child.with_tag(clr_invalid(child.tag())) == *curr {
                    if !is_invalid(child.tag()) {
                        let res = mem::replace(&mut curr.deref_mut().val, MaybeUninit::uninit())
                            .assume_init();
                        return Some(res);
                    } else if is_delinv(child.tag()) {
                        return None;
                    }
                }
            }
        }
    }

    pub unsafe fn insert(&self, key: usize, val: T) -> Option<T> {
        let guard = &epoch::pin();
        let coord = Inner::<T>::key_to_coord(key);
        let mut new_node = Owned::new(MdNode::with_coord(coord, val));

        loop {
            let curr = &mut self.head.load(Relaxed, epoch::unprotected());
            let pred: &mut Shared<'_, MdNode<T>> = &mut Shared::null();
            let ad: &mut Shared<'_, MdDesc<T>> = &mut Shared::null();

            let (curr_atomic, mut location) = Self::locate_pred(&coord, ad, pred, curr, guard);

            if location.exists() && !is_delinv(curr.tag()) {
                let curr_value = mem::replace(&mut curr.deref_mut().val, MaybeUninit::uninit());
                let mut new = curr.with_tag(0x0);
                mem::swap(&mut new_node.val, &mut new.deref_mut().val);

                if curr_atomic
                    .compare_and_set_weak(*curr, new, Ordering::Relaxed, guard)
                    .is_ok()
                {
                    return Some(curr_value.assume_init());
                } else {
                    mem::swap(&mut new.deref_mut().val, &mut new_node.val);
                    continue;
                }
            }

            if !location.is_conflict() {
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
                    .get_unchecked(location.pd)
                    .load(Ordering::Acquire, guard);
                if is_delinv(child.tag()) {
                    *curr = curr.with_tag(set_delinv(curr.tag()));
                    location.try_bump_to_max();
                }

                let desc = Self::fill_new_node(new_node.borrow_mut(), *curr, &mut location, guard);

                match pred_ref
                    .children
                    .get_unchecked(location.pd)
                    .compare_and_set_weak(*curr, new_node, SeqCst, guard)
                {
                    Ok(mut new_node) => {
                        let desc = desc.load(Relaxed, guard);
                        if !desc.is_null() {
                            Self::finish_inserting(new_node.deref_mut(), desc, guard);
                        }

                        self.len.fetch_add(1, Ordering::Relaxed);
                        return Option::None;
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
        &self,
        pred: &mut Shared<'t, MdNode<T>>,
        curr: &mut Shared<'t, MdNode<T>>,
        pred_dim: usize,
        dim: usize,
        guard: &Guard,
    ) -> Option<T> {
        if dim == DIMENSION {
            let pred_child = &pred.as_ref().unwrap().children.get_unchecked(pred_dim);
            if pred_child.load(SeqCst, guard) == *curr
                && pred_child
                    .compare_and_set(*curr, curr.with_tag(set_delinv(curr.tag())), SeqCst, guard)
                    .is_ok()
            {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(
                    mem::replace(&mut curr.deref_mut().val, MaybeUninit::uninit()).assume_init(),
                );
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

        let (_, location) = Self::locate_pred(&coord, ad, pred, curr, guard);
        location.exists()
    }

    #[inline]
    fn key_to_coord(mut key: usize) -> [u8; DIMENSION] {
        [(); DIMENSION].map(|_| {
            let k = (key % BASIS) as u8;
            key /= BASIS;
            k
        })

        // for d in &mut coord {
        //     unsafe {
        //         *d = (key % BASIS) as u8;
        //     }
        //     key /= BASIS;
        // }

        // coord
    }

    unsafe fn locate_pred<'t>(
        coord: &[u8; DIMENSION],
        ad: &mut Shared<'t, MdDesc<T>>,
        pred: &mut Shared<'t, MdNode<T>>,
        curr: &mut Shared<'t, MdNode<T>>,
        guard: &'t Guard,
    ) -> (Atomic<MdNode<T>>, Location) {
        let mut location = Location::default();
        let mut pred_new = Shared::null();
        let mut curr_atomic = Atomic::null();

        'outer: while location.cd < DIMENSION {
            let coord_base = location.current_coord(coord);
            while let Some(curr_ref) = curr.as_ref() {
                if coord_base > location.current_coord(&curr_ref.coord) {
                    location.mark_dimension_as_done();
                    pred_new = *curr;

                    *ad = curr_ref.pending.load(Relaxed, guard);
                    if let Some(ad_ref) = ad.as_ref() {
                        if ad_ref.location.curr_selection_contains(location.pd) {
                            Self::finish_inserting(curr_ref, *ad, guard);
                        }
                    }

                    let child = curr_ref.children.get_unchecked(location.cd);
                    curr_atomic = child.clone();
                    let child = child.load(Ordering::Acquire, guard);
                    *curr = child;
                    if curr.is_null() {
                        break 'outer;
                    }
                } else {
                    break;
                }
            }

            if let Some(curr) = curr.as_ref() {
                if coord_base >= location.current_coord(&curr.coord) {
                    location.goto_next_dimension();
                    continue;
                }
            }

            break;
        }

        *pred = pred_new;
        // *curr = *curr.with_tag(0x0); // TODO: set tag to 0x0 here?
        (curr_atomic, location)
    }

    fn fill_new_node<'a>(
        new_node: &mut MdNode<T>,
        curr: Shared<'a, MdNode<T>>,
        location: &mut Location,
        guard: &Guard,
    ) -> Atomic<MdDesc<T>> {
        let desc = if location.is_conflict() {
            Atomic::null()
        } else {
            let curr_untagged = Atomic::null();
            curr_untagged.store(curr.with_tag(clr_delinv(curr.tag())), Relaxed);
            Atomic::new(MdDesc {
                curr: curr_untagged,
                location: *location,
            })
        };

        for i in 0..location.pd {
            unsafe {
                new_node.children.get_unchecked(i).store(
                    new_node
                        .children
                        .get_unchecked(i)
                        .load(Relaxed, guard)
                        .with_tag(0x1),
                    Relaxed,
                );
            }
        }

        if location.cd < DIMENSION {
            unsafe {
                new_node
                    .children
                    .get_unchecked(location.cd)
                    .store(curr, Relaxed);
            }
        }

        new_node.pending.store(desc.load(Relaxed, guard), Relaxed);

        desc
    }

    pub unsafe fn finish_inserting(n: &MdNode<T>, desc: Shared<'_, MdDesc<T>>, guard: &Guard) {
        let desc_ref = desc.deref(); // Safe unwrap
        let location = desc_ref.location;
        let curr = &desc_ref.curr;

        for i in location.prev_selection() {
            let child = &curr
                .load(SeqCst, guard)
                .as_ref()
                .unwrap()
                .children
                .get_unchecked(i);
            let child = child.fetch_or(0x1, SeqCst, guard);
            let child = child.with_tag(clr_adpinv(child.tag()));

            if !child.is_null() && n.children.get_unchecked(i).load(Relaxed, guard).is_null() {
                let _ = n.children.get_unchecked(i).compare_and_set_weak(
                    Shared::null(),
                    child,
                    SeqCst,
                    guard,
                );
            }
        }

        if n.pending.load(Relaxed, guard) == desc {
            if let Ok(p) = n
                .pending
                .compare_and_set_weak(desc, Shared::null(), SeqCst, guard)
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

unsafe impl<K: Send + Sync, V: Send + Sync, S: Send + Sync> Send for MdMap<K, V, S> {}
unsafe impl<K: Send + Sync, V: Send + Sync, S: Send + Sync> Sync for MdMap<K, V, S> {}

impl<K: Hash, T> MdMap<K, T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            list: Inner::new(),
            hasher: Default::default(),
            phantom_data: std::marker::PhantomData,
        }
    }
}

// #[derive(Clone, Copy)]
// pub struct FakeHasher(u64);

// impl Hasher for FakeHasher {
//     #[inline]
//     fn finish(&self) -> u64 {
//         if self.0 == 0 {
//             u64::MAX
//         } else {
//             self.0
//         }
//     }

//     #[inline]
//     fn write(&mut self, bytes: &[u8]) {
//         if bytes.len() == 4 {
//             self.0 = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 0, 0, 0, 0]);
//         } else {
//             self.0 = u64::from_le_bytes([
//                 bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
//             ]);
//         }
//     }
// }

// #[derive(Default, Clone)]
// pub struct FakeHashBuilder;

// impl BuildHasher for FakeHashBuilder {
//     type Hasher = FakeHasher;

//     fn build_hasher(&self) -> Self::Hasher {
//         FakeHasher(0)
//     }
// }

impl<K: Hash, T, S: BuildHasher> MdMap<K, T, S> {
    /// Creates an empty [`MdMap`] which will use the given hash builder to hash keys.
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            list: Inner::new(),
            hasher,
            phantom_data: std::marker::PhantomData,
        }
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn hash_usize<V: Hash>(&self, item: &V) -> usize {
        let mut hasher = self.hasher.build_hasher();
        item.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// An iterator visiting all entries in the map
    pub fn iter(&'_ self) -> Iter<'_, '_, T> {
        unsafe {
            let guard = &epoch::unprotected();
            self.list.iter(guard)
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated,
    /// and the old value is returned.
    pub fn insert(&self, key: K, value: T) -> Option<T> {
        let key = self.hash_usize(&key);
        unsafe { self.list.insert(key, value) }
    }

    /// Returns a reference to the value corresponding to the key.

    /// The key may be any borrowed form of the map’s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
    pub fn get<Q>(&self, key: &Q) -> Option<Entry<'_, '_, T>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        unsafe { self.list.get(key) }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map’s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        unsafe {
            let guard = &epoch::pin();
            self.list.contains(key, guard)
        }
    }

    /// Removes a key from the map, returning the value at the key if the key was
    /// previously in the map.
    ///
    /// The key may be any borrowed form of the map’s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
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
        let list = MdMap::<usize, usize>::new();
        list.insert(123, 10);
        assert_eq!(*list.get(&123).unwrap(), 10);
    }

    #[test]
    fn test_update() {
        let list = MdMap::<usize, usize>::new();
        list.insert(100, 1);
        assert_eq!(*list.get(&100).unwrap(), 1);
        list.insert(100, 2);
        assert_eq!(*list.get(&100).unwrap(), 2);
    }

    #[test]
    fn test_remove() {
        let list = MdMap::new();
        list.insert(1, 10);
        assert_eq!(*list.get(&1).unwrap(), 10);
        assert_eq!(list.remove(1), Some(10));
        let res = list.get(&1);
        assert!(res.is_none(), "got: {:?}", res.map(|v| *v));
    }

    #[test]
    fn test_multiple() {
        let list = MdMap::new();
        for i in 0..100 {
            list.insert(i, i);
        }

        for i in 0..100 {
            assert_eq!(list.get(&i).map(|v| *v), Some(i), "key: {}", i);
        }

        for i in 0..100 {
            assert_eq!(list.remove(i), Some(i), "key: {}", i);
        }
    }

    #[test]
    fn test_parallel() {
        let mdlist = MdMap::new();
        let md_ref = &mdlist;
        (1..100_000).into_par_iter().for_each(|i| {
            assert!(matches!(md_ref.insert(i, i), None));
        });

        (1..100_000).into_par_iter().for_each(|i| {
            assert!(md_ref.contains_key(&i));
            let got = md_ref.get(&i).map(|v| *v);
            assert_eq!(got, Some(i), "key: {}, got: {:?}", i, got);
        });
    }

    #[test]
    #[ignore = "manual inspection of key entropy"]
    fn test_key_to_coord() {
        for i in 0..(1 << 8) {
            let coord = Inner::<usize>::key_to_coord(i);
            println!("{:^2?}", coord);
        }
    }
}
