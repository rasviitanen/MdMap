use crossbeam_epoch::{self as epoch, Atomic, CompareExchangeError, Guard, Owned, Shared};
use crossbeam_utils::{Backoff, CachePadded};
use std::{
    borrow::BorrowMut,
    mem::{self, MaybeUninit},
    ops::Range,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

#[inline]
const fn set_adpinv(p: usize) -> usize {
    p | 0x1
}
#[inline]
const fn clr_adpinv(p: usize) -> usize {
    p & !0x1
}
#[inline]
const fn is_adpinv(p: usize) -> bool {
    p & 0x1 != 0
}

#[inline]
const fn set_delinv(p: usize) -> usize {
    p | 0x2
}
#[inline]
const fn clr_delinv(p: usize) -> usize {
    p & !0x2
}
#[inline]
const fn is_delinv(p: usize) -> bool {
    p & 0x2 != 0
}

#[inline]
const fn clr_invalid(p: usize) -> usize {
    p & !0x3
}
#[inline]
const fn is_invalid(p: usize) -> bool {
    p & 0x3 != 0
}

pub struct Ref<'g, T> {
    pub(crate) val: &'g T,
}

impl<'g, T> std::ops::Deref for Ref<'g, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val
    }
}

pub struct Iter<'t, 'g, T, const BASE: usize, const DIM: usize> {
    parent: &'t MdList<T, BASE, DIM>,
    guard: &'g Guard,
    stack: Vec<&'t Atomic<MdNode<T, BASE, DIM>>>,
    current: Option<&'t MdNode<T, BASE, DIM>>,
    returned_prematurely: bool,
}

impl<'t: 'g, 'g, T: 't, const BASE: usize, const DIM: usize> Iterator
    for Iter<'t, 'g, T, BASE, DIM>
{
    type Item = Ref<'g, T>;

    fn next(&mut self) -> Option<Ref<'g, T>> {
        unsafe {
            let guard = &*(self.guard as *const _);

            if self.returned_prematurely {
                self.returned_prematurely = false;
                for d in 0..DIM {
                    let child = &self.current.unwrap().children.get_unchecked(d);
                    if !child.load(Ordering::SeqCst, guard).is_null() {
                        self.stack.push(child);
                    }
                }
            }

            while let Some(node) = self.stack.pop().map(|n| n.load(Ordering::SeqCst, guard)) {
                if node.is_null() || is_delinv(node.tag()) {
                    continue;
                }

                let node = node.as_ref().unwrap();
                self.current = Some(node);

                // Skip the root node
                // FIXME: we should mark this node
                // and include it if set
                if node.coord != [0; DIM] {
                    self.returned_prematurely = true;
                    return Some(Ref {
                        val: node.val.assume_init_ref(),
                    });
                }

                for d in 0..DIM {
                    let child = &node.children.get_unchecked(d);
                    if !child.load(Ordering::SeqCst, guard).is_null() {
                        self.stack.push(child);
                    }
                }
            }

            None
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MdDesc<T, const BASE: usize, const DIM: usize> {
    location: Location<DIM>,
    pub(crate) curr: Atomic<MdNode<T, BASE, DIM>>,
}

pub struct MdNode<T, const BASE: usize, const DIM: usize> {
    pub coord: [u8; DIM],
    val: MaybeUninit<T>,
    pub(crate) pending: Atomic<MdDesc<T, BASE, DIM>>,
    children: [Atomic<Self>; DIM],
}

impl<T, const BASE: usize, const DIM: usize> Default for MdNode<T, BASE, DIM> {
    fn default() -> Self {
        Self::new_uninit(0)
    }
}

impl<T: std::fmt::Debug, const BASE: usize, const DIM: usize> std::fmt::Debug
    for MdNode<T, BASE, DIM>
{
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
                .field(
                    "pending",
                    &self.pending.load(Ordering::Relaxed, guard).as_ref(),
                )
                .field(
                    "children",
                    &self
                        .children
                        .iter()
                        .filter_map(|x| {
                            let ptr = x.load(Ordering::Relaxed, guard);
                            ptr.as_ref().map(|x| (ptr.tag(), x))
                        })
                        .collect::<Vec<_>>(),
                )
                .finish()
        }
    }
}

impl<T, const BASE: usize, const DIM: usize> MdNode<T, BASE, DIM> {
    fn unsafe_drop_value(&mut self) {
        unsafe {
            ptr::drop_in_place(self.val.as_mut_ptr());
        }
    }

    fn unsafe_drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.val.as_mut_ptr());

            let pending = mem::replace(&mut self.pending, Atomic::null());
            let pending = pending.load_consume(epoch::unprotected());
            if !pending.is_null() && !is_adpinv(pending.tag()) {
                pending.into_owned();
            }

            let children = mem::replace(&mut self.children, [(); DIM].map(|_| Atomic::null()));
            for child in children {
                let child_ref = child.load(Ordering::Relaxed, epoch::unprotected());
                if !child_ref.is_null() && !is_adpinv(child_ref.tag()) {
                    child.into_owned().unsafe_drop();
                }
            }
        }
    }
}

impl<T, const BASE: usize, const DIM: usize> MdNode<T, BASE, DIM> {
    pub fn new(key: usize, val: T) -> Self {
        Self {
            coord: MdList::<T, BASE, DIM>::key_to_coord(key),
            val: MaybeUninit::new(val),
            pending: Atomic::null(),
            children: [(); DIM].map(|_| Atomic::null()),
        }
    }

    #[must_use = "must be initialized"]
    pub fn new_uninit(key: usize) -> Self {
        Self {
            coord: MdList::<T, BASE, DIM>::key_to_coord(key),
            val: MaybeUninit::uninit(),
            pending: Atomic::null(),
            children: [(); DIM].map(|_| Atomic::null()),
        }
    }

    pub fn with_coord(coord: [u8; DIM], val: T) -> Self {
        Self {
            coord,
            val: MaybeUninit::new(val),
            pending: Atomic::null(),
            children: [(); DIM].map(|_| Atomic::null()),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Location<const DIM: usize> {
    pub pd: usize,
    pub cd: usize,
}

impl<const DIM: usize> Location<DIM> {
    #[inline]
    fn exists(&self) -> bool {
        self.cd == DIM
    }

    #[inline]
    fn try_bump_to_max(&mut self) {
        if self.cd == DIM - 1 {
            self.cd = DIM;
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
    fn reset(&mut self) {
        self.cd = 0;
        self.pd = 0;
    }

    #[inline]
    fn step_back(&mut self) {
        self.cd = self.pd;
    }

    #[inline]
    fn goto_next_dimension(&mut self) {
        self.cd += 1;
    }

    #[inline]
    fn current_coord(&self, coord: &[u8; DIM]) -> u8 {
        unsafe { *coord.get_unchecked(self.cd) }
    }

    #[inline]
    fn prev_selection(&self) -> Range<usize> {
        self.pd..self.cd
    }

    #[inline]
    fn curr_selection_contains(&self, dim: usize) -> bool {
        dim >= self.pd && dim <= self.cd
    }
}

pub enum Entry<'a, T, const BASE: usize, const DIM: usize> {
    Occupied(OccupiedEntry<'a, T, BASE, DIM>),
    Vacant(VacantEntry<'a, T, BASE, DIM>),
}

pub struct EntryPosition<'a, T, const BASE: usize, const DIM: usize> {
    coord: [u8; DIM],
    guard: &'a Guard,
    head: &'a Atomic<MdNode<T, BASE, DIM>>,
    location: Location<DIM>,
    pred: Shared<'a, MdNode<T, BASE, DIM>>,
    curr: Shared<'a, MdNode<T, BASE, DIM>>,
}

impl<'a, T, const BASE: usize, const DIM: usize> EntryPosition<'a, T, BASE, DIM> {
    fn insert(&mut self, val: T, mut is_update: bool) -> Option<T> {
        unsafe {
            let mut new_node = Owned::new(MdNode::with_coord(self.coord, val));
            let mut location = self.location;
            let backoff = Backoff::new();

            loop {
                if let Some(pred_ref) = self.pred.as_ref() {
                    let pred_child = pred_ref
                        .children
                        .get_unchecked(location.pd)
                        .load(Ordering::Acquire, &self.guard);
                    let mut expected = self.curr;

                    if is_delinv(pred_child.tag()) {
                        expected = self.curr.with_tag(set_delinv(self.curr.tag()));
                        location.try_bump_to_max();
                    }

                    if pred_child == expected.into() {
                        MdList::fill_new_node(
                            new_node.borrow_mut(),
                            expected,
                            &mut location,
                            &self.guard,
                        );

                        match pred_ref
                            .children
                            .get_unchecked(location.pd)
                            .compare_exchange_weak(
                                expected,
                                new_node,
                                Ordering::SeqCst,
                                Ordering::Relaxed,
                                &self.guard,
                            ) {
                            Ok(mut new_node) => {
                                let desc = new_node
                                    .deref()
                                    .pending
                                    .load(Ordering::Relaxed, &self.guard);
                                if !desc.is_null() {
                                    if let Some(curr_ref) = self.curr.as_ref() {
                                        let pending =
                                            curr_ref.pending.load(Ordering::Relaxed, &self.guard);
                                        if !pending.is_null() {
                                            MdList::finish_inserting(
                                                curr_ref,
                                                pending,
                                                &self.guard,
                                            );
                                        }
                                    }

                                    MdList::finish_inserting(
                                        new_node.deref_mut(),
                                        desc,
                                        &self.guard,
                                    );
                                }

                                if is_update {
                                    return Some(
                                        std::mem::replace(
                                            &mut expected.deref_mut().val,
                                            MaybeUninit::uninit(),
                                        )
                                        .assume_init(),
                                    );
                                }

                                return None;
                            }
                            Err(err) => {
                                new_node = err.new;
                            }
                        }
                    }

                    if is_adpinv(pred_child.tag()) {
                        self.pred = Shared::null();
                        self.curr = self.head.load(Ordering::Relaxed, epoch::unprotected());
                        location.reset();
                    } else if pred_child.with_tag(0x0) != self.curr {
                        self.curr = self.pred;
                        location.step_back();
                    } else {
                        // Do nothing
                    }

                    let desc = mem::replace(&mut new_node.pending, Atomic::null())
                        .load_consume(&epoch::unprotected());
                    if !desc.is_null() {
                        drop(desc.to_owned());
                    }
                }

                let found_location = MdList::locate_pred(
                    &self.coord,
                    location,
                    &mut self.pred,
                    &mut self.curr,
                    self.guard,
                );
                location = found_location;
                is_update = location.exists() && !is_invalid(self.curr.tag());
                backoff.spin();
            }
        }
    }
}

pub struct VacantEntry<'a, T, const BASE: usize, const DIM: usize>(EntryPosition<'a, T, BASE, DIM>);

impl<'a, T, const BASE: usize, const DIM: usize> VacantEntry<'a, T, BASE, DIM> {
    #[inline]
    pub fn insert(mut self, value: T) {
        self.0.insert(value, false);
    }
}

pub struct OccupiedEntry<'a, T, const BASE: usize, const DIM: usize>(
    EntryPosition<'a, T, BASE, DIM>,
);

impl<'a, T, const BASE: usize, const DIM: usize> OccupiedEntry<'a, T, BASE, DIM> {
    #[inline]
    pub fn get(&self) -> Ref<'_, T> {
        unsafe {
            let curr_ref = self.0.curr.deref();
            Ref {
                val: curr_ref.val.assume_init_ref(),
            }
        }
    }

    #[inline]
    pub fn insert(mut self, value: T) -> Option<T> {
        self.0.insert(value, true)
    }
}

pub struct MdList<T, const BASE: usize, const DIM: usize> {
    head: Atomic<MdNode<T, BASE, DIM>>,
    len: CachePadded<AtomicUsize>,
}

impl<T, const BASE: usize, const DIM: usize> Default for MdList<T, BASE, DIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'g, T, const BASE: usize, const DIM: usize> MdList<T, BASE, DIM> {
    #[must_use]
    pub fn new() -> Self {
        let head = Atomic::new(MdNode::new_uninit(0));
        Self {
            head,
            len: Default::default(),
        }
    }

    pub fn retain(&self, mut f: impl FnMut(&mut T) -> bool) {
        let guard = &epoch::pin();
        let mut stack = vec![&self.head];
        unsafe {
            while let Some(n) = stack.pop() {
                let mut node = n.load(Ordering::SeqCst, guard);
                if node.is_null() || is_delinv(node.tag()) {
                    continue;
                }

                let node_ref = node.as_ref().unwrap();

                // Skip the root node
                // FIXME: we should mark this node
                // and include it if set
                if node_ref.coord != [0; DIM] {
                    if !f(node.deref_mut().val.assume_init_mut()) {
                        let marked = node.with_tag(set_delinv(node.tag()));
                        let _ = n.compare_exchange_weak(
                            node,
                            marked,
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                            guard,
                        );
                    }
                }

                for d in 0..DIM {
                    let child = &node_ref.children.get_unchecked(d);
                    if !child.load(Ordering::SeqCst, guard).is_null() {
                        stack.push(child);
                    }
                }
            }
        }
    }

    pub fn head(&self) -> &Atomic<MdNode<T, BASE, DIM>> {
        &self.head
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        // FIXME: handle over/underflow here...
        self.len.load(Ordering::Relaxed)
    }

    pub fn iter(&self, guard: &'g Guard) -> Iter<'_, 'g, T, BASE, DIM> {
        Iter {
            parent: self,
            stack: vec![&self.head],
            guard,
            current: None,
            returned_prematurely: false,
        }
    }

    pub fn get(&self, key: usize) -> Option<Ref<'_, T>> {
        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();

        unsafe {
            let curr = &mut self.head.load(Ordering::Relaxed, epoch::unprotected());
            let guard = &*(&epoch::pin() as *const _);
            let location = Self::locate_pred(&coord, Location::default(), pred, curr, guard);
            if location.exists() {
                if is_invalid(curr.tag()) {
                    return None;
                }

                let curr_ref = curr.as_ref()?;
                return Some(Ref {
                    val: curr_ref.val.assume_init_ref(),
                });
            }
        }

        None
    }

    pub unsafe fn remove(&self, key: usize) -> Option<T> {
        let guard = &epoch::pin();
        let coord = Self::key_to_coord(key);
        loop {
            let curr = &mut self.head.load(Ordering::Relaxed, epoch::unprotected());
            let pred: &mut Shared<'_, MdNode<T, BASE, DIM>> = &mut Shared::null();
            let location = Self::locate_pred(&coord, Location::default(), pred, curr, guard);

            if !location.exists() {
                // Could not find node to delete
                return None;
            }

            let marked = curr.with_tag(set_delinv(curr.tag()));
            if let Some(pred_ref) = pred.as_ref() {
                let child = pred_ref
                    .children
                    .get_unchecked(location.pd)
                    .load(Ordering::SeqCst, guard);

                // mark node for deletion
                let _new_child = match pred_ref
                    .children
                    .get_unchecked(location.pd)
                    .compare_exchange_weak(
                        *curr,
                        marked,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                        guard,
                    ) {
                    Ok(new) => new,
                    Err(CompareExchangeError { current, .. }) => current,
                };

                if child.with_tag(clr_invalid(child.tag())) == *curr {
                    if !is_invalid(child.tag()) {
                        // This is not safe yet
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

    pub fn entry(&self, key: usize) -> Entry<'_, T, BASE, DIM> {
        unsafe {
            let guard = &*(&epoch::pin() as *const _);
            let coord = Self::key_to_coord(key);
            let mut location = Location::default();
            let mut curr = self.head.load(Ordering::Relaxed, epoch::unprotected());
            let mut pred: Shared<'_, MdNode<T, BASE, DIM>> = Shared::null();

            let found_location = Self::locate_pred(&coord, location, &mut pred, &mut curr, guard);
            location = found_location;
            let occupied = location.exists() && !is_delinv(curr.tag());

            let entry_pos = EntryPosition {
                coord,
                guard,
                head: &self.head,
                location: found_location,
                curr,
                pred,
            };

            if occupied {
                Entry::Occupied(OccupiedEntry(entry_pos))
            } else {
                Entry::Vacant(VacantEntry(entry_pos))
            }
        }
    }

    pub fn insert(&self, key: usize, val: T) -> Option<T> {
        match self.entry(key) {
            Entry::Occupied(entry) => {
                return entry.insert(val);
            }
            Entry::Vacant(entry) => {
                entry.insert(val);
            }
        }
        None
    }

    pub fn contains(&self, key: usize, guard: &'g Guard) -> bool {
        let coord = Self::key_to_coord(key);
        let pred = &mut Shared::null();
        let curr = &mut self.head.load(Ordering::Relaxed, guard);

        let location = unsafe { Self::locate_pred(&coord, Location::default(), pred, curr, guard) };

        location.exists()
    }

    #[inline]
    fn key_to_coord(mut key: usize) -> [u8; DIM] {
        [(); DIM].map(|_| {
            let k = (key % BASE) as u8;
            key /= BASE;
            k
        })
    }

    unsafe fn locate_pred<'t>(
        coord: &[u8; DIM],
        mut location: Location<DIM>,
        pred: &mut Shared<'t, MdNode<T, BASE, DIM>>,
        curr: &mut Shared<'t, MdNode<T, BASE, DIM>>,
        guard: &'t Guard,
    ) -> Location<DIM> {
        while location.cd < DIM {
            while !curr.is_null()
                && location.current_coord(coord) > location.current_coord(&curr.deref().coord)
            {
                location.mark_dimension_as_done();
                *pred = *curr;

                let curr_ref = curr.deref();
                let pending = curr_ref.pending.load(Ordering::Relaxed, guard);
                if !pending.is_null()
                    && pending
                        .deref()
                        .location
                        .curr_selection_contains(location.pd)
                {
                    Self::finish_inserting(curr_ref, pending, guard);
                }

                let child = curr_ref
                    .children
                    .get_unchecked(location.cd)
                    .load(Ordering::Relaxed, guard);
                *curr = child.with_tag(clr_adpinv(child.tag()));
            }

            if curr.is_null()
                || location.current_coord(coord) < location.current_coord(&curr.deref().coord)
            {
                return location;
            } else {
                location.goto_next_dimension();
            }
        }

        location
    }

    fn fill_new_node<'a>(
        new_node: &mut MdNode<T, BASE, DIM>,
        curr: Shared<'a, MdNode<T, BASE, DIM>>,
        location: &mut Location<DIM>,
        guard: &'a Guard,
    ) {
        let desc = if location.is_conflict() {
            None
        } else {
            let curr_untagged = Atomic::null();
            curr_untagged.store(curr.with_tag(clr_delinv(curr.tag())), Ordering::Relaxed);
            Some(Owned::new(MdDesc {
                curr: curr_untagged,
                location: *location,
            }))
        };

        for i in 0..location.pd {
            unsafe {
                new_node.children.get_unchecked(i).store(
                    new_node
                        .children
                        .get_unchecked(i)
                        .load(Ordering::Relaxed, guard)
                        .with_tag(0x1),
                    Ordering::Relaxed,
                );
            }
        }

        if location.cd < DIM {
            unsafe {
                new_node
                    .children
                    .get_unchecked(location.cd)
                    .store(curr, Ordering::Relaxed);
            }
        }

        if let Some(desc) = desc {
            new_node.pending.store(desc, Ordering::Relaxed);
        } else {
            new_node.pending.store(Shared::null(), Ordering::Relaxed);
        }
    }

    unsafe fn finish_inserting(
        n: &MdNode<T, BASE, DIM>,
        desc: Shared<'_, MdDesc<T, BASE, DIM>>,
        guard: &Guard,
    ) {
        let desc_ref = desc.deref(); // Safe unwrap
        let location = desc_ref.location;
        let curr = &desc_ref.curr;

        let curr_ref = &curr.load(Ordering::SeqCst, guard).as_ref().unwrap();

        for i in location.prev_selection() {
            let child = curr_ref.children.get_unchecked(i);
            let child = child.fetch_or(0x1, Ordering::Relaxed, guard);
            let child = child.with_tag(clr_adpinv(child.tag()));

            if !child.is_null()
                && n.children
                    .get_unchecked(i)
                    .load(Ordering::SeqCst, guard)
                    .is_null()
            {
                let _ = n.children.get_unchecked(i).compare_exchange_weak(
                    Shared::null(),
                    child,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                    guard,
                );
            }
        }

        if n.pending.load(Ordering::Relaxed, guard) == desc {
            if n.pending
                .compare_exchange_weak(
                    desc,
                    Shared::null(),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                    guard,
                )
                .is_ok()
            {
                // if !desc.is_null() {
                //     guard.defer_unchecked(move || {
                //         drop(desc.into_owned());
                //     });
                // }
            }
        }
    }
}

impl<T: std::fmt::Debug, const BASE: usize, const DIM: usize> std::fmt::Debug
    for MdList<T, BASE, DIM>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let guard = &epoch::pin();

            f.debug_struct("MdList")
                .field(
                    "head",
                    &self.head.load(Ordering::Relaxed, guard).as_ref().unwrap(),
                )
                .field("len", &self.len.load(Ordering::Relaxed))
                .finish()
        }
    }
}

impl<T, const BASE: usize, const DIM: usize> std::ops::Drop for MdList<T, BASE, DIM> {
    fn drop(&mut self) {
        unsafe {
            mem::replace(&mut self.head, Atomic::null())
                .into_owned()
                .unsafe_drop();
        }
    }
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use super::*;

    #[test]
    #[ignore = "manual inspection of key entropy"]
    fn test_key_to_coord() {
        dbg!(mem::size_of::<MdNode<usize, 16, 16>>());
        dbg!(mem::size_of::<CachePadded<usize>>());
        for i in 0..(1 << 2) {
            let coord = MdList::<usize, 16, 16>::key_to_coord(i);
            println!("{:^2?}", coord);
        }
    }

    #[test]
    fn test_update() {
        let list = MdList::<usize, 16, 16>::new();
        assert_eq!(list.insert(1, 1), None);
        assert_eq!(list.insert(1, 2), Some(1));
    }

    #[test]
    fn test_insert() {
        loom::model(|| {
            let list1 = std::sync::Arc::new(MdList::<usize, 16, 16>::new());
            let list2 = list1.clone();

            // use 5 since it's greater than the 4 used for the sanitize feature
            let jh = loom::thread::spawn(move || {
                assert_eq!(list1.insert(1, 1), None);
                assert_eq!(list1.insert(1, 2), Some(1));
                assert_eq!(list1.insert(1, 3), Some(2));
            });

            jh.join().unwrap();
            assert_eq!(list2.get(1).as_deref(), Some(&3));
        });
    }

    #[test]
    fn test_insert_hold_ref() {
        let list = MdList::<usize, 16, 16>::new();
        assert_eq!(list.insert(1, 1), None);
        let v = list.insert(1, 2);
        assert_eq!(list.insert(1, 3), Some(2));
        assert_eq!(v, Some(1));
    }

    #[test]
    fn test_entry() {
        let list = MdList::<usize, 16, 16>::new();
        match list.entry(1) {
            Entry::Occupied(_) => panic!(),
            Entry::Vacant(e) => e.insert(123),
        }
        assert_eq!(list.get(1).as_deref(), Some(&123));
        match list.entry(1) {
            Entry::Occupied(e) => {
                assert_eq!(*e.get(), 123);
                e.insert(456);
            }
            Entry::Vacant(_) => panic!(),
        }
        assert_eq!(list.get(1).as_deref(), Some(&456))
    }

    #[test]
    fn test_retain() {
        let list = MdList::<usize, 16, 16>::new();
        list.insert(1, 1);
        list.insert(2, 2);
        list.insert(3, 3);
        list.insert(4, 4);
        list.retain(|v| *v % 2 == 0);
        assert_eq!(list.get(1).as_deref(), None);
        assert_eq!(list.get(2).as_deref(), Some(&2));
        assert_eq!(list.get(3).as_deref(), None);
        assert_eq!(list.get(4).as_deref(), Some(&4));
    }

    #[test]
    fn test_get() {
        let list = MdList::<usize, 16, 16>::new();
        list.insert(1, 1);
        assert_eq!(list.get(1).as_deref(), Some(&1));
    }

    #[test]
    fn test_get_hold_ref() {
        let list = MdList::<usize, 16, 16>::new();
        list.insert(1, 1);
        let r = list.get(1);
        list.insert(1, 2);
        // This is unsafe for now :)
        // assert_eq!(r.as_deref(), Some(&1));
        assert_eq!(list.get(1).as_deref(), Some(&2));
    }

    #[test]
    fn test_parallel() {
        let map = MdList::<usize, 16, 16>::default();
        let md_ref = &map;

        (1..1_00).into_par_iter().for_each(|i| {
            assert!(matches!(md_ref.insert(i, i), None));
        });

        (1..1_00).into_par_iter().for_each(|i| {
            assert!(md_ref.contains(i, &epoch::pin()), "key: {}", i);
            let got = md_ref.get(i).map(|v| *v);
            assert_eq!(got, Some(i), "key: {}, got: {:?}", i, got);
        });
    }
}
