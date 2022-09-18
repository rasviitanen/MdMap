use core::marker::PhantomData;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::collections::VecDeque;
use std::ops::Deref;

use crate::cachepadded::CachePadded;
use crate::ebr::{pin, Atomic, Guard, Owned, Shared};

#[derive(Debug)]
pub struct AdoptDesc<const DIM: usize> {
    dp: usize,
    dc: usize,
    curr: Owned<Node<DIM>>,
}

#[derive(Debug)]
pub struct Pred<'g, const DIM: usize> {
    pred: Shared<'g, Node<DIM>>,
    curr: Shared<'g, Node<DIM>>,
    dp: usize,
    dc: usize,
}

#[derive(Debug)]
pub struct Node<const DIM: usize> {
    adesc: Atomic<AdoptDesc<DIM>>,
    children: [Atomic<Node<DIM>>; DIM],
    dimension: usize,
    coords: [u8; DIM],
}

impl<const DIM: usize> Node<DIM> {
    unsafe fn unsafe_drop(&self, guard: &Guard) {
        let pending = self.adesc.load(Relaxed, guard);
        if !pending.is_null() && !pending.tag() != 0x1 {
            guard.defer_destroy(pending);
        }

        for child in &self.children {
            let child = child.load(Relaxed, guard);
            if !child.is_null() && !child.tag() != 0x1 {
                child.deref().unsafe_drop(guard);
            }
        }

        Self::finalize(self, &guard);
    }
}

impl<const DIM: usize> IsElement<DIM, Node<DIM>> for Node<DIM> {
    fn entry_of(entry: &Node<DIM>) -> &Node<DIM> {
        entry
    }

    unsafe fn element_of(entry: &Node<DIM>) -> &Node<DIM> {
        entry
    }

    unsafe fn finalize(entry: &Node<DIM>, guard: &Guard) {
        guard.defer_destroy(Shared::from(Self::element_of(entry) as *const _));
    }
}

pub struct NodeWithValue<const DIM: usize, V> {
    node: CachePadded<Node<DIM>>,
    value: V,
}

impl<const DIM: usize, V> std::ops::Deref for NodeWithValue<DIM, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<const DIM: usize, V> NodeWithValue<DIM, V> {
    #[inline]
    pub fn new(coords: [u8; DIM], value: V) -> Self {
        Self {
            node: CachePadded::from(Node::new(coords)),
            value,
        }
    }
}

impl<const DIM: usize, V> NodeWithValue<DIM, V> {
    fn entry_offset() -> usize {
        use std::mem::MaybeUninit;
        let local: MaybeUninit<Self> = MaybeUninit::uninit();

        // MaybeUninit is repr(transparent so we can treat a pointer as a pointer to the inner value)
        let local_ref: &Self = unsafe { &*(&local as *const MaybeUninit<Self> as *const Self) };
        let entry_ref: &Node<DIM> = &local_ref.node;

        let local_ptr = local_ref as *const Self;
        let entry_ptr = entry_ref as *const Node<DIM>;

        entry_ptr as usize - local_ptr as usize
    }
}

impl<const DIM: usize, V> IsElement<DIM, NodeWithValue<DIM, V>> for NodeWithValue<DIM, V> {
    fn entry_of(entry: &NodeWithValue<DIM, V>) -> &Node<DIM> {
        &entry.node
    }

    unsafe fn element_of(entry: &Node<DIM>) -> &NodeWithValue<DIM, V> {
        let ptr = (entry as *const Node<DIM> as usize - Self::entry_offset()) as *const Self;
        &*ptr
    }

    unsafe fn finalize(entry: &Node<DIM>, guard: &Guard) {
        guard.defer_destroy(Shared::from(Self::element_of(entry) as *const _));
    }
}

pub trait IsElement<const DIM: usize, T> {
    fn entry_of(_: &T) -> &Node<DIM>;
    unsafe fn element_of(_: &Node<DIM>) -> &T;
    unsafe fn finalize(_: &Node<DIM>, _: &Guard);
}

pub struct Iter<'g, const DIM: usize, T, C: IsElement<DIM, T>> {
    guard: &'g Guard,
    needle: &'g [u8],
    stack: VecDeque<Shared<'g, Node<DIM>>>,
    _marker: PhantomData<(&'g T, C)>,
}

impl<'g, const DIM: usize, T: 'g, C: IsElement<DIM, T>> Iterator for Iter<'g, DIM, T, C> {
    type Item = &'g T;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            while let Some(node) = self.stack.pop_front() {
                if node.tag() == 0x2 {
                    continue;
                }

                for d in 0..DIM {
                    let child = node.deref().children[d].load(Relaxed, self.guard);
                    if !child.is_null() {
                        self.stack.push_front(child);
                    }
                }

                if !node.deref().coords.starts_with(self.needle) {
                    return None;
                }

                if node.deref().coords != [0; DIM] {
                    return Some(C::element_of(node.deref()));
                }
            }
        }

        None
    }
}

impl<const DIM: usize> Default for Node<DIM> {
    fn default() -> Self {
        Self {
            children: [(); DIM].map(|_| Atomic::null()),
            adesc: Atomic::null(),
            dimension: 0,
            coords: [0; DIM],
        }
    }
}

impl<const DIM: usize> Node<DIM> {
    #[inline]
    fn new(coords: [u8; DIM]) -> Self {
        Self {
            children: [(); DIM].map(|_| Atomic::null()),
            adesc: Atomic::null(),
            dimension: 0,
            coords,
        }
    }
}

pub struct List<const DIM: usize, T, C: IsElement<DIM, T> = T> {
    head: Atomic<Node<DIM>>,
    _marker: PhantomData<(T, C)>,
}

impl<const DIM: usize, T, C: IsElement<DIM, T>> List<DIM, T, C> {
    const UNMARKED: usize = 0;
    const ADP: usize = 1;
    const DEL: usize = 2;
    const ALL: usize = Self::ADP | Self::DEL;

    pub fn new() -> Self {
        Self {
            head: Atomic::new(Node::default()),
            _marker: PhantomData,
        }
    }

    pub(crate) unsafe fn finish_inserting<'g>(
        node: Shared<'g, Node<DIM>>,
        adesc: Shared<'g, AdoptDesc<DIM>>,
        guard: &'g Guard,
    ) {
        let ad = adesc.deref();
        let curr = &ad.curr;
        for i in ad.dp..ad.dc {
            let mut child = curr.children[i].load(Relaxed, guard);
            while curr.children[i]
                .compare_and_set_weak(
                    child,
                    child.with_tag(child.tag() | Self::ADP),
                    Release,
                    guard,
                )
                .is_err()
            {
                child = curr.children[i].load(Relaxed, guard);
            }

            if node.deref().children[i].load(Relaxed, guard).is_null() {
                let _ = node.deref().children[i].compare_and_set_weak(
                    Shared::null(),
                    child.with_tag(child.tag() & !Self::ADP),
                    Release,
                    guard,
                );
            }
        }
    }

    pub(crate) unsafe fn locate_pred<'g>(
        &'g self,
        coords: [u8; DIM],
        guard: &'g Guard,
    ) -> Pred<'g, DIM> {
        let (mut dp, mut dc) = (0, 0);
        let mut parent = Shared::null();
        let mut curr = self.head.load(Relaxed, &guard);
        while dc < DIM {
            while !curr.is_null() && coords[dc] > curr.deref().coords[dc] {
                dp = dc;
                parent = curr;
                let ad = curr.deref().adesc.load(Relaxed, guard);
                if !ad.is_null() && dp >= ad.deref().dp && dp <= ad.deref().dc {
                    Self::finish_inserting(curr, ad, guard);
                }
                curr = curr.deref().children[dc].load(Relaxed, guard).with_tag(0x0);
            }

            if curr.is_null() || coords[dc] < curr.deref().coords[dc] {
                break;
            } else {
                dc += 1;
            }
        }

        Pred {
            pred: parent,
            curr,
            dp,
            dc,
        }
    }

    pub(crate) unsafe fn get<'g>(&'g self, coords: [u8; DIM], guard: &'g Guard) -> Option<&T> {
        let head = self.head.load(Relaxed, guard);
        if head.is_null() {
            return None;
        }

        let p = self.locate_pred(coords, guard);
        if p.dc == DIM && (p.curr.tag() & Self::DEL == 0) {
            return Some(C::element_of(p.curr.deref()));
        }
        None
    }

    pub(crate) unsafe fn insert<'g>(&'g self, container: Shared<'g, T>, guard: &'g Guard) {
        let mut ad = Shared::null();
        let entry: &Node<DIM> = C::entry_of(container.deref());
        loop {
            let mut p = self.locate_pred(entry.coords, guard);

            // if p.dc == DIM && (p.curr.tag() & Self::DEL == 0) {
            //     // Node already exists
            //     C::finalize(entry, guard);
            //     return;
            // }

            if let Some(curr) = p.curr.as_ref() {
                ad = curr.adesc.load(Relaxed, guard);
            }

            if !ad.is_null() && p.dp != p.dc {
                Self::finish_inserting(p.curr, ad, guard);
            }

            if (p.pred.deref().children[p.dp].load(Relaxed, guard).tag() & Self::DEL) != 0 {
                p.curr.with_tag(p.curr.tag() | Self::DEL);
                if p.dc == DIM - 1 {
                    p.dc = DIM;
                }
            }

            ad = Shared::null();

            if p.dp != p.dc {
                ad = Owned::new(AdoptDesc {
                    curr: p.curr.into_owned(),
                    dp: p.dp,
                    dc: p.dc,
                })
                .into_shared(guard);
            }

            for i in 0..p.dp {
                entry.children[i].store(Shared::null().with_tag(Self::ADP), Relaxed);
            }

            for i in p.dp..DIM {
                entry.children[i].store(
                    entry.children[i]
                        .load(Acquire, guard)
                        .with_tag(Self::UNMARKED),
                    Relaxed,
                );
            }

            if p.dc < DIM {
                entry.children[p.dc].store(p.curr, Relaxed);
            }

            entry.adesc.store(ad, Relaxed);

            let entry_ptr = Shared::from(entry as *const _);
            if p.pred.deref().children[p.dp]
                .compare_and_set_weak(p.curr, entry_ptr, Release, guard)
                .is_ok()
            {
                if !ad.is_null() {
                    Self::finish_inserting(entry_ptr, ad, guard);
                }
                return;
            }
        }
    }

    pub fn starts_with<'g>(&'g self, needle: &'g [u8], guard: &'g Guard) -> Iter<'g, DIM, T, C> {
        let mut coords = [0; DIM];
        for (idx, byte) in needle.iter().copied().enumerate() {
            coords[idx] = byte;
        }
        let pred = unsafe { Self::locate_pred(&self, coords, guard) };

        Iter {
            guard,
            needle,
            stack: VecDeque::from([pred.curr]),
            _marker: PhantomData,
        }
    }
}

impl<const DIM: usize, T, C: IsElement<DIM, T>> Drop for List<DIM, T, C> {
    fn drop(&mut self) {
        unsafe {
            let guard = pin();
            self.head.load(Relaxed, &guard).deref().unsafe_drop(&guard);
        }
    }
}

pub struct MdList<K, T, const DIM: usize = 16> {
    list: List<DIM, NodeWithValue<DIM, T>>,
    _ph: core::marker::PhantomData<K>,
}

impl<const DIM: usize, K: ToCoords<DIM>, T> MdList<K, T, DIM> {
    pub fn new() -> Self {
        Self {
            list: List::<DIM, NodeWithValue<DIM, T>>::new(),
            _ph: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn insert(&self, key: K, value: T) {
        unsafe {
            let guard = crate::ebr::pin();
            let elem = Owned::new(NodeWithValue::new(key.to_coords(), value)).into_shared(&guard);
            self.list.insert(elem, &guard)
        }
    }

    #[inline]
    pub fn get(&self, key: K) -> Option<&T> {
        unsafe {
            let guard = crate::ebr::unprotected();
            self.list.get(key.to_coords(), &guard).map(|v| v.deref())
        }
    }

    pub fn starts_with<'q, Q: ?Sized + AsRef<[u8]>>(
        &'q self,
        prefix: &'q Q,
    ) -> impl 'q + Iterator<Item = &T> {
        unsafe {
            let guard = crate::ebr::unprotected();
            self.list
                .starts_with(prefix.as_ref(), &guard)
                .map(|v| v.deref())
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        unsafe {
            let guard = crate::ebr::unprotected();
            self.list.starts_with(&[], &guard).map(|v| v.deref())
        }
    }
}

pub trait ToCoords<const DIM: usize> {
    fn to_coords(self) -> [u8; DIM];
}

impl ToCoords<16> for usize {
    fn to_coords(mut self) -> [u8; 16] {
        [(); 16].map(|_| {
            let k = (self % 16) as u8;
            self /= 16;
            k
        })
    }
}

impl<'a, const DIM: usize> ToCoords<DIM> for &'a str {
    fn to_coords(self) -> [u8; DIM] {
        let mut coords = [0; DIM];
        let bytes = self.as_bytes();
        assert!(bytes.len() < DIM);
        for (idx, byte) in bytes.into_iter().enumerate() {
            coords[idx] = *byte;
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use super::*;
    use crate::ebr::collector::Collector;
    use crate::ebr::Owned;

    #[test]
    fn insert() {
        let collector = Collector::new();
        let handle = collector.register();
        let guard = handle.pin();

        let l: List<4, Node<4>> = List::new();

        let e1 = Owned::new(Node::new([1, 0, 0, 0])).into_shared(&guard);
        let e2 = Owned::new(Node::new([2, 0, 0, 0])).into_shared(&guard);
        let e3 = Owned::new(Node::new([3, 0, 0, 0])).into_shared(&guard);

        unsafe {
            l.insert(e1, &guard);
            l.insert(e2, &guard);
            l.insert(e3, &guard);
            assert_eq!(
                l.get([1, 0, 0, 0], &guard).unwrap() as *const Node<4>,
                e1.as_raw()
            );
            assert_eq!(
                l.get([2, 0, 0, 0], &guard).unwrap() as *const Node<4>,
                e2.as_raw()
            );
            assert_eq!(
                l.get([3, 0, 0, 0], &guard).unwrap() as *const Node<4>,
                e3.as_raw()
            );
        }
    }

    #[test]
    fn insert_with_value() {
        let collector = Collector::new();
        let handle = collector.register();
        let guard = handle.pin();

        let l: List<4, NodeWithValue<4, usize>> = List::new();

        let e1 = Owned::new(NodeWithValue::new([1, 0, 0, 0], 10)).into_shared(&guard);
        let e2 = Owned::new(NodeWithValue::new([2, 0, 0, 0], 20)).into_shared(&guard);
        let e3 = Owned::new(NodeWithValue::new([3, 0, 0, 0], 30)).into_shared(&guard);

        let e4 = Owned::new(NodeWithValue::new([0, 1, 0, 0], 40)).into_shared(&guard);
        let e5 = Owned::new(NodeWithValue::new([1, 1, 0, 0], 50)).into_shared(&guard);
        let e6 = Owned::new(NodeWithValue::new([2, 1, 0, 0], 60)).into_shared(&guard);

        unsafe {
            l.insert(e1, &guard);
            l.insert(e2, &guard);
            l.insert(e3, &guard);
            l.insert(e4, &guard);
            l.insert(e5, &guard);
            l.insert(e6, &guard);
            assert_eq!(l.get([1, 0, 0, 0], &guard).unwrap().value, 10);
            assert_eq!(l.get([2, 0, 0, 0], &guard).unwrap().value, 20);
            assert_eq!(l.get([3, 0, 0, 0], &guard).unwrap().value, 30);
            assert_eq!(l.get([0, 1, 0, 0], &guard).unwrap().value, 40);
            assert_eq!(l.get([1, 1, 0, 0], &guard).unwrap().value, 50);
            assert_eq!(l.get([2, 1, 0, 0], &guard).unwrap().value, 60);
        }
    }

    #[test]
    fn test_iter() {
        let collector = Collector::new();
        let handle = collector.register();
        let guard = handle.pin();

        let l = List::<16, _, NodeWithValue<16, _>>::new();

        let e1 = Owned::new(NodeWithValue::new(ToCoords::to_coords(1), 1)).into_shared(&guard);
        let e2 = Owned::new(NodeWithValue::new(ToCoords::to_coords(2), 2)).into_shared(&guard);
        let e3 = Owned::new(NodeWithValue::new(ToCoords::to_coords(3), 3)).into_shared(&guard);
        let e4 = Owned::new(NodeWithValue::new(ToCoords::to_coords(4), 4)).into_shared(&guard);
        let e5 = Owned::new(NodeWithValue::new(ToCoords::to_coords(5), 5)).into_shared(&guard);
        let e6 = Owned::new(NodeWithValue::new(ToCoords::to_coords(6), 6)).into_shared(&guard);

        unsafe {
            l.insert(e1, &guard);
            l.insert(e2, &guard);
            l.insert(e3, &guard);
            l.insert(e4, &guard);
            l.insert(e5, &guard);
            l.insert(e6, &guard);

            let mut iter = l.starts_with(&[], &guard);
            dbg!(iter.next().unwrap().value);
            dbg!(iter.next().unwrap().value);
            dbg!(iter.next().unwrap().value);
            dbg!(iter.next().unwrap().value);
            dbg!(iter.next().unwrap().value);
            dbg!(iter.next().unwrap().value);
        }
    }

    #[test]
    fn test_parallel() {
        let l = MdList::new();

        let keys = (300..1_000).cycle().take(4_000).collect::<Vec<_>>();
        keys.par_iter().for_each(|i| {
            l.insert(*i, *i);
        });

        keys.par_iter().for_each(|i| {
            assert_eq!(l.get(*i), Some(i), "key: {}", i);
        });
    }

    #[test]
    fn test_string_coords() {
        let l = MdList::<_, _, 32>::new();

        l.insert("user", "user");
        l.insert("user#123", "user#123");
        l.insert("user#456", "user#456");
        l.insert("a", "a");
        l.insert("ab", "ab");
        l.insert("abc", "abc");
        l.insert("b", "b");
        l.insert("bc", "bc");
        l.insert("ba", "ba");
        l.insert("br", "br");

        assert_eq!(
            l.starts_with("user#").cloned().collect::<Vec<_>>(),
            vec!["user#123", "user#456"]
        );

        assert_eq!(
            l.starts_with("a").cloned().collect::<Vec<_>>(),
            vec!["a", "ab", "abc"]
        );

        assert_eq!(l.starts_with("ba").cloned().collect::<Vec<_>>(), vec!["ba"]);

        assert_eq!(
            l.starts_with("").cloned().collect::<Vec<_>>(),
            vec!["a", "ab", "abc", "b", "ba", "bc", "br", "user", "user#123", "user#456",]
        );
    }
}
