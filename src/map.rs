use crate::list::{Iter, MdList, Ref};
use crossbeam_epoch as epoch;
use std::hash::{BuildHasher, Hash, Hasher};
use std::{borrow::Borrow, collections::hash_map::RandomState};

#[derive(Debug)]
pub struct MdMap<K, V, const BASE: usize, const DIM: usize, S = RandomState> {
    list: MdList<V, BASE, DIM>,
    hasher: S,
    phantom_data: std::marker::PhantomData<K>,
}

unsafe impl<K: Send + Sync, V: Send + Sync, const BASE: usize, const DIM: usize, S: Send + Sync>
    Send for MdMap<K, V, BASE, DIM, S>
{
}
unsafe impl<K: Send + Sync, V: Send + Sync, const BASE: usize, const DIM: usize, S: Send + Sync>
    Sync for MdMap<K, V, BASE, DIM, S>
{
}

impl<K: Hash, V> Default for MdMap<K, V, 16, 16> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash, T, const BASE: usize, const DIM: usize> MdMap<K, T, BASE, DIM> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            list: MdList::new(),
            hasher: Default::default(),
            phantom_data: Default::default(),
        }
    }
}

impl<K: Hash, T, const BASE: usize, const DIM: usize, S: BuildHasher> MdMap<K, T, BASE, DIM, S> {
    /// Creates an empty [`MdMap`] which will use the given hash builder to hash keys.
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            list: MdList::new(),
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
    pub fn iter(&'_ self) -> Iter<'_, '_, T, BASE, DIM> {
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
    pub fn insert(&self, key: K, value: T) -> Option<Ref<'_, T>> {
        let key = self.hash_usize(&key);
        self.list.insert(key, value)
    }

    /// Returns a reference to the value corresponding to the key.

    /// The key may be any borrowed form of the map???s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
    pub fn get<Q>(&self, key: &Q) -> Option<Ref<'_, T>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        self.list.get(key)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map???s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        let guard = &epoch::pin();
        self.list.contains(key, guard)
    }

    /// Removes a key from the map, returning the value at the key if the key was
    /// previously in the map.
    ///
    /// The key may be any borrowed form of the map???s key type,
    /// but [`Hash`] and [`Eq`] on the borrowed form must match those for the key type.
    pub unsafe fn remove<Q>(&self, key: &Q) -> Option<T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key = self.hash_usize(&key);
        self.list.remove(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_insert() {
        let map = MdMap::default();
        map.insert(123, 10);
        assert_eq!(*map.get(&123).unwrap(), 10);
    }

    #[test]
    fn test_update() {
        let map = MdMap::default();
        map.insert(100, 1);
        assert_eq!(*map.get(&100).unwrap(), 1);
        map.insert(100, 2);
        assert_eq!(*map.get(&100).unwrap(), 2);
    }

    #[test]
    fn test_remove() {
        let map = MdMap::default();
        map.insert(1, 10);
        assert_eq!(*map.get(&1).unwrap(), 10);
        assert_eq!(unsafe { map.remove(&1) }, Some(10));
        let res = map.get(&1);
        assert!(res.is_none(), "got: {:?}", res.map(|v| *v));
    }

    #[test]
    fn test_multiple() {
        let map = MdMap::default();
        for i in 0..100 {
            map.insert(i, i);
        }

        for i in 0..100 {
            assert_eq!(map.get(&i).map(|v| *v), Some(i), "key: {}", i);
        }

        for i in 0..100 {
            assert_eq!(unsafe { map.remove(&i) }, Some(i), "key: {}", i);
        }
    }

    #[test]
    fn test_parallel() {
        let map = MdMap::default();
        let md_ref = &map;
        (1..10_000).into_par_iter().for_each(|i| {
            assert!(matches!(md_ref.insert(i, i), None));
        });

        (1..10_000).into_par_iter().for_each(|i| {
            assert!(md_ref.contains_key(&i), "key: {}", i);
            let got = md_ref.get(&i).map(|v| *v);
            assert_eq!(got, Some(i), "key: {}, got: {:?}", i, got);
        });
    }
}
