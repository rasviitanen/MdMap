use bustle::*;
use dashmap::DashMap;
use mdmap::{FakeHashBuilder, MdMap};
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct MdMapTable<K>(Arc<MdMap<K, (), RandomState>>);

impl<K> Collection for MdMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + std::hash::Hash + Eq,
{
    type Handle = Self;
    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(MdMap::with_hasher(Default::default())))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for MdMapTable<K>
where
    K: Send + From<u64> + Copy + 'static + std::hash::Hash + Eq,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        !matches!(self.0.insert(*key, ()), mdmap::InsertStatus::AlreadyExists);
        true
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        self.0.remove(*key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        matches!(self.0.insert(*key, ()), mdmap::InsertStatus::AlreadyExists)
    }
}

#[derive(Clone)]
pub struct DashMapTable<K>(Arc<DashMap<K, ()>>);

impl<K> Collection for DashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + Hash + Eq + std::fmt::Debug,
{
    type Handle = Self;

    fn with_capacity(capacity: usize) -> Self {
        Self(Arc::new(DashMap::new()))
    }

    fn pin(&self) -> Self::Handle {
        self.clone()
    }
}

impl<K> CollectionHandle for DashMapTable<K>
where
    K: Send + Sync + From<u64> + Copy + 'static + Hash + Eq + std::fmt::Debug,
{
    type Key = K;

    fn get(&mut self, key: &Self::Key) -> bool {
        self.0.get(key).is_some()
    }

    fn insert(&mut self, key: &Self::Key) -> bool {
        self.0.insert(*key, ()).is_none()
    }

    fn remove(&mut self, key: &Self::Key) -> bool {
        todo!()
        // self.0.remove(key).is_some()
    }

    fn update(&mut self, key: &Self::Key) -> bool {
        todo!()
        // self.0.get_mut(key).map(|mut v| *v = ()).is_some()
    }
}

fn main() {
    let n = 8;
    let mix = Mix {
        read: 90,
        insert: 10,
        remove: 0,
        update: 0,
        upsert: 0,
    };

    // for n in 1..=3 {
    let mut wl = Workload::new(12, mix);
    let wl = wl.initial_capacity_log2(n).operations(1024.0);

    // wl.run::<MdMapTable<u64>>();
    wl.run::<MdMapTable<u64>>();
    wl.run::<DashMapTable<u64>>();
    // }
}
