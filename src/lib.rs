#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

//! [`MdMap`] is a lock-free data structure with a map-like interface.
//!
//! Items in [`MdMap`] are stored in a multi-dimensional linked list.
//! This makes it possible to achieve logarithmic search performance
//! while allowing many threads to operate on the list in parallel.
//! An effect of the multi-dimensional list is that keys are sorted,
//! which makes this suitable for things like priority queues.
// mod graph;
pub mod cachepadded;
mod ebr;
pub mod lazy;
pub(crate) mod list;
mod map;
pub mod mdlist;

pub use list::MdList;
pub use map::{Entry, MdMap, OccupiedEntry, VacantEntry};
