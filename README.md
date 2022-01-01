<div align="center">
  <br></br>

  <h2>MdMap</h2>
  <p>
    MdMap is a lock-free multi-dimensional list which can replace e.g. HashMap or DashMap
    for fast concurrent/parallel access.
  </p>

<sub>💥 UNSTABLE 💥 <br> MdMap has not been thoroughly tested and is probably filled with bugs</sub>

</div>

## Performance

Items in MdMap are stored in a multi-dimensional linked list.
This makes it possible to achieve logarithmic search performance, and allow many threads to operate on the list in parallel.

Here are some _experimental_ benchmarks, where we see that MdMap performs really well for get operations.

![get](./get.svg)

Insert is not as quick, but still faster than Mutex<Hasmap>.

![insert](./insert.svg)

**Disclaimer**: MdMap is still under development and I am working on improving its performance. As of now, Dashmap is faster for most workloads.

## Todo

- Does not yet support updating values
- Does not yet support the value `0` as the key, as this is reserved for the head
- Check for memory leaks
- Test
- Add hashing to allow hashable types as keys

## Based on these papers

- Zachary Painter, Christina Peterson, Damian Dechev. _Lock-Free Transactional Adjacency List._
- Deli Zhang and Damian Dechev. _An efficient lock-free logarithmic search data structure based on multi-dimensional list._
