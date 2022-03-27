<div align="center">
  <br></br>

  <h1>🌌 MdMap</h1>
  <p>
    MdMap is a lock-free data structure that can act as a replacment for  e.g. Mutex&lt;HashMap&gt; or DashMap
  </p>

<sub>🚧 UNSTABLE 🚧 <br> MdMap has not been thoroughly tested and is likely filled with bugs</sub>

</div>

Items in MdMap are stored in a multi-dimensional linked list.
This makes it possible to achieve logarithmic search performance while allowing many threads to operate on the list in parallel.
An effect of the multi-dimensional list is that keys are sorted, which makes this suitable for things like priority queues.

## Performance

Here are some _experimental_ and **INCREDIBLY BIASED** benchmarks, where we see that MdMap performs really when contention is high. One can imagine using `MdMap` for applications that have huge pressure on just a few keys.

### Benchmark 1

```
- executing 1000 ops
- map size: 1000 entries (pre-filled)
- operation distribution: 8% insert; 92% get
- operationg on keys: [100, 200, 300, 400, 500, 600, 700]
- key distributions:
  100: 74%
  200: 12%
  300: 6%
  400: 3%
  500: 3%
  600: 1%
  700: 1%
```

| MdMap     | DashMap   | Mutex<HashMap> | RwLock<HashMap> |
| --------- | --------- | -------------- | --------------- |
| 64.733 us | 144.78 us | 202.91 us      | 182.69 us       |

![get](./violin.svg)
**Disclaimer**: MdMap is still under development and I am working on improving its performance. DashMap is faster for most realistic workloads.

## Todo

- Does not yet support updating values safely
- Does not yet support the value `0` as the key, which is reserved for the head
- Check for memory leaks
- Figure out a better hashing situation
- Test

## Based on these papers

- Zachary Painter, Christina Peterson, Damian Dechev. _Lock-Free Transactional Adjacency List._
- Deli Zhang and Damian Dechev. _An efficient lock-free logarithmic search data structure based on multi-dimensional list._
