use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    cell::RefCell,
    hash::Hash,
    mem, ptr,
};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Shared};
use crossbeam_utils::atomic::AtomicCell;
use epoch::{Owned, Pointer};
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

use crate::{
    list::{MdDesc, MdNode},
    MdList, MdMap,
};

thread_local!(static HELPSTACK: RefCell<Vec<*const u8>> = RefCell::new(Vec::new()));

#[inline]
fn set_mark(p: usize) -> usize {
    p | 1
}
#[inline]
fn clr_mark(p: usize) -> usize {
    p & !1
}
#[inline]
fn clr_markd(p: usize) -> usize {
    p & !1
}
#[inline]
fn is_marked(p: usize) -> bool {
    (p & 1) != 0
}
#[inline]
fn is_delinv(p: usize) -> bool {
    (p & 2) != 0
}
#[inline]
fn set_delinv(p: usize) -> usize {
    p | 2
}

pub struct Node<T, E> {
    pub key: usize,
    value: AtomicCell<Option<T>>,
    node_desc: Atomic<NodeDesc<T, E>>,
    next: Atomic<Self>,
    pub out_edges: Option<MdMap<T, E, 16, 16>>,
    pub in_edges: Option<MdMap<T, E, 16, 16>>,
}

impl<T, E> Node<T, E> {
    #[inline]
    fn new(
        key: usize,
        value: Option<T>,
        next: Atomic<Self>,
        node_desc: Atomic<NodeDesc<T, E>>,
        out_edges: Option<MdMap<T, E, 16, 16>>,
        in_edges: Option<MdMap<T, E, 16, 16>>,
    ) -> Self {
        Self {
            key,
            value: AtomicCell::new(value),
            next,
            node_desc,
            out_edges,
            in_edges,
        }
    }
}

pub struct OpDesc<T, E> {
    status: OpStatus,
    ops: Vec<Operator<T, E>>,
}

pub struct NodeDesc<T, E> {
    pub desc: Owned<OpDesc<T, E>>,
    pub opid: usize,
    pub override_as_find: bool,
    pub override_as_delete: bool,
}

impl<T, E> NodeDesc<T, E> {
    pub fn new(desc: OpDesc<T, E>, opid: usize) -> Self {
        Self {
            desc: Atomic::new(desc),
            opid,
            override_as_find: false,
            override_as_delete: false,
        }
    }
}

pub struct Operator<T, E> {
    pub optype: OpType<T, E>,
}

#[derive(Clone)]
pub enum OpType<T, E> {
    Find(usize),
    Insert(usize, Option<T>),
    Delete(usize),
    InsertEdge(usize, usize, Option<E>, bool),
    DeleteEdge(usize, usize, bool),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpStatus {
    Active,
    Committed,
    Aborted,
}

#[derive(Debug, Clone)]
pub enum ReturnCode<R> {
    Success,
    Inserted,
    Deleted(R),
    Found(R),
    Skip,
    Fail(String),
}

pub struct Desc<T, E> {
    pub status: AtomicCell<OpStatus>,
    pub size: usize,
    pub ops: Vec<Operator<T, E>>,
    pub pending: Vec<AtomicCell<bool>>,
}

impl<T, E> Desc<T, E> {
    #[must_use]
    pub fn alloc(ops: Vec<Operator<T, E>>) -> *mut Self {
        unsafe {
            let layout = Self::get_layout();
            #[allow(clippy::cast_ptr_alignment)]
            let ptr = alloc(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            ptr::write(&mut (*ptr).status, AtomicCell::new(OpStatus::Active));

            let size = ops.len();
            ptr::write(&mut (*ptr).size, size);

            ptr::write(&mut (*ptr).ops, ops);

            ptr::write(
                &mut (*ptr).pending,
                (0..size).map(|_| AtomicCell::new(true)).collect(),
            );

            ptr
        }
    }

    /// Deallocates a node.
    ///
    /// This function will not run any destructors.
    ///
    /// # Safety
    ///
    /// Be careful not to deallocate data that is still in use.
    pub unsafe fn dealloc(ptr: *mut Self) {
        let layout = Self::get_layout();
        dealloc(ptr as *mut u8, layout);
    }

    /// Returns the layout of a node with the given `height`.
    unsafe fn get_layout() -> Layout {
        let size_self = mem::size_of::<Self>();
        let align_self = mem::align_of::<Self>();

        Layout::from_size_align_unchecked(size_self, align_self)
    }

    #[must_use]
    pub fn empty() -> Self {
        Self {
            status: AtomicCell::new(OpStatus::Committed),
            size: 0,
            ops: Vec::new(),
            pending: Vec::new(),
        }
    }
}

pub struct Graph<T, E> {
    head: Atomic<Node<T, E>>,
    tail: Atomic<Node<T, E>>,
}

impl<T: Hash, E> Graph<T, E> {
    pub fn execute_ops(
        &self,
        desc: *const Desc<T, E>,
        sender: std::sync::mpsc::Sender<ReturnCode<Atomic<Node<T, E>>>>,
        guard: &Guard,
    ) {
        HELPSTACK.with(|hs| {
            hs.replace(Vec::new());
        });

        unsafe {
            self.help_ops(desc, 0, &Some(sender), guard);
        }

        // // Check execution status
        // let op_status = desc
        //     .status
        //     .load();

        // op_status
    }

    #[inline]
    unsafe fn help_ops(
        &self,
        desc: OpDesc<T, E>,
        mut opid: usize,
        sender: &Option<std::sync::mpsc::Sender<ReturnCode<Atomic<Node<T, E>>>>>,
        guard: &Guard,
    ) {
        // FIXME:(rasmus) Safe deref_mut()?
        match (*desc).status.load() {
            OpStatus::Active => {}
            _ => return,
        }

        // Cyclic dependency check
        HELPSTACK.with(|hs| {
            for d in hs.borrow().iter() {
                if std::ptr::eq(*d as *const _, desc) {
                    (*desc)
                        .status
                        .compare_exchange(OpStatus::Active, OpStatus::Aborted);
                    return;
                }
            }

            hs.borrow_mut().push(desc as *const _);

            let mut ret = ReturnCode::Success;

            // Vertex nodes
            let mut del_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();
            let mut del_pred_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();
            let mut ins_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();
            let mut ins_pred_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();

            // Edge nodes
            let mut md_del_nodes: Vec<Shared<'_, MdNode<T, 16, 16>>> = Vec::new();
            let mut md_del_pred_nodes: Vec<Shared<'_, MdNode<T, 16, 16>>> = Vec::new();
            let mut md_del_parent_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();
            let mut md_del_dims: Vec<usize> = Vec::new();
            let mut md_del_pred_dims: Vec<usize> = Vec::new();

            // Edge Nodes
            let mut md_ins_nodes: Vec<Shared<'_, MdNode<T, 16, 16>>> = Vec::new();
            let mut md_ins_pred_nodes: Vec<Shared<'_, MdNode<T, 16, 16>>> = Vec::new();
            let mut md_ins_parent_nodes: Vec<Shared<'_, Node<T, E>>> = Vec::new();
            let mut md_ins_dims: Vec<usize> = Vec::new();
            let mut md_ins_pred_dims: Vec<usize> = Vec::new();

            while let OpStatus::Active = (*desc).status.load() {
                if let ReturnCode::Fail(_) = ret {
                    break;
                }

                if opid >= (*desc).size {
                    break;
                }
                let op = &(*desc).ops[opid];

                match &op.optype {
                    OpType::Insert(key, value) => {
                        let mut inserted = Shared::null();
                        let mut pred = Shared::null();
                        ret = self.insert_vertex(
                            *key,
                            value.take(), // FIXME: give back on failure
                            desc,
                            opid,
                            &mut inserted,
                            &mut pred,
                            guard,
                        );

                        ins_nodes.push(inserted);
                        ins_pred_nodes.push(pred);
                    }

                    OpType::InsertEdge(vertex, edge, value, direction_in) => {
                        let mut inserted = Shared::null();
                        let mut md_pred = Shared::null();
                        let mut parent = Shared::null();

                        let mut dim = 0;
                        let mut pred_dim = 0;

                        self.insert_edge(
                            *vertex,
                            *edge,
                            value.as_ref(),
                            *direction_in,
                            desc,
                            opid,
                            inserted,
                            md_pred,
                            parent,
                            dim,
                            pred_dim,
                            guard,
                        );

                        md_ins_nodes.push(inserted);
                        md_ins_pred_nodes.push(md_pred);
                        md_ins_parent_nodes.push(parent);
                        md_ins_dims.push(dim);
                        md_ins_pred_dims.push(pred_dim);
                    }

                    OpType::Delete(vertex) => {
                        let mut deleted = Shared::null();
                        let mut pred = Shared::null();

                        self.delete_vertex(*vertex, desc, opid, deleted, pred, guard);

                        del_nodes.push(deleted);
                        del_pred_nodes.push(pred);
                    }

                    OpType::DeleteEdge(vertex, edge, direction_in) => {
                        let mut deleted = Shared::null();
                        let mut md_pred = Shared::null();
                        let mut parent = Shared::null();

                        let mut dim = 0;
                        let mut pred_dim = 0;

                        self.delete_edge(
                            *vertex,
                            *edge,
                            *direction_in,
                            desc,
                            opid,
                            deleted,
                            md_pred,
                            parent,
                            dim,
                            pred_dim,
                            guard,
                        );

                        md_del_nodes.push(deleted);
                        md_del_pred_nodes.push(md_pred);
                        md_del_parent_nodes.push(parent);
                        md_del_dims.push(dim);
                        md_del_pred_dims.push(pred_dim);
                    }

                    OpType::Find(key) => {
                        ret = self.find(*key, desc, opid, guard);
                    }
                }

                opid += 1;

                sender.as_ref().map(|tx| tx.send(ret.clone()));
            }

            hs.borrow_mut().pop();

            if let ReturnCode::Fail(_) = ret {
                if desc
                    .stastus
                    .compare_exchange(OpStatus::Active, OpStatus::Aborted)
                    .is_ok()
                {
                    // FIXME:(rasmus) call mark for deletion here
                    // Self::mark_for_deletion(
                    //     &ins_nodes,
                    //     &ins_pred_nodes,
                    //     &md_ins_pred_nodes,
                    //     &md_ins_pred_nodes,
                    //     // &md_ins_parent_nodes,
                    //     &md_ins_dims,
                    //     &md_ins_pred_dims,
                    //     desc,
                    //     guard,
                    // );
                }
            } else if (*desc)
                .status
                .compare_exchange(OpStatus::Active, OpStatus::Committed)
                .is_ok()
            {
                // Self::mark_for_deletion(
                //     &del_nodes,
                //     &del_pred_nodes,
                //     &md_del_nodes,
                //     &md_del_pred_nodes,
                //     // &md_del_parent_nodes,
                //     &md_del_dims,
                //     &md_del_pred_dims,
                //     desc,
                //     guard,
                // )
            }
        });
    }

    unsafe fn insert_vertex(
        &self,
        key: usize,
        value: Option<T>,
        desc: NodeDesc<T, E>,
        opid: usize,
        inserted: &mut Shared<Node<T, E>>,
        pred: &mut Shared<Node<T, E>>,
        guard: &Guard,
    ) -> ReturnCode<Atomic<Node<T, E>>> {
        let guard = &*(guard as *const _);
        *inserted = Shared::null();

        let current = &mut self.head.load_consume(guard);

        let n_desc = Atomic::new(NodeDesc::new(desc, opid));
        loop {
            // Check if node is physically in the list
            if Self::is_node_exist(*current, key) {
                // If the node is physically in the list, it may be possible to simply update the descriptor
                let current_ref = &current.as_ref().unwrap();
                let current_desc = &current_ref.node_desc;

                //Check if node descriptor is marked for deletion
                //If it has, we cannot update the descriptor and must perform physical removal
                let g_current_desc = current_desc.load(SeqCst, guard);
                if is_marked(g_current_desc.tag()) {
                    if !is_marked(current_ref.next.load(SeqCst, epoch::unprotected()).tag()) {
                        current_ref.next.fetch_or(0x1, SeqCst, epoch::unprotected());
                    }
                    *current = self.head.load_consume(guard);
                    continue;
                }

                self.finish_pending_txn(g_current_desc, desc, guard);

                if Self::is_same_operation(
                    g_current_desc.as_ref().unwrap(),
                    // We are the only one accessing n_desc...
                    n_desc.load(Relaxed, epoch::unprotected()).as_ref().unwrap(),
                ) {
                    return ReturnCode::Skip;
                }

                // Check is node is logically in the list
                if Self::is_key_exist(g_current_desc.as_ref().unwrap(), guard) {
                    // The Node is in the list, but it is not certain that it has the new value.
                    // For this reason, we update the Node.
                    // FIXME:(rasmus) This returns Fail in the original code...
                    current.as_ref().unwrap().value.store(value);
                    return ReturnCode::Success;
                } else {
                    if let OpStatus::Active = desc.desc.status {
                    } else {
                        return ReturnCode::Fail("Transaction is inactive".into());
                    }

                    if current
                        .as_ref()
                        .unwrap()
                        .node_desc
                        .compare_and_set(
                            g_current_desc,
                            n_desc.load(Relaxed, epoch::unprotected()),
                            SeqCst,
                            guard,
                        )
                        .is_ok()
                    {
                        *inserted = *current;
                        return ReturnCode::Inserted;
                        // return ReturnCode::Inserted(RefEntry { node: *inserted });
                    }
                }
            } else {
                if let OpStatus::Active = desc.desc.status {
                } else {
                    return ReturnCode::Fail("Transaction is inactive".into());
                }

                let mut new_node = None;
                if new_node.is_none() {
                    let in_edges = MdMap::default();
                    let out_edges = MdMap::default();

                    in_edges.list.head().load(SeqCst, guard).deref_mut().pending = n_desc.clone();
                    out_edges
                        .list
                        .head()
                        .load(SeqCst, guard)
                        .deref_mut()
                        .pending = n_desc.desc.clone();

                    new_node.replace(Node::new(
                        key,
                        value,
                        Atomic::null(),
                        n_desc.clone(),
                        Some(in_edges),
                        Some(out_edges),
                    ));
                }

                new_node.as_ref().unwrap().next.store(*current, Relaxed);

                let next = &pred.as_ref().unwrap().next;
                if let Ok(p) =
                    next.compare_and_set(*current, Owned::new(new_node.unwrap()), SeqCst, guard)
                {
                    *inserted = p;
                    return ReturnCode::Inserted;
                }

                *current = if is_marked(next.load(SeqCst, epoch::unprotected()).tag()) {
                    *pred
                } else {
                    self.head.load(SeqCst, guard)
                };
            }
        }
    }

    fn insert_edge(
        &self,
        vertex: usize,
        edge: usize,
        value: Option<&E>,
        direction_in: bool,
        desc: *const Desc<T, E>,
        opid: usize,
        inserted: Shared<MdNode<T, 16, 16>>,
        md_pred: Shared<MdNode<T, 16, 16>>,
        parent: Shared<Node<T, E>>,
        dim: usize,
        pred_dim: usize,
        guard: &Guard,
    ) {
        todo!()
    }

    fn delete_vertex(
        &self,
        vertex: usize,
        desc: *const Desc<T, E>,
        opid: usize,
        deleted: Shared<Node<T, E>>,
        pred: Shared<Node<T, E>>,
        guard: &Guard,
    ) -> () {
        todo!()
    }

    fn delete_edge(
        &self,
        vertex: usize,
        edge: usize,
        direction_in: bool,
        desc: *const Desc<T, E>,
        opid: usize,
        deleted: Shared<MdNode<T, 16, 16>>,
        md_pred: Shared<MdNode<T, 16, 16>>,
        parent: Shared<Node<T, E>>,
        dim: usize,
        pred_dim: usize,
        guard: &Guard,
    ) {
        todo!()
    }

    unsafe fn find(
        &self,
        key: usize,
        desc: *const Desc<T, E>,
        opid: usize,
        guard: &Guard,
    ) -> ReturnCode<Atomic<Node<T, E>>> {
        // Hack to bind lifetime of guard to self.
        let guard = &*(guard as *const _);

        let pred = &mut Shared::null();
        let current = &mut self.head.load(SeqCst, guard);

        let mut n_desc = Atomic::null();

        loop {
            self.locate_pred(pred, current, key, guard);
            if Self::is_node_exist(*current, key) {
                let current_ref = current.as_ref().unwrap();
                let current_desc = &current_ref.node_desc;

                let g_current_desc = current_desc.load(SeqCst, guard);
                if is_marked(g_current_desc.tag()) {
                    if !is_marked(current_ref.next.load(SeqCst, epoch::unprotected()).tag()) {
                        current_ref.next.fetch_or(0x1, SeqCst, epoch::unprotected());
                    }
                    *current = self.head.load(SeqCst, guard);
                    continue;
                }

                self.finish_pending_txn(g_current_desc, desc, guard);

                if n_desc.load(SeqCst, guard).is_null() {
                    n_desc = Atomic::new(NodeDesc::new(desc, opid));
                }

                let current_desc_ref = g_current_desc.as_ref().expect("No current desc");

                if Self::is_same_operation(
                    current_desc_ref,
                    n_desc.load(SeqCst, guard).as_ref().unwrap(),
                ) {
                    return ReturnCode::Skip;
                }

                if Self::is_key_exist(current_desc_ref, guard) {
                    if let OpStatus::Active = (*desc).status.load() {
                    } else {
                        return ReturnCode::Fail("Transaction is Inactive".into());
                    }

                    if current_ref
                        .node_desc
                        .compare_and_set(g_current_desc, n_desc.load(SeqCst, guard), SeqCst, guard)
                        .is_ok()
                    {
                        return ReturnCode::Success;
                        // return ReturnCode::Found(RefEntry { node: *current });
                    }
                } else {
                    return ReturnCode::Fail("Requested key does not exist".into());
                }
            } else {
                return ReturnCode::Fail("Reqested node does not exist".into());
            }
        }
    }

    #[inline]
    unsafe fn finish_pending_txn(
        &self,
        node_desc: Shared<NodeDesc<T, E>>,
        desc: *const Desc<T, E>,
        guard: &Guard,
    ) {
        if let Some(node_desc_ref) = node_desc.as_ref() {
            let g_node_inner_desc = node_desc_ref.desc;

            if std::ptr::eq(g_node_inner_desc, desc) {
                return;
            }

            let optype = &(*g_node_inner_desc).ops[node_desc_ref.opid].optype;
            if let OpType::Delete(_) = optype {
                if (*g_node_inner_desc).pending[node_desc_ref.opid].load() {
                    self.help_ops(desc, node_desc_ref.opid, &None, guard);
                    return;
                }
            }

            self.help_ops(&*node_desc_ref.desc, node_desc_ref.opid, &None, guard);
        }
    }

    #[inline]
    unsafe fn is_node_exist(node: Shared<Node<T, E>>, key: usize) -> bool {
        !node.is_null() && node.as_ref().unwrap().key == key
    }

    #[inline]
    unsafe fn is_mdnode_exist(node: Shared<MdNode<T, 16, 16>>, coord: [u8; 16]) -> bool {
        !node.is_null() && node.as_ref().unwrap().coord == coord
    }

    #[inline]
    fn is_same_operation(desc: &NodeDesc<T, E>, other: &NodeDesc<T, E>) -> bool {
        desc.desc.into_usize() == other.desc.into_usize() && desc.opid == other.opid
    }

    #[inline]
    fn is_node_active(desc: &NodeDesc<T, E>, _guard: &Guard) -> bool {
        unsafe {
            if let OpStatus::Committed = desc.desc.status {
                true
            } else {
                false
            }
        }
    }

    /// Checks if a node is logically within the list
    #[inline]
    unsafe fn is_key_exist(node_desc: &NodeDesc<T, E>, guard: &Guard) -> bool {
        let is_node_active = Self::is_node_active(node_desc, guard);
        let opoptype = &node_desc.desc.ops[node_desc.opid].optype;

        match opoptype {
            OpType::Find(..) => return true,
            OpType::Insert(..) | OpType::InsertEdge(..) => {
                if is_node_active {
                    return true;
                }
            }
            OpType::Delete(..) | OpType::DeleteEdge(..) => {
                if !is_node_active {
                    return true;
                }
            }
        }

        node_desc.override_as_find || (!is_node_active && node_desc.override_as_delete)
    }

    #[inline]
    unsafe fn locate_pred<'a>(
        &self,
        pred: &mut Shared<'a, Node<T, E>>,
        current: &mut Shared<'a, Node<T, E>>,
        key: usize,
        guard: &Guard,
    ) {
        // Hack to bind lifetime of guard to self.
        let guard = &*(guard as *const _);
        let pred_next = &mut Shared::null();

        while let Some(curr_ref) = current.as_ref() {
            if curr_ref.key >= key {
                break;
            }
            *pred = *current;
            let pred_n = &pred.as_ref().unwrap().next;
            *pred_next = pred_n
                .load(SeqCst, guard)
                .with_tag(clr_mark(pred_n.load(SeqCst, guard).tag()));
            *current = *pred_next;

            while is_marked(current.as_ref().unwrap().next.load(SeqCst, guard).tag()) {
                let next = current.as_ref().unwrap().next.load(SeqCst, guard);
                *current = next.with_tag(clr_mark(next.tag()));
            }

            if current != pred_next {
                //Failed to remove deleted nodes, start over from pred
                if pred_n
                    .compare_and_set(*pred_next, *current, SeqCst, guard)
                    .is_err()
                {
                    *current = self.head.load(SeqCst, guard);
                }
            }
        }
    }
}
