//! Python GC edges for values shared by Rust evaluators.
//!
//! Rust aliases contain weak references to the Python payload nodes. Each interpreter or
//! escaping native handle holds actual Python references to the nodes its native values
//! reach. Once Rust execution parks, all root sets are prepared before any old set is
//! released. During execution, the old sets and newly made payloads keep the working set
//! alive, including across reentrant Python collections. Python traversal reads only
//! these stable root sets, never a Rust evaluator that may be running on another thread.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use klong_core::table::Table;
use klong_core::tree::{Node, Tree};
use klong_core::value::{Func, Slot};
use klong_core::Value;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyReferenceError;
use pyo3::prelude::*;
use pyo3::PyClass;
use pyo3::types::{PyWeakrefMethods, PyWeakrefReference};

use super::{registered_callback, table, HostArr, KFn, Klong, NativeDict, NativeObject, NativeTable};

#[pyclass(weakref)]
struct Payload {
    object: Mutex<Option<Py<PyAny>>>,
    link: Arc<PayloadLink>,
}

/// The native value and its payload share this weak link. If an owner is
/// resurrected, its old root set can reconnect the link without a second
/// owning reference to the user's Python object.
#[derive(Default)]
struct PayloadLink(Mutex<Option<Py<PyWeakrefReference>>>);

impl PayloadLink {
    fn get<'py>(&self, py: Python<'py>) -> Option<Bound<'py, Payload>> {
        self.0.lock().unwrap().as_ref()?.bind(py).upgrade()?.downcast_into().ok()
    }

    fn connect(&self, payload: &Bound<'_, Payload>) -> PyResult<()> {
        let weak = PyWeakrefReference::new(payload)?.unbind();
        let previous = self.0.lock().unwrap().replace(weak);
        drop(previous);
        Ok(())
    }
}

#[pymethods]
impl Payload {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&*self.object.lock().unwrap())
    }

    fn __clear__(&self) {
        let object = self.object.lock().unwrap().take();
        drop(object);
    }
}

pub(super) type ValueRef = Arc<ValueReference>;

pub(super) struct ValueReference {
    reference: Reference,
    identity: usize,
}

enum Reference {
    Traced(Arc<PayloadLink>),
    Owned(Py<PyAny>),
}

impl ValueReference {
    pub(super) fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<ValueRef> {
        let identity = object.as_ptr() as usize;
        let reference = if let Some(engine) = active_engine() {
            let link = Arc::new(PayloadLink::default());
            let payload = Py::new(py, Payload { object: Mutex::new(Some(object)), link: link.clone() })?;
            link.connect(payload.bind(py))?;
            graph().lock().unwrap().pending.entry(engine).or_default().push(payload);
            Reference::Traced(link)
        } else {
            // Standalone compiler conversions may produce opaque, temporary host data
            // outside a Klong invocation. They do not participate in interpreter sharing.
            Reference::Owned(object)
        };
        Ok(Arc::new(Self { reference, identity }))
    }

    pub(super) fn owned(object: Py<PyAny>) -> ValueRef {
        Arc::new(Self { identity: object.as_ptr() as usize, reference: Reference::Owned(object) })
    }

    pub(super) fn bind<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match &self.reference {
            Reference::Owned(object) => Some(object.clone_ref(py)),
            Reference::Traced(link) => link.get(py).and_then(|payload| {
                let payload = payload.borrow();
                let object = payload.object.lock().unwrap().as_ref().map(|object| object.clone_ref(py));
                object
            }),
        };
        value.map(|value| value.into_bound(py))
            .ok_or_else(|| PyReferenceError::new_err("the Python value was collected"))
    }

    pub(super) fn identity(&self) -> usize {
        self.identity
    }

    pub(super) fn pin(&self, py: Python<'_>) {
        let (Some(engine), Reference::Traced(link)) = (active_engine(), &self.reference) else { return };
        if let Some(node) = link.get(py) {
            graph().lock().unwrap().pending.entry(engine).or_default().push(node.unbind());
        }
    }
}

#[derive(Default)]
pub(super) struct Roots {
    nodes: RefCell<Vec<Py<Payload>>>,
    registration: RefCell<Option<Py<PyWeakrefReference>>>,
}

impl Roots {
    pub(super) fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for node in self.nodes.borrow().iter() {
            visit.call(node)?;
        }
        Ok(())
    }

    pub(super) fn clear(&self) {
        drop(self.nodes.take());
    }

    fn replace(&self, mut nodes: Walker<'_>, retired: &mut Vec<Vec<Py<Payload>>>) {
        nodes.walk();
        retired.push(self.nodes.replace(nodes.payloads));
    }
}

/// Engine identities do not retain the Python interpreters. A sharing group is used
/// only to know which native state must be parked at the same time.
pub(super) type EngineId = usize;

pub(super) fn engine_id() -> EngineId {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[derive(Default)]
struct Graph {
    active: HashMap<EngineId, usize>,
    groups: HashMap<EngineId, EngineId>,
    dirty: HashSet<EngineId>,
    retired: HashSet<EngineId>,
    pending: HashMap<EngineId, Vec<Py<Payload>>>,
    owners: Vec<(EngineId, Py<PyWeakrefReference>)>,
    refreshing: bool,
}

impl Graph {
    fn group(&self, engine: EngineId) -> EngineId {
        self.groups.get(&engine).copied().unwrap_or(engine)
    }

    fn ready_groups(&self) -> HashSet<EngineId> {
        let blocked: HashSet<_> = self.active.keys().map(|engine| self.group(*engine)).collect();
        self.dirty.iter().map(|engine| self.group(*engine))
            .filter(|group| !blocked.contains(group)).collect()
    }

    fn unregister(&mut self, engine: EngineId) -> Vec<(EngineId, Py<PyWeakrefReference>)> {
        let (removed, live) = std::mem::take(&mut self.owners).into_iter()
            .partition::<Vec<_>, _>(|(owner, _)| *owner == engine);
        self.owners = live;
        if self.dirty.contains(&engine) && self.groups.contains_key(&engine) {
            self.retired.insert(engine);
        } else {
            self.groups.remove(&engine);
            if !self.pending.contains_key(&engine) {
                self.dirty.remove(&engine);
            }
        }
        removed
    }

    fn merge_groups(&mut self, source: EngineId, current: EngineId) {
        let from = self.group(source);
        let to = self.group(current);
        if from != to {
            for group in self.groups.values_mut() {
                if *group == from {
                    *group = to;
                }
            }
            self.groups.insert(source, to);
            self.groups.insert(current, to);
            self.dirty.insert(source);
        }
    }

    fn drain_ready(&mut self, ready: &HashSet<EngineId>) -> (
        Vec<(EngineId, Py<PyWeakrefReference>)>,
        HashMap<EngineId, Vec<Py<Payload>>>,
    ) {
        let dirty = std::mem::take(&mut self.dirty);
        self.dirty = dirty.into_iter().filter(|engine| !ready.contains(&self.group(*engine))).collect();
        let (owners, deferred) = std::mem::take(&mut self.owners).into_iter()
            .partition::<Vec<_>, _>(|(engine, _)| ready.contains(&self.group(*engine)));
        self.owners = deferred;
        let (pending, deferred) = std::mem::take(&mut self.pending).into_iter()
            .partition::<HashMap<_, _>, _>(|(engine, _)| ready.contains(&self.group(*engine)));
        self.pending = deferred;
        // Classify the pins first. A dirty engine's group mapping is needed
        // until its pending references are included in the right publication.
        if !self.retired.is_empty() {
            let groups = &mut self.groups;
            self.retired.retain(|engine| {
                let group = groups.get(engine).copied().unwrap_or(*engine);
                if ready.contains(&group) {
                    groups.remove(engine);
                    false
                } else {
                    true
                }
            });
        }
        (owners, pending)
    }
}

fn graph() -> &'static Mutex<Graph> {
    static GRAPH: OnceLock<Mutex<Graph>> = OnceLock::new();
    GRAPH.get_or_init(|| Mutex::new(Graph::default()))
}

pub(super) fn registered<T: PyClass>(
    owner: Bound<'_, T>,
    engine: EngineId,
    roots: fn(&T) -> &Roots,
) -> PyResult<Bound<'_, T>> {
    roots(&owner.borrow()).register(owner.as_any(), engine)?;
    Ok(owner)
}

impl Roots {
    pub(super) fn needs_registration(&self, py: Python<'_>) -> bool {
        self.registration.borrow().as_ref().and_then(|weak| weak.bind(py).upgrade()).is_none()
    }

    pub(super) fn register(&self, owner: &Bound<'_, PyAny>, engine: EngineId) -> PyResult<()> {
        if !self.needs_registration(owner.py()) {
            return Ok(());
        }
        // CPython clears weak references before calling finalizers. A resurrected
        // owner has its original native state and needs a new registry entry.
        let py = owner.py();
        for node in self.nodes.borrow().iter() {
            let payload = node.bind(py);
            let link = &payload.borrow().link;
            if link.get(py).is_none() {
                link.connect(payload)?;
            }
        }
        let weak = PyWeakrefReference::new(owner)?.unbind();
        let previous = self.registration.replace(Some(weak.clone_ref(py)));
        let mut graph = graph().lock().unwrap();
        graph.owners.push((engine, weak));
        graph.dirty.insert(engine);
        drop(graph);
        drop(previous);
        Ok(())
    }
}

pub(super) fn unregister(engine: EngineId) {
    let weak = graph().lock().unwrap().unregister(engine);
    drop(weak);
}

thread_local! {
    static SCOPES: RefCell<Vec<EngineId>> = const { RefCell::new(Vec::new()) };
}

fn active_engine() -> Option<EngineId> {
    SCOPES.with(|stack| stack.borrow().last().copied())
}

impl Roots {
    /// A Python view can be the last owner of an anonymous native value. Pin its
    /// existing payloads until the receiving roots have been published.
    pub(super) fn import(&self, py: Python<'_>, source: EngineId) {
        let Some(current) = active_engine() else { return };
        let pinned = self.nodes.borrow().iter().map(|node| node.clone_ref(py)).collect::<Vec<_>>();
        let mut graph = graph().lock().unwrap();
        if !pinned.is_empty() {
            graph.pending.entry(current).or_default().extend(pinned);
        }
        if source != current && source != 0 {
            graph.merge_groups(source, current);
        }
    }
}

pub(super) struct Scope<'py> {
    py: Python<'py>,
    engine: EngineId,
}

impl<'py> Scope<'py> {
    pub(super) fn enter(py: Python<'py>, engine: EngineId) -> Self {
        Self::enter_mode(py, engine, true)
    }

    pub(super) fn observe(py: Python<'py>, engine: EngineId) -> Self {
        Self::enter_mode(py, engine, false)
    }

    fn enter_mode(py: Python<'py>, engine: EngineId, roots_may_change: bool) -> Self {
        SCOPES.with(|stack| stack.borrow_mut().push(engine));
        let mut graph = graph().lock().unwrap();
        *graph.active.entry(engine).or_default() += 1;
        if roots_may_change {
            graph.dirty.insert(engine);
        }
        Self { py, engine }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        let last = SCOPES.with(|stack| stack.borrow_mut().pop());
        debug_assert_eq!(last, Some(self.engine));
        let mut state = graph().lock().unwrap();
        let active = state.active.get_mut(&self.engine).unwrap();
        *active -= 1;
        if *active == 0 {
            state.active.remove(&self.engine);
        }
        if state.refreshing || state.ready_groups().is_empty() {
            return;
        }
        state.refreshing = true;
        drop(state);

        let py = self.py;
        loop {
            let (owners, pending) = {
                let mut state = graph().lock().unwrap();
                let ready = state.ready_groups();
                if ready.is_empty() {
                    state.refreshing = false;
                    return;
                }
                state.drain_ready(&ready)
            };

            let mut live = Vec::with_capacity(owners.len());
            let mut retired = Vec::new();
            for (engine, weak) in owners {
                if let Some(owner) = weak.bind(py).upgrade() {
                    refresh(py, &owner, &mut retired);
                    live.push((engine, weak));
                }
            }
            graph().lock().unwrap().owners.extend(live);
            // Publish the new sets before releasing previous references and allowing
            // finalizers to reenter. Imports by a finalizer pin their source payloads.
            drop(retired);
            drop(pending);
        }
    }
}

enum Edge {
    Value(Value),
    Tree(Rc<Tree>),
}

pub(super) struct Walker<'py> {
    py: Python<'py>,
    seen: HashSet<usize>,
    work: Vec<Edge>,
    referenced: HashSet<usize>,
    payloads: Vec<Py<Payload>>,
}

impl<'py> Walker<'py> {
    fn new(py: Python<'py>) -> Self {
        Self { py, seen: HashSet::new(), work: Vec::new(), referenced: HashSet::new(), payloads: Vec::new() }
    }

    pub(super) fn reference(&mut self, reference: &ValueRef) {
        let Reference::Traced(link) = &reference.reference else { return };
        if !self.referenced.insert(Arc::as_ptr(link) as usize) {
            return;
        }
        if let Some(node) = link.get(self.py) {
            self.payloads.push(node.unbind());
        }
    }

    pub(super) fn tree(&mut self, tree: &Rc<Tree>) {
        if self.seen.insert(Rc::as_ptr(tree) as usize) {
            self.work.push(Edge::Tree(tree.clone()));
        }
    }

    fn slot(&mut self, slot: &Slot) {
        match slot {
            Slot::Hole => {}
            Slot::Val(value) => self.value(value),
            Slot::Code(tree, _) => self.tree(tree),
        }
    }

    pub(super) fn value(&mut self, value: &Value) {
        let identity = match value {
            Value::List(list) if list.may_reach_host_values() => Rc::as_ptr(list) as usize,
            Value::Dict(dict) => Rc::as_ptr(dict) as usize,
            Value::Fn(function) => Rc::as_ptr(function) as usize,
            Value::Ext(ext) => Rc::as_ptr(ext) as usize,
            _ => return,
        };
        if self.seen.insert(identity) {
            self.work.push(Edge::Value(value.clone()));
        }
    }

    fn walk(&mut self) {
        // Dictionaries may nest beyond the ordinary array depth limit.
        while let Some(edge) = self.work.pop() {
            match edge {
                Edge::Tree(tree) => {
                    for node in &tree.nodes {
                        if let Node::Lit(value) | Node::DictLit(value) | Node::CopyList(value) = node {
                            self.value(value);
                        }
                    }
                }
                Edge::Value(value) => self.expand(&value),
            }
        }
    }

    fn expand(&mut self, value: &Value) {
        match value {
            Value::List(list) => list.iter().for_each(|item| self.value(item)),
            Value::Dict(dict) => {
                let dict = dict.borrow();
                for value in dict.keys().iter().chain(dict.vals()) {
                    self.value(value);
                }
            }
            Value::Fn(function) => match &**function {
                Func::Native { f, captures, .. } => {
                    if let Some(references) = registered_callback(f) {
                        self.reference(&references.callable);
                        self.reference(&references.original);
                    }
                    if let Some(captures) = captures {
                        for value in captures.iter() {
                            self.value(value);
                        }
                    }
                }
                Func::Code { tree, .. } => self.tree(tree),
                Func::Proj { f, args, .. } => {
                    self.value(f);
                    for slot in args {
                        self.slot(slot);
                    }
                }
                Func::Lazy(_) => {}
            },
            Value::Ext(ext) => {
                if let Some(host) = ext.0.as_any().downcast_ref::<HostArr>() {
                    self.reference(&host.obj);
                    self.reference(&host.table);
                } else if let Some(table) = Table::from_value(value) {
                    let data = table.data();
                    if let Some(schema) = table::python_schema(data.metadata()) {
                        self.reference(schema);
                    }
                    for (name, column) in data.columns() {
                        self.value(name);
                        self.value(column);
                    }
                }
            }
            _ => {}
        }
    }
}

fn refresh(py: Python<'_>, owner: &Bound<'_, PyAny>, retired: &mut Vec<Vec<Py<Payload>>>) {
    let mut walker = Walker::new(py);
    if let Ok(engine) = owner.downcast::<Klong>() {
        let engine = engine.borrow();
        engine.visit_values(&mut walker);
        engine.roots.replace(walker, retired);
    } else if let Ok(func) = owner.downcast::<KFn>() {
        let func = func.borrow();
        walker.value(&func.f);
        func.roots.replace(walker, retired);
    } else if let Ok(dict) = owner.downcast::<NativeDict>() {
        let dict = dict.borrow();
        walker.value(&Value::Dict(dict.d.clone()));
        dict.roots.replace(walker, retired);
    } else if let Ok(object) = owner.downcast::<NativeObject>() {
        let object = object.borrow();
        walker.value(&object.v);
        object.roots.replace(walker, retired);
    } else if let Ok(table) = owner.downcast::<NativeTable>() {
        let table = table.borrow();
        walker.value(&table.value);
        table.roots.replace(walker, retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parked_group_reclaims_dead_keys_without_draining_an_active_group() {
        let (source, retired, blocked, other) = (1, 2, 3, 4);
        let mut graph = Graph::default();
        graph.merge_groups(source, retired);
        graph.merge_groups(blocked, other);
        graph.dirty.extend([retired, other]);
        graph.pending.insert(retired, Vec::new());
        graph.pending.insert(other, Vec::new());
        graph.active.insert(blocked, 1);
        graph.unregister(retired);
        graph.unregister(other);

        let ready = graph.ready_groups();
        assert_eq!(ready, HashSet::from([retired]));
        let (_, pending) = graph.drain_ready(&ready);
        assert!(pending.contains_key(&retired));
        assert!(!pending.contains_key(&other));
        assert!(graph.pending.contains_key(&other));
        assert!(graph.groups.contains_key(&other));
        assert!(!graph.groups.contains_key(&retired));
        assert_eq!(graph.group(source), retired);

        graph.active.remove(&blocked);
        let ready = graph.ready_groups();
        let (_, pending) = graph.drain_ready(&ready);
        assert!(pending.contains_key(&other));
        graph.unregister(source);
        graph.unregister(blocked);
        assert!(graph.groups.is_empty());
        assert!(graph.dirty.is_empty());
        assert!(graph.pending.is_empty());
    }

    #[test]
    fn retired_keys_follow_merges_and_a_group_can_park_without_owners() {
        let (source, first, second) = (1, 2, 3);
        let mut graph = Graph::default();
        graph.merge_groups(source, first);
        graph.dirty.insert(first);
        graph.pending.insert(first, Vec::new());
        graph.unregister(first);

        graph.merge_groups(source, second);
        graph.dirty.insert(second);
        graph.pending.insert(second, Vec::new());
        graph.unregister(second);
        graph.unregister(source);

        let ready = graph.ready_groups();
        assert_eq!(ready, HashSet::from([second]));
        let (owners, pending) = graph.drain_ready(&ready);
        assert!(owners.is_empty());
        assert!(pending.contains_key(&first));
        assert!(pending.contains_key(&second));
        assert!(graph.groups.is_empty());
        assert!(graph.dirty.is_empty());
        assert!(graph.pending.is_empty());
    }

    #[test]
    fn unshared_engine_without_pending_pins_can_leave_without_parking() {
        let engine = 1;
        let mut graph = Graph::default();
        graph.dirty.insert(engine);
        graph.unregister(engine);
        assert!(graph.dirty.is_empty());
        assert!(graph.retired.is_empty());
    }
}
