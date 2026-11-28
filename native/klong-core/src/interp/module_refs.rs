//! A module owns its variables and exports. A function refers back to the
//! scope in which it runs. Count the owning references inside this graph so
//! that an extra reference (including one held by the embedder) is a root.
//! Following only reachable edges lets a module own its own projections.
use super::{Func, ModScope, Rc, Slot, Value, Vec};

#[derive(Default)]
struct Node {
    value: Option<Value>,
    incoming: usize,
    edges: Vec<usize>,
}

struct Graph {
    nodes: Vec<Node>,
    objects: Vec<(usize, usize)>,
    base: u16,
    modules: usize,
}

// Collection is optional. Every workspace allocation can fail without changing
// the module scopes. Vec growth and insertion must reserve before writing.
fn push<T>(items: &mut Vec<T>, value: T) -> Option<()> {
    items.try_reserve(1).ok()?;
    items.push(value);
    Some(())
}

fn identity(v: &Value) -> Option<(usize, usize)> {
    match v {
        Value::Fn(rc) => Some((Rc::as_ptr(rc) as usize, Rc::strong_count(rc))),
        Value::Dict(rc) => Some((Rc::as_ptr(rc) as usize, Rc::strong_count(rc))),
        // A mixed list containing only data cannot own a module. Avoid adding
        // every scalar record in a large dataset to the reachability graph.
        #[cfg(feature = "host")]
        Value::List(rc) if rc.may_reach_host_values() => Some((Rc::as_ptr(rc) as usize, Rc::strong_count(rc))),
        #[cfg(not(feature = "host"))]
        Value::List(rc) if rc.iter().any(|v| matches!(v, Value::Fn(_) | Value::Dict(_) | Value::List(_))) =>
            Some((Rc::as_ptr(rc) as usize, Rc::strong_count(rc))),
        _ => None,
    }
}

impl Graph {
    fn object(&mut self, key: usize, value: &Value) -> Option<usize> {
        match self.objects.binary_search_by_key(&key, |entry| entry.0) {
            Ok(i) => Some(self.objects[i].1),
            Err(i) => {
                let id = self.nodes.len();
                self.objects.try_reserve(1).ok()?;
                push(&mut self.nodes, Node { value: Some(value.clone()), ..Node::default() })?;
                self.objects.insert(i, (key, id));
                Some(id)
            }
        }
    }

    fn own_object(&mut self, owner: usize, key: usize, value: &Value) -> Option<usize> {
        let child = self.object(key, value)?;
        push(&mut self.nodes[owner].edges, child)?;
        self.nodes[child].incoming += 1;
        Some(child)
    }

    fn owns(&mut self, owner: usize, value: &Value) -> Option<()> {
        if let Some((key, _)) = identity(value) {
            self.own_object(owner, key, value)?;
        }
        Some(())
    }

    fn needs_scope(&mut self, owner: usize, id: u16) -> Option<()> {
        if let Some(i) = id.checked_sub(self.base).and_then(|n| n.checked_sub(1)) {
            if (i as usize) < self.modules {
                push(&mut self.nodes[owner].edges, i as usize)?;
            }
        }
        Some(())
    }

    // Trace shared containers once, including dictionary cycles. A host may
    // have a mutable borrow: defer releasing modules until the next sweep.
    fn children(&mut self, id: usize, value: &Value) -> Option<()> {
        match value {
            Value::Dict(d) => {
                let d = d.try_borrow().ok()?;
                for v in d.keys().iter().chain(d.vals()) {
                    self.owns(id, v)?;
                }
            }
            Value::List(list) => {
                for v in list.iter() {
                    self.owns(id, v)?;
                }
            }
            Value::Fn(f) => match &**f {
                Func::Code { tree, .. } => self.needs_scope(id, tree.module.get())?,
                Func::Lazy(lazy) => self.needs_scope(id, lazy.ctx.mid)?,
                Func::Proj { f, args, .. } => {
                    self.owns(id, f)?;
                    for slot in args {
                        match slot {
                            Slot::Val(v) => self.owns(id, v)?,
                            Slot::Code(tree, _) => self.needs_scope(id, tree.module.get())?,
                            Slot::Hole => {}
                        }
                    }
                }
                #[cfg(feature = "host")]
                Func::Native { captures: Some(values), .. } => {
                    for v in values {
                        self.owns(id, v)?;
                    }
                }
                Func::Native { .. } => {}
            },
            _ => {}
        }
        Some(())
    }
}

pub(super) fn live_modules(scopes: &[ModScope], base: u16) -> Option<Vec<bool>> {
    let n = scopes.len();
    let mut nodes = Vec::new();
    nodes.try_reserve_exact(n).ok()?;
    nodes.resize_with(n, Node::default);
    let mut graph = Graph { nodes, objects: Vec::new(), modules: n, base };
    for (i, scope) in scopes.iter().enumerate() {
        for (_, v) in &scope.vars {
            graph.owns(i, v)?;
        }
        if let Some(export @ Value::Dict(d)) = &scope.export {
            let namespace = graph.own_object(i, Rc::as_ptr(d) as usize, export)?;
            // Holding the namespace or any directly exported function retains
            // its module, even if the function is a native callable.
            push(&mut graph.nodes[namespace].edges, i)?;
            for v in d.try_borrow().ok()?.vals() {
                if let Value::Fn(f) = v {
                    let function = graph.object(Rc::as_ptr(f) as usize, v)?;
                    push(&mut graph.nodes[function].edges, i)?;
                }
            }
        }
        for weak in &scope.projections {
            if let Some(f) = weak.upgrade() {
                graph.object(Rc::as_ptr(&f) as usize, &Value::Fn(f))?;
            }
        }
    }
    let mut i = n;
    while i < graph.nodes.len() {
        if let Some(v) = graph.nodes[i].value.clone() {
            graph.children(i, &v)?;
        }
        i += 1;
    }

    let mut live = Vec::new();
    live.try_reserve_exact(graph.nodes.len()).ok()?;
    live.resize(graph.nodes.len(), false);
    let mut work = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate().skip(n) {
        if let Some((_, count)) = node.value.as_ref().and_then(identity) {
            // This graph itself holds exactly one clone of each object.
            if count > node.incoming + 1 {
                live[i] = true;
                push(&mut work, i)?;
            }
        }
    }
    while let Some(i) = work.pop() {
        for &j in &graph.nodes[i].edges {
            if !live[j] {
                live[j] = true;
                push(&mut work, j)?;
            }
        }
    }
    live.truncate(n);
    Some(live)
}
