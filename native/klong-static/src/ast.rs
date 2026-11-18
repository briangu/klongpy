//! The tree of `klong-core` as the checker reads it: owned nodes with the names as text and the spans the Python reader attached
//! (`span`, `op_span`, `arg_spans`), so that every rule can be written against what it points at.
use klong_core::sym::Interner;
use std::rc::Rc;
use klong_core::tree::{Aux, Node as T, NodeId, Tree, NONE, SEMI};
use klong_core::value::Value;

pub type Sp = (usize, usize);

#[derive(Clone, Debug)]
pub struct Node {
    pub k: Kind,
    /// Where the reader says the node is (calls, operations, conditionals, literal lists and dictionaries, function literals): `None` for
    /// a number, a name, a string, a data list.
    pub span: Option<Sp>,
    pub op_span: Option<Sp>,
    pub arg_spans: Vec<Option<Sp>>,
}

#[derive(Clone, Debug)]
pub enum Bad {
    Sym(String),
    Char,
    Str(String),
    Dict,
    Op(String),
    Ragged,
}

#[derive(Clone, Debug)]
pub enum Scalar {
    Int(i128),
    Flt(f64),
    Sym(String),
    Char,
    Str(String),
    Op(String),
    Dict,
}

#[derive(Clone, Debug)]
pub enum Lit {
    /// A rectangular list of numbers: its shape (an empty list is `[0]`), whether any is a float, the values flattened.
    Num { shape: Vec<usize>, float: bool, ints: Vec<i128>, flts: Vec<f64> },
    /// One element of an object array that is not a list.
    Scalar(Scalar),
    /// An object array in NumPy's terms: the first thing in it that is not a number, and its rows (`value[i]`).
    Bad(Bad, Vec<Lit>),
}

#[derive(Clone, Debug)]
pub enum Kind {
    Missing,
    Sym(String),
    QSym(String),
    Int(i128),
    Flt(f64),
    Char,
    Str(String),
    /// A data list
    Arr(Lit),
    /// A bare operator
    Op(String),
    Monad(String, Rc<Node>),
    Dyad(String, Rc<Node>, Rc<Node>),
    /// An operator with a number of operands other than one or two
    OpN(String, usize),
    /// A call, a projection or an infix call; `kcall`: no argument is left out
    Call { callee: Rc<Node>, args: Vec<Rc<Node>>, kcall: bool },
    /// A function literal
    FnLit { stmts: Vec<Rc<Node>>, spans: Vec<Option<Sp>>, params: Vec<String>, locals: Vec<String> },
    /// The body of a function literal called in place
    Body(Vec<Rc<Node>>),
    /// An adverb chain: the verb, the adverb glyphs, and the operand (`l` for a dyadic one)
    Chain { verb: Rc<Node>, advs: Vec<String>, dyadic: bool, l: Rc<Node>, r: Rc<Node> },
    Cond(Vec<Rc<Node>>),
    EList(Vec<Rc<Node>>),
    EDict(Vec<Rc<Node>>),
    Destructure(Vec<String>),
    /// A dictionary literal, or a list that holds one (copied when it is evaluated)
    DictLit(bool),
}

impl Node {
    fn new(k: Kind) -> Node {
        Node { k, span: None, op_span: None, arg_spans: Vec::new() }
    }
    pub fn missing() -> Node {
        Node::new(Kind::Missing)
    }
    pub fn is_missing(&self) -> bool {
        matches!(self.k, Kind::Missing)
    }
    /// Python `_arg_span(n, i, default)`.
    pub fn arg_span(&self, i: usize, default: Sp) -> Sp {
        match self.arg_spans.get(i) {
            Some(Some(s)) => *s,
            _ => default,
        }
    }
    /// `_is_fn_literal`
    pub fn is_fn_literal(&self) -> bool {
        matches!(self.k, Kind::FnLit { .. })
    }
    pub fn name(&self) -> Option<&str> {
        match &self.k {
            Kind::Sym(s) => Some(s),
            _ => None,
        }
    }
}

pub struct Conv<'a> {
    pub tree: &'a Tree,
    pub syms: &'a Interner,
}

fn sp(s: (u32, u32)) -> Sp {
    (s.0 as usize, s.1 as usize)
}
fn osp(s: (u32, u32)) -> Option<Sp> {
    if s == (0, 0) {
        None
    } else {
        Some(sp(s))
    }
}

impl<'a> Conv<'a> {
    fn aux(&self) -> Option<&Aux> {
        self.tree.aux.as_deref()
    }
    fn name(&self, s: klong_core::sym::SymId) -> String {
        self.syms.name(s).to_string()
    }

    pub fn conv(&self, id: NodeId) -> Node {
        if id == NONE {
            return Node::missing();
        }
        if id == SEMI {
            return Node::new(Kind::Str(";".to_string()));
        }
        let t = self.tree;
        let span = t.span(id).map(sp);
        let args_of = |i: NodeId| -> Vec<Option<Sp>> {
            self.aux().and_then(|a| a.args.get(&i.0)).map(|v| v.iter().map(|s| osp(*s)).collect()).unwrap_or_default()
        };
        let op_of = |i: NodeId| -> Option<Sp> { self.aux().and_then(|a| a.ops.get(&i.0)).map(|s| sp(*s)) };
        match t.node(id) {
            T::Lit(v) => Node::new(self.value(v)),
            T::Sym(s) => Node::new(Kind::Sym(self.name(*s))),
            T::QSym(s) => Node::new(Kind::QSym(self.name(*s))),
            T::OpVal { op, .. } => Node::new(Kind::Op(self.name(*op))),
            T::Monad { op, arg, .. } => {
                let a = Rc::new(self.conv(*arg));
                let glyph = self.name(*op);
                // (an evaluated dictionary literal is a list of expressions and would be taken for the argument list)
                let k = if matches!(a.k, Kind::EDict(_)) { Kind::OpN(glyph, 1) } else { Kind::Monad(glyph, a) };
                Node { k, span, op_span: op_of(id), arg_spans: args_of(id) }
            }
            T::Dyad { op, l, r, .. } => {
                let (l, r) = (Rc::new(self.conv(*l)), Rc::new(self.conv(*r)));
                Node { k: Kind::Dyad(self.name(*op), l, r), span, op_span: op_of(id), arg_spans: args_of(id) }
            }
            T::Fn { a, args, has_args, arity, call, .. } => {
                if !*has_args {
                    return self.fn_literal(id, *a, *arity, span);
                }
                let callee = match t.node(*a) {
                    T::Body { .. } | T::Seq(_) => Node::new(Kind::Body(self.body_stmts(*a))),
                    _ => self.conv(*a),
                };
                let args: Vec<Rc<Node>> = t.ids(*args).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect();
                Node { k: Kind::Call { callee: Rc::new(callee), args, kcall: *call }, span, op_span: op_of(id), arg_spans: args_of(id) }
            }
            T::Adv { verb, advs, dyad, l, r, .. } => {
                let advs: Vec<String> = t.ids(*advs).iter().map(|&q| self.name(klong_core::sym::SymId(q))).collect();
                let (lc, rc) = (Rc::new(self.conv(*l)), Rc::new(self.conv(*r)));
                let verb = Rc::new(self.conv(*verb));
                Node {
                    k: Kind::Chain { verb, advs, dyadic: *dyad, l: lc, r: rc },
                    span,
                    op_span: op_of(id),
                    arg_spans: args_of(id),
                }
            }
            T::Cond(c, a, b) => {
                let v = vec![Rc::new(self.conv(*c)), Rc::new(self.conv(*a)), Rc::new(self.conv(*b))];
                Node { k: Kind::Cond(v), span, op_span: None, arg_spans: args_of(id) }
            }
            T::EList(r) => {
                let v = t.ids(*r).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect();
                Node { k: Kind::EList(v), span, op_span: None, arg_spans: args_of(id) }
            }
            T::EDict(r) => {
                let v = t.ids(*r).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect();
                Node { k: Kind::EDict(v), span, op_span: None, arg_spans: args_of(id) }
            }
            T::DictLit(_) => Node { k: Kind::DictLit(true), span, op_span: None, arg_spans: Vec::new() },
            T::CopyList(_) => Node { k: Kind::DictLit(false), span, op_span: None, arg_spans: Vec::new() },
            T::Destructure(r) => {
                let names = t.ids(*r).iter().map(|&q| self.name(klong_core::sym::SymId(q))).collect();
                Node::new(Kind::Destructure(names))
            }
            T::Seq(_) | T::Body { .. } => Node::new(Kind::Body(self.body_stmts(id))),
        }
    }

    /// The highest of x=1 y=2 z=3 mentioned in the body, a nested function literal and data left out (KlongPy `_scan_args`).
    fn max_arg(&self, id: NodeId) -> u8 {
        if !Tree::is_real(id) {
            return 0;
        }
        let each = |r: klong_core::tree::Range| self.tree.ids(r).iter().map(|&q| self.max_arg(NodeId(q))).max().unwrap_or(0);
        match self.tree.node(id) {
            T::Sym(s) | T::QSym(s) => match self.syms.name(*s) {
                "x" => 1,
                "y" => 2,
                "z" => 3,
                _ => 0,
            },
            T::Fn { a, args, has_args, .. } => {
                let mut m = if *has_args && matches!(self.tree.node(*a), T::Sym(_) | T::QSym(_)) { self.max_arg(*a) } else { 0 };
                if *has_args {
                    m = m.max(each(*args));
                }
                m
            }
            T::Monad { arg, .. } => self.max_arg(*arg),
            T::Dyad { l, r, .. } => self.max_arg(*l).max(self.max_arg(*r)),
            T::Adv { verb, l, r, .. } => self.max_arg(*verb).max(self.max_arg(*l)).max(self.max_arg(*r)),
            T::Cond(c, a, b) => self.max_arg(*c).max(self.max_arg(*a)).max(self.max_arg(*b)),
            T::EList(r) | T::EDict(r) | T::Seq(r) => each(*r),
            T::Body { stmts, .. } => each(*stmts),
            _ => 0,
        }
    }

    fn body_stmts(&self, id: NodeId) -> Vec<Rc<Node>> {
        match self.tree.node(id) {
            T::Body { stmts, .. } => self.tree.ids(*stmts).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect(),
            T::Seq(r) => self.tree.ids(*r).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect(),
            _ => vec![Rc::new(self.conv(id))],
        }
    }

    fn fn_literal(&self, id: NodeId, body: NodeId, _arity: u8, span: Option<Sp>) -> Node {
        let t = self.tree;
        // the arity of a literal is the highest of x, y, z its body mentions (KlongPy `max_arg`), not what a body that is itself a function
        // value (`{g(1;)}`) takes
        let arity = self.max_arg(body);
        let (stmts, params, locals) = match t.node(body) {
            T::Body { stmts, locals, params, has_params } => {
                let stmts = t.ids(*stmts).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect();
                let locals = t.ids(*locals).iter().map(|&q| self.name(klong_core::sym::SymId(q))).collect();
                let params = if *has_params {
                    t.ids(*params).iter().map(|&q| self.name(klong_core::sym::SymId(q))).collect()
                } else {
                    ["x", "y", "z"][..(arity as usize).min(3)].iter().map(|s| s.to_string()).collect()
                };
                (stmts, params, locals)
            }
            T::Seq(r) => (t.ids(*r).iter().map(|&q| Rc::new(self.conv(NodeId(q)))).collect(), ["x", "y", "z"][..(arity as usize).min(3)].iter().map(|s| s.to_string()).collect(), Vec::new()),
            _ => (vec![Rc::new(self.conv(body))], ["x", "y", "z"][..(arity as usize).min(3)].iter().map(|s| s.to_string()).collect(), Vec::new()),
        };
        let mut spans: Vec<Option<Sp>> = self.aux().and_then(|a| a.stmts.get(&id.0)).map(|v| v.iter().map(|s| osp(*s)).collect()).unwrap_or_default();
        spans.resize(stmts.len(), None);
        Node { k: Kind::FnLit { stmts, spans, params, locals }, span, op_span: None, arg_spans: Vec::new() }
    }

    /// A data value in the tree.
    fn value(&self, v: &Value) -> Kind {
        match v {
            Value::Int(i) => Kind::Int(*i as i128),
            Value::Flt(f) => Kind::Flt(*f as f64),
            Value::Char(_) => Kind::Char,
            Value::Str(s) => Kind::Str(s.as_str().to_string()),
            Value::Big(b) => Kind::Int(big_i128(b)),
            Value::Sym(s) => Kind::QSym(self.name(*s)),
            Value::Ints(_) | Value::Flts(_) | Value::List(_) => Kind::Arr(self.literal(v)),
            Value::Prim(s) => Kind::Op(self.name(*s)),
            _ => Kind::Arr(Lit::Scalar(Scalar::Dict)),
        }
    }

    fn scalar(&self, v: &Value) -> Scalar {
        match v {
            Value::Int(i) => Scalar::Int(*i as i128),
            Value::Flt(f) => Scalar::Flt(*f as f64),
            Value::Big(b) => Scalar::Int(big_i128(b)),
            Value::Sym(s) => Scalar::Sym(self.name(*s)),
            Value::Char(_) => Scalar::Char,
            Value::Str(s) => Scalar::Str(s.as_str().to_string()),
            Value::Prim(s) => Scalar::Op(self.name(*s)),
            _ => Scalar::Dict,
        }
    }

    /// The array a list is as KlongPy builds it (`kg_asarray`, bottom up): a list of equal-shaped arrays is one array of one more
    /// dimension, a list of anything else is a one-dimensional object array.
    fn arr(&self, v: &Value) -> Arr {
        let Ok(items) = v.items() else { return Arr::Ragged(Vec::new()) };
        if items.is_empty() {
            return Arr::Num { shape: vec![0], float: false, ints: Vec::new(), flts: Vec::new() };
        }
        let els: Vec<El> = items
            .iter()
            .map(|it| match it {
                Value::Ints(_) | Value::Flts(_) | Value::List(_) => El::Arr(self.arr(it)),
                other => El::Leaf(self.scalar(other)),
            })
            .collect();
        // every element a leaf
        if els.iter().all(|e| matches!(e, El::Leaf(_))) {
            let leaves: Vec<Scalar> = els.into_iter().map(|e| if let El::Leaf(s) = e { s } else { unreachable!() }).collect();
            return Arr::from_leaves(vec![leaves.len()], leaves);
        }
        // every element an array of one shape
        if els.iter().all(|e| matches!(e, El::Arr(a) if a.shape().is_some())) {
            let first = if let El::Arr(a) = &els[0] { a.shape().unwrap() } else { unreachable!() };
            if els.iter().all(|e| matches!(e, El::Arr(a) if a.shape().as_ref() == Some(&first))) {
                let mut shape = vec![els.len()];
                shape.extend(first);
                let mut leaves = Vec::new();
                for e in els {
                    if let El::Arr(a) = e {
                        leaves.extend(a.leaves());
                    }
                }
                return Arr::from_leaves(shape, leaves);
            }
        }
        Arr::Ragged(els)
    }

    pub fn literal(&self, v: &Value) -> Lit {
        self.arr(v).lit()
    }
}

fn big_i128(b: &impl ToString) -> i128 {
    let t = b.to_string();
    t.parse::<i128>().unwrap_or(if t.starts_with('-') { i128::MIN } else { i128::MAX })
}

enum El {
    Leaf(Scalar),
    Arr(Arr),
}

/// An array as NumPy has it: numbers of one dtype, an N-dimensional object array, or a one-dimensional object array of arrays and scalars.
enum Arr {
    Num { shape: Vec<usize>, float: bool, ints: Vec<i128>, flts: Vec<f64> },
    Obj { shape: Vec<usize>, leaves: Vec<Scalar> },
    Ragged(Vec<El>),
}

fn is_num(s: &Scalar) -> bool {
    matches!(s, Scalar::Int(_) | Scalar::Flt(_))
}

fn bad_of(s: &Scalar) -> Bad {
    match s {
        Scalar::Sym(n) => Bad::Sym(n.clone()),
        Scalar::Char => Bad::Char,
        Scalar::Str(t) => Bad::Str(t.clone()),
        Scalar::Op(g) => Bad::Op(g.clone()),
        _ => Bad::Dict,
    }
}

impl Arr {
    fn from_leaves(shape: Vec<usize>, leaves: Vec<Scalar>) -> Arr {
        if leaves.iter().all(is_num) {
            let float = leaves.iter().any(|l| matches!(l, Scalar::Flt(_)));
            let (mut ints, mut flts) = (Vec::new(), Vec::new());
            for l in &leaves {
                match l {
                    Scalar::Int(i) => {
                        ints.push(*i);
                        flts.push(*i as f64);
                    }
                    Scalar::Flt(f) => flts.push(*f),
                    _ => {}
                }
            }
            if float {
                ints.clear();
            }
            Arr::Num { shape, float, ints, flts }
        } else {
            Arr::Obj { shape, leaves }
        }
    }

    /// The shape of an array that is a stack (not a one-dimensional object array of mixed elements).
    fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Arr::Num { shape, .. } | Arr::Obj { shape, .. } => Some(shape.clone()),
            Arr::Ragged(_) => None,
        }
    }

    fn leaves(&self) -> Vec<Scalar> {
        match self {
            Arr::Num { float, ints, flts, .. } => {
                if *float {
                    flts.iter().map(|f| Scalar::Flt(*f)).collect()
                } else {
                    ints.iter().map(|i| Scalar::Int(*i)).collect()
                }
            }
            Arr::Obj { leaves, .. } => leaves.clone(),
            Arr::Ragged(_) => Vec::new(),
        }
    }

    /// Python `_first_non_number` over this object array.
    fn first_bad(&self) -> Option<Bad> {
        match self {
            Arr::Num { .. } => None,
            Arr::Obj { leaves, .. } => leaves.iter().find(|l| !is_num(l)).map(bad_of),
            Arr::Ragged(els) => {
                for e in els {
                    match e {
                        El::Leaf(s) if is_num(s) => continue,
                        El::Leaf(s) => return Some(bad_of(s)),
                        El::Arr(a) => {
                            return match a {
                                Arr::Num { .. } => Some(Bad::Ragged),
                                other => match other.first_bad() {
                                    Some(Bad::Ragged) | None => Some(Bad::Ragged),
                                    Some(b) => Some(b),
                                },
                            };
                        }
                    }
                }
                None
            }
        }
    }

    fn lit(&self) -> Lit {
        match self {
            Arr::Num { shape, float, ints, flts } => Lit::Num { shape: shape.clone(), float: *float, ints: ints.clone(), flts: flts.clone() },
            Arr::Obj { shape, leaves } => {
                let bad = self.first_bad().unwrap_or(Bad::Ragged);
                // value[i]: a leaf of a one-dimensional array, else the array of the leaves of that row
                let n = shape[0];
                let per: usize = shape[1..].iter().product::<usize>().max(1);
                let rows: Vec<Lit> = (0..n)
                    .map(|i| {
                        if shape.len() == 1 {
                            Lit::Scalar(leaves[i].clone())
                        } else {
                            Arr::from_leaves(shape[1..].to_vec(), leaves[i * per..(i + 1) * per].to_vec()).lit()
                        }
                    })
                    .collect();
                Lit::Bad(bad, rows)
            }
            Arr::Ragged(els) => {
                let bad = self.first_bad().unwrap_or(Bad::Ragged);
                let rows: Vec<Lit> = els
                    .iter()
                    .map(|e| match e {
                        El::Leaf(s) => Lit::Scalar(s.clone()),
                        El::Arr(a) => a.lit(),
                    })
                    .collect();
                Lit::Bad(bad, rows)
            }
        }
    }
}
