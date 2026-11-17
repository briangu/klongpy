//! The parse tree: one arena per parsed source (`Vec<Node>` indexed by `NodeId`, with a
//! parallel `Vec<Span>`).
//!
//! The node kinds follow KlongPy's tree classes, because the canonical dump
//! (`conformance/README.md`) and the calling rules are defined on them:
//!
//! | here | KlongPy |
//! |---|---|
//! | `Lit` | a number, char, string or data list |
//! | `Sym`, `QSym` | `KGSym`, `KGQSym` (x, y, z are `Sym` 0, 1, 2) |
//! | `OpVal` | a bare `KGOp` |
//! | `Monad`, `Dyad` | `KGFn(KGOp, ...)` |
//! | `Fn` | `KGFn` / `KGCall` whose callee is not an operator: function literal, call, projection, infix call |
//! | `Adv` | `KGCall([KGAdverb...], None)` |
//! | `Cond`, `EList`, `EDict` | `KGCond`, `KGExprArray`, `KGExprDict` |
//! | `Seq`, `Body` | a statement list, `KGBody` |
//! | `DictLit`, `CopyList` | `KGCall(copy_lambda, ...)` |
use alloc::rc::Rc;
use alloc::vec::Vec;

use crate::error::{Source, Span};
use crate::sym::SymId;
use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(pub u32);

/// An omitted argument or operand (Python `None`).
pub const NONE: NodeId = NodeId(u32::MAX);
/// Parser-internal: a `;` (or a newline acting as one) where an expression was read.
pub const SEMI: NodeId = NodeId(u32::MAX - 1);

/// A run of `u32` (node ids or symbol ids) in `Tree::extra`.
#[derive(Clone, Copy, Default)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

pub enum Node {
    Lit(Value),
    Sym(SymId),
    QSym(SymId),
    OpVal { op: SymId, arity: u8 },
    Monad { op: SymId, arg: NodeId, missing: bool },
    Dyad { op: SymId, l: NodeId, r: NodeId, missing: bool },
    /// `a` is the callee: a `Sym`, another `Fn`, or a function body (one statement, a `Seq` or
    /// a `Body`). `args` are argument expressions, `NONE` for a hole.
    /// `literal`: the call of a function literal (`{x}(1)`), whose callee is a body and not a name to look up.
    Fn { a: NodeId, args: Range, has_args: bool, arity: u8, call: bool, literal: bool },
    /// `advs` holds the adverb glyphs; `first_arity` is the arity the first adverb was read
    /// with (1 after a factor, 2 between operands), the others have arity 1. A monadic chain
    /// has its operand in `r`.
    Adv { verb: NodeId, verb_arity: u8, advs: Range, first_arity: u8, dyad: bool, l: NodeId, r: NodeId },
    Cond(NodeId, NodeId, NodeId),
    EList(Range),
    EDict(Range),
    /// A dictionary literal: copied each time it is evaluated.
    DictLit(Value),
    /// A list literal that holds a dictionary literal: copied each time it is evaluated.
    CopyList(Value),
    Seq(Range),
    Body { stmts: Range, locals: Range, params: Range, has_params: bool },
    /// The left side of `[a b c]::v`.
    Destructure(Range),
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub spans: Vec<Span>,
    pub extra: Vec<u32>,
    pub source: Option<Rc<Source>>,
    /// Set by a load: the function literals of this tree are kept as text (`Func::Lazy`), parsed in this context when called.
    pub lazy: Option<Rc<crate::value::LazyCtx>>,
    /// The arena could not grow: the tree is incomplete and the parse is a `memory` error.
    pub oom: bool,
}

impl Tree {
    pub fn new(source: Option<Rc<Source>>) -> Tree {
        Tree { nodes: Vec::new(), spans: Vec::new(), extra: Vec::new(), source, lazy: None, oom: false }
    }
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0 as usize]
    }
    pub fn span(&self, id: NodeId) -> Option<Span> {
        match self.spans.get(id.0 as usize) {
            Some(&(0, 0)) | None => None,
            Some(&s) => Some(s),
        }
    }
    pub fn add(&mut self, n: Node, span: Span) -> NodeId {
        // Growth cannot be allowed to abort, and a hundred call sites cannot each handle a
        // failure: the arena records it and the parser reports it (`parse::parse`). Node 0
        // exists by then (`reserve`), so the id returned is valid.
        if self.nodes.try_reserve(1).is_err() || self.spans.try_reserve(1).is_err() {
            self.oom = true;
            return NodeId(0);
        }
        self.nodes.push(n);
        self.spans.push(span);
        NodeId(self.nodes.len() as u32 - 1)
    }
    /// Give back the room the arenas grew beyond their nodes (a tree that is kept, as a function's body is, keeps no slack).
    pub fn shrink(&mut self) {
        self.nodes.shrink_to_fit();
        self.spans.shrink_to_fit();
        self.extra.shrink_to_fit();
    }
    /// Room for the first nodes; false when there is no memory for them.
    pub fn reserve(&mut self) -> bool {
        self.nodes.try_reserve(8).is_ok() && self.spans.try_reserve(8).is_ok() && {
            self.nodes.push(Node::Lit(crate::value::Value::Undefined));
            self.spans.push((0, 0));
            true
        }
    }
    pub fn range(&mut self, ids: &[u32]) -> Range {
        let start = self.extra.len() as u32;
        if self.extra.try_reserve(ids.len()).is_err() {
            self.oom = true;
            return Range { start: 0, len: 0 };
        }
        self.extra.extend_from_slice(ids);
        Range { start, len: ids.len() as u32 }
    }
    pub fn ids(&self, r: Range) -> &[u32] {
        &self.extra[r.start as usize..(r.start + r.len) as usize]
    }
    pub fn is_real(id: NodeId) -> bool {
        id != NONE && id != SEMI
    }
    /// KlongPy `isinstance(x, KGFn)`.
    pub fn is_kgfn(&self, id: NodeId) -> bool {
        Tree::is_real(id)
            && matches!(
                self.node(id),
                Node::Monad { .. } | Node::Dyad { .. } | Node::Fn { .. } | Node::Adv { .. } | Node::DictLit(_) | Node::CopyList(_)
            )
    }
    /// A `KGFn` that is neither an operator application nor an adverb chain.
    pub fn is_plain_fn(&self, id: NodeId) -> bool {
        Tree::is_real(id) && matches!(self.node(id), Node::Fn { .. })
    }
    /// A statement list (KlongPy: a Python `list` or one of its subclasses).
    pub fn is_seq(&self, id: NodeId) -> bool {
        Tree::is_real(id) && matches!(self.node(id), Node::Seq(_) | Node::Body { .. } | Node::Cond(..) | Node::EList(_) | Node::EDict(_))
    }
}
