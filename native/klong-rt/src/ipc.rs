//! The IPC endpoint (feature `ipc`): `.cli .clid .clic .clit .srv .async` and the hooks
//! `.srv.o .srv.c .srv.e .srv.auth`, over the kdb+ protocol of `klong-ipc` and whatever
//! transports the platform has. The native twin of `klongpy/ipc/endpoint.py` and
//! `klongpy/sys_fn_ipc.py`; design `docs/ipc_capabilities.md`.
//!
//! One task runs everything. A client call sends its request and then loops: move bytes,
//! take the next event of that connection, serve it if it is a request (that is the nested
//! serving that makes a callback from the server work), until its own response arrives, its
//! deadline passes or the console sends Ctrl-C. Requests that arrive while nothing waits are
//! served from `Runtime::poll_events`, one per connection per pass, in arrival order.
//!
//! Trust: a connected peer can evaluate any Klong on this node. Listen only on a network
//! where every peer is trusted; there is no encryption, and credentials travel in clear.
use alloc::boxed::Box;
use alloc::rc::{Rc, Weak};
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use klong_core::error::{Kind as ErrKind, Result};
use klong_core::fmt::format_value;
use klong_core::tree::{Node, NodeId, Tree};
use klong_core::value::{try_push, try_str, try_vec, Ext, ExtBox, Func, NativeFn, Value};
use klong_core::{Error, Interp, SymId};
use klong_ipc::codec::{ext_ref, KdbError, KdbLambda};
use klong_ipc::rel::{self, Rel};
use klong_ipc::session::{self, Event, Kind, Pending, PendingId, Reason, Session};

use crate::events::{Events, Work};
use crate::link::{Link, LinkError, ListenerId, PacketLink, StreamLink};
use crate::{one, Platform};

pub struct IpcConfig {
    /// Largest message accepted, bytes. A device sets it from its heap: decoding can take
    /// several times the message (up to about 70 bytes per byte for a dictionary of bytes).
    pub max_message: usize,
    /// Deadline of a call on a new connection, microseconds; `None` for no limit (`.clit`).
    pub call_timeout_us: Option<u64>,
    pub handshake_timeout_us: u64,
    /// Retry budget of the packet reliability layer (0: fail fast).
    pub retries: u8,
    /// Keepalive of the packet reliability layer, milliseconds; 0 turns it off.
    pub link_idle_ms: u32,
    /// Compress what is sent when it is worth it (feature `compress`).
    pub compress: bool,
    /// Bytes read from a link at a time.
    pub read_chunk: usize,
}

impl Default for IpcConfig {
    fn default() -> IpcConfig {
        IpcConfig { max_message: session::MAX_MESSAGE, call_timeout_us: Some(30_000_000), handshake_timeout_us: 5_000_000, retries: 8, link_idle_ms: 0, compress: false, read_chunk: if cfg!(target_pointer_width = "64") { 65536 } else { 1024 } }
    }
}

struct RelLink {
    link: Box<dyn PacketLink>,
    rel: Rel,
    /// A packet the link was too busy to take.
    stuck: Vec<u8>,
    buf: Vec<u8>,
}

enum ConnLink {
    Stream(Box<dyn StreamLink>),
    Rel(RelLink),
}

struct AsyncReq {
    id: PendingId,
    cb: Value,
    lookup: Option<SymId>,
}

/// One connection, either side.
pub struct Conn {
    link: RefCell<Option<ConnLink>>,
    sess: RefCell<Session>,
    /// Framed bytes the link has not taken yet, and how many of them it has.
    out: RefCell<(Vec<u8>, usize)>,
    peer: String,
    server_side: bool,
    ready: Cell<bool>,
    /// `.srv.o` was called for it.
    opened: Cell<bool>,
    hs_deadline: Cell<u64>,
    timeout_us: Cell<Option<u64>>,
    /// Responses that arrived and were not yet taken by their callers.
    results: RefCell<Vec<(PendingId, Result<Value>)>>,
    asyncs: RefCell<Vec<AsyncReq>>,
    /// Why the connection is over, once it is.
    down: RefCell<Option<String>>,
    /// The function handle of this connection (kept so that it is made once).
    handle: RefCell<Option<Value>>,
}

impl Conn {
    fn is_down(&self) -> bool {
        self.down.try_borrow().map(|d| d.is_some()).unwrap_or(true)
    }

    fn down_error(&self) -> Error {
        let text = self.down.try_borrow().ok().and_then(|d| d.clone()).unwrap_or_default();
        Error::system(text)
    }

    fn queue(&self, bytes: &[u8]) -> Result<()> {
        let mut out = self.out.try_borrow_mut().map_err(|_| busy())?;
        if out.1 > 0 && out.1 == out.0.len() {
            out.0.clear();
            out.1 = 0;
        }
        let _tag = klong_core::htag::Scope::new(klong_core::htag::IPC);
        out.0.try_reserve(bytes.len()).map_err(|_| Error::memory())?;
        out.0.extend_from_slice(bytes);
        Ok(())
    }

    fn unsent(&self) -> bool {
        self.out.try_borrow().map(|o| o.1 < o.0.len()).unwrap_or(false)
    }

    fn take_result(&self, id: PendingId) -> Option<Result<Value>> {
        let mut r = self.results.try_borrow_mut().ok()?;
        let i = r.iter().position(|(p, _)| *p == id)?;
        Some(r.remove(i).1)
    }
}

struct Listener {
    transport: usize,
    id: ListenerId,
}

struct HandleEntry {
    f: Weak<Func>,
    /// Weak: the function's closure holds the connection while the function lives, and an entry that held it too kept a
    /// closed connection (its buffers, an echo's 4 KB) until the next entry pruned this one.
    conn: Weak<Conn>,
    /// The global of the peer a proxy calls; `None` for the handle of the connection.
    sym: Option<SymId>,
}

pub struct Ipc {
    pub cfg: RefCell<IpcConfig>,
    conns: RefCell<Vec<Rc<Conn>>>,
    listener: RefCell<Option<Listener>>,
    /// The function values that stand for connections (handles and remote function proxies),
    /// so that `.clic`, `.clit`, `.clid` and `.async` know them.
    handles: RefCell<Vec<HandleEntry>>,
    scratch: RefCell<Vec<u8>>,
}

impl Ipc {
    pub(crate) fn new() -> Ipc {
        Ipc { cfg: RefCell::new(IpcConfig::default()), conns: RefCell::new(Vec::new()), listener: RefCell::new(None), handles: RefCell::new(Vec::new()), scratch: RefCell::new(Vec::new()) }
    }

    pub(crate) fn active(&self) -> bool {
        if self.listener.try_borrow().map(|l| l.is_some()).unwrap_or(true) {
            return true;
        }
        match self.conns.try_borrow() {
            Ok(c) => c.iter().any(|c| !c.is_down() && (c.server_side || c.unsent() || c.sess.try_borrow().map(|s| s.has_pending()).unwrap_or(true))),
            Err(_) => true,
        }
    }

    pub(crate) fn next_deadline(&self) -> Option<u64> {
        let conns = self.conns.try_borrow().ok()?;
        let mut best: Option<u64> = None;
        let mut take = |d: Option<u64>| {
            if let Some(d) = d {
                best = Some(best.map_or(d, |b| b.min(d)));
            }
        };
        for c in conns.iter() {
            if !c.ready.get() {
                take(Some(c.hs_deadline.get()));
            }
            take(c.sess.try_borrow().ok().and_then(|s| s.next_deadline()));
            if let Ok(l) = c.link.try_borrow() {
                if let Some(ConnLink::Rel(r)) = l.as_ref() {
                    take(r.rel.next_wake().map(|ms| ms.saturating_mul(1000)));
                }
            }
        }
        best
    }

    /// Connections that are open (for tests and `.mem`-style inspection).
    pub fn connections(&self) -> usize {
        self.conns.try_borrow().map(|c| c.iter().filter(|c| !c.is_down()).count()).unwrap_or(0)
    }

    /// Bytes this endpoint holds for one connection: buffered input and unsent output.
    pub fn buffered(&self) -> usize {
        self.conns.try_borrow().map(|c| c.iter().map(|c| c.sess.try_borrow().map(|s| s.buffered()).unwrap_or(0) + c.out.try_borrow().map(|o| o.0.len()).unwrap_or(0)).sum()).unwrap_or(0)
    }
}

fn busy() -> Error {
    Error::internal("runtime state in use")
}

enum Pump {
    Idle,
    Moved,
    /// The link is over: why, and whether the peer simply closed.
    Over(String, bool),
}

fn link_over(e: LinkError, peer: &str) -> Pump {
    match e {
        LinkError::Closed => Pump::Over(klong_core::msg!("connection closed by peer: {}", peer), true),
        other => Pump::Over(klong_core::msg!("connection lost ({}): {}", other.text(), peer), false),
    }
}

/// Wire name of an error kind (`docs/errors.md`, as `klongpy/errors.py`).
fn wire_name(k: ErrKind) -> &'static str {
    match k {
        ErrKind::Syntax => "parse",
        ErrKind::Name => "name",
        ErrKind::Arity => "rank",
        ErrKind::Type => "type",
        ErrKind::Shape => "length",
        ErrKind::Range => "index",
        ErrKind::System => "os",
        ErrKind::Memory => "wsfull",
        ErrKind::Interrupt => "stop",
        ErrKind::Internal => "internal",
        ErrKind::Remote => "",
    }
}

fn kind_of_wire(name: &str) -> Option<ErrKind> {
    Some(match name {
        "parse" => ErrKind::Syntax,
        "name" => ErrKind::Name,
        "rank" => ErrKind::Arity,
        "type" => ErrKind::Type,
        "length" => ErrKind::Shape,
        "index" => ErrKind::Range,
        "os" => ErrKind::System,
        "wsfull" => ErrKind::Memory,
        "stop" => ErrKind::Interrupt,
        "internal" => ErrKind::Internal,
        _ => return None,
    })
}

/// `<wire name>: <message>` for an error raised while serving a request: no position.
fn wire_error_text(e: &Error) -> String {
    match e.kind {
        // an error of the peer's peer goes on as it came
        ErrKind::Remote => klong_core::msg!("{}", e.message),
        k => klong_core::msg!("{}: {}", wire_name(k), e.message),
    }
}

/// The error a peer's error object stands for: the kind its text names, else `remote`.
fn remote_error(text: &[u8], peer: &str) -> Error {
    let text = String::from_utf8_lossy(text);
    let e = match text.split_once(": ") {
        Some((head, rest)) => match kind_of_wire(head) {
            Some(k) => Error::new(k, klong_core::msg!("{}", rest)),
            None => Error::new(ErrKind::Remote, klong_core::msg!("{}", text)),
        },
        None => Error::new(ErrKind::Remote, klong_core::msg!("{}", text)),
    };
    e.with_hint(klong_core::msg!("reported by {}", peer))
}

/// Arity of a display-form lambda text (`:monad`, `:fn/5`).
fn display_arity(text: &str) -> Option<u8> {
    match text {
        ":nilad" => Some(0),
        ":monad" => Some(1),
        ":dyad" => Some(2),
        ":triad" => Some(3),
        _ => text.strip_prefix(":fn/").and_then(|n| n.parse().ok()),
    }
}

/// The function a source text defines: the text is parsed (nothing runs) and must be exactly
/// one function literal, or the definition `fn(a;b)::{...}` of one with named parameters
/// (`fn` is a placeholder: the function is kept and nothing is defined).
fn parse_function(ip: &mut Interp, text: &str) -> Option<Value> {
    let p = ip.parse(text, "<ipc>").ok()?;
    if p.stmts.len() != 1 {
        return None;
    }
    let literal = |id: NodeId| matches!(p.tree.node(id), Node::Fn { has_args: false, call: false, .. });
    let id = p.stmts[0];
    let node = match p.tree.node(id) {
        _ if literal(id) => id,
        Node::Dyad { op, l, r, missing: false } if ip.syms.name(*op) == "::" => named_literal(ip, &p.tree, *l, *r)?,
        Node::Fn { has_args: true, call: false, arity: 2, args, .. } => {
            let ids = p.tree.ids(*args);
            named_literal(ip, &p.tree, NodeId(*ids.first()?), NodeId(*ids.get(1)?))?
        }
        _ => return None,
    };
    Some(Value::Fn(Rc::new(Func::Code { tree: p.tree.clone(), node })))
}

/// The literal `r` of `fn(a;b)::r` (`l` is the placeholder name), when `r` has named parameters.
fn named_literal(ip: &Interp, tree: &Tree, l: NodeId, r: NodeId) -> Option<NodeId> {
    let Node::Sym(s) = tree.node(l) else { return None };
    if ip.syms.name(*s).split('`').next() != Some(klong_ipc::codec::WIRE_NAME) || !Tree::is_real(r) {
        return None;
    }
    match tree.node(r) {
        Node::Fn { a, has_args: false, call: false, .. } if Tree::is_real(*a) && matches!(tree.node(*a), Node::Body { has_params: true, .. }) => Some(r),
        _ => None,
    }
}

fn arity_of(v: &Value) -> Option<u8> {
    match v {
        Value::Fn(f) => Some(klong_core::fmt::func_arity(f)),
        Value::Prim(_) => Some(2),
        _ => None,
    }
}

/// A `kg` lambda that arrived as an argument or a value to set: the function its text
/// defines, when it defines one. Anything else stays as it is.
fn local_function(ip: &mut Interp, v: Value) -> Value {
    if let Some(l) = ext_ref::<KdbLambda>(&v) {
        if l.is_klong() {
            if let Some(f) = l.text_str().and_then(|t| if display_arity(t).is_some() { None } else { parse_function(ip, t) }) {
                return f;
            }
        }
    }
    v
}

/// What `.clid` returns: `d?:name` asks the peer for a global, `d,[:name value]` sets one.
pub struct DictHandle<P: Platform + 'static> {
    ev: Weak<Events<P>>,
    conn: Rc<Conn>,
}

impl<P: Platform + 'static> DictHandle<P> {
    fn key(ip: &mut Interp, k: &Value) -> Result<SymId> {
        match k {
            Value::Sym(s) => Ok(*s),
            Value::Str(s) => {
                let id = ip.syms.intern(s.as_str());
                match ip.syms.failure() {
                    Some(e) => Err(e),
                    None => Ok(id),
                }
            }
            other => Err(Error::typ(klong_core::msg!("a remote dictionary key is a symbol, got {}", other.describe()))),
        }
    }
}

impl<P: Platform + 'static> Ext for DictHandle<P> {
    fn type_name(&self) -> &str {
        "clid"
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
    fn find(&self, ip: &mut Interp, y: &Value) -> Option<Result<Value>> {
        let ev = self.ev.upgrade()?;
        Some(Self::key(ip, y).and_then(|k| ev.call(ip, &self.conn, Request::Lookup(k))))
    }
    fn join(&self, ip: &mut Interp, y: &Value) -> Option<Result<Value>> {
        let ev = self.ev.upgrade()?;
        let r = (|| {
            if !y.is_list() || y.len() != Some(2) {
                return Err(Error::typ("a remote dictionary takes a [key value] pair"));
            }
            let k = Self::key(ip, &y.item(0))?;
            let mut items = try_vec(3)?;
            items.push(Value::text("set")?);
            items.push(Value::Sym(k));
            items.push(y.item(1));
            ev.call(ip, &self.conn, Request::Call(items))?;
            Ok(ev.dict_handle(&self.conn))
        })();
        Some(r)
    }
}

enum Request<'a> {
    /// Sent as it is: a string is evaluated by the peer, data comes back unchanged.
    Value(&'a Value),
    /// A symbol atom: the value of that global; a function comes back as a proxy.
    Lookup(SymId),
    /// A call envelope: callee and arguments.
    Call(Vec<Value>),
}

impl<P: Platform + 'static> Events<P> {
    // ----- bytes -------------------------------------------------------------------------

    /// Move bytes both ways on one connection: at most one read.
    fn pump(&self, conn: &Conn) -> Pump {
        let now_ms = self.now() / 1000;
        let (Ok(mut link), Ok(mut out), Ok(mut sess), Ok(mut chunk)) = (conn.link.try_borrow_mut(), conn.out.try_borrow_mut(), conn.sess.try_borrow_mut(), self.ipc.scratch.try_borrow_mut()) else {
            return Pump::Idle;
        };
        let Some(link) = link.as_mut() else { return Pump::Idle };
        let want = self.ipc.cfg.try_borrow().map(|c| c.read_chunk).unwrap_or(1024).max(64);
        if chunk.len() != want {
            chunk.clear();
            let _tag = klong_core::htag::Scope::new(klong_core::htag::IPC);
            if chunk.try_reserve_exact(want).is_err() {
                return Pump::Over(String::from("out of memory"), false);
            }
            chunk.resize(want, 0);
        }
        let mut moved = false;
        let (buf, pos) = &mut *out;
        match link {
            ConnLink::Stream(s) => {
                while *pos < buf.len() {
                    match s.write(&buf[*pos..]) {
                        Ok(0) => break,
                        Ok(n) => {
                            *pos += n;
                            moved = true;
                        }
                        Err(e) => return link_over(e, &conn.peer),
                    }
                }
                match s.read(&mut chunk) {
                    Ok(0) => {}
                    Ok(n) => {
                        moved = true;
                        if sess.feed(&chunk[..n]).is_err() {
                            return Pump::Over(klong_core::msg!("out of memory receiving from {}", conn.peer), false);
                        }
                    }
                    Err(e) => return link_over(e, &conn.peer),
                }
            }
            ConnLink::Rel(r) => {
                // packets in
                for _ in 0..64 {
                    match r.link.recv(&mut r.buf) {
                        Ok(Some(n)) => {
                            moved = true;
                            r.rel.on_packet(&r.buf[..n.min(r.buf.len())], now_ms);
                        }
                        Ok(None) => break,
                        Err(e) => return link_over(e, &conn.peer),
                    }
                }
                // stream out
                if *pos < buf.len() {
                    let n = r.rel.write(&buf[*pos..]);
                    *pos += n;
                    moved |= n > 0;
                }
                // packets out
                let mut clear = true;
                if !r.stuck.is_empty() {
                    match r.link.send(&r.stuck) {
                        Ok(true) => r.stuck.clear(),
                        Ok(false) => clear = false,
                        Err(e) => return link_over(e, &conn.peer),
                    }
                }
                if clear {
                    for _ in 0..64 {
                        let Some(n) = r.rel.poll_packet(now_ms, &mut r.buf) else { break };
                        moved = true;
                        match r.link.send(&r.buf[..n]) {
                            Ok(true) => {}
                            Ok(false) => {
                                r.stuck.clear();
                                r.stuck.extend_from_slice(&r.buf[..n]);
                                break;
                            }
                            Err(e) => return link_over(e, &conn.peer),
                        }
                    }
                }
                // stream in
                let n = r.rel.read(&mut chunk);
                if n > 0 {
                    moved = true;
                    if sess.feed(&chunk[..n]).is_err() {
                        return Pump::Over(klong_core::msg!("out of memory receiving from {}", conn.peer), false);
                    }
                }
                if let Some(e) = r.rel.error() {
                    if n == 0 {
                        return Pump::Over(klong_core::msg!("connection lost ({}): {}", e.text(), conn.peer), e == rel::Error::Reset);
                    }
                }
            }
        }
        if *pos > 0 && *pos == buf.len() {
            if buf.capacity() > 65536 {
                *buf = Vec::new();
            } else {
                buf.clear();
            }
            *pos = 0;
        }
        if moved {
            Pump::Moved
        } else {
            Pump::Idle
        }
    }

    /// One unit of work on a connection: the next event if one is complete, else move bytes.
    /// False when there was nothing to do.
    fn step(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>) -> bool {
        if conn.is_down() {
            return false;
        }
        let ev = match conn.sess.try_borrow_mut() {
            Ok(mut s) => s.next_event(&mut ip.syms),
            Err(_) => None,
        };
        if let Some(ev) = ev {
            self.on_event(ip, conn, ev);
            return true;
        }
        match self.pump(conn) {
            Pump::Idle => false,
            Pump::Moved => true,
            Pump::Over(text, clean) => {
                let ev = conn.sess.try_borrow_mut().ok().and_then(|mut s| s.feed_eof());
                match ev {
                    Some(Event::Rejected) => self.on_event(ip, conn, Event::Rejected),
                    Some(Event::Closed { reason, pending }) => {
                        let clean = clean && reason == Reason::Eof;
                        self.conn_down(ip, conn, text, pending, clean);
                    }
                    _ => self.conn_down(ip, conn, text, Vec::new(), clean),
                }
                true
            }
        }
    }

    /// Ctrl-C on the console, or the interrupt flag of the embedder.
    fn interrupted(&self, ip: &Interp) -> bool {
        if let Some(flag) = ip.cfg.interrupt {
            if flag.load(core::sync::atomic::Ordering::Relaxed) {
                return true;
            }
        }
        let Ok(mut p) = self.shared.platform.try_borrow_mut() else { return false };
        self.shared.scan_console(&mut p)
    }

    // ----- connections -------------------------------------------------------------------

    fn new_conn(&self, link: Link, peer: String, server_side: bool) -> Result<Rc<Conn>> {
        let (cfg, timeout, hs, retries, idle) = {
            let c = self.ipc.cfg.try_borrow().map_err(|_| busy())?;
            (session::Config { max_message: c.max_message, ..session::Config::default() }, c.call_timeout_us, c.handshake_timeout_us, c.retries, c.link_idle_ms)
        };
        let now = self.now();
        let (mut timeout, mut hs, mut retries, mut idle) = (timeout, hs, retries, idle);
        let link = match link {
            Link::Stream(s) => ConnLink::Stream(s),
            Link::Packet(p) => {
                let t = p.tuning();
                hs = t.handshake_timeout_us.unwrap_or(hs);
                retries = t.retries.unwrap_or(retries);
                idle = t.idle_ms.unwrap_or(idle);
                if t.call_timeout_us.is_some() {
                    timeout = t.call_timeout_us;
                }
                let mut rc = rel::Config::new(p.mtu(), p.reliable(), p.rto_hint_ms());
                rc.retries = retries;
                rc.idle_ms = idle;
                let seed = self.shared.platform.try_borrow_mut().map(|mut p| p.entropy()).unwrap_or(now);
                let id = [seed as u8, (seed >> 8) as u8, (seed >> 16) as u8];
                let r = if server_side { Rel::server(rc, id, now / 1000) } else { Rel::client(rc, id, now / 1000) };
                let _tag = klong_core::htag::Scope::new(klong_core::htag::IPC);
                let mut buf = try_vec(rc.mtu)?;
                buf.resize(rc.mtu, 0);
                ConnLink::Rel(RelLink { link: p, rel: r, stuck: Vec::new(), buf })
            }
        };
        let sess = if server_side { Session::server(cfg) } else { Session::client(cfg) };
        let conn = Rc::new(Conn {
            link: RefCell::new(Some(link)),
            sess: RefCell::new(sess),
            out: RefCell::new((Vec::new(), 0)),
            peer,
            server_side,
            ready: Cell::new(false),
            opened: Cell::new(false),
            hs_deadline: Cell::new(now + hs),
            timeout_us: Cell::new(timeout),
            results: RefCell::new(Vec::new()),
            asyncs: RefCell::new(Vec::new()),
            down: RefCell::new(None),
            handle: RefCell::new(None),
        });
        let mut conns = self.ipc.conns.try_borrow_mut().map_err(|_| busy())?;
        conns.try_reserve(1).map_err(|_| Error::memory())?;
        conns.push(conn.clone());
        Ok(conn)
    }

    /// The connection is over: close the link, fail what was waiting on it, run the hooks.
    fn conn_down(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, text: String, mut pending: Vec<Pending>, clean: bool) {
        match conn.down.try_borrow_mut() {
            Ok(mut d) => {
                if d.is_some() {
                    return;
                }
                *d = Some(text.clone());
            }
            Err(_) => return,
        }
        if let Ok(mut s) = conn.sess.try_borrow_mut() {
            pending.extend(s.close());
        }
        if let Ok(mut l) = conn.link.try_borrow_mut() {
            match l.take() {
                Some(ConnLink::Stream(mut s)) => s.close(),
                Some(ConnLink::Rel(mut r)) => {
                    // tell the peer, once; a lost RST is found by its own timeouts
                    r.rel.close();
                    if let Some(n) = r.rel.poll_packet(self.now() / 1000, &mut r.buf) {
                        let _ = r.link.send(&r.buf[..n]);
                    }
                    r.link.close();
                }
                None => {}
            }
        }
        // every pending request fails with the same error
        let asyncs: Vec<AsyncReq> = conn.asyncs.try_borrow_mut().map(|mut a| core::mem::take(&mut *a)).unwrap_or_default();
        for p in pending {
            if asyncs.iter().any(|a| a.id == p.id) {
                self.log(&klong_core::msg!(".async: remote call failed: {}", text));
            } else if let Ok(mut r) = conn.results.try_borrow_mut() {
                if r.try_reserve(1).is_ok() {
                    r.push((p.id, Err(Error::system(klong_core::msg!("{}", text)))));
                }
            }
        }
        if let Ok(mut conns) = self.ipc.conns.try_borrow_mut() {
            conns.retain(|c| !Rc::ptr_eq(c, conn));
        }
        if conn.server_side && conn.opened.get() {
            if !clean && Self::has_hook(ip, ".srv.e") {
                if let Ok(t) = Value::text(&text) {
                    let h = self.handle(ip, conn);
                    self.hook(ip, ".srv.e", &[h, t]);
                }
            }
            if Self::has_hook(ip, ".srv.c") {
                let h = self.handle(ip, conn);
                self.hook(ip, ".srv.c", &[h]);
            }
        }
        // (the handle keeps the connection alive and the connection its handle: let go)
        if let Ok(mut h) = conn.handle.try_borrow_mut() {
            *h = None;
        }
        // and forget the handles nobody holds any more
        if let Ok(mut h) = self.ipc.handles.try_borrow_mut() {
            h.retain(|e| e.f.strong_count() > 0);
            if h.is_empty() {
                *h = Vec::new();
            }
        }
    }

    fn has_hook(ip: &Interp, name: &str) -> bool {
        matches!(ip.global_path(name), Some(Value::Fn(_) | Value::Prim(_)))
    }

    /// Call a hook (`.srv.o`, `.srv.c`, `.srv.e`) if the program defined it as a function.
    fn hook(&self, ip: &mut Interp, name: &str, args: &[Value]) -> Option<Value> {
        let f = ip.global_path(name)?;
        if !matches!(f, Value::Fn(_) | Value::Prim(_)) {
            return None;
        }
        let mut v = Vec::new();
        v.try_reserve(args.len()).ok()?;
        v.extend_from_slice(args);
        let n = args.len() as u8;
        match ip.apply(&f, v, Some(n)) {
            Ok(r) => Some(r),
            Err(e) => {
                self.log(&klong_core::msg!("IPC: error in the {} hook: {}", name, e.render()));
                None
            }
        }
    }

    fn on_event(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, ev: Event) {
        match ev {
            Event::Accepted { .. } => conn.ready.set(true),
            Event::Rejected => {
                let text = klong_core::msg!("{} closed the connection during the handshake (bad credentials, or it does not speak the kdb+ protocol)", conn.peer);
                self.conn_down(ip, conn, text, Vec::new(), false);
            }
            Event::Handshake { user, password, .. } => {
                // .srv.auth(user; password): 1 accepts, 0 refuses; no hook accepts everyone
                let ok = match ip.global(".srv.auth") {
                    Some(f) if matches!(f, Value::Fn(_) | Value::Prim(_)) => match (Value::text(&user), Value::text(&password)) {
                        (Ok(u), Ok(p)) => matches!(self.hook(ip, ".srv.auth", &[u, p]), Some(Value::Int(i)) if i != 0),
                        _ => false,
                    },
                    _ => true,
                };
                let reply = if ok { conn.sess.try_borrow_mut().ok().and_then(|mut s| s.accept().ok()) } else { None };
                match reply {
                    Some(b) if conn.queue(&b).is_ok() => {
                        conn.ready.set(true);
                        conn.opened.set(true);
                        if Self::has_hook(ip, ".srv.o") {
                            let h = self.handle(ip, conn);
                            self.hook(ip, ".srv.o", &[h]);
                        }
                    }
                    _ => {
                        if let Ok(mut s) = conn.sess.try_borrow_mut() {
                            s.reject();
                        }
                        self.conn_down(ip, conn, klong_core::msg!("access denied for {} from {}", user, conn.peer), Vec::new(), true);
                    }
                }
            }
            Event::Message { kind: Kind::Response, value, pending, .. } => {
                self.message_cpu();
                let Some(id) = pending else { return };
                let a = conn.asyncs.try_borrow_mut().ok().and_then(|mut a| a.iter().position(|q| q.id == id).map(|i| a.remove(i)));
                match a {
                    Some(a) => match self.result_value(ip, conn, value, a.lookup) {
                        Ok(v) => {
                            let mut args = Vec::new();
                            if args.try_reserve(1).is_ok() {
                                args.push(v);
                                let _ = self.queue(Work { f: a.cb, args, then: None, what: ".async" });
                            }
                        }
                        Err(e) => self.log(&klong_core::msg!(".async: remote call failed: {}", e.render())),
                    },
                    None => {
                        if let Ok(mut r) = conn.results.try_borrow_mut() {
                            if r.try_reserve(1).is_ok() {
                                r.push((id, value));
                            }
                        }
                    }
                }
            }
            Event::Message { kind, value, .. } => {
                self.message_cpu();
                self.serve(ip, conn, kind, value)
            }
            Event::Failure { error, pending } => {
                let text = klong_core::msg!("protocol error from {}: {}", conn.peer, error);
                self.conn_down(ip, conn, text, pending, false);
            }
            Event::Closed { reason, pending } => {
                let (text, clean) = match reason {
                    Reason::Eof => (klong_core::msg!("connection closed by peer: {}", conn.peer), true),
                    Reason::Truncated => (klong_core::msg!("connection lost (truncated): {}", conn.peer), false),
                    Reason::Timeout => (klong_core::msg!("call timed out: {}", conn.peer), false),
                };
                self.conn_down(ip, conn, text, pending, clean);
            }
        }
    }

    // ----- serving -----------------------------------------------------------------------

    /// Serve one request and answer it.
    fn serve(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, kind: Kind, value: Result<Value>) {
        let begun = match conn.sess.try_borrow_mut() {
            Ok(mut s) => match s.begin_service(kind) {
                Ok(svc) => Ok(svc),
                Err(_) => Err(s.refuse(kind).unwrap_or_default()),
            },
            Err(_) => return,
        };
        let mut svc = match begun {
            Ok(svc) => svc,
            Err(refusal) => {
                let _ = conn.queue(&refusal);
                return;
            }
        };
        // Tell the peer what has arrived before a long evaluation starts: on a packet link it
        // would otherwise retransmit, and give up, while this node computes.
        if let Ok(mut l) = conn.link.try_borrow_mut() {
            if let Some(ConnLink::Rel(r)) = l.as_mut() {
                r.rel.ack_soon();
            }
        }
        let _ = self.pump(conn);
        // .cli.h is the handle of the client being served
        let prev = ip.global(".cli.h");
        let h = self.handle(ip, conn);
        let _ = ip.define_global(".cli.h", h);
        let result = value.and_then(|v| self.evaluate_request(ip, v));
        let _ = ip.define_global(".cli.h", prev.unwrap_or(Value::Undefined));
        let compress = self.ipc.cfg.try_borrow().map(|c| c.compress).unwrap_or(false);
        if kind == Kind::Sync {
            let framed = match conn.sess.try_borrow_mut() {
                Ok(mut s) => {
                    let first = match &result {
                        Ok(v) => s.send_response(&mut svc, &ip.syms, v, compress),
                        Err(e) => s.send_error(&mut svc, &wire_error_text(e)),
                    };
                    // a result with no kdb+ form is an error for the caller
                    match first {
                        Ok(b) => Ok(b),
                        Err(e) if result.is_ok() => s.send_error(&mut svc, &wire_error_text(&e)),
                        Err(e) => Err(e),
                    }
                }
                Err(_) => Err(busy()),
            };
            if let Ok(bytes) = framed {
                let _ = conn.queue(&bytes);
            }
        } else if let Err(e) = &result {
            // nobody waits for the answer: the error goes to .srv.e and the log
            let text = wire_error_text(e);
            self.log(&klong_core::msg!("IPC: error serving a request from {}: {}", conn.peer, text));
            if Self::has_hook(ip, ".srv.e") {
                if let Ok(t) = Value::text(&text) {
                    let h = self.handle(ip, conn);
                    self.hook(ip, ".srv.e", &[h, t]);
                }
            }
        }
        let failure = conn.sess.try_borrow_mut().ok().and_then(|mut s| s.end_service(svc));
        if let Some(Event::Failure { error, pending }) = failure {
            let text = klong_core::msg!("protocol error from {}: {}", conn.peer, error);
            self.conn_down(ip, conn, text, pending, false);
        }
        let _ = self.pump(conn);
    }

    /// Do what a request message asks.
    fn evaluate_request(&self, ip: &mut Interp, value: Value) -> Result<Value> {
        match &value {
            Value::Sym(s) => {
                let name = try_str(ip.syms.name(*s))?;
                return ip.global_path(&name).ok_or_else(|| Error::name(klong_core::msg!("symbol not found: {}", name)));
            }
            Value::Str(text) => {
                let text = try_str(text.as_str())?;
                return Ok(ip.run_global(&text, "<ipc>")?.unwrap_or(Value::Undefined));
            }
            _ => {}
        }
        // a call: a list whose first item is a symbol or a string (and that has arguments),
        // or a function of ours sent as text
        let is_call = match &value {
            Value::List(items) => match items.first() {
                Some(Value::Sym(_)) | Some(Value::Str(_)) => items.len() >= 2,
                Some(first) => ext_ref::<KdbLambda>(first).is_some_and(|l| l.is_klong()),
                None => false,
            },
            _ => false,
        };
        if !is_call {
            return Ok(value);
        }
        let mut items = value.items()?;
        let callee = items.remove(0);
        // ("set"; name; value) assigns a global unless the program has its own `set`
        if items.len() == 2 {
            let named_set = match &callee {
                Value::Str(s) => s.as_str() == "set",
                Value::Sym(s) => ip.syms.name(*s) == "set",
                _ => false,
            };
            if let (true, Value::Sym(target)) = (named_set, &items[0]) {
                if ip.global("set").is_none() {
                    let name = try_str(ip.syms.name(*target))?;
                    let v = local_function(ip, items.remove(1));
                    ip.define_global(&name, v)?;
                    return Ok(Value::Undefined);
                }
            }
        }
        let target = match &callee {
            Value::Sym(s) => {
                let name = try_str(ip.syms.name(*s))?;
                ip.global(&name).ok_or_else(|| Error::name(klong_core::msg!("symbol not found: {}", name)))?
            }
            Value::Str(text) => {
                let text = try_str(text.as_str())?;
                ip.run_global(&text, "<ipc>")?.unwrap_or(Value::Undefined)
            }
            other => {
                let text = ext_ref::<KdbLambda>(other).and_then(|l| l.text_str()).unwrap_or("");
                parse_function(ip, text).ok_or_else(|| Error::typ(klong_core::msg!("cannot run function text from the peer: {}", text)))?
            }
        };
        let arity = arity_of(&target).ok_or_else(|| Error::typ(klong_core::msg!("not callable: {}", format_value(&ip.syms, &callee, true).unwrap_or_default())))?;
        let mut args = try_vec(items.len())?;
        for a in items {
            args.push(local_function(ip, a));
        }
        if arity == 0 && args.len() == 1 && matches!(args[0], Value::Undefined) {
            args.clear(); // q's (`f; ::) calls f with no arguments
        }
        let n = args.len().min(255) as u8;
        ip.apply(&target, args, Some(n))
    }

    // ----- calling -----------------------------------------------------------------------

    /// What a response means to the caller: an error object raises, a function of the peer
    /// becomes a proxy (after a lookup) or a local function.
    fn result_value(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, value: Result<Value>, lookup: Option<SymId>) -> Result<Value> {
        let v = value?;
        if let Some(e) = ext_ref::<KdbError>(&v) {
            return Err(remote_error(&e.text, &conn.peer));
        }
        let lam = match ext_ref::<KdbLambda>(&v) {
            Some(l) if l.is_klong() => l,
            _ => return Ok(v),
        };
        let text = lam.text_str().unwrap_or("");
        let (arity, local) = match display_arity(text) {
            Some(n) => (Some(n), None),
            None => match parse_function(ip, text) {
                Some(f) => (arity_of(&f), Some(f)),
                None => (None, None),
            },
        };
        match (lookup, arity) {
            (Some(sym), Some(n)) => self.proxy(ip, conn, sym, n),
            _ => Ok(local.unwrap_or(v)),
        }
    }

    /// The processor time of a message (see `Platform::message_cpu`).
    fn message_cpu(&self) {
        if let Ok(mut p) = self.shared.platform.try_borrow_mut() {
            p.message_cpu();
        }
    }

    fn send(&self, ip: &Interp, conn: &Conn, req: &Request, blocking: bool) -> Result<PendingId> {
        if conn.is_down() {
            return Err(conn.down_error());
        }
        self.message_cpu();
        let deadline = conn.timeout_us.get().map(|t| self.now().saturating_add(t));
        let compress = self.ipc.cfg.try_borrow().map(|c| c.compress).unwrap_or(false);
        let (bytes, id) = {
            let mut s = conn.sess.try_borrow_mut().map_err(|_| busy())?;
            match req {
                Request::Value(v) => s.send_sync(&ip.syms, v, blocking, deadline, compress)?,
                Request::Lookup(sym) => s.send_sync(&ip.syms, &Value::Sym(*sym), blocking, deadline, compress)?,
                Request::Call(items) => s.send_call(&ip.syms, items, blocking, deadline, compress)?,
            }
        };
        if let Err(e) = conn.queue(&bytes) {
            if let Ok(mut s) = conn.sess.try_borrow_mut() {
                s.forget(id);
            }
            return Err(e);
        }
        Ok(id)
    }

    /// A blocking call: send, then serve this connection until the answer is there.
    fn call(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, req: Request) -> Result<Value> {
        let lookup = match &req {
            Request::Lookup(s) => Some(*s),
            _ => None,
        };
        let id = self.send(ip, conn, &req, true)?;
        drop(req);
        let seconds = conn.timeout_us.get().unwrap_or(0) as f64 / 1e6;
        let value = loop {
            if let Some(r) = conn.take_result(id) {
                break r;
            }
            if conn.is_down() {
                break Err(conn.down_error());
            }
            if self.step(ip, conn) {
                self.idle.set(0);
                continue;
            }
            let now = self.now();
            let late = conn.sess.try_borrow_mut().ok().and_then(|mut s| s.check_deadlines(now));
            if let Some(Event::Closed { pending, .. }) = late {
                // no request ids: a call that times out cannot be skipped, the connection goes
                let text = klong_core::msg!("call timed out after {}s: {}", seconds, conn.peer);
                self.conn_down(ip, conn, text, pending, false);
                continue;
            }
            if self.interrupted(ip) {
                let text = klong_core::msg!("interrupted while waiting: {}", conn.peer);
                self.conn_down(ip, conn, text, Vec::new(), false);
                let _ = conn.take_result(id);
                return Err(Error::interrupt());
            }
            self.idle_wait(self.ipc.next_deadline());
        };
        self.result_value(ip, conn, value, lookup)
    }

    /// The same request without waiting: `cb` is called with the answer from `poll_events`.
    fn call_async(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, req: Request, cb: Value) -> Result<()> {
        let lookup = match &req {
            Request::Lookup(s) => Some(*s),
            _ => None,
        };
        conn.asyncs.try_borrow_mut().map_err(|_| busy())?.try_reserve(1).map_err(|_| Error::memory())?;
        let id = self.send(ip, conn, &req, false)?;
        conn.asyncs.try_borrow_mut().map_err(|_| busy())?.push(AsyncReq { id, cb, lookup });
        let _ = self.pump(conn);
        Ok(())
    }

    /// The request `handle(x)` makes: a list that starts with a symbol is a call, a symbol
    /// a lookup, anything else goes as it is.
    fn request_of(x: &Value) -> Result<Request<'_>> {
        match x {
            Value::Sym(s) => Ok(Request::Lookup(*s)),
            Value::List(items) if matches!(items.first(), Some(Value::Sym(_))) => {
                let mut v = try_vec(items.len().max(2))?;
                v.extend(items.iter().cloned());
                if v.len() == 1 {
                    v.push(Value::Undefined); // q's way to call with no arguments
                }
                Ok(Request::Call(v))
            }
            other => Ok(Request::Value(other)),
        }
    }

    fn proxy_request(sym: SymId, a: &[Value]) -> Result<Request<'static>> {
        let mut v = try_vec(a.len() + 2)?;
        v.push(Value::Sym(sym));
        v.extend(a.iter().cloned());
        if a.is_empty() {
            v.push(Value::Undefined);
        }
        Ok(Request::Call(v))
    }

    // ----- handles -----------------------------------------------------------------------

    fn remember(&self, f: &Rc<Func>, conn: &Rc<Conn>, sym: Option<SymId>) -> Result<()> {
        let mut h = self.ipc.handles.try_borrow_mut().map_err(|_| busy())?;
        h.retain(|e| e.f.strong_count() > 0);
        h.try_reserve(1).map_err(|_| Error::memory())?;
        h.push(HandleEntry { f: Rc::downgrade(f), conn: Rc::downgrade(conn), sym });
        Ok(())
    }

    /// The connection a value stands for: a handle, a remote dictionary, a proxy.
    fn conn_of(&self, v: &Value) -> Option<Rc<Conn>> {
        self.remote_of(v).map(|(c, _)| c)
    }

    /// The connection, and for a proxy the name it calls.
    fn remote_of(&self, v: &Value) -> Option<(Rc<Conn>, Option<SymId>)> {
        match v {
            Value::Fn(f) => {
                let h = self.ipc.handles.try_borrow().ok()?;
                h.iter().find(|e| e.f.strong_count() > 0 && core::ptr::eq(e.f.as_ptr(), Rc::as_ptr(f))).and_then(|e| e.conn.upgrade().map(|c| (c, e.sym)))
            }
            Value::Ext(e) => e.0.as_any().downcast_ref::<DictHandle<P>>().map(|d| (d.conn.clone(), None)),
            _ => None,
        }
    }

    /// The function that stands for a connection: `h("text")`, `h(:name)`, `h(:f,args)`.
    fn handle(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>) -> Value {
        if let Ok(h) = conn.handle.try_borrow() {
            if let Some(v) = h.as_ref() {
                return v.clone();
            }
        }
        let (w, c) = (Rc::downgrade(self), conn.clone());
        let f: NativeFn = Rc::new(move |ip: &mut Interp, a: &[Value]| {
            let ev = w.upgrade().ok_or_else(|| Error::system("no runtime"))?;
            let req = Self::request_of(one(a, "a connection")?)?;
            ev.call(ip, &c, req)
        });
        let func = Rc::new(Func::Native { name: ip.syms.intern(".cli.fn"), arity: 1, f });
        let _ = self.remember(&func, conn, None);
        let v = Value::Fn(func);
        if !conn.is_down() {
            if let Ok(mut h) = conn.handle.try_borrow_mut() {
                *h = Some(v.clone());
            }
        }
        v
    }

    /// A function of the peer, called by name.
    fn proxy(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>, sym: SymId, arity: u8) -> Result<Value> {
        let (w, c) = (Rc::downgrade(self), conn.clone());
        let f: NativeFn = Rc::new(move |ip: &mut Interp, a: &[Value]| {
            let ev = w.upgrade().ok_or_else(|| Error::system("no runtime"))?;
            ev.call(ip, &c, Self::proxy_request(sym, a)?)
        });
        let func = Rc::new(Func::Native { name: ip.syms.intern(".cli.proxy"), arity, f });
        // (which name it calls is what `.async` needs to know)
        self.remember(&func, conn, Some(sym))?;
        Ok(Value::Fn(func))
    }

    fn dict_handle(self: &Rc<Self>, conn: &Rc<Conn>) -> Value {
        Value::Ext(Rc::new(ExtBox(Box::new(DictHandle { ev: Rc::downgrade(self), conn: conn.clone() }))))
    }

    // ----- addresses ---------------------------------------------------------------------

    fn transport_index(&self, name: &str) -> Option<usize> {
        let mut p = self.shared.platform.try_borrow_mut().ok()?;
        p.transports().iter().position(|t| t.name() == name)
    }

    /// (transport, address for it, user, password, the address as shown in messages).
    fn address(&self, ip: &Interp, x: &Value, listen: bool) -> Result<(usize, String, String, String, String)> {
        let text = match x {
            Value::Int(port) => klong_core::msg!("{}", port),
            Value::Str(s) => try_str(s.as_str())?,
            other => return Err(Error::typ(klong_core::msg!("bad address {}: expected a port or \"host:port\"", format_value(&ip.syms, other, false).unwrap_or_default()))),
        };
        let missing = |name: &str| Error::system(klong_core::msg!("this platform has no {} transport", name));
        if let Some((head, rest)) = text.split_once(':') {
            // "name:address" for a transport the platform has; the address is passed untouched
            if head != "tcp" || !rest.is_empty() {
                if let Some(i) = self.transport_index(head) {
                    if head != "tcp" {
                        return Ok((i, try_str(rest)?, String::new(), String::new(), text));
                    }
                }
            }
        }
        let rest = text.strip_prefix("tcp:").unwrap_or(&text);
        let i = self.transport_index("tcp").ok_or_else(|| missing("tcp"))?;
        let bad = || Error::typ(klong_core::msg!("bad address {}: expected a port or \"host:port\"", text));
        let mut parts = rest.splitn(4, ':');
        let first = parts.next().unwrap_or("");
        let (host, port) = match parts.next() {
            Some(port) => (first, port),
            None => (if listen { "" } else { "localhost" }, first),
        };
        if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) || port.len() > 5 {
            return Err(bad());
        }
        let user = try_str(parts.next().unwrap_or(""))?;
        let password = try_str(parts.next().unwrap_or(""))?;
        let addr = klong_core::msg!("{}:{}", host, port);
        let shown = if listen { try_str(&addr)? } else { klong_core::msg!("{}:{}", host, port) };
        Ok((i, addr, user, password, shown))
    }

    /// Open a connection and shake hands; an error when the peer cannot be reached or refuses.
    fn connect(self: &Rc<Self>, ip: &mut Interp, x: &Value) -> Result<Rc<Conn>> {
        let (t, addr, user, password, shown) = self.address(ip, x, false)?;
        let link = {
            let mut p = self.shared.platform.try_borrow_mut().map_err(|_| busy())?;
            let tr = p.transports().get_mut(t).ok_or_else(|| Error::system("no such transport"))?;
            tr.connect(&addr).map_err(|e| Error::system(klong_core::msg!("cannot connect to {}: {}", shown, e.text())))?
        };
        let conn = self.new_conn(link, shown, false)?;
        let hello = conn.sess.try_borrow_mut().map_err(|_| busy()).and_then(|mut s| s.hello(&user, &password));
        let hello = match hello {
            Ok(h) => h,
            Err(e) => {
                self.conn_down(ip, &conn, String::from("handshake not sent"), Vec::new(), true);
                return Err(e);
            }
        };
        conn.queue(&hello)?;
        while !conn.ready.get() {
            if conn.is_down() {
                return Err(conn.down_error());
            }
            if self.step(ip, &conn) {
                self.idle.set(0);
                continue;
            }
            if self.now() >= conn.hs_deadline.get() {
                let text = klong_core::msg!("handshake timed out: {}", conn.peer);
                self.conn_down(ip, &conn, text, Vec::new(), false);
                continue;
            }
            if self.interrupted(ip) {
                self.conn_down(ip, &conn, String::from("interrupted"), Vec::new(), false);
                return Err(Error::interrupt());
            }
            self.idle_wait(Some(conn.hs_deadline.get()));
        }
        Ok(conn)
    }

    /// Close a connection of ours: what is queued goes out first (briefly).
    fn close(self: &Rc<Self>, ip: &mut Interp, conn: &Rc<Conn>) {
        let until = self.now() + 1_000_000;
        while conn.unsent() && !conn.is_down() && self.now() < until {
            if !matches!(self.pump(conn), Pump::Moved) {
                self.idle_wait(Some(until));
            }
        }
        self.conn_down(ip, conn, klong_core::msg!("connection closed: {}", conn.peer), Vec::new(), true);
    }

    fn serve_start(self: &Rc<Self>, ip: &mut Interp, x: &Value) -> Result<Value> {
        if matches!(x, Value::Int(0)) {
            // .srv(0): stop listening and drop the clients
            let old = self.ipc.listener.try_borrow_mut().map_err(|_| busy())?.take();
            let Some(l) = old else { return Ok(Value::Int(0)) };
            if let Ok(mut p) = self.shared.platform.try_borrow_mut() {
                if let Some(t) = p.transports().get_mut(l.transport) {
                    t.unlisten(l.id);
                }
            }
            let conns: Vec<Rc<Conn>> = self.ipc.conns.try_borrow().map(|c| c.iter().filter(|c| c.server_side).cloned().collect()).unwrap_or_default();
            for c in conns {
                self.close(ip, &c);
            }
            return Ok(Value::Int(1));
        }
        if self.ipc.listener.try_borrow().map_err(|_| busy())?.is_some() {
            return Ok(Value::Int(0));
        }
        let (t, addr, _, _, shown) = self.address(ip, x, true)?;
        let id = {
            let mut p = self.shared.platform.try_borrow_mut().map_err(|_| busy())?;
            let tr = p.transports().get_mut(t).ok_or_else(|| Error::system("no such transport"))?;
            tr.listen(&addr).map_err(|e| Error::system(klong_core::msg!("cannot listen on {}: {}", shown, e.text())))?
        };
        *self.ipc.listener.try_borrow_mut().map_err(|_| busy())? = Some(Listener { transport: t, id });
        Ok(Value::Int(1))
    }

    /// The IPC part of `poll_events`.
    pub(crate) fn ipc_poll(self: &Rc<Self>, ip: &mut Interp) -> bool {
        let mut did = false;
        let listening = self.ipc.listener.try_borrow().ok().and_then(|l| l.as_ref().map(|l| (l.transport, l.id)));
        if let Some((t, id)) = listening {
            loop {
                let got = match self.shared.platform.try_borrow_mut() {
                    Ok(mut p) => p.transports().get_mut(t).and_then(|tr| tr.accept(id)),
                    Err(_) => None,
                };
                let Some((link, peer)) = got else { break };
                did = true;
                if self.new_conn(link, peer, true).is_err() {
                    break;
                }
            }
        }
        let conns: Vec<Rc<Conn>> = self.ipc.conns.try_borrow().map(|c| c.clone()).unwrap_or_default();
        for c in conns {
            if c.is_down() {
                continue;
            }
            if self.step(ip, &c) {
                did = true;
                continue;
            }
            let now = self.now();
            if !c.ready.get() {
                // a packet link that waits for its first SYN has no handshake to time out yet
                let listening = c.link.try_borrow().map(|l| matches!(l.as_ref(), Some(ConnLink::Rel(r)) if r.rel.state() == rel::State::Listen)).unwrap_or(false);
                if listening {
                    c.hs_deadline.set(now + self.ipc.cfg.try_borrow().map(|c| c.handshake_timeout_us).unwrap_or(5_000_000));
                } else if now >= c.hs_deadline.get() {
                    self.conn_down(ip, &c, klong_core::msg!("handshake timed out: {}", c.peer), Vec::new(), true);
                    did = true;
                }
                continue;
            }
            let late = c.sess.try_borrow_mut().ok().and_then(|mut s| s.check_deadlines(now));
            if let Some(Event::Closed { pending, .. }) = late {
                self.conn_down(ip, &c, klong_core::msg!("call timed out: {}", c.peer), pending, false);
                did = true;
            }
        }
        did
    }
}

fn is_function(v: &Value) -> bool {
    matches!(v, Value::Fn(_) | Value::Prim(_))
}

pub(crate) fn register<P: Platform + 'static>(interp: &mut Interp, ev: &Rc<Events<P>>) {
    macro_rules! sysfn {
        ($name:expr, $arity:expr, |$ev:ident, $ip:ident, $a:ident| $body:expr) => {{
            let w = Rc::downgrade(ev);
            interp.register(
                $name,
                $arity,
                Rc::new(move |$ip: &mut Interp, $a: &[Value]| {
                    let $ev = w.upgrade().ok_or_else(|| Error::system("no runtime"))?;
                    $body
                }),
            );
        }};
    }
    sysfn!(".cli", 1, |ev, ip, a| {
        let x = one(a, ".cli")?;
        match ev.conn_of(x) {
            Some(c) => Ok(ev.handle(ip, &c)),
            None => {
                let c = ev.connect(ip, x)?;
                Ok(ev.handle(ip, &c))
            }
        }
    });
    sysfn!(".clid", 1, |ev, ip, a| {
        let x = one(a, ".clid")?;
        match ev.conn_of(x) {
            Some(c) => Ok(ev.dict_handle(&c)),
            None => {
                let c = ev.connect(ip, x)?;
                Ok(ev.dict_handle(&c))
            }
        }
    });
    sysfn!(".clic", 1, |ev, ip, a| {
        match ev.conn_of(one(a, ".clic")?) {
            Some(c) if !c.is_down() => {
                ev.close(ip, &c);
                Ok(Value::Int(1))
            }
            _ => Ok(Value::Int(0)),
        }
    });
    sysfn!(".clit", 2, |ev, _ip, a| {
        let (x, y) = match a {
            [x, y] => (x, y),
            _ => return Err(Error::arity(".clit takes 2 arguments")),
        };
        let Some(c) = ev.conn_of(x) else { return Ok(Value::Int(0)) };
        let seconds = match y {
            Value::Int(i) => *i as f64,
            Value::Flt(f) => *f as f64,
            _ => return Err(Error::typ("y must be a number of seconds")),
        };
        if !(seconds >= 0.0) {
            return Err(Error::range("y must be a number of seconds, 0 or more"));
        }
        c.timeout_us.set(if seconds == 0.0 { None } else { Some((seconds * 1e6) as u64) });
        Ok(Value::Int(1))
    });
    sysfn!(".srv", 1, |ev, ip, a| ev.serve_start(ip, one(a, ".srv")?));
    sysfn!(".async", 2, |ev, ip, a| {
        let (x, y) = match a {
            [x, y] => (x.clone(), y.clone()),
            _ => return Err(Error::arity(".async takes 2 arguments")),
        };
        if !is_function(&x) {
            return Err(Error::typ("x must be a function"));
        }
        if !is_function(&y) {
            return Err(Error::typ("y must be a function"));
        }
        let arity = arity_of(&x).unwrap_or(1);
        let remote = ev.remote_of(&x);
        let w = Rc::downgrade(&ev);
        let f: NativeFn = Rc::new(move |ip: &mut Interp, a: &[Value]| {
            let ev = w.upgrade().ok_or_else(|| Error::system("no runtime"))?;
            match &remote {
                // a connection or a remote function: the request goes out now, y gets the answer
                Some((conn, None)) => {
                    let req = Events::<P>::request_of(one(a, "a connection")?)?;
                    ev.call_async(ip, conn, req, y.clone())?;
                }
                // (a proxy: its own call would block, so its request is sent from here)
                Some((conn, Some(sym))) => ev.call_async(ip, conn, Events::<P>::proxy_request(*sym, a)?, y.clone())?,
                // a local function: it runs from the event loop, then y gets its result
                None => {
                    let mut args = try_vec(a.len())?;
                    for q in a {
                        try_push(&mut args, q.clone())?;
                    }
                    ev.queue(Work { f: x.clone(), args, then: Some(y.clone()), what: ".async" })?;
                }
            }
            Ok(Value::Int(1))
        });
        Ok(Value::Fn(Rc::new(Func::Native { name: ip.syms.intern(".async.fn"), arity, f })))
    });
    for name in [".srv.o", ".srv.c", ".srv.e", ".srv.auth", ".cli.h"] {
        let s = interp.syms.intern(name);
        interp.set_global(s, Value::Undefined);
    }
}
