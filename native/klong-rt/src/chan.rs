//! Channels and the file functions (feature `files`): `.cin .cout .cerr`, `.ic .oc .ac .cc`,
//! `.fc .tc .mi .rl .r .rs .fl .df`, and `.l`. Storage is reached only through
//! [`FileSystem`](crate::FileSystem).
use alloc::rc::{Rc, Weak};
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use klong_core::error::Result;
use klong_core::value::{Ext, ExtBox, Value, WILDCARD};
use klong_core::{Error, Interp};

use crate::platform::{FileId, FsError, Mode, Platform};
use crate::{is_false, one, text_arg, Shared};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    ConsoleIn,
    ConsoleOut,
    ConsoleErr,
    File,
}

/// A channel: the console, or an open file. Dropping the last reference closes the file.
pub struct Chan<P: Platform + 'static> {
    target: Target,
    input: bool,
    file: Cell<Option<FileId>>,
    /// A read has gone past the end (`.mi` answers 0 after that).
    eof: Cell<bool>,
    /// Input read from the file or the console and not yet consumed.
    buf: RefCell<Vec<u8>>,
    rt: Weak<Shared<P>>,
}

impl<P: Platform + 'static> Ext for Chan<P> {
    fn type_name(&self) -> &str {
        // as KlongPy prints them
        match (self.input, self.target) {
            (true, _) => "inchan.0",
            (false, Target::ConsoleErr) => "outchan.2",
            (false, _) => "outchan.1",
        }
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<P: Platform + 'static> Chan<P> {
    fn close(&self) {
        if let Some(id) = self.file.take() {
            if let Some(sh) = self.rt.upgrade() {
                if let Ok(mut p) = sh.platform.try_borrow_mut() {
                    if let Some(fs) = p.fs() {
                        fs.close(id);
                    }
                }
            }
        }
    }
}

impl<P: Platform + 'static> Drop for Chan<P> {
    fn drop(&mut self) {
        self.close();
    }
}

fn fs_error(e: FsError, path: &str) -> Error {
    Error::system(klong_core::msg!("{}: {}", e.text(), path))
}

fn no_fs() -> Error {
    Error::system("this platform has no file system")
}

fn busy() -> Error {
    Error::internal("runtime state in use")
}

fn new_chan<P: Platform + 'static>(sh: &Rc<Shared<P>>, target: Target, input: bool, file: Option<FileId>) -> Value {
    let c: Chan<P> = Chan { target, input, file: Cell::new(file), eof: Cell::new(false), buf: RefCell::new(Vec::new()), rt: Rc::downgrade(sh) };
    Value::Ext(Rc::new(ExtBox(alloc::boxed::Box::new(c))))
}

fn chan_of<P: Platform + 'static>(v: &Value) -> Option<&Chan<P>> {
    match v {
        Value::Ext(e) => e.0.as_any().downcast_ref::<Chan<P>>(),
        _ => None,
    }
}

fn open<P: Platform + 'static>(sh: &Rc<Shared<P>>, path: &str, mode: Mode) -> Result<Value> {
    let id = {
        let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
        let fs = p.fs().ok_or_else(no_fs)?;
        fs.open(path, mode).map_err(|e| fs_error(e, path))?
    };
    Ok(new_chan(sh, Target::File, mode == Mode::Read, Some(id)))
}

/// The input the console channel has read from the host and the program has not used (what `.r` or `.rl` left), taken out of the channel: a host
/// that read it from a stream of its own can put it back there. Bytes, as read.
pub(crate) fn take_console_unread<P: Platform + 'static>(sh: &Rc<Shared<P>>) -> Vec<u8> {
    let cin = match sh.io.try_borrow() {
        Ok(io) => io.cin.clone(),
        Err(_) => return Vec::new(),
    };
    match chan_of::<P>(&cin).and_then(|c| c.buf.try_borrow_mut().ok().map(|mut b| core::mem::take(&mut *b))) {
        Some(b) => b,
        None => Vec::new(),
    }
}

/// Write to an output channel.
pub(crate) fn write<P: Platform + 'static>(sh: &Rc<Shared<P>>, to: &Value, mut bytes: &[u8]) -> Result<()> {
    let c = chan_of::<P>(to).ok_or_else(|| Error::typ("not a channel"))?;
    let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
    match c.target {
        Target::ConsoleErr => p.console_error(bytes),
        Target::File => {
            let id = c.file.get().ok_or_else(|| Error::system("the channel is closed"))?;
            let fs = p.fs().ok_or_else(no_fs)?;
            while !bytes.is_empty() {
                match fs.write(id, bytes) {
                    Ok(0) => return Err(Error::system("could not write to the file")),
                    Ok(n) => bytes = &bytes[n.min(bytes.len())..],
                    Err(e) => return Err(Error::system(e.text())),
                }
            }
        }
        _ => p.console_write(bytes),
    }
    Ok(())
}

/// Read more input into the channel's buffer. Returns the number of bytes added: 0 at the end
/// of a file, or when the console has nothing waiting.
fn fill<P: Platform + 'static>(sh: &Rc<Shared<P>>, c: &Chan<P>) -> Result<usize> {
    let mut chunk = [0u8; 256];
    let mut buf = c.buf.try_borrow_mut().map_err(|_| busy())?;
    let n = match c.target {
        Target::File => {
            let id = match c.file.get() {
                Some(id) => id,
                None => return Ok(0),
            };
            let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
            let fs = p.fs().ok_or_else(no_fs)?;
            fs.read(id, &mut chunk).map_err(|e| Error::system(e.text()))?
        }
        _ => {
            // what was typed while an evaluation ran comes first
            let mut pending = sh.pending.try_borrow_mut().map_err(|_| busy())?;
            if !pending.is_empty() {
                let n = pending.len().min(chunk.len());
                chunk[..n].copy_from_slice(&pending[..n]);
                pending.drain(..n);
                n
            } else {
                let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
                p.console_input(&mut chunk)
            }
        }
    };
    buf.try_reserve(n).map_err(|_| Error::memory())?;
    buf.extend_from_slice(&chunk[..n]);
    Ok(n)
}

fn lossy(bytes: &[u8]) -> Result<String> {
    let mut s = String::new();
    s.try_reserve(bytes.len()).map_err(|_| Error::memory())?;
    match core::str::from_utf8(bytes) {
        Ok(t) => s.push_str(t),
        // text that is not UTF-8 is read with replacement characters, as KlongPy's
        // text-mode files would refuse or replace it
        Err(_) => s.push_str(&String::from_utf8_lossy(bytes)),
    }
    Ok(s)
}

/// The whole of a file as text (`.l`, `boot.kg`).
pub(crate) fn read_file<P: Platform + 'static>(sh: &Rc<Shared<P>>, path: &str) -> Result<String> {
    let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
    let fs = p.fs().ok_or_else(no_fs)?;
    let id = fs.open(path, Mode::Read).map_err(|e| fs_error(e, path))?;
    let mut data: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 512];
    let r = loop {
        match fs.read(id, &mut chunk) {
            Ok(0) => break Ok(()),
            Ok(n) => {
                if data.try_reserve(n).is_err() {
                    break Err(Error::memory());
                }
                data.extend_from_slice(&chunk[..n]);
            }
            Err(e) => break Err(fs_error(e, path)),
        }
    };
    fs.close(id);
    r?;
    match String::from_utf8(data) {
        Ok(s) => Ok(s),
        Err(_) => Err(Error::system(klong_core::msg!("not UTF-8 text: {}", path))),
    }
}

pub(crate) fn register<P: Platform + 'static>(interp: &mut Interp, shared: &Rc<Shared<P>>) {
    let cin = new_chan(shared, Target::ConsoleIn, true, None);
    let cout = new_chan(shared, Target::ConsoleOut, false, None);
    let cerr = new_chan(shared, Target::ConsoleErr, false, None);
    for (name, v) in [(".cin", &cin), (".cout", &cout), (".cerr", &cerr)] {
        let s = interp.syms.intern(name);
        interp.set_global(s, v.clone());
    }
    if let Ok(mut io) = shared.io.try_borrow_mut() {
        io.from = cin.clone();
        io.to = cout.clone();
        io.cin = cin;
        io.cout = cout;
    }

    let sh = shared.clone();
    interp.register(".ic", 1, Rc::new(move |_: &mut Interp, a: &[Value]| open(&sh, text_arg(a, ".ic")?, Mode::Read)));
    let sh = shared.clone();
    interp.register(".oc", 1, Rc::new(move |_: &mut Interp, a: &[Value]| open(&sh, text_arg(a, ".oc")?, Mode::Write)));
    let sh = shared.clone();
    interp.register(".ac", 1, Rc::new(move |_: &mut Interp, a: &[Value]| open(&sh, text_arg(a, ".ac")?, Mode::Append)));
    interp.register(
        ".cc",
        1,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            match chan_of::<P>(one(a, ".cc")?) {
                // the standard channels cannot be closed
                Some(c) if c.target == Target::File => c.close(),
                Some(_) => {}
                None => return Err(Error::typ(".cc needs a channel")),
            }
            Ok(ip.no_value())
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".fc",
        1,
        Rc::new(move |_: &mut Interp, a: &[Value]| {
            let x = one(a, ".fc")?;
            let mut io = sh.io.try_borrow_mut().map_err(|_| busy())?;
            let new = if is_false(x) { io.cin.clone() } else { x.clone() };
            match chan_of::<P>(&new) {
                Some(c) if c.input => {}
                Some(_) => return Err(Error::typ("an output channel cannot be used for input")),
                None => return Err(Error::typ(".fc needs an input channel")),
            }
            Ok(core::mem::replace(&mut io.from, new))
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".tc",
        1,
        Rc::new(move |_: &mut Interp, a: &[Value]| {
            let x = one(a, ".tc")?;
            let mut io = sh.io.try_borrow_mut().map_err(|_| busy())?;
            let new = if is_false(x) { io.cout.clone() } else { x.clone() };
            match chan_of::<P>(&new) {
                Some(c) if !c.input => {}
                Some(_) => return Err(Error::typ("an input channel cannot be a to channel")),
                None => return Err(Error::typ(".tc needs an output channel")),
            }
            Ok(core::mem::replace(&mut io.to, new))
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".mi",
        0,
        Rc::new(move |_: &mut Interp, _: &[Value]| {
            let from = sh.io.try_borrow().map_err(|_| busy())?.from.clone();
            let c = chan_of::<P>(&from).ok_or_else(|| Error::typ("no input channel"))?;
            Ok(Value::bool(!c.eof.get()))
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".rl",
        0,
        Rc::new(move |_: &mut Interp, _: &[Value]| {
            let from = sh.io.try_borrow().map_err(|_| busy())?.from.clone();
            let c = chan_of::<P>(&from).ok_or_else(|| Error::typ("no input channel"))?;
            let line = loop {
                let found = c.buf.try_borrow().map_err(|_| busy())?.iter().position(|&b| b == b'\n');
                if let Some(i) = found {
                    let mut buf = c.buf.try_borrow_mut().map_err(|_| busy())?;
                    let line = lossy(&buf[..i])?;
                    buf.drain(..=i);
                    break line;
                }
                if fill(&sh, c)? == 0 {
                    let mut buf = c.buf.try_borrow_mut().map_err(|_| busy())?;
                    if buf.is_empty() {
                        c.eof.set(true);
                    }
                    let line = lossy(&buf)?;
                    buf.clear();
                    break line;
                }
            };
            // KlongPy strips the line end with Python's rstrip, which takes all trailing blanks
            Value::text(line.trim_end())
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".r",
        0,
        Rc::new(move |ip: &mut Interp, _: &[Value]| {
            let from = sh.io.try_borrow().map_err(|_| busy())?.from.clone();
            let c = chan_of::<P>(&from).ok_or_else(|| Error::typ("no input channel"))?;
            // Read as much as the object needs and no more: an object may span lines, so more input is asked for while the text ends inside
            // it (or where an atom could go on); what follows the object stays in the channel, and in the host's stream when the host has not
            // been asked for it. At the end of input an object that is not closed is taken as the reader takes it.
            loop {
                let mut buf = c.buf.try_borrow_mut().map_err(|_| busy())?;
                // a chunk may end inside a character: read what is whole, unless nothing more can come
                let whole = match core::str::from_utf8(&buf) {
                    Ok(_) => buf.len(),
                    Err(e) if e.error_len().is_none() => e.valid_up_to(),
                    Err(_) => {
                        let fixed = lossy(&buf)?;
                        *buf = fixed.into_bytes();
                        buf.len()
                    }
                };
                let text = core::str::from_utf8(&buf[..whole]).unwrap_or("");
                let step = ip.read_datum_ex(text);
                drop(buf);
                let more = match &step {
                    Ok(Some((_, _, true))) => false,
                    _ => fill(&sh, c)? != 0,
                };
                if more {
                    continue;
                }
                let mut buf = c.buf.try_borrow_mut().map_err(|_| busy())?;
                return match step? {
                    Some((used, v, _)) => {
                        let used = used.min(buf.len());
                        buf.drain(..used);
                        Ok(v)
                    }
                    None => {
                        buf.clear();
                        c.eof.set(true);
                        // at the end of input: the value :undefined (as KlongPy; a function that returns nothing has no value, this has one)
                        Ok(Value::Undefined)
                    }
                };
            }
        }),
    );
    interp.register(
        ".rs",
        1,
        Rc::new(|ip: &mut Interp, a: &[Value]| {
            let text = klong_core::value::try_str(text_arg(a, ".rs")?)?;
            match ip.read_datum(&text)? {
                Some((_, v)) => Ok(v),
                None => Ok(Value::Undefined),
            }
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".fl",
        WILDCARD,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            // writes are not buffered by the runtime: there is nothing to flush, only to check
            let to = match a.first() {
                Some(v) => v.clone(),
                None => sh.io.try_borrow().map_err(|_| busy())?.to.clone(),
            };
            match chan_of::<P>(&to) {
                Some(c) if !c.input => Ok(ip.no_value()),
                Some(_) => Err(Error::typ("an input channel cannot be flushed")),
                None => Err(Error::typ(".fl needs an output channel")),
            }
        }),
    );
    let sh = shared.clone();
    interp.register(
        ".df",
        1,
        Rc::new(move |ip: &mut Interp, a: &[Value]| {
            let path = text_arg(a, ".df")?;
            let mut p = sh.platform.try_borrow_mut().map_err(|_| busy())?;
            let fs = p.fs().ok_or_else(no_fs)?;
            fs.remove(path).map_err(|e| fs_error(e, path))?;
            Ok(ip.no_value())
        }),
    );
}
