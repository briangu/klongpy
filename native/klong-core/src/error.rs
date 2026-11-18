//! Errors: the kinds of `docs/errors.md`, and their rendering.
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use core::fmt::Write;

/// `[start, end)` byte offsets into a source text.
pub type Span = (u32, u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Syntax,
    Name,
    Arity,
    Type,
    Shape,
    Range,
    System,
    Memory,
    Interrupt,
    /// Always a bug in this runtime.
    Internal,
    /// An error reported by an IPC peer in words that are not one of the kinds above (a kdb+
    /// signal such as `length`, or free text).
    Remote,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Syntax => "syntax",
            Kind::Name => "name",
            Kind::Arity => "arity",
            Kind::Type => "type",
            Kind::Shape => "shape",
            Kind::Range => "range",
            Kind::System => "system",
            Kind::Memory => "memory",
            Kind::Interrupt => "interrupt",
            Kind::Internal => "internal",
            Kind::Remote => "remote",
        }
    }
}

/// A piece of program text and where it came from.
pub struct Source {
    pub name: String,
    /// Owned, or (a library in the image's flash) borrowed: then no copy of it is made in RAM.
    pub text: Cow<'static, str>,
}

impl Source {
    /// 1-based line and column (in bytes) of an offset.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let off = (offset as usize).min(self.text.len());
        let before = &self.text.as_bytes()[..off];
        let line = before.iter().filter(|&&b| b == b'\n').count() as u32 + 1;
        let start = before.iter().rposition(|&b| b == b'\n').map(|p| p + 1).unwrap_or(0);
        (line, (off - start) as u32 + 1)
    }

    pub fn line_text(&self, line: u32) -> &str {
        self.text.split('\n').nth(line.saturating_sub(1) as usize).unwrap_or("")
    }
}

pub struct ErrorData {
    pub kind: Kind,
    pub message: Cow<'static, str>,
    pub span: Option<Span>,
    pub source: Option<Rc<Source>>,
    pub hint: Option<Cow<'static, str>>,
    /// Raised by an operator on operands it does not take (what NumPy's own exceptions are in KlongPy): the operator's application
    /// rewrites the message to name the whole operands (`cannot apply + (Plus) to list of 3 and list of 2`).
    pub generic: bool,
}

/// A Klong error. One pointer wide, so that `Result<Value, Error>` stays small.
pub struct Error(Box<ErrorData>);

impl core::ops::Deref for Error {
    type Target = ErrorData;
    fn deref(&self) -> &ErrorData {
        &self.0
    }
}

impl core::ops::DerefMut for Error {
    fn deref_mut(&mut self) -> &mut ErrorData {
        &mut self.0
    }
}

impl Error {
    pub fn new(kind: Kind, message: impl Into<Cow<'static, str>>) -> Error {
        Error(Box::new(ErrorData { kind, message: message.into(), span: None, source: None, hint: None, generic: false }))
    }
    pub fn with_span(mut self, span: Span) -> Error {
        if self.0.span.is_none() {
            self.0.span = Some(span);
        }
        self
    }
    pub fn generic(mut self) -> Error {
        self.0.generic = true;
        self
    }
    pub fn with_hint(mut self, hint: impl Into<Cow<'static, str>>) -> Error {
        self.0.hint = Some(hint.into());
        self
    }
    pub fn syntax(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Syntax, m)
    }
    pub fn name(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Name, m)
    }
    pub fn arity(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Arity, m)
    }
    pub fn typ(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Type, m)
    }
    pub fn shape(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Shape, m)
    }
    pub fn range(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Range, m)
    }
    pub fn system(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::System, m)
    }
    /// Allocation failed. The message is static: nothing is allocated for it beyond the box.
    pub fn memory() -> Error {
        Error::new(Kind::Memory, "out of memory")
    }
    pub fn interrupt() -> Error {
        Error::new(Kind::Interrupt, "interrupted")
    }
    pub fn internal(m: impl Into<Cow<'static, str>>) -> Error {
        Error::new(Kind::Internal, m)
    }

    /// The text of docs/errors.md:
    ///
    /// ```text
    /// name:line:col: <kind> error: <message>
    ///   <source line>
    ///       ^
    ///   hint: ...
    /// ```
    pub fn render(&self) -> String {
        let mut s = String::new();
        let located = match (&self.source, self.span) {
            (Some(src), Some(span)) => Some((src, span)),
            _ => None,
        };
        if let Some((src, span)) = located {
            let (line, col) = src.line_col(span.0);
            if !src.name.is_empty() && !src.name.starts_with('<') {
                let _ = write!(s, "{}:", src.name);
            }
            let _ = write!(s, "{}:{}: ", line, col);
        }
        let _ = write!(s, "{} error: {}", self.kind.name(), self.message);
        if let Some((src, span)) = located {
            let (line, col) = src.line_col(span.0);
            let text = src.line_text(line);
            let shown: String = text.chars().take(120).map(|c| if c == '\t' { ' ' } else { c }).collect();
            let width = (span.1.saturating_sub(span.0)).max(1) as usize;
            let col0 = (col as usize - 1).min(shown.len());
            let carets = width.min(shown.len().saturating_sub(col0)).max(1);
            let _ = write!(s, "\n  {}\n  {}{}", shown, " ".repeat(col0), "^".repeat(carets));
        }
        if let Some(h) = &self.hint {
            let _ = write!(s, "\n  hint: {}", h);
        }
        s
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} error: {}", self.kind.name(), self.message)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// The longest message text [`msg!`] builds, in bytes.
pub const MSG_MAX: usize = 300;

/// The text of a message under construction (see [`msg!`]).
pub struct Msg {
    text: String,
    full: bool,
}

impl Msg {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Msg {
        Msg { text: String::new(), full: false }
    }
    pub fn finish(self) -> String {
        self.text
    }
}

impl core::fmt::Write for Msg {
    fn write_str(&mut self, t: &str) -> core::fmt::Result {
        if self.full {
            return Ok(());
        }
        let room = MSG_MAX - self.text.len().min(MSG_MAX);
        let mut n = t.len().min(room);
        while !t.is_char_boundary(n) {
            n -= 1;
        }
        let cut = n < t.len();
        if self.text.try_reserve(n + 3).is_err() {
            self.full = true;
            return Ok(());
        }
        self.text.push_str(&t[..n]);
        if cut {
            self.text.push_str("...");
            self.full = true;
        }
        Ok(())
    }
}

/// `format!` for the text of an error message: a message quotes names, tokens, paths and
/// templates, which are as long as the program makes them, so the text is cut at [`MSG_MAX`]
/// bytes and never asks the allocator for more (a message that cannot be stored at all is
/// empty; the kind of the error and its position remain).
#[macro_export]
macro_rules! msg {
    ($($arg:tt)*) => {{
        let mut m = $crate::error::Msg::new();
        let _ = core::fmt::Write::write_fmt(&mut m, format_args!($($arg)*));
        m.finish()
    }};
}
