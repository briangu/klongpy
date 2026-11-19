//! klong-tools: what reads Klong text outside the evaluator, in Rust. The Python package keeps no lexer, parser or evaluator of the
//! language; it asks these.
//!
//! ```text
//!     ast::ast_text(text, name)            the canonical parse tree (`kg ast`, `kgpy ast`)
//!     ast::ast_main(prog, args, ..)        the body of `kg ast`
//!     repl::check_text(text)               is a REPL entry complete, incomplete or in error
//!     repl::complete(line, names, cmds, ..) tab completion
//!     func::function_form(text)            is a text one function literal, as an IPC peer sends it
//! ```
//!
//! Offsets that cross to a caller are byte offsets here; the extension converts them to characters.
pub mod ast;
pub mod func;
pub mod repl;

/// The native stack of the threads that parse: the parser recurses.
pub const STACK_BYTES: usize = 256 << 20;

/// Run `f` on a thread with a stack of `STACK_BYTES`; a panic comes back as None.
pub fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    std::thread::Builder::new().stack_size(STACK_BYTES).spawn(f).ok()?.join().ok()
}

/// An error of the evaluator or its parser, in a form that can leave the thread that made it.
#[derive(Clone, Debug)]
pub struct ToolError {
    /// `syntax`, `range`, `memory`, ... (`klong_core::Kind::name`)
    pub kind: &'static str,
    pub message: String,
    /// Byte offsets into `source_text`.
    pub span: Option<(usize, usize)>,
    pub hint: Option<String>,
    pub source_name: String,
    pub source_text: String,
    /// As `klongpy.errors.KlongError.render` lays it out.
    pub rendered: String,
}

impl From<klong_core::Error> for ToolError {
    fn from(e: klong_core::Error) -> ToolError {
        let (source_name, source_text) = match &e.source {
            Some(s) => (s.name.clone(), s.text.to_string()),
            None => (String::new(), String::new()),
        };
        ToolError {
            kind: e.kind.name(),
            message: e.message.to_string(),
            span: e.span.map(|(a, b)| (a as usize, b as usize)),
            hint: e.hint.as_ref().map(|h| h.to_string()),
            source_name,
            source_text,
            rendered: e.render(),
        }
    }
}

impl ToolError {
    pub fn system(m: &str) -> ToolError {
        klong_core::Error::system(m.to_string()).into()
    }
}

/// A character offset for each byte offset, for texts that are not ASCII.
pub struct CharOffsets {
    starts: Option<Vec<usize>>,
}

impl CharOffsets {
    pub fn new(text: &str) -> CharOffsets {
        CharOffsets { starts: if text.is_ascii() { None } else { Some(text.char_indices().map(|(b, _)| b).collect()) } }
    }
    pub fn at(&self, byte: usize) -> usize {
        match &self.starts {
            None => byte,
            Some(s) => s.partition_point(|&b| b < byte),
        }
    }
}
