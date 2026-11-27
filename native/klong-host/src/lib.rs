//! klong-host: the `Platform` of desktop operating systems.
//!
//! Time, the console and files. TCP, pseudo-terminal links and the lossy
//! test link live in klong-rt and klong-ipc.
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use klong_rt::{FileId, FileSystem, FsError, Mode, Platform};

pub const DEFAULT_LOAD_PATH: &str = if cfg!(windows) { ".;lib" } else { ".:lib" };

pub fn load_path_env() -> String {
    std::env::var("KLONGPATH").ok().filter(|path| !path.is_empty()).unwrap_or_else(|| String::from(DEFAULT_LOAD_PATH))
}

pub fn parse_load_path(path: &str) -> impl Iterator<Item = String> + '_ {
    std::env::split_paths(path)
        .filter_map(|entry| entry.into_os_string().into_string().ok())
        .filter(|entry| !entry.is_empty())
        .map(|entry| module_path(&entry).into_owned())
}

/// Represent a host path with the separators used by the module search.
pub fn module_path(path: &str) -> std::borrow::Cow<'_, str> {
    #[cfg(windows)]
    if path.contains('\\') {
        return std::borrow::Cow::Owned(path.replace('\\', "/"));
    }
    std::borrow::Cow::Borrowed(path)
}

// Windows verbatim paths (from canonicalize) require native separators when passed to the OS.
pub fn os_path(path: &str) -> std::borrow::Cow<'_, str> {
    #[cfg(windows)]
    if path.starts_with("//?/") {
        return std::borrow::Cow::Owned(path.replace('/', "\\"));
    }
    std::borrow::Cow::Borrowed(path)
}

/// An absolute source path for the loader and desktop analysis tools.
pub fn absolute_module_path(path: &str) -> String {
    #[cfg(windows)]
    if let Ok(absolute) = std::path::absolute(os_path(path).as_ref()) {
        return module_path(&absolute.to_string_lossy()).into_owned();
    }
    if klong_core::module::is_absolute(path) {
        return String::from(path);
    }
    let dir = std::env::current_dir().map(|p| module_path(&p.to_string_lossy()).into_owned()).unwrap_or_default();
    klong_core::module::join(&dir, path)
}

#[cfg(feature = "duckdb")]
mod database;
#[cfg(feature = "binary-files")]
mod binary_files;
#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(any(feature = "binary-files", feature = "sqlite"))]
mod local_file;
pub mod net;

/// Where what a program prints with `.p .d .w` goes.
pub enum Console {
    Stdout,
    Stderr,
    /// Collected in memory (tests).
    Buffer(Vec<u8>),
}

pub struct HostPlatform {
    pub features: Vec<&'static str>,
    start: Instant,
    #[cfg(feature = "duckdb")]
    sql_idle: Option<std::rc::Rc<database::IdlePublication>>,
    pub console: Console,
    pub fs: Files,
    /// The networks `.cli` and `.srv` can use: `tcp`, and whatever a test adds.
    pub transports: Vec<Box<dyn klong_rt::link::Transport>>,
    /// Where console output goes when the embedder takes it: `(bytes, is_error)`, called at each write (`.p .d .w` and `.cerr`).
    pub sink: Option<Box<dyn FnMut(&[u8], bool)>>,
    /// Where the console channel's input comes from when the embedder supplies it: fills the buffer, blocks until there is input,
    /// returns 0 at the end of input. Used by `.rl .r .mi` only.
    pub source: Option<Box<dyn FnMut(&mut [u8]) -> usize>>,
}

/// Where the file functions go: the operating system, or a file system in memory (fuzzing
/// and tests must not touch real files).
pub enum Files {
    Host(HostFs),
    Mem(klong_rt::memfs::MemFs),
}

impl HostPlatform {
    pub fn new(console: Console) -> HostPlatform {
        let features = vec![
            #[cfg(feature = "duckdb")] "duckdb",
            #[cfg(feature = "binary-files")] "binary-files",
            #[cfg(feature = "sqlite")] "sqlite",
        ];
        HostPlatform { features, start: Instant::now(), #[cfg(feature = "duckdb")] sql_idle: None, console, fs: Files::Host(HostFs::default()), transports: vec![Box::new(net::TcpTransport::new())], sink: None, source: None }
    }

    pub fn enable_sql_idle(&mut self, trace: bool) {
        #[cfg(feature = "duckdb")]
        { self.sql_idle = Some(std::rc::Rc::new(database::IdlePublication::new(trace))); }
        #[cfg(not(feature = "duckdb"))]
        let _ = trace;
    }

    pub fn run_sql_idle(&mut self, ip: &klong_core::Interp, activity: bool) -> bool {
        #[cfg(feature = "duckdb")]
        if let Some(idle) = &self.sql_idle {
            if activity {
                idle.activity();
            } else {
                return idle.step(ip);
            }
        }
        let _ = (ip, activity);
        false
    }
}

/// The files of the operating system. Writes are not buffered here.
#[derive(Default)]
pub struct HostFs {
    open: Vec<Option<std::fs::File>>,
}

fn fs_error(e: &std::io::Error) -> FsError {
    match e.kind() {
        std::io::ErrorKind::NotFound => FsError::NotFound,
        std::io::ErrorKind::PermissionDenied => FsError::Denied,
        std::io::ErrorKind::InvalidInput => FsError::Invalid,
        _ => FsError::Io,
    }
}

impl HostFs {
    fn file(&mut self, f: FileId) -> Result<&mut std::fs::File, FsError> {
        self.open.get_mut(f.0 as usize).and_then(|o| o.as_mut()).ok_or(FsError::Invalid)
    }
}

impl FileSystem for HostFs {
    fn open(&mut self, path: &str, mode: Mode) -> Result<FileId, FsError> {
        let mut o = std::fs::OpenOptions::new();
        match mode {
            Mode::Read => o.read(true),
            Mode::Write => o.write(true).create(true).truncate(true),
            Mode::Append => o.append(true).create(true),
            Mode::Update => o.read(true).write(true).create(true),
        };
        let file = o.open(os_path(path).as_ref()).map_err(|e| fs_error(&e))?;
        if mode == Mode::Read && file.metadata().map(|m| m.is_dir()).unwrap_or(false) {
            return Err(FsError::Invalid);
        }
        let slot = match self.open.iter().position(|f| f.is_none()) {
            Some(i) => i,
            None => {
                self.open.push(None);
                self.open.len() - 1
            }
        };
        self.open[slot] = Some(file);
        Ok(FileId(slot as u32))
    }
    fn read(&mut self, f: FileId, buf: &mut [u8]) -> Result<usize, FsError> {
        self.file(f)?.read(buf).map_err(|e| fs_error(&e))
    }
    fn write(&mut self, f: FileId, buf: &[u8]) -> Result<usize, FsError> {
        self.file(f)?.write(buf).map_err(|e| fs_error(&e))
    }
    fn seek(&mut self, f: FileId, pos: u64) -> Result<(), FsError> {
        // beyond the end is allowed; the operating system fills the gap with zeros
        self.file(f)?.seek(SeekFrom::Start(pos)).map(|_| ()).map_err(|e| fs_error(&e))
    }
    fn close(&mut self, f: FileId) {
        if let Some(o) = self.open.get_mut(f.0 as usize) {
            *o = None;
        }
    }
    fn remove(&mut self, path: &str) -> Result<(), FsError> {
        std::fs::remove_file(path).map_err(|e| fs_error(&e))
    }
    fn rename(&mut self, from: &str, to: &str) -> Result<(), FsError> {
        std::fs::rename(from, to).map_err(|e| fs_error(&e))
    }
    fn list(&mut self, dir: &str, each: &mut dyn FnMut(&str, u64)) -> Result<(), FsError> {
        let dir = if dir.is_empty() { "." } else { dir };
        for entry in std::fs::read_dir(dir).map_err(|e| fs_error(&e))? {
            let entry = entry.map_err(|e| fs_error(&e))?;
            let meta = entry.metadata().map_err(|e| fs_error(&e))?;
            if meta.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    each(name, meta.len());
                }
            }
        }
        Ok(())
    }
}

impl Platform for HostPlatform {
    fn features(&self) -> &[&str] { &self.features }

    #[cfg(any(feature = "duckdb", feature = "binary-files", feature = "sqlite"))]
    fn register(&self, ip: &mut klong_core::Interp) {
        #[cfg(feature = "duckdb")]
        database::register(ip, self.sql_idle.clone());
        #[cfg(any(feature = "binary-files", feature = "sqlite"))]
        if matches!(self.fs, Files::Host(_)) {
            #[cfg(feature = "binary-files")]
            binary_files::register(ip);
            #[cfg(feature = "sqlite")]
            sqlite::register(ip);
        }
    }

    fn now_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
    fn wall_clock(&self) -> Option<f64> {
        SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
    }
    fn idle(&mut self) {}
    fn real_path(&self, path: &str) -> Option<String> {
        let p = if path.is_empty() { "." } else { path };
        std::fs::canonicalize(os_path(p).as_ref()).ok().and_then(|r| r.to_str().map(|s| self.module_path(s).into_owned()))
    }
    #[cfg(windows)]
    fn module_path<'a>(&self, path: &'a str) -> std::borrow::Cow<'a, str> {
        module_path(path)
    }
    fn console_read(&mut self, _buf: &mut [u8]) -> usize {
        // The host console is line-buffered standard input, read by the CLI between
        // evaluations; nothing is read while one runs (no Ctrl-C byte: the terminal sends a
        // signal instead).
        0
    }
    fn console_input(&mut self, buf: &mut [u8]) -> usize {
        match &mut self.source {
            Some(f) => f(buf),
            None => 0,
        }
    }
    fn console_write(&mut self, bytes: &[u8]) {
        if let Some(f) = &mut self.sink {
            return f(bytes, false);
        }
        match &mut self.console {
            Console::Stdout => {
                let _ = std::io::stdout().write_all(bytes);
            }
            Console::Stderr => {
                let _ = std::io::stderr().write_all(bytes);
            }
            Console::Buffer(b) => b.extend_from_slice(bytes),
        }
    }
    fn console_error(&mut self, bytes: &[u8]) {
        if let Some(f) = &mut self.sink {
            return f(bytes, true);
        }
        match &mut self.console {
            Console::Buffer(b) => b.extend_from_slice(bytes),
            _ => {
                let _ = std::io::stderr().write_all(bytes);
            }
        }
    }
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        Some(match &mut self.fs {
            Files::Host(f) => f,
            Files::Mem(f) => f,
        })
    }
    fn name(&self) -> &str {
        "host"
    }
    fn cwd(&self) -> Option<String> {
        std::env::current_dir().ok().map(|p| self.module_path(&p.to_string_lossy()).into_owned())
    }
    fn transports(&mut self) -> &mut [Box<dyn klong_rt::link::Transport>] {
        &mut self.transports
    }
    fn wait(&mut self, deadline_us: Option<u64>) {
        let now = self.now_us();
        let us = deadline_us.map_or(1000, |d| d.saturating_sub(now)).min(5000);
        if us > 0 {
            if let [transport] = self.transports.as_mut_slice() {
                if transport.wait_ready(us) {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_micros(us));
        }
    }
    fn exit(&mut self, code: i32) {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        std::process::exit(code);
    }
}
