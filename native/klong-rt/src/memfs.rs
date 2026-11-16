//! A file system in memory: for tests, for fuzzing the file functions, and for a board that
//! wants `.l` and channels without storage. Allocation is fallible (`NoSpace`).
use alloc::string::String;
use alloc::vec::Vec;

use crate::platform::{FileId, FileSystem, FsError, Mode};

struct File {
    name: String,
    data: Vec<u8>,
}

struct Open {
    file: usize,
    pos: usize,
    write: bool,
    append: bool,
}

#[derive(Default)]
pub struct MemFs {
    files: Vec<Option<File>>,
    open: Vec<Option<Open>>,
    /// Total bytes the files may hold (0 = no limit).
    pub capacity: usize,
}

impl MemFs {
    pub fn new() -> MemFs {
        MemFs::default()
    }

    fn find(&self, path: &str) -> Option<usize> {
        self.files.iter().position(|f| matches!(f, Some(f) if f.name == path))
    }

    fn used(&self) -> usize {
        self.files.iter().flatten().map(|f| f.data.len()).sum()
    }

    /// Put a file in place (tests).
    pub fn put(&mut self, path: &str, data: &[u8]) -> Result<(), FsError> {
        let id = self.open(path, Mode::Write)?;
        let r = self.write(id, data).map(|_| ());
        self.close(id);
        r
    }

    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.find(path).and_then(|i| self.files[i].as_ref()).map(|f| &f.data[..])
    }

    fn handle(&mut self, f: FileId) -> Result<&mut Open, FsError> {
        self.open.get_mut(f.0 as usize).and_then(|o| o.as_mut()).ok_or(FsError::Invalid)
    }
}

impl FileSystem for MemFs {
    fn open(&mut self, path: &str, mode: Mode) -> Result<FileId, FsError> {
        if path.is_empty() {
            return Err(FsError::Invalid);
        }
        let file = match (self.find(path), mode) {
            (Some(i), _) => i,
            (None, Mode::Read) => return Err(FsError::NotFound),
            (None, _) => {
                let mut name = String::new();
                name.try_reserve_exact(path.len()).map_err(|_| FsError::NoSpace)?;
                name.push_str(path);
                let slot = match self.files.iter().position(|f| f.is_none()) {
                    Some(i) => i,
                    None => {
                        self.files.try_reserve(1).map_err(|_| FsError::NoSpace)?;
                        self.files.push(None);
                        self.files.len() - 1
                    }
                };
                self.files[slot] = Some(File { name, data: Vec::new() });
                slot
            }
        };
        if mode == Mode::Write {
            if let Some(f) = self.files[file].as_mut() {
                f.data = Vec::new();
            }
        }
        let o = Open { file, pos: 0, write: mode != Mode::Read, append: mode == Mode::Append };
        let slot = match self.open.iter().position(|o| o.is_none()) {
            Some(i) => i,
            None => {
                self.open.try_reserve(1).map_err(|_| FsError::Busy)?;
                self.open.push(None);
                self.open.len() - 1
            }
        };
        self.open[slot] = Some(o);
        Ok(FileId(slot as u32))
    }

    fn read(&mut self, f: FileId, buf: &mut [u8]) -> Result<usize, FsError> {
        let (file, pos) = {
            let o = self.handle(f)?;
            (o.file, o.pos)
        };
        let data = &self.files.get(file).and_then(|f| f.as_ref()).ok_or(FsError::Invalid)?.data;
        let n = buf.len().min(data.len().saturating_sub(pos));
        buf[..n].copy_from_slice(&data[pos..pos + n]);
        self.handle(f)?.pos = pos + n;
        Ok(n)
    }

    fn write(&mut self, f: FileId, buf: &[u8]) -> Result<usize, FsError> {
        let (file, pos, append) = {
            let o = self.handle(f)?;
            if !o.write {
                return Err(FsError::Denied);
            }
            (o.file, o.pos, o.append)
        };
        let used = self.used();
        let capacity = self.capacity;
        let data = &mut self.files.get_mut(file).and_then(|f| f.as_mut()).ok_or(FsError::Invalid)?.data;
        let pos = if append { data.len() } else { pos };
        let end = pos.checked_add(buf.len()).ok_or(FsError::NoSpace)?;
        if end > data.len() {
            let more = end - data.len();
            if capacity != 0 && used + more > capacity {
                return Err(FsError::NoSpace);
            }
            data.try_reserve(more).map_err(|_| FsError::NoSpace)?;
            // a gap left by seek reads as zero
            data.resize(end, 0);
        }
        data[pos..end].copy_from_slice(buf);
        self.handle(f)?.pos = end;
        Ok(buf.len())
    }

    fn seek(&mut self, f: FileId, pos: u64) -> Result<(), FsError> {
        let pos = usize::try_from(pos).map_err(|_| FsError::Invalid)?;
        self.handle(f)?.pos = pos;
        Ok(())
    }

    fn close(&mut self, f: FileId) {
        if let Some(o) = self.open.get_mut(f.0 as usize) {
            *o = None;
        }
    }

    fn remove(&mut self, path: &str) -> Result<(), FsError> {
        let i = self.find(path).ok_or(FsError::NotFound)?;
        self.files[i] = None;
        for o in self.open.iter_mut() {
            if matches!(o, Some(o) if o.file == i) {
                *o = None;
            }
        }
        Ok(())
    }

    fn rename(&mut self, from: &str, to: &str) -> Result<(), FsError> {
        let i = self.find(from).ok_or(FsError::NotFound)?;
        if to.is_empty() {
            return Err(FsError::Invalid);
        }
        let mut name = String::new();
        name.try_reserve_exact(to.len()).map_err(|_| FsError::NoSpace)?;
        name.push_str(to);
        if let Some(j) = self.find(to) {
            if j != i {
                self.files[j] = None;
            }
        }
        if let Some(f) = self.files[i].as_mut() {
            f.name = name;
        }
        Ok(())
    }

    fn list(&mut self, dir: &str, each: &mut dyn FnMut(&str, u64)) -> Result<(), FsError> {
        let prefix = dir.trim_end_matches('/');
        for f in self.files.iter().flatten() {
            let rest = if prefix.is_empty() || prefix == "." {
                Some(f.name.as_str())
            } else {
                f.name.strip_prefix(prefix).and_then(|r| r.strip_prefix('/'))
            };
            if let Some(r) = rest {
                if !r.contains('/') {
                    each(r, f.data.len() as u64);
                }
            }
        }
        Ok(())
    }
}
