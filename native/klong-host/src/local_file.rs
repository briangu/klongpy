//! Durable creation of local files shared by the binary and SQLite interfaces.
use klong_core::Error;
use std::{
    fs::{File, OpenOptions},
    io,
    path::Path,
};

pub(super) fn io_error(e: impl std::fmt::Display) -> Error {
    Error::system(e.to_string())
}

pub(super) fn sync_parent(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        })?
        .sync_all()?;
    }
    let _ = path;
    Ok(())
}

pub(super) fn options() -> OpenOptions {
    let mut o = OpenOptions::new();
    o.read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o
}

// Only open a new inode here. Closing an extra descriptor to an existing SQLite
// database releases this process's POSIX record locks.
pub(super) fn create_file(path: &Path) -> io::Result<Option<File>> {
    match options().create_new(true).open(path) {
        Ok(f) => {
            f.sync_all()?;
            sync_parent(path)?;
            Ok(Some(f))
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(None),
        Err(e) => Err(e),
    }
}
